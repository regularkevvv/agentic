// Package providerhttp carries the JSON-over-HTTP transport shared by the
// providers that POST a JSON body and read one back: bearer authentication, a
// bounded retry on 429 and transient 5xx responses, a bounded read of the
// response body, and an error that reports a failure without quoting the
// request back.
//
// It resolves no credentials of its own. Which environment variable a provider
// reads, and whether an absent token is an error at all, differs between a
// vendor's client and a generic one, so it stays in the calling package.
package providerhttp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// maxRetryAfter caps how long a server-supplied Retry-After header can park a
// request inside a single call.
const maxRetryAfter = 30 * time.Second

// defaultMaxResponseBytes bounds a decoded response body. A multi-vector
// response over a batch of long documents is tens of millions of floats, and
// reading one without a ceiling is a way to exhaust memory from a remote body.
const defaultMaxResponseBytes = 64 << 20

// maxDetailLength bounds the provider message carried in an APIError.
const maxDetailLength = 200

// Config is the transport half of a provider's configuration. A provider
// embeds it in its own config struct, so a caller's option writes the field
// this package reads.
type Config struct {
	// Token is sent as a bearer credential. An empty Token sends no
	// Authorization header at all, so a provider whose endpoints require one
	// must reject the empty case in its constructor rather than let it become
	// an anonymous request.
	Token string

	// HTTPClient replaces the default client, for proxies, instrumentation, or
	// tests.
	HTTPClient *http.Client

	// MaxRetries and MaxResponseBytes are pointers so that an unset option is
	// distinguishable from a deliberate zero, which is invalid for the second
	// and meaningful for the first.
	MaxRetries       *int
	MaxResponseBytes *int64
}

// Client is one provider's HTTP transport.
type Client struct {
	provider         string
	token            string
	httpClient       *http.Client
	maxRetries       int
	maxResponseBytes int64

	// Backoff computes the wait before the next attempt. It is a field so that
	// a provider's transport tests can drive the retry paths without sleeping.
	Backoff func(attempt int, header http.Header) time.Duration
}

// New resolves cfg against the package defaults.
//
// provider prefixes every error this client produces and must be the calling
// package's name: the caller of a failing Encode knows which constructor they
// used and nothing else about which transport ran.
func New(provider string, cfg Config) (*Client, error) {
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}

	maxRetries := 2
	if cfg.MaxRetries != nil {
		if *cfg.MaxRetries < 0 {
			return nil, errors.New(provider + ": max retries must be non-negative")
		}
		maxRetries = *cfg.MaxRetries
	}

	maxResponseBytes := int64(defaultMaxResponseBytes)
	if cfg.MaxResponseBytes != nil {
		if *cfg.MaxResponseBytes <= 0 {
			return nil, errors.New(provider + ": max response bytes must be positive")
		}
		maxResponseBytes = *cfg.MaxResponseBytes
	}

	return &Client{
		provider:         provider,
		token:            cfg.Token,
		httpClient:       httpClient,
		maxRetries:       maxRetries,
		maxResponseBytes: maxResponseBytes,
		Backoff:          retryWait,
	}, nil
}

// APIError reports a non-200 response from a provider.
//
// It carries the status and the provider's own bounded error message, never
// the raw body: a rejected request's error quotes the offending input back at
// you, and an error that may contain a user's document cannot be logged
// freely.
type APIError struct {
	// Provider names the client that produced this error. Providers sharing
	// this transport share the type, so a program using more than one reads
	// this field rather than distinguishing them by type assertion.
	Provider string

	// Status is the HTTP status code.
	Status int

	// Endpoint is the URL that was called, with any query string removed.
	Endpoint string

	// Detail is the provider's error message, bounded in length. It is empty
	// when the body was not a recognizable error envelope.
	Detail string
}

func (e *APIError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: status %d", e.Provider, e.Status)
	if e.Endpoint != "" {
		fmt.Fprintf(&b, " from %s", e.Endpoint)
	}
	if e.Detail != "" {
		fmt.Fprintf(&b, ": %s", e.Detail)
	}
	return b.String()
}

// Post sends body as JSON to url and returns the raw 200 response body,
// retrying bounded 429 and transient 5xx responses. A 503 is retried because a
// cold model answers with one while it loads, which is a wait rather than a
// failure.
func (c *Client) Post(ctx context.Context, url string, body any) ([]byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	// The loop carries no bound of its own: every attempt either returns or
	// waits for the next one, and the attempt numbered lastAttempt returns
	// whatever it got. A bounded loop would need a trailing return that no
	// input can reach.
	lastAttempt := c.maxRetries
	for attempt := 0; ; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		// No token means no header: an anonymous request is a deliberate
		// configuration, decided by the constructor rather than here.
		if c.token != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.token)
		}
		httpReq.Header.Set("Content-Type", "application/json")

		httpResp, err := c.httpClient.Do(httpReq)
		if err != nil {
			// A canceled or expired context keeps its own cause: a caller's
			// shutdown must not look like a transient outage to a retry
			// wrapper further up.
			if ctx.Err() != nil || attempt == lastAttempt {
				return nil, err
			}
			if err := sleepRetry(ctx, c.Backoff(attempt, nil)); err != nil {
				return nil, err
			}
			continue
		}

		payload, readErr := io.ReadAll(io.LimitReader(httpResp.Body, c.maxResponseBytes+1))
		closeErr := httpResp.Body.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		if int64(len(payload)) > c.maxResponseBytes {
			return nil, fmt.Errorf("%s: response exceeds the %d byte limit", c.provider, c.maxResponseBytes)
		}

		if httpResp.StatusCode != http.StatusOK {
			apiErr := newAPIError(c.provider, httpResp.StatusCode, url, payload)
			if !retryableStatus(httpResp.StatusCode) || attempt == lastAttempt {
				return nil, apiErr
			}
			if err := sleepRetry(ctx, c.Backoff(attempt, httpResp.Header)); err != nil {
				return nil, err
			}
			continue
		}

		return payload, nil
	}
}

// newAPIError extracts the safe parts of an error body. A rejected request is
// answered with {"error": ...} or {"detail": ...} by every provider using this
// transport; anything else is reported by status alone.
func newAPIError(provider string, status int, url string, payload []byte) *APIError {
	apiErr := &APIError{Provider: provider, Status: status, Endpoint: stripQuery(url)}

	var envelope struct {
		Error  json.RawMessage `json:"error"`
		Detail json.RawMessage `json:"detail"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return apiErr
	}
	for _, field := range []json.RawMessage{envelope.Error, envelope.Detail} {
		if len(field) == 0 {
			continue
		}
		var text string
		if err := json.Unmarshal(field, &text); err == nil {
			apiErr.Detail = truncate(text, maxDetailLength)
		} else {
			apiErr.Detail = truncate(string(field), maxDetailLength)
		}
		break
	}
	return apiErr
}

// stripQuery removes any query string, which on some deployments carries a
// signed access parameter.
func stripQuery(url string) string {
	if i := strings.IndexByte(url, '?'); i >= 0 {
		return url[:i]
	}
	return url
}

func truncate(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return s[:limit] + "..."
}

func retryableStatus(status int) bool {
	return status == http.StatusTooManyRequests ||
		status == http.StatusInternalServerError ||
		status == http.StatusBadGateway ||
		status == http.StatusServiceUnavailable ||
		status == http.StatusGatewayTimeout
}

// retryWait returns how long to wait before the next attempt. A valid
// Retry-After header replaces the exponential backoff, and the result is
// jittered either way so that clients throttled by one burst do not retry in
// lockstep.
func retryWait(attempt int, header http.Header) time.Duration {
	if header != nil {
		if delay, ok := parseRetryAfter(header.Get("Retry-After"), time.Now()); ok {
			return withJitter(delay)
		}
	}
	return withJitter(retryDelay(attempt))
}

// parseRetryAfter interprets a Retry-After header in either of its RFC 9110
// forms — delay in seconds, or an HTTP-date — relative to now.
func parseRetryAfter(value string, now time.Time) (time.Duration, bool) {
	if value == "" {
		return 0, false
	}

	var delay time.Duration
	if seconds, err := strconv.Atoi(value); err == nil {
		if seconds > int(maxRetryAfter/time.Second) {
			return maxRetryAfter, true
		}
		delay = time.Duration(seconds) * time.Second
	} else if deadline, err := http.ParseTime(value); err == nil {
		delay = deadline.Sub(now)
	} else {
		return 0, false
	}

	if delay < 0 {
		return 0, true
	}
	if delay > maxRetryAfter {
		return maxRetryAfter, true
	}
	return delay, true
}

func retryDelay(attempt int) time.Duration {
	delay := time.Duration(1<<attempt) * time.Second
	if delay > 4*time.Second {
		return 4 * time.Second
	}
	return delay
}

func withJitter(d time.Duration) time.Duration {
	if d <= 0 {
		return 0
	}
	return d + rand.N(d/4+1)
}

func sleepRetry(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
