package huggingface

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// defaultRouterURL is the Inference Providers router, which serves the shared
// hosted models.
const defaultRouterURL = "https://router.huggingface.co"

// maxRetryAfter caps how long a server-supplied Retry-After header can park a
// request inside a single Encode call.
const maxRetryAfter = 30 * time.Second

// defaultMaxResponseBytes bounds a decoded response body. A multi-vector
// response over a batch of long documents is tens of millions of floats, and
// reading one without a ceiling is a way to exhaust memory from a remote body.
const defaultMaxResponseBytes = 64 << 20

// maxDetailLength bounds the provider message carried in an APIError.
const maxDetailLength = 200

// client is the shared Hugging Face HTTP transport. The dedicated and shared
// encoders each embed one, so authentication, retry, and bounds are defined
// once and behave identically on both routes.
type client struct {
	token            string
	httpClient       *http.Client
	maxRetries       int
	maxResponseBytes int64

	// backoff computes the wait before the next attempt. It is a field so
	// transport tests can drive the retry paths without sleeping.
	backoff func(attempt int, header http.Header) time.Duration
}

type transportConfig struct {
	token            string
	httpClient       *http.Client
	maxRetries       *int
	maxResponseBytes *int64
}

// newClient resolves a transportConfig, applying the HF_TOKEN fallback and the
// package defaults.
func newClient(cfg transportConfig) (*client, error) {
	token := cfg.token
	if token == "" {
		token = os.Getenv("HF_TOKEN")
	}
	if token == "" {
		token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	if token == "" {
		return nil, errors.New("huggingface: token not set (use WithToken or the HF_TOKEN env var)")
	}

	httpClient := cfg.httpClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}

	maxRetries := 2
	if cfg.maxRetries != nil {
		if *cfg.maxRetries < 0 {
			return nil, errors.New("huggingface: max retries must be non-negative")
		}
		maxRetries = *cfg.maxRetries
	}

	maxResponseBytes := int64(defaultMaxResponseBytes)
	if cfg.maxResponseBytes != nil {
		if *cfg.maxResponseBytes <= 0 {
			return nil, errors.New("huggingface: max response bytes must be positive")
		}
		maxResponseBytes = *cfg.maxResponseBytes
	}

	return &client{
		token:            token,
		httpClient:       httpClient,
		maxRetries:       maxRetries,
		maxResponseBytes: maxResponseBytes,
		backoff:          retryWait,
	}, nil
}

// APIError reports a non-200 response from a Hugging Face endpoint.
//
// It carries the status and Hugging Face's own bounded error message, never
// the raw body: a rejected request's error quotes the offending input back at
// you, and an error that may contain a user's document cannot be logged
// freely.
type APIError struct {
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
	fmt.Fprintf(&b, "huggingface: status %d", e.Status)
	if e.Endpoint != "" {
		fmt.Fprintf(&b, " from %s", e.Endpoint)
	}
	if e.Detail != "" {
		fmt.Fprintf(&b, ": %s", e.Detail)
	}
	return b.String()
}

// post sends body as JSON to url and returns the raw 200 response body,
// retrying bounded 429 and transient 5xx responses. A 503 is included because
// Hugging Face answers a cold model with one while it loads.
func (c *client) post(ctx context.Context, url string, body any) ([]byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	attempts := c.maxRetries + 1
	var lastErr error
	for attempt := range attempts {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Authorization", "Bearer "+c.token)
		httpReq.Header.Set("Content-Type", "application/json")

		httpResp, err := c.httpClient.Do(httpReq)
		if err != nil {
			// A canceled or expired context keeps its own cause: a caller's
			// shutdown must not look like a transient outage to a retry
			// wrapper further up.
			if ctx.Err() != nil || attempt == attempts-1 {
				return nil, err
			}
			lastErr = err
			if err := sleepRetry(ctx, c.backoff(attempt, nil)); err != nil {
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
			return nil, fmt.Errorf("huggingface: response exceeds the %d byte limit", c.maxResponseBytes)
		}

		if httpResp.StatusCode != http.StatusOK {
			apiErr := newAPIError(httpResp.StatusCode, url, payload)
			if !retryableStatus(httpResp.StatusCode) || attempt == attempts-1 {
				return nil, apiErr
			}
			lastErr = apiErr
			if err := sleepRetry(ctx, c.backoff(attempt, httpResp.Header)); err != nil {
				return nil, err
			}
			continue
		}

		return payload, nil
	}
	return nil, lastErr
}

// newAPIError extracts the safe parts of an error body. Hugging Face answers a
// rejected request with {"error": ...}; anything else is reported by status
// alone.
func newAPIError(status int, url string, payload []byte) *APIError {
	apiErr := &APIError{Status: status, Endpoint: stripQuery(url)}

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
