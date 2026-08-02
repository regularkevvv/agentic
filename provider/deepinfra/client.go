package deepinfra

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

const defaultBaseURL = "https://api.deepinfra.com/v1"

// maxRetryAfter caps how long a server-supplied Retry-After header can park a
// request. Honoring a multi-minute hint verbatim would strand a caller inside
// a single Encode call for longer than most request deadlines allow, so a
// longer hint is clamped and the attempt is retried sooner.
const maxRetryAfter = 30 * time.Second

// defaultMaxResponseBytes bounds a decoded response body.
//
// Multi-vector output makes a small request into a very large response: a
// batch of long documents at 1024 dimensions per token is tens of millions of
// floats, and reading that without a ceiling is a straightforward way to
// exhaust a process's memory from a remote body. 64 MiB is comfortably above
// a sane batch and far below anything that threatens a server.
const defaultMaxResponseBytes = 64 << 20

// maxDetailLength bounds the provider message carried in an APIError.
const maxDetailLength = 200

// client is the DeepInfra HTTP transport.
type client struct {
	apiKey           string
	baseURL          string
	httpClient       *http.Client
	maxRetries       int
	maxResponseBytes int64

	// backoff computes the wait before the next attempt. It is a field so
	// that transport tests can drive the retry paths without sleeping.
	backoff func(attempt int, header http.Header) time.Duration
}

type transportConfig struct {
	apiKey           string
	baseURL          string
	httpClient       *http.Client
	maxRetries       *int
	maxResponseBytes *int64
}

func newClient(cfg transportConfig) (*client, error) {
	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("DEEPINFRA_TOKEN")
	}
	if apiKey == "" {
		return nil, errors.New("deepinfra: API token not set (use WithAPIToken or the DEEPINFRA_TOKEN env var)")
	}

	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	baseURL = strings.TrimSuffix(baseURL, "/")

	httpClient := cfg.httpClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}

	maxRetries := 2
	if cfg.maxRetries != nil {
		if *cfg.maxRetries < 0 {
			return nil, errors.New("deepinfra: max retries must be non-negative")
		}
		maxRetries = *cfg.maxRetries
	}

	maxResponseBytes := int64(defaultMaxResponseBytes)
	if cfg.maxResponseBytes != nil {
		if *cfg.maxResponseBytes <= 0 {
			return nil, errors.New("deepinfra: max response bytes must be positive")
		}
		maxResponseBytes = *cfg.maxResponseBytes
	}

	return &client{
		apiKey:           apiKey,
		baseURL:          baseURL,
		httpClient:       httpClient,
		maxRetries:       maxRetries,
		maxResponseBytes: maxResponseBytes,
		backoff:          retryWait,
	}, nil
}

// APIError reports a non-200 response from DeepInfra.
//
// It deliberately does not carry the raw response body. A provider's
// validation error quotes the offending input back at you, and an error that
// may contain a user's document cannot be logged freely. Only the status, the
// request ID, and DeepInfra's own bounded "detail" message survive.
type APIError struct {
	// Status is the HTTP status code.
	Status int

	// RequestID is DeepInfra's request identifier when the response carried
	// one, for correlating with provider-side logs.
	RequestID string

	// Detail is the provider's error message, bounded in length. It is empty
	// when the body was not a recognizable error envelope.
	Detail string
}

func (e *APIError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "deepinfra: status %d", e.Status)
	if e.RequestID != "" {
		fmt.Fprintf(&b, " (request %s)", e.RequestID)
	}
	if e.Detail != "" {
		fmt.Fprintf(&b, ": %s", e.Detail)
	}
	return b.String()
}

// post sends body as JSON to path and returns the raw 200 response body,
// retrying bounded 429 and transient 5xx responses. Ordinary 4xx contract and
// authentication failures are not retried: they will fail identically on the
// next attempt and only cost another request.
func (c *client) post(ctx context.Context, path string, body any) ([]byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	attempts := c.maxRetries + 1
	var lastErr error
	for attempt := range attempts {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")

		httpResp, err := c.httpClient.Do(httpReq)
		if err != nil {
			// A canceled or expired context is returned with its own cause:
			// a caller's shutdown must not be indistinguishable from a
			// transient outage, or a retry wrapper above will treat it as one.
			if ctx.Err() != nil {
				return nil, err
			}
			if attempt == attempts-1 {
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
			return nil, fmt.Errorf("deepinfra: response exceeds the %d byte limit", c.maxResponseBytes)
		}

		if httpResp.StatusCode != http.StatusOK {
			apiErr := newAPIError(httpResp.StatusCode, payload)
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

// newAPIError extracts the safe parts of an error body. DeepInfra answers a
// rejected request with FastAPI's {"detail": ...} envelope; anything else is
// reported by status alone.
func newAPIError(status int, payload []byte) *APIError {
	apiErr := &APIError{Status: status}

	var envelope struct {
		Detail    json.RawMessage `json:"detail"`
		Error     string          `json:"error"`
		RequestID string          `json:"request_id"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return apiErr
	}
	apiErr.RequestID = truncate(envelope.RequestID, maxDetailLength)

	switch {
	case len(envelope.Detail) > 0:
		var text string
		if err := json.Unmarshal(envelope.Detail, &text); err == nil {
			apiErr.Detail = truncate(text, maxDetailLength)
		} else {
			apiErr.Detail = truncate(string(envelope.Detail), maxDetailLength)
		}
	case envelope.Error != "":
		apiErr.Detail = truncate(envelope.Error, maxDetailLength)
	}
	return apiErr
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
// Retry-After header replaces the exponential backoff — the server knows when
// its own window reopens better than a doubling counter does — and the result
// is jittered either way so that a fleet of clients throttled by one burst
// does not resynchronize onto the same retry instant.
func retryWait(attempt int, header http.Header) time.Duration {
	if header != nil {
		if delay, ok := parseRetryAfter(header.Get("Retry-After"), time.Now()); ok {
			return withJitter(delay)
		}
	}
	return withJitter(retryDelay(attempt))
}

// parseRetryAfter interprets a Retry-After header in either of its RFC 9110
// forms — delay in seconds, or an HTTP-date — relative to now. It reports
// false for an absent or malformed value. A date already in the past yields a
// zero delay rather than a negative one, and any value above maxRetryAfter is
// clamped.
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

// sleepRetry waits for delay, returning early with the context's error if it
// is canceled first. A retry must never sleep past the caller's deadline.
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
