package pinecone

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

const defaultBaseURL = "https://api.pinecone.io"

// defaultAPIVersion pins the dated API contract this package was written
// against. Pinecone versions its API by date header, so pinning is what keeps
// a response shape from changing under a deployment without warning.
const defaultAPIVersion = "2025-04"

// maxRetryAfter caps how long a server-supplied Retry-After header can park a
// request inside a single Encode call.
const maxRetryAfter = 30 * time.Second

// defaultMaxResponseBytes bounds a decoded response body.
const defaultMaxResponseBytes = 32 << 20

// maxDetailLength bounds the provider message carried in an APIError.
const maxDetailLength = 200

type client struct {
	apiKey           string
	baseURL          string
	apiVersion       string
	httpClient       *http.Client
	maxRetries       int
	maxResponseBytes int64

	// backoff computes the wait before the next attempt. It is a field so
	// transport tests can drive the retry paths without sleeping.
	backoff func(attempt int, header http.Header) time.Duration
}

type transportConfig struct {
	apiKey           string
	baseURL          string
	apiVersion       string
	httpClient       *http.Client
	maxRetries       *int
	maxResponseBytes *int64
}

func newClient(cfg transportConfig) (*client, error) {
	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("PINECONE_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New("pinecone: API key not set (use WithAPIKey or the PINECONE_API_KEY env var)")
	}

	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	apiVersion := cfg.apiVersion
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}

	httpClient := cfg.httpClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}

	maxRetries := 2
	if cfg.maxRetries != nil {
		if *cfg.maxRetries < 0 {
			return nil, errors.New("pinecone: max retries must be non-negative")
		}
		maxRetries = *cfg.maxRetries
	}

	maxResponseBytes := int64(defaultMaxResponseBytes)
	if cfg.maxResponseBytes != nil {
		if *cfg.maxResponseBytes <= 0 {
			return nil, errors.New("pinecone: max response bytes must be positive")
		}
		maxResponseBytes = *cfg.maxResponseBytes
	}

	return &client{
		apiKey:           apiKey,
		baseURL:          strings.TrimSuffix(baseURL, "/"),
		apiVersion:       apiVersion,
		httpClient:       httpClient,
		maxRetries:       maxRetries,
		maxResponseBytes: maxResponseBytes,
		backoff:          retryWait,
	}, nil
}

// APIError reports a non-200 response from the Pinecone Inference API.
//
// It carries the status and Pinecone's own bounded error code and message,
// never the raw body: a rejected request's error quotes the offending input
// back at you, and an error that may contain a user's document cannot be
// logged freely.
type APIError struct {
	// Status is the HTTP status code.
	Status int

	// Code is Pinecone's machine-readable error code, when the body carried
	// one.
	Code string

	// Detail is Pinecone's error message, bounded in length.
	Detail string
}

func (e *APIError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "pinecone: status %d", e.Status)
	if e.Code != "" {
		fmt.Fprintf(&b, " (%s)", e.Code)
	}
	if e.Detail != "" {
		fmt.Fprintf(&b, ": %s", e.Detail)
	}
	return b.String()
}

// post sends body as JSON to path and returns the raw 200 response body,
// retrying bounded 429 and transient 5xx responses.
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
		httpReq.Header.Set("Api-Key", c.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("X-Pinecone-Api-Version", c.apiVersion)

		httpResp, err := c.httpClient.Do(httpReq)
		if err != nil {
			// A canceled or expired context keeps its own cause, so a caller's
			// shutdown is not retried as a transient outage.
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
			return nil, fmt.Errorf("pinecone: response exceeds the %d byte limit", c.maxResponseBytes)
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

// newAPIError extracts the safe parts of an error body. Pinecone answers a
// rejected request with {"error": {"code": ..., "message": ...}}; the details
// array is deliberately dropped, because it names the offending field and
// quotes its value.
func newAPIError(status int, payload []byte) *APIError {
	apiErr := &APIError{Status: status}

	var envelope struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return apiErr
	}
	apiErr.Code = truncate(envelope.Error.Code, maxDetailLength)
	switch {
	case envelope.Error.Message != "":
		apiErr.Detail = truncate(envelope.Error.Message, maxDetailLength)
	case envelope.Message != "":
		apiErr.Detail = truncate(envelope.Message, maxDetailLength)
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

// retryWait returns how long to wait before the next attempt, honoring a valid
// Retry-After header and jittering either way.
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
