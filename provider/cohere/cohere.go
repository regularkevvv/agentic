// Package cohere provides Cohere Embedder and Reranker implementations for
// Agentic.
//
// Cohere offers retrieval-tuned embedding models with a required input-type
// parameter, and a cross-encoder reranking model. This package implements
// retrieval.Embedder (POST /v2/embed) and retrieval.Reranker (POST /v2/rerank). It does
// not implement a chat model.
//
// The HTTP client is hand-rolled rather than taken from the Cohere Go SDK, so
// this package adds no module dependencies. Both the Embedder and the Reranker
// share one unexported client, so authentication, retry and backoff behave
// identically across the two APIs.
package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// defaultBaseURL is the Cohere API root. Unlike Voyage AI, Cohere carries the
// API version in the request path (/v2/embed, /v2/rerank) rather than in the
// base URL, so the version is appended by each endpoint and not baked in here.
const defaultBaseURL = "https://api.cohere.com"

// defaultTimeout bounds a single HTTP attempt.
const defaultTimeout = 120 * time.Second

// defaultMaxRetries matches the SDK-backed providers in this repo.
const defaultMaxRetries = 2

// client holds the transport state shared by the Embedder and the Reranker.
//
// Nothing on it is endpoint-specific: post reads only apiKey, baseURL,
// httpClient and maxRetries, which is why one instance can serve both /v2/embed
// and /v2/rerank without either constructor duplicating the plumbing.
type client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	maxRetries int
}

// newClient resolves the transport settings shared by both constructors,
// falling back to the CO_API_KEY and COHERE_API_KEY environment variables when
// no key is supplied. Cohere's own tooling reads CO_API_KEY, so it is tried
// first, with COHERE_API_KEY accepted as the more explicit spelling.
func newClient(apiKey, baseURL string, httpClient *http.Client, maxRetries *int) (*client, error) {
	if apiKey == "" {
		apiKey = os.Getenv("CO_API_KEY")
	}
	if apiKey == "" {
		apiKey = os.Getenv("COHERE_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("cohere: API key not set (use WithAPIKey or the CO_API_KEY env var)")
	}

	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	if httpClient == nil {
		httpClient = &http.Client{Timeout: defaultTimeout}
	}

	retries := defaultMaxRetries
	if maxRetries != nil {
		if *maxRetries < 0 {
			return nil, fmt.Errorf("cohere: max retries must be non-negative")
		}
		retries = *maxRetries
	}

	return &client{
		apiKey:     apiKey,
		baseURL:    baseURL,
		httpClient: httpClient,
		maxRetries: retries,
	}, nil
}

// errorEnvelope is Cohere's error body. It is {"id": ..., "message": ...},
// unlike Voyage AI's {"detail": ...}, so the message field is what carries the
// human-readable cause and must be surfaced.
type errorEnvelope struct {
	ID      string `json:"id"`
	Message string `json:"message"`
}

// statusError renders a non-200 response into an error, preferring the
// envelope's message field and falling back to the raw body when the response
// is not the documented shape — a proxy or gateway in front of the API may
// return HTML or plain text, which must still reach the caller legibly.
func statusError(path string, status int, payload []byte) error {
	var envelope errorEnvelope
	if err := json.Unmarshal(payload, &envelope); err == nil && envelope.Message != "" {
		return fmt.Errorf("cohere %s: status %d: %.300s", path, status, envelope.Message)
	}
	return fmt.Errorf("cohere %s: status %d: %.300s", path, status, payload)
}

// post sends body as JSON to path and returns the raw 200 response body,
// retrying transient failures with capped exponential backoff.
func (c *client) post(ctx context.Context, path string, body any) ([]byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	attempts := c.maxRetries + 1
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "application/json")

		httpResp, err := c.httpClient.Do(httpReq)
		if err != nil {
			if ctx.Err() != nil || attempt == attempts-1 {
				return nil, err
			}
			lastErr = err
			if err := sleepRetry(ctx, retryDelay(attempt)); err != nil {
				return nil, err
			}
			continue
		}

		payload, readErr := io.ReadAll(httpResp.Body)
		closeErr := httpResp.Body.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}

		if httpResp.StatusCode != http.StatusOK {
			err := statusError(path, httpResp.StatusCode, payload)
			if !retryableStatus(httpResp.StatusCode) || attempt == attempts-1 {
				return nil, err
			}
			lastErr = err
			if err := sleepRetry(ctx, retryDelay(attempt)); err != nil {
				return nil, err
			}
			continue
		}

		return payload, nil
	}
	return nil, lastErr
}

// retryableStatus reports whether a status code is worth a second attempt.
func retryableStatus(status int) bool {
	return status == http.StatusTooManyRequests ||
		status == http.StatusInternalServerError ||
		status == http.StatusBadGateway ||
		status == http.StatusServiceUnavailable ||
		status == http.StatusGatewayTimeout
}

// retryDelay returns the backoff before the attempt after the given one,
// doubling from one second and capped at four.
func retryDelay(attempt int) time.Duration {
	delay := time.Duration(1<<attempt) * time.Second
	if delay > 4*time.Second {
		return 4 * time.Second
	}
	return delay
}

// sleepRetry waits for delay, returning early with the context's error if it is
// cancelled first.
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

// billedUnits is Cohere's per-request billing block, reported under meta. Only
// the fields this package maps are declared: embedding requests are metered in
// input tokens, reranking requests in search units.
type billedUnits struct {
	InputTokens int `json:"input_tokens"`
	SearchUnits int `json:"search_units"`
}

// meta is the metadata block both endpoints return.
type meta struct {
	BilledUnits billedUnits `json:"billed_units"`
	Tokens      struct {
		InputTokens int `json:"input_tokens"`
	} `json:"tokens"`
}
