package voyageai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"time"
)

const defaultBaseURL = "https://api.voyageai.com/v1"

// maxRetryAfter caps how long a server-supplied Retry-After header can park a
// request. Voyage occasionally answers a sustained overage with a multi-minute
// Retry-After; honoring that verbatim would strand a caller inside a single
// Embed or Rerank call for longer than most request deadlines allow, so a
// longer hint is clamped and the attempt is simply retried sooner.
const maxRetryAfter = 30 * time.Second

// client is the shared Voyage AI HTTP transport. Embedder and Reranker each
// embed one, so authentication, retry policy and backoff are defined once and
// behave identically on both endpoints.
type client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	maxRetries int
}

// transportConfig holds the transport knobs common to every Voyage AI
// constructor. Both Option and RerankerOption write into one of these, which is
// what lets New and NewReranker share newClient without sharing an option type.
type transportConfig struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	maxRetries *int
}

// newClient resolves a transportConfig into a usable client, applying the
// VOYAGE_API_KEY fallback and the package defaults.
func newClient(cfg transportConfig) (*client, error) {
	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("VOYAGE_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("voyageai: API key not set (use WithAPIKey or the VOYAGE_API_KEY env var)")
	}

	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	httpClient := cfg.httpClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}

	maxRetries := 2
	if cfg.maxRetries != nil {
		if *cfg.maxRetries < 0 {
			return nil, fmt.Errorf("voyageai: max retries must be non-negative")
		}
		maxRetries = *cfg.maxRetries
	}

	return &client{
		apiKey:     apiKey,
		baseURL:    baseURL,
		httpClient: httpClient,
		maxRetries: maxRetries,
	}, nil
}

// post sends body as JSON to path and returns the raw 200 response body,
// retrying transient failures with jittered, capped exponential backoff. A
// Retry-After header on a retryable response overrides the computed backoff.
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

		httpResp, err := c.httpClient.Do(httpReq)
		if err != nil {
			if ctx.Err() != nil || attempt == attempts-1 {
				return nil, err
			}
			lastErr = err
			if err := sleepRetry(ctx, retryWait(attempt, nil)); err != nil {
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
			err := fmt.Errorf("voyageai %s: status %d: %.300s", path, httpResp.StatusCode, payload)
			if !retryableStatus(httpResp.StatusCode) || attempt == attempts-1 {
				return nil, err
			}
			lastErr = err
			if err := sleepRetry(ctx, retryWait(attempt, httpResp.Header)); err != nil {
				return nil, err
			}
			continue
		}

		return payload, nil
	}
	return nil, lastErr
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
// is jittered either way so that a fleet of clients rate-limited by the same
// burst does not resynchronize onto the same retry instant.
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

// withJitter spreads a delay over [d, d*1.25) so concurrent clients that were
// throttled together do not retry in lockstep. A non-positive delay is
// returned unchanged.
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
