// Package voyageai provides a Voyage AI Embedder implementation for Agentic.
// Voyage AI (by MongoDB) offers retrieval-tuned embedding models with
// first-class query/document input types; it has no chat API, so this package
// implements only core.Embedder.
package voyageai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
)

const defaultBaseURL = "https://api.voyageai.com/v1"

// Embedder implements core.Embedder using the Voyage AI embeddings API.
type Embedder struct {
	model      string
	apiKey     string
	baseURL    string
	httpClient *http.Client
	truncation *bool
	maxRetries int
}

// Option configures the Voyage AI Embedder.
type Option func(*config)

type config struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	truncation *bool
	maxRetries *int
}

// WithAPIKey sets the API key. If not set, the VOYAGE_API_KEY env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithBaseURL sets a custom base URL (default https://api.voyageai.com/v1).
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}

// WithTruncation controls whether inputs longer than the model's context
// length are truncated (the API default) or rejected with an error (false).
func WithTruncation(truncation bool) Option {
	return func(c *config) { c.truncation = &truncation }
}

// WithMaxRetries sets how many times a request is retried on 429 and 5xx
// responses (default 2, matching the SDK-backed providers).
func WithMaxRetries(retries int) Option {
	return func(c *config) { c.maxRetries = &retries }
}

// New creates a new Voyage AI Embedder.
//
// Examples:
//
//	embedder, err := voyageai.New("voyage-3.5", voyageai.WithAPIKey("pa-..."))
//	embedder, err := voyageai.New("voyage-code-3") // key from VOYAGE_API_KEY
func New(model string, opts ...Option) (*Embedder, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

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

	return &Embedder{
		model:      model,
		apiKey:     apiKey,
		baseURL:    baseURL,
		httpClient: httpClient,
		truncation: cfg.truncation,
		maxRetries: maxRetries,
	}, nil
}

// MustNew is like New but panics on error.
func MustNew(model string, opts ...Option) *Embedder {
	e, err := New(model, opts...)
	if err != nil {
		panic(err)
	}
	return e
}

type embedRequest struct {
	Input           []string `json:"input"`
	Model           string   `json:"model"`
	InputType       string   `json:"input_type,omitempty"`
	Truncation      *bool    `json:"truncation,omitempty"`
	OutputDimension int      `json:"output_dimension,omitempty"`
}

type embedResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// Embed implements core.Embedder.
func (e *Embedder) Embed(ctx context.Context, req *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	body := embedRequest{
		Input:           req.Input,
		Model:           e.model,
		InputType:       string(req.InputType),
		Truncation:      e.truncation,
		OutputDimension: req.Dimensions,
	}

	payload, err := e.post(ctx, "/embeddings", body)
	if err != nil {
		return nil, err
	}

	var resp embedResponse
	if err := json.Unmarshal(payload, &resp); err != nil {
		return nil, fmt.Errorf("voyageai embeddings: decode response: %w", err)
	}
	if len(resp.Data) != len(req.Input) {
		return nil, fmt.Errorf("voyageai embeddings: got %d vectors for %d inputs", len(resp.Data), len(req.Input))
	}

	// Place vectors by the response's index field rather than list order.
	vectors := make([][]float64, len(req.Input))
	for _, item := range resp.Data {
		if item.Index < 0 || item.Index >= len(vectors) {
			return nil, fmt.Errorf("voyageai embeddings: vector index %d out of range for %d inputs", item.Index, len(req.Input))
		}
		vectors[item.Index] = item.Embedding
	}

	return &core.EmbeddingResponse{
		Vectors: vectors,
		Model:   resp.Model,
		Usage:   core.EmbeddingUsage{TotalTokens: resp.Usage.TotalTokens},
	}, nil
}

// Name implements core.Embedder.
func (e *Embedder) Name() string {
	return e.model
}

func (e *Embedder) post(ctx context.Context, path string, body any) ([]byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	attempts := e.maxRetries + 1
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+path, bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Authorization", "Bearer "+e.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")

		httpResp, err := e.httpClient.Do(httpReq)
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
			err := fmt.Errorf("voyageai %s: status %d: %.300s", path, httpResp.StatusCode, payload)
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

func retryableStatus(status int) bool {
	return status == http.StatusTooManyRequests ||
		status == http.StatusInternalServerError ||
		status == http.StatusBadGateway ||
		status == http.StatusServiceUnavailable ||
		status == http.StatusGatewayTimeout
}

func retryDelay(attempt int) time.Duration {
	delay := time.Duration(1<<attempt) * time.Second
	if delay > 4*time.Second {
		return 4 * time.Second
	}
	return delay
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

// Compile-time check that Embedder implements core.Embedder.
var _ core.Embedder = (*Embedder)(nil)
