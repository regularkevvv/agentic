// Package voyageai provides Voyage AI implementations of retrieval.Embedder and
// retrieval.Reranker for Agentic. Voyage AI (by MongoDB) offers retrieval-tuned
// embedding models with first-class query/document input types, plus
// cross-encoder rerank models for second-stage ranking; it has no chat API, so
// this package implements no core.Model.
package voyageai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/retrieval"
)

// Embedder implements retrieval.Embedder using the Voyage AI embeddings API.
type Embedder struct {
	*client

	model      string
	truncation *bool
}

// Option configures the Voyage AI Embedder. Options are not interchangeable
// with RerankerOption, so a rerank-only knob cannot reach New by mistake.
type Option func(*config)

type config struct {
	transportConfig

	truncation *bool
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

// WithTruncation sets the default for inputs longer than the model's context
// length: true truncates them, false rejects the request with an error.
//
// This is only a default. A per-call retrieval.EmbeddingRequest.Truncate takes
// precedence over it. If neither is set, the Voyage API's own default applies,
// which is to truncate silently — prefer WithTruncation(false) when indexing,
// because storing the vector of a clipped document as if it covered the whole
// document is a quiet, expensive retrieval failure.
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

	c, err := newClient(cfg.transportConfig)
	if err != nil {
		return nil, err
	}

	return &Embedder{
		client:     c,
		model:      model,
		truncation: cfg.truncation,
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
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// Embed implements retrieval.Embedder.
func (e *Embedder) Embed(ctx context.Context, req *retrieval.EmbeddingRequest) (*retrieval.EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	body := embedRequest{
		Input:           req.Input,
		Model:           e.model,
		InputType:       string(req.InputType),
		Truncation:      e.resolveTruncation(req.Truncate),
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
	vectors := make([][]float32, len(req.Input))
	for _, item := range resp.Data {
		if item.Index < 0 || item.Index >= len(vectors) {
			return nil, fmt.Errorf("voyageai embeddings: vector index %d out of range for %d inputs", item.Index, len(req.Input))
		}
		vectors[item.Index] = item.Embedding
	}

	return &retrieval.EmbeddingResponse{
		Vectors: vectors,
		Model:   resp.Model,
		Usage:   retrieval.EmbeddingUsage{TotalTokens: resp.Usage.TotalTokens},
	}, nil
}

// Name implements retrieval.Embedder.
func (e *Embedder) Name() string {
	return e.model
}

// ModelMetadata reports semantic provider and transport identity.
func (e *Embedder) ModelMetadata() core.ModelMetadata {
	return core.ModelMetadataForEndpoint("voyage_ai", "embeddings", e.baseURL)
}

// resolveTruncation picks the truncation flag to send on the wire.
//
// Precedence is per-request over constructor: a caller who says "reject this
// document rather than clip it" for one indexing call must win over an
// Embedder built with the opposite default. When neither is set the field is
// omitted and the Voyage API's own default (truncate) applies.
func (e *Embedder) resolveTruncation(requested *bool) *bool {
	if requested != nil {
		return requested
	}
	return e.truncation
}

// Compile-time check that Embedder implements retrieval.Embedder.
var _ retrieval.Embedder = (*Embedder)(nil)
