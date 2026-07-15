package openai

import (
	"context"
	"fmt"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// Embedder implements core.Embedder using the OpenAI Embeddings API. It
// supports both the official OpenAI API and OpenAI-compatible providers via
// the WithBaseURL option.
type Embedder struct {
	client *openai.Client
	model  string
}

// NewEmbedder creates a new OpenAI Embedder. It accepts the same options as
// New.
//
// Examples:
//
//	// OpenAI
//	embedder, err := openai.NewEmbedder("text-embedding-3-small", openai.WithAPIKey("sk-..."))
//
//	// OpenAI-compatible provider
//	embedder, err := openai.NewEmbedder("some-embedding-model",
//	    openai.WithAPIKey("..."),
//	    openai.WithBaseURL("https://my-provider.com/v1"),
//	)
func NewEmbedder(model string, opts ...Option) (*Embedder, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	var reqOpts []option.RequestOption
	if cfg.apiKey != "" {
		reqOpts = append(reqOpts, option.WithAPIKey(cfg.apiKey))
	}
	if cfg.baseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(cfg.baseURL))
	}
	if cfg.organization != "" {
		reqOpts = append(reqOpts, option.WithOrganization(cfg.organization))
	}
	reqOpts = append(reqOpts, cfg.extraOpts...)

	client := openai.NewClient(reqOpts...)

	return &Embedder{
		client: &client,
		model:  model,
	}, nil
}

// MustNewEmbedder is like NewEmbedder but panics on error.
func MustNewEmbedder(model string, opts ...Option) *Embedder {
	e, err := NewEmbedder(model, opts...)
	if err != nil {
		panic(err)
	}
	return e
}

// Embed implements core.Embedder. The request's InputType is ignored: the
// OpenAI embeddings API has no input-type parameter.
func (e *Embedder) Embed(ctx context.Context, req *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	params := openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: req.Input},
		Model: e.model,
	}
	if req.Dimensions > 0 {
		params.Dimensions = openai.Int(int64(req.Dimensions))
	}

	resp, err := e.client.Embeddings.New(ctx, params)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) != len(req.Input) {
		return nil, fmt.Errorf("openai embeddings: got %d vectors for %d inputs", len(resp.Data), len(req.Input))
	}

	// Place vectors by the response's index field rather than list order.
	vectors := make([][]float64, len(req.Input))
	for _, item := range resp.Data {
		if item.Index < 0 || item.Index >= int64(len(vectors)) {
			return nil, fmt.Errorf("openai embeddings: vector index %d out of range for %d inputs", item.Index, len(req.Input))
		}
		vectors[item.Index] = item.Embedding
	}

	return &core.EmbeddingResponse{
		Vectors: vectors,
		Model:   resp.Model,
		Usage: core.EmbeddingUsage{
			PromptTokens: int(resp.Usage.PromptTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
		},
	}, nil
}

// Name implements core.Embedder.
func (e *Embedder) Name() string {
	return e.model
}

// Compile-time check that Embedder implements core.Embedder.
var _ core.Embedder = (*Embedder)(nil)
