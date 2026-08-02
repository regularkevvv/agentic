package ollama

import (
	"fmt"
	"os"

	"github.com/regularkevvv/agentic/internal/retrieval"
	oaiProvider "github.com/regularkevvv/agentic/provider/openai"
)

// DefaultEmbeddingModel is the model used when NewEmbedder is called with an
// empty model name. It is Ollama's most widely pulled embedding model.
const DefaultEmbeddingModel = "nomic-embed-text"

// Embedder wraps the OpenAI Embedder for Ollama's OpenAI-compatible
// /v1/embeddings endpoint. It inherits Embed and Name.
type Embedder struct {
	*oaiProvider.Embedder
}

// NewEmbedder creates a new Ollama Embedder. It accepts the same options as
// New, so the host resolution rules are identical: an explicit WithHost wins,
// then OLLAMA_HOST, then localhost:11434, and a schemeless host such as
// "127.0.0.1:11434" is normalized to http.
//
// An empty model name selects DefaultEmbeddingModel.
//
// Examples:
//
//	embedder, err := ollama.NewEmbedder("")                 // nomic-embed-text on localhost
//	embedder, err := ollama.NewEmbedder("mxbai-embed-large")
//	embedder, err := ollama.NewEmbedder("nomic-embed-text", ollama.WithHost("gpu-server:11434"))
//
// EmbeddingRequest.Dimensions is forwarded as the OpenAI-compatible
// "dimensions" field. Only models trained for it (Matryoshka-style embeddings)
// vary their output width in response; other models return their native width
// regardless. EmbeddingRequest.InputType is ignored, and Truncate=true is
// rejected, both inherited from the OpenAI embedder.
func NewEmbedder(model string, opts ...Option) (*Embedder, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	if model == "" {
		model = DefaultEmbeddingModel
	}

	baseURL, err := buildBaseURL(cfg.host)
	if err != nil {
		return nil, err
	}

	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("OLLAMA_API_KEY")
	}
	if apiKey == "" {
		// The OpenAI SDK requires a key but Ollama doesn't — use a placeholder.
		apiKey = "ollama"
	}

	oaiEmbedder, err := oaiProvider.NewEmbedder(model,
		oaiProvider.WithBaseURL(baseURL),
		oaiProvider.WithAPIKey(apiKey),
	)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}

	return &Embedder{Embedder: oaiEmbedder}, nil
}

// MustNewEmbedder is like NewEmbedder but panics on error.
func MustNewEmbedder(model string, opts ...Option) *Embedder {
	e, err := NewEmbedder(model, opts...)
	if err != nil {
		panic(err)
	}
	return e
}

// Compile-time check that Embedder implements retrieval.Embedder.
var _ retrieval.Embedder = (*Embedder)(nil)
