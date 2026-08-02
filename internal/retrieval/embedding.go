package retrieval

import (
	"context"
	"errors"
)

// Embedder is the core abstraction for embedding providers.
// Implement this interface to add support for a new provider.
type Embedder interface {
	// Embed generates one vector per input text, in input order.
	Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)

	// Name returns the embedding model identifier (e.g., "text-embedding-3-small").
	Name() string
}

// EmbeddingInputType tells the provider which side of a retrieval task the
// inputs sit on. Retrieval-tuned providers (Voyage AI) prepend a task-specific
// instruction before vectorizing, which embeds queries near their answering
// documents instead of near other similarly-phrased queries. Providers without
// the concept (OpenAI) ignore it.
type EmbeddingInputType string

const (
	// EmbeddingInputNone embeds the raw text with no task instruction.
	EmbeddingInputNone EmbeddingInputType = ""

	// EmbeddingInputQuery marks the inputs as search queries.
	EmbeddingInputQuery EmbeddingInputType = "query"

	// EmbeddingInputDocument marks the inputs as documents to be stored and
	// searched against.
	EmbeddingInputDocument EmbeddingInputType = "document"
)

// EmbeddingRequest is a batch embedding request.
type EmbeddingRequest struct {
	// Input is the list of texts to embed. Must be non-empty. Provider batch
	// limits apply (OpenAI: 2048 inputs, 300k tokens per request; Voyage AI:
	// 1000 inputs).
	Input []string

	// InputType marks the inputs as queries or documents for retrieval-tuned
	// providers. Leave as EmbeddingInputNone for similarity, clustering, or
	// classification use.
	InputType EmbeddingInputType

	// Dimensions overrides the output vector dimension when the model
	// supports it (OpenAI text-embedding-3 and later; Voyage AI voyage-3.5,
	// voyage-3-large, voyage-4 family, voyage-code-3: 256, 512, 1024, or
	// 2048). Zero means the model default.
	Dimensions int

	// Truncate controls what a provider does with an input longer than the
	// model's context window. True truncates it to fit; false rejects the
	// request with an error. Nil uses the provider's own default.
	//
	// Prefer false when indexing: silently embedding the first N tokens of a
	// long document and storing the vector as if it represented the whole
	// document is a quiet, expensive retrieval failure.
	Truncate *bool
}

// Validate checks that the request is well-formed.
func (r *EmbeddingRequest) Validate() error {
	if len(r.Input) == 0 {
		return errors.New("input cannot be empty")
	}
	for _, text := range r.Input {
		if text == "" {
			return errors.New("input texts cannot be empty strings")
		}
	}
	switch r.InputType {
	case EmbeddingInputNone, EmbeddingInputQuery, EmbeddingInputDocument:
	default:
		return errors.New("input type must be query, document, or empty")
	}
	if r.Dimensions < 0 {
		return errors.New("dimensions must be non-negative")
	}
	return nil
}

// EmbeddingResponse holds one vector per request input, in input order.
type EmbeddingResponse struct {
	// Vectors are the embedding vectors, Vectors[i] corresponding to Input[i].
	//
	// Element type is float32: no provider transmits more than float32 of
	// precision, and at 3072 dimensions the narrower type halves the memory
	// an index costs to hold.
	Vectors [][]float32

	// Model is the model name reported by the provider.
	Model string

	// Usage reports token consumption for the request.
	Usage EmbeddingUsage
}

// EmbeddingUsage reports token consumption for an embeddings request. Fields
// a provider does not report are zero (Voyage AI reports only TotalTokens).
type EmbeddingUsage struct {
	PromptTokens int
	TotalTokens  int
}
