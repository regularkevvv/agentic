package test

import (
	"context"
	"hash/fnv"

	"github.com/regularkevvv/agentic/internal/core"
)

// TestEmbedder is a mock Embedder implementation for testing without API
// calls. It returns deterministic vectors derived from each input text, so
// equal texts embed identically and different texts embed differently.
type TestEmbedder struct {
	name  string
	dims  int
	calls []core.EmbeddingRequest
}

// NewTestEmbedder creates a TestEmbedder producing vectors of the given
// dimension (default 8 if dims <= 0).
func NewTestEmbedder(dims int) *TestEmbedder {
	if dims <= 0 {
		dims = 8
	}
	return &TestEmbedder{
		name: "test:embedder",
		dims: dims,
	}
}

// Embed implements core.Embedder.
func (e *TestEmbedder) Embed(ctx context.Context, req *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	e.calls = append(e.calls, *req)

	dims := e.dims
	if req.Dimensions > 0 {
		dims = req.Dimensions
	}

	vectors := make([][]float64, len(req.Input))
	totalTokens := 0
	for i, text := range req.Input {
		vectors[i] = deterministicVector(text, string(req.InputType), dims)
		totalTokens += len(text)
	}

	return &core.EmbeddingResponse{
		Vectors: vectors,
		Model:   e.name,
		Usage:   core.EmbeddingUsage{PromptTokens: totalTokens, TotalTokens: totalTokens},
	}, nil
}

// Name implements core.Embedder.
func (e *TestEmbedder) Name() string {
	return e.name
}

// Calls returns all requests received.
func (e *TestEmbedder) Calls() []core.EmbeddingRequest {
	return e.calls
}

// CallCount returns the number of Embed calls made.
func (e *TestEmbedder) CallCount() int {
	return len(e.calls)
}

// Reset clears the recorded calls.
func (e *TestEmbedder) Reset() {
	e.calls = nil
}

// deterministicVector derives a stable pseudo-vector from the text, the input
// type, and the position within the vector.
func deterministicVector(text, inputType string, dims int) []float64 {
	vec := make([]float64, dims)
	for i := range vec {
		h := fnv.New64a()
		_, _ = h.Write([]byte(text))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(inputType))
		_, _ = h.Write([]byte{byte(i), byte(i >> 8)})
		// Map the hash onto [-1, 1).
		vec[i] = float64(int64(h.Sum64())) / float64(1<<63)
	}
	return vec
}

// Compile-time check that TestEmbedder implements core.Embedder.
var _ core.Embedder = (*TestEmbedder)(nil)
