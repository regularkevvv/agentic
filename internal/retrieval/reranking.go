package retrieval

import (
	"context"
	"errors"
)

// Reranker is the core abstraction for reranking providers.
// Implement this interface to add support for a new provider.
//
// A reranker is a cross-encoder: it reads the query and each document together
// and scores the pair directly, rather than comparing vectors computed
// independently of one another. That is markedly more accurate than embedding
// similarity and markedly more expensive, so the usual arrangement is two
// stages — retrieve a broad shortlist with an [Embedder], then narrow it with
// a Reranker.
type Reranker interface {
	// Rerank scores each document against the query and returns the results
	// ordered by descending score.
	Rerank(ctx context.Context, req *RerankRequest) (*RerankResponse, error)

	// Name returns the reranking model identifier (e.g., "rerank-2.5").
	Name() string
}

// RerankRequest is a single-query reranking request.
type RerankRequest struct {
	// Query is the search query the documents are scored against.
	// Must be non-empty.
	Query string

	// Documents are the candidate texts to score, typically the shortlist
	// returned by a vector search. Must be non-empty; provider per-call
	// limits apply.
	//
	// To carry metadata alongside a document, keep a parallel slice in the
	// caller and join on RerankResult.Index.
	Documents []string

	// TopN caps how many results are returned, keeping the highest-scoring
	// documents. Zero returns a result for every document. Values larger
	// than len(Documents) are clamped.
	TopN int
}

// Validate checks that the request is well-formed.
func (r *RerankRequest) Validate() error {
	if r.Query == "" {
		return errors.New("query cannot be empty")
	}
	if len(r.Documents) == 0 {
		return errors.New("documents cannot be empty")
	}
	for _, doc := range r.Documents {
		if doc == "" {
			return errors.New("documents cannot be empty strings")
		}
	}
	if r.TopN < 0 {
		return errors.New("top n must be non-negative")
	}
	return nil
}

// RerankResponse holds the scored documents, ordered by descending score.
type RerankResponse struct {
	// Results are the scored documents, highest score first. When the request
	// set TopN this holds at most TopN entries, otherwise one entry per
	// request document.
	Results []RerankResult

	// Model is the model name reported by the provider, or the configured
	// model name for providers that do not echo one.
	Model string

	// Usage reports consumption for the request.
	Usage RerankUsage
}

// RerankResult is one document's relevance score.
type RerankResult struct {
	// Index is the position of this document in the request's Documents
	// slice, always in range. Use it to join back to whatever metadata the
	// caller holds alongside the text.
	Index int

	// Score is the relevance of the document to the query, higher meaning
	// more relevant.
	//
	// Scores are ordinal within a single response and nothing more. Their
	// range and distribution differ by provider and by model, so never
	// compare scores across providers, across models, or across separate
	// calls, and never hard-code an absolute relevance threshold. Rank, or
	// cut by position.
	Score float64

	// Document is the request document at Index, copied through for
	// convenience. Providers fill this from the request slice rather than
	// from the API response, so it is present even for providers that do not
	// echo documents back.
	Document string
}

// RerankUsage reports consumption for a reranking request. Fields a provider
// does not report are zero.
type RerankUsage struct {
	// TotalTokens is the number of tokens billed, for providers that meter
	// reranking by token (Voyage AI).
	TotalTokens int

	// SearchUnits is the number of search units billed, for providers that
	// meter one query against up to N documents as a single unit (Cohere).
	SearchUnits int
}
