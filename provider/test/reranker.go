package test

import (
	"context"
	"hash/fnv"
	"sort"
	"strings"

	"github.com/regularkevvv/agentic/internal/retrieval"
)

// TestReranker is a mock Reranker implementation for testing without API
// calls. It scores each document by how many of the query's words it contains,
// so a document that shares more terms with the query ranks higher and the
// ordering is meaningful rather than arbitrary. Ties break deterministically.
type TestReranker struct {
	name  string
	calls []retrieval.RerankRequest
}

// NewTestReranker creates a TestReranker.
func NewTestReranker() *TestReranker {
	return &TestReranker{name: "test:reranker"}
}

// Rerank implements retrieval.Reranker.
func (r *TestReranker) Rerank(ctx context.Context, req *retrieval.RerankRequest) (*retrieval.RerankResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	r.calls = append(r.calls, *req)

	queryTerms := strings.Fields(strings.ToLower(req.Query))

	results := make([]retrieval.RerankResult, len(req.Documents))
	for i, doc := range req.Documents {
		lower := strings.ToLower(doc)
		overlap := 0
		for _, term := range queryTerms {
			if strings.Contains(lower, term) {
				overlap++
			}
		}
		results[i] = retrieval.RerankResult{
			Index: i,
			// Term overlap dominates; the hash only breaks ties, keeping the
			// order stable without making it look arbitrary.
			Score:    float64(overlap) + tieBreaker(doc),
			Document: doc,
		}
	}

	sort.SliceStable(results, func(a, b int) bool {
		return results[a].Score > results[b].Score
	})

	if req.TopN > 0 && req.TopN < len(results) {
		results = results[:req.TopN]
	}

	totalTokens := 0
	for _, doc := range req.Documents {
		totalTokens += len(strings.Fields(doc))
	}

	return &retrieval.RerankResponse{
		Results: results,
		Model:   r.name,
		Usage:   retrieval.RerankUsage{TotalTokens: totalTokens, SearchUnits: 1},
	}, nil
}

// Name implements retrieval.Reranker.
func (r *TestReranker) Name() string {
	return r.name
}

// Calls returns all requests received.
func (r *TestReranker) Calls() []retrieval.RerankRequest {
	return r.calls
}

// CallCount returns the number of Rerank calls made.
func (r *TestReranker) CallCount() int {
	return len(r.calls)
}

// Reset clears the recorded calls.
func (r *TestReranker) Reset() {
	r.calls = nil
}

// tieBreaker maps a document onto a small stable value in [0, 1) so equal
// term-overlap scores still order deterministically.
func tieBreaker(doc string) float64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(doc))
	return float64(h.Sum64()%1000) / 1000
}
