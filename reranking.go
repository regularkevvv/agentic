package agentic

import "context"

// Rerank scores documents against a query and returns them ordered by
// descending relevance, keeping the topN highest-scoring. Pass 0 for topN to
// score every document.
//
// Rerankers are cross-encoders and cost far more per document than an
// embedding comparison, so the usual arrangement is to retrieve a broad
// shortlist with [EmbedQuery] and narrow it here.
//
// Scores are ordinal within one response: rank by them, but never threshold on
// them or compare them across providers or models. See [RerankResult].
func Rerank(ctx context.Context, r Reranker, query string, documents []string, topN int) (*RerankResponse, error) {
	return r.Rerank(ctx, &RerankRequest{Query: query, Documents: documents, TopN: topN})
}
