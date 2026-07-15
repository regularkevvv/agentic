package agentic

import "context"

// EmbedQuery embeds texts as search queries. Retrieval-tuned providers embed
// queries into the region of vector space where their answering documents
// live; use this for the search side of a retrieval task.
func EmbedQuery(ctx context.Context, e Embedder, texts ...string) (*EmbeddingResponse, error) {
	return e.Embed(ctx, &EmbeddingRequest{Input: texts, InputType: EmbeddingInputQuery})
}

// EmbedDocuments embeds texts as documents to be stored and searched against;
// use this for the indexing side of a retrieval task.
func EmbedDocuments(ctx context.Context, e Embedder, texts ...string) (*EmbeddingResponse, error) {
	return e.Embed(ctx, &EmbeddingRequest{Input: texts, InputType: EmbeddingInputDocument})
}
