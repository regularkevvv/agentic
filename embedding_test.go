package agentic

import (
	"context"
	"testing"

	testprovider "github.com/regularkevvv/agentic/provider/test"
)

func TestEmbedQueryAndDocumentsSetInputType(t *testing.T) {
	embedder := testprovider.NewTestEmbedder(4)

	if _, err := EmbedQuery(context.Background(), embedder, "what is the budget?"); err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if _, err := EmbedDocuments(context.Background(), embedder, "the budget is $2,000", "the brand is LoomSignal"); err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}

	calls := embedder.Calls()
	if len(calls) != 2 {
		t.Fatalf("calls = %d, want 2", len(calls))
	}
	if calls[0].InputType != EmbeddingInputQuery {
		t.Errorf("EmbedQuery input type = %q, want query", calls[0].InputType)
	}
	if len(calls[0].Input) != 1 {
		t.Errorf("EmbedQuery inputs = %d, want 1", len(calls[0].Input))
	}
	if calls[1].InputType != EmbeddingInputDocument {
		t.Errorf("EmbedDocuments input type = %q, want document", calls[1].InputType)
	}
	if len(calls[1].Input) != 2 {
		t.Errorf("EmbedDocuments inputs = %d, want 2", len(calls[1].Input))
	}
}

func TestEmbedderAlias(t *testing.T) {
	// The root aliases must be interchangeable with the core types.
	var embedder Embedder = testprovider.NewTestEmbedder(4)

	resp, err := embedder.Embed(context.Background(), &EmbeddingRequest{
		Input:     []string{"hello"},
		InputType: EmbeddingInputNone,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 1 {
		t.Fatalf("vectors = %d, want 1", len(resp.Vectors))
	}
	usage := EmbeddingUsage(resp.Usage)
	if usage.TotalTokens == 0 {
		t.Error("usage should be populated")
	}
}
