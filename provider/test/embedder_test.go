package test

import (
	"context"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestTestEmbedderDeterministic(t *testing.T) {
	embedder := NewTestEmbedder(4)

	first, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"hello", "world"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	second, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"hello"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(first.Vectors) != 2 || len(first.Vectors[0]) != 4 {
		t.Fatalf("vectors = %dx%d, want 2x4", len(first.Vectors), len(first.Vectors[0]))
	}
	for i := range first.Vectors[0] {
		if first.Vectors[0][i] != second.Vectors[0][i] {
			t.Fatalf("same text embedded differently: %v vs %v", first.Vectors[0], second.Vectors[0])
		}
	}
	if first.Vectors[0][0] == first.Vectors[1][0] && first.Vectors[0][1] == first.Vectors[1][1] {
		t.Fatal("different texts produced identical vectors")
	}
}

func TestTestEmbedderInputTypeChangesVector(t *testing.T) {
	embedder := NewTestEmbedder(4)

	query, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input: []string{"hello"}, InputType: core.EmbeddingInputQuery,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	document, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input: []string{"hello"}, InputType: core.EmbeddingInputDocument,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	same := true
	for i := range query.Vectors[0] {
		if query.Vectors[0][i] != document.Vectors[0][i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("query and document embeddings should differ")
	}
}

func TestTestEmbedderDimensionsOverride(t *testing.T) {
	embedder := NewTestEmbedder(4)

	resp, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input: []string{"hello"}, Dimensions: 16,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors[0]) != 16 {
		t.Fatalf("dimensions = %d, want 16", len(resp.Vectors[0]))
	}
}

func TestTestEmbedderRecordsCalls(t *testing.T) {
	embedder := NewTestEmbedder(0)

	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input: []string{"a"}, InputType: core.EmbeddingInputQuery,
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if embedder.CallCount() != 1 {
		t.Fatalf("CallCount = %d, want 1", embedder.CallCount())
	}
	if embedder.Name() != "test:embedder" {
		t.Errorf("Name = %q, want test:embedder", embedder.Name())
	}
	if got := embedder.Calls()[0].InputType; got != core.EmbeddingInputQuery {
		t.Errorf("recorded input type = %q, want query", got)
	}

	embedder.Reset()
	if embedder.CallCount() != 0 {
		t.Fatalf("CallCount after Reset = %d, want 0", embedder.CallCount())
	}
}

func TestTestEmbedderRejectsInvalidRequest(t *testing.T) {
	embedder := NewTestEmbedder(4)
	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{}); err == nil {
		t.Fatal("Embed should reject an empty request")
	}
	if embedder.CallCount() != 0 {
		t.Fatal("invalid requests should not be recorded")
	}
}
