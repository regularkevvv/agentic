package test_test

import (
	"context"
	"testing"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestTestRerankerRanksByTermOverlap(t *testing.T) {
	r := test.NewTestReranker()

	resp, err := r.Rerank(context.Background(), &retrieval.RerankRequest{
		Query: "go concurrency channels",
		Documents: []string{
			"a recipe for sourdough bread",
			"go concurrency uses channels and goroutines",
			"channels in go",
		},
	})
	if err != nil {
		t.Fatalf("Rerank() = %v, want nil", err)
	}

	if len(resp.Results) != 3 {
		t.Fatalf("got %d results, want 3", len(resp.Results))
	}
	// The document matching all three query terms must outrank the two-term
	// match, which must outrank the unrelated one.
	if resp.Results[0].Index != 1 {
		t.Errorf("top result index = %d, want 1 (the three-term match)", resp.Results[0].Index)
	}
	if resp.Results[len(resp.Results)-1].Index != 0 {
		t.Errorf("last result index = %d, want 0 (the unrelated document)", resp.Results[len(resp.Results)-1].Index)
	}

	for i := 1; i < len(resp.Results); i++ {
		if resp.Results[i-1].Score < resp.Results[i].Score {
			t.Errorf("results not sorted by descending score: %v", resp.Results)
			break
		}
	}
}

func TestTestRerankerDocumentMatchesIndex(t *testing.T) {
	r := test.NewTestReranker()
	docs := []string{"alpha", "beta gamma", "gamma"}

	resp, err := r.Rerank(context.Background(), &retrieval.RerankRequest{Query: "gamma", Documents: docs})
	if err != nil {
		t.Fatalf("Rerank() = %v, want nil", err)
	}

	// Index and Document must stay consistent after sorting, or a caller
	// joining on Index against their own metadata gets the wrong row.
	for _, got := range resp.Results {
		if got.Index < 0 || got.Index >= len(docs) {
			t.Fatalf("index %d out of range for %d documents", got.Index, len(docs))
		}
		if got.Document != docs[got.Index] {
			t.Errorf("result{Index: %d, Document: %q}, want document %q", got.Index, got.Document, docs[got.Index])
		}
	}
}

func TestTestRerankerHonorsTopN(t *testing.T) {
	r := test.NewTestReranker()
	docs := []string{"alpha", "beta", "gamma", "delta"}

	for _, tt := range []struct {
		topN int
		want int
	}{
		{topN: 0, want: 4},
		{topN: 2, want: 2},
		{topN: 4, want: 4},
		{topN: 10, want: 4},
	} {
		resp, err := r.Rerank(context.Background(), &retrieval.RerankRequest{
			Query: "alpha", Documents: docs, TopN: tt.topN,
		})
		if err != nil {
			t.Fatalf("Rerank(TopN=%d) = %v, want nil", tt.topN, err)
		}
		if len(resp.Results) != tt.want {
			t.Errorf("TopN=%d returned %d results, want %d", tt.topN, len(resp.Results), tt.want)
		}
	}
}

func TestTestRerankerValidatesRequest(t *testing.T) {
	r := test.NewTestReranker()

	if _, err := r.Rerank(context.Background(), &retrieval.RerankRequest{Documents: []string{"a"}}); err == nil {
		t.Error("Rerank() with an empty query = nil error, want a validation error")
	}
	if _, err := r.Rerank(context.Background(), &retrieval.RerankRequest{Query: "q"}); err == nil {
		t.Error("Rerank() with no documents = nil error, want a validation error")
	}
	if r.CallCount() != 0 {
		t.Errorf("CallCount() = %d, want 0 — invalid requests must not be recorded", r.CallCount())
	}
}

func TestTestRerankerRecordsCalls(t *testing.T) {
	r := test.NewTestReranker()

	if r.Name() != "test:reranker" {
		t.Errorf("Name() = %q, want %q", r.Name(), "test:reranker")
	}

	for range 3 {
		if _, err := r.Rerank(context.Background(), &retrieval.RerankRequest{
			Query: "q", Documents: []string{"a", "b"},
		}); err != nil {
			t.Fatalf("Rerank() = %v, want nil", err)
		}
	}
	if r.CallCount() != 3 {
		t.Errorf("CallCount() = %d, want 3", r.CallCount())
	}
	if got := r.Calls(); len(got) != 3 || got[0].Query != "q" {
		t.Errorf("Calls() = %+v, want 3 recorded requests", got)
	}

	r.Reset()
	if r.CallCount() != 0 {
		t.Errorf("CallCount() after Reset = %d, want 0", r.CallCount())
	}
}
