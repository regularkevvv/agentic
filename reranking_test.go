package agentic_test

import (
	"context"
	"testing"

	"github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestRerankFacade(t *testing.T) {
	r := test.NewTestReranker()
	docs := []string{
		"an unrelated document about baking",
		"go channels carry values between goroutines",
		"channels",
	}

	resp, err := agentic.Rerank(context.Background(), r, "go channels", docs, 2)
	if err != nil {
		t.Fatalf("Rerank() = %v, want nil", err)
	}

	if len(resp.Results) != 2 {
		t.Fatalf("got %d results, want 2 (TopN)", len(resp.Results))
	}
	if resp.Results[0].Index != 1 {
		t.Errorf("top result index = %d, want 1", resp.Results[0].Index)
	}
	if resp.Results[0].Document != docs[1] {
		t.Errorf("top result document = %q, want %q", resp.Results[0].Document, docs[1])
	}

	calls := r.Calls()
	if len(calls) != 1 {
		t.Fatalf("made %d calls, want 1", len(calls))
	}
	if calls[0].Query != "go channels" || calls[0].TopN != 2 || len(calls[0].Documents) != 3 {
		t.Errorf("request = %+v, want the facade arguments passed through verbatim", calls[0])
	}
}

func TestRerankFacadePropagatesValidationErrors(t *testing.T) {
	r := test.NewTestReranker()

	if _, err := agentic.Rerank(context.Background(), r, "", []string{"a"}, 0); err == nil {
		t.Error("Rerank() with an empty query = nil error, want a validation error")
	}
	if _, err := agentic.Rerank(context.Background(), r, "q", nil, 0); err == nil {
		t.Error("Rerank() with no documents = nil error, want a validation error")
	}
}
