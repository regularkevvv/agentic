package voyageai_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/provider/voyageai"
)

// rerankDocs is the request slice every reranker transport test scores.
var rerankDocs = []string{"alpha", "bravo", "charlie"}

// newFixtureReranker serves body verbatim on every request.
func newFixtureReranker(t *testing.T, body string) *voyageai.Reranker {
	t.Helper()
	return newHandlerReranker(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	})
}

func newHandlerReranker(t *testing.T, handler http.HandlerFunc, opts ...voyageai.RerankerOption) *voyageai.Reranker {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]voyageai.RerankerOption{
		voyageai.WithRerankerAPIKey("test-key"),
		voyageai.WithRerankerBaseURL(server.URL),
		voyageai.WithRerankerHTTPClient(server.Client()),
	}, opts...)

	reranker, err := voyageai.NewReranker("rerank-2.5", opts...)
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	return reranker
}

func rerank(t *testing.T, r *voyageai.Reranker) (*retrieval.RerankResponse, error) {
	t.Helper()
	return r.Rerank(context.Background(), &retrieval.RerankRequest{
		Query:     "which one",
		Documents: rerankDocs,
	})
}

// TestRerankOutOfRangeIndex pins that an index the API hands back which does
// not address the request slice produces an error rather than a panic. A
// misbehaving proxy that renumbers or reorders results must not be able to
// crash a caller, and must never silently pair a score with the wrong text.
func TestRerankOutOfRangeIndex(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "index past the end",
			body: `{"data":[{"index":7,"relevance_score":0.9}],"model":"rerank-2.5","usage":{"total_tokens":3}}`,
		},
		{
			name: "index one past the end",
			body: `{"data":[{"index":3,"relevance_score":0.9}],"model":"rerank-2.5","usage":{"total_tokens":3}}`,
		},
		{
			name: "negative index",
			body: `{"data":[{"index":-2,"relevance_score":0.9}],"model":"rerank-2.5","usage":{"total_tokens":3}}`,
		},
		{
			name: "valid results followed by an out-of-range one",
			body: `{"data":[
				{"index":0,"relevance_score":0.9},
				{"index":99,"relevance_score":0.8}
			],"model":"rerank-2.5","usage":{"total_tokens":3}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := rerank(t, newFixtureReranker(t, tt.body))
			if err == nil {
				t.Fatal("Rerank should reject an out-of-range result index")
			}
			if !strings.Contains(err.Error(), "out of range") {
				t.Errorf("error = %v, want it to mention the out-of-range index", err)
			}
		})
	}
}

// TestRerankOutOfOrderIndexes pins that results arriving in neither score nor
// index order come back sorted by descending score, with each Document taken
// from the request slice at its own index.
func TestRerankOutOfOrderIndexes(t *testing.T) {
	reranker := newFixtureReranker(t, `{
		"object":"list",
		"data":[
			{"index":1,"relevance_score":0.25},
			{"index":2,"relevance_score":0.97},
			{"index":0,"relevance_score":0.61}
		],
		"model":"rerank-2.5",
		"usage":{"total_tokens":26}
	}`)

	resp, err := rerank(t, reranker)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}

	want := []retrieval.RerankResult{
		{Index: 2, Score: 0.97, Document: "charlie"},
		{Index: 0, Score: 0.61, Document: "alpha"},
		{Index: 1, Score: 0.25, Document: "bravo"},
	}
	if len(resp.Results) != len(want) {
		t.Fatalf("results = %v, want %v", resp.Results, want)
	}
	for i := range want {
		if resp.Results[i] != want[i] {
			t.Errorf("results[%d] = %v, want %v", i, resp.Results[i], want[i])
		}
	}
}

// TestRerankDocumentsComeFromTheRequest pins that Document is filled from the
// request slice, not from whatever the API echoes back. return_documents is
// never sent, so a response carrying documents is either a proxy's invention
// or stale, and must be ignored.
func TestRerankDocumentsComeFromTheRequest(t *testing.T) {
	reranker := newFixtureReranker(t, `{
		"data":[
			{"index":0,"relevance_score":0.9,"document":"WRONG"},
			{"index":1,"relevance_score":0.4,"document":"ALSO WRONG"}
		],
		"model":"rerank-2.5",
		"usage":{"total_tokens":3}
	}`)

	resp, err := rerank(t, reranker)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	for _, res := range resp.Results {
		if res.Document != rerankDocs[res.Index] {
			t.Errorf("results[%d].Document = %q, want %q from the request slice",
				res.Index, res.Document, rerankDocs[res.Index])
		}
	}
}

// TestRerankHonorsRetryAfter pins that a 429 carrying Retry-After is retried
// after that hint rather than the computed backoff. The header says zero
// seconds, so a run that honors it finishes promptly while one that ignores it
// would sit out a full second of exponential backoff.
func TestRerankHonorsRetryAfter(t *testing.T) {
	var calls int
	reranker := newHandlerReranker(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "0")
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"index":0,"relevance_score":0.9}],"model":"rerank-2.5","usage":{"total_tokens":3}}`)
	})

	resp, err := rerank(t, reranker)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2", calls)
	}
	if len(resp.Results) != 1 || resp.Results[0].Document != "alpha" {
		t.Errorf("results = %v, want the single scored document", resp.Results)
	}
}

// TestEmbedHonorsRetryAfter pins that the same Retry-After handling reaches
// the Embedder, since both endpoints share one transport.
func TestEmbedHonorsRetryAfter(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "0")
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, singleVectorResponse)
	}))
	t.Cleanup(server.Close)

	embedder, err := voyageai.New("voyage-3.5",
		voyageai.WithAPIKey("test-key"),
		voyageai.WithBaseURL(server.URL),
		voyageai.WithHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2", calls)
	}
}

// TestRerankInvalidBaseURL pins that a base URL that cannot form a request
// surfaces as an error rather than a panic.
func TestRerankInvalidBaseURL(t *testing.T) {
	reranker, err := voyageai.NewReranker("rerank-2.5",
		voyageai.WithRerankerAPIKey("test-key"),
		voyageai.WithRerankerBaseURL("http://exa\x7fmple.com"),
	)
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	if _, err := rerank(t, reranker); err == nil {
		t.Fatal("Rerank should fail when the base URL cannot form a request")
	}
}

// TestRerankCanceledContext pins that a dead context aborts before the call.
func TestRerankCanceledContext(t *testing.T) {
	reranker := newFixtureReranker(t, `{"data":[{"index":0,"relevance_score":0.9}],"model":"rerank-2.5","usage":{"total_tokens":3}}`)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := reranker.Rerank(ctx, &retrieval.RerankRequest{
		Query:     "which one",
		Documents: rerankDocs,
	}); err == nil {
		t.Fatal("Rerank should fail with a canceled context")
	}
}
