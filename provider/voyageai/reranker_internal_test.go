package voyageai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

const rerankFixture = `{
	"object":"list",
	"data":[
		{"index":2,"relevance_score":0.9},
		{"index":0,"relevance_score":0.1},
		{"index":1,"relevance_score":0.5}
	],
	"model":"rerank-2.5",
	"usage":{"total_tokens":26}
}`

func newTestReranker(t *testing.T, handler http.HandlerFunc, opts ...RerankerOption) *Reranker {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]RerankerOption{
		WithRerankerAPIKey("test-key"),
		WithRerankerBaseURL(server.URL),
		WithRerankerHTTPClient(server.Client()),
	}, opts...)

	reranker, err := NewReranker("rerank-2.5", opts...)
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	return reranker
}

// jsonHandler replies with a fixed body and records the decoded request.
func jsonHandler(t *testing.T, body string, got *map[string]any) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if got != nil {
			if err := json.NewDecoder(r.Body).Decode(got); err != nil {
				t.Errorf("decode request body: %v", err)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}
}

func TestRerankRequestAndResponse(t *testing.T) {
	var gotAuth, gotPath string
	var gotBody map[string]any
	reranker := newTestReranker(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		jsonHandler(t, rerankFixture, &gotBody)(w, r)
	})

	docs := []string{"alpha", "bravo", "charlie"}
	resp, err := reranker.Rerank(context.Background(), &core.RerankRequest{
		Query:     "which one",
		Documents: docs,
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}

	if gotAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", gotAuth)
	}
	if gotPath != "/rerank" {
		t.Errorf("path = %q, want /rerank", gotPath)
	}
	if got, want := gotBody["model"], "rerank-2.5"; got != want {
		t.Errorf("model = %v, want %v", got, want)
	}
	if got, want := gotBody["query"], "which one"; got != want {
		t.Errorf("query = %v, want %v", got, want)
	}
	if _, ok := gotBody["top_k"]; ok {
		t.Errorf("top_k should be omitted when TopN is zero, got %v", gotBody["top_k"])
	}
	if _, ok := gotBody["top_n"]; ok {
		t.Error("top_n must not be sent; Voyage calls the field top_k")
	}
	if _, ok := gotBody["return_documents"]; ok {
		t.Error("return_documents must not be sent")
	}
	if _, ok := gotBody["truncation"]; ok {
		t.Errorf("truncation should be omitted unless set, got %v", gotBody["truncation"])
	}

	// Fixture is out of score order: results must come back sorted descending.
	want := []core.RerankResult{
		{Index: 2, Score: 0.9, Document: "charlie"},
		{Index: 1, Score: 0.5, Document: "bravo"},
		{Index: 0, Score: 0.1, Document: "alpha"},
	}
	if len(resp.Results) != len(want) {
		t.Fatalf("results = %v, want %v", resp.Results, want)
	}
	for i := range want {
		if resp.Results[i] != want[i] {
			t.Errorf("results[%d] = %v, want %v", i, resp.Results[i], want[i])
		}
	}
	if resp.Model != "rerank-2.5" {
		t.Errorf("model = %q, want rerank-2.5", resp.Model)
	}
	if resp.Usage.TotalTokens != 26 {
		t.Errorf("total tokens = %d, want 26", resp.Usage.TotalTokens)
	}
	if resp.Usage.SearchUnits != 0 {
		t.Errorf("search units = %d, want 0 (Voyage meters by token)", resp.Usage.SearchUnits)
	}
}

// TestRerankTopNMapsToTopK pins the field rename and the clamp to the document
// count.
func TestRerankTopNMapsToTopK(t *testing.T) {
	tests := []struct {
		name        string
		topN        int
		wantPresent bool
		want        float64
	}{
		{name: "zero omits the field", topN: 0, wantPresent: false},
		{name: "in range", topN: 2, wantPresent: true, want: 2},
		{name: "clamped to the document count", topN: 9, wantPresent: true, want: 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBody map[string]any
			reranker := newTestReranker(t, jsonHandler(t, rerankFixture, &gotBody))

			if _, err := reranker.Rerank(context.Background(), &core.RerankRequest{
				Query:     "q",
				Documents: []string{"alpha", "bravo", "charlie"},
				TopN:      tt.topN,
			}); err != nil {
				t.Fatalf("Rerank: %v", err)
			}

			got, present := gotBody["top_k"]
			if present != tt.wantPresent {
				t.Fatalf("top_k present = %v (%v), want %v", present, got, tt.wantPresent)
			}
			if tt.wantPresent && got != tt.want {
				t.Errorf("top_k = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestRerankTrimsToTopN pins that a server ignoring top_k cannot widen the
// result set past what the caller asked for.
func TestRerankTrimsToTopN(t *testing.T) {
	reranker := newTestReranker(t, jsonHandler(t, rerankFixture, nil))

	resp, err := reranker.Rerank(context.Background(), &core.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha", "bravo", "charlie"},
		TopN:      2,
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(resp.Results) != 2 {
		t.Fatalf("results = %v, want the top 2", resp.Results)
	}
	if resp.Results[0].Index != 2 || resp.Results[1].Index != 1 {
		t.Errorf("results = %v, want the two highest scores", resp.Results)
	}
}

func TestRerankSendsTruncation(t *testing.T) {
	var gotBody map[string]any
	reranker := newTestReranker(t, jsonHandler(t, rerankFixture, &gotBody), WithRerankerTruncation(false))

	if _, err := reranker.Rerank(context.Background(), &core.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha", "bravo", "charlie"},
	}); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if got, want := gotBody["truncation"], false; got != want {
		t.Errorf("truncation = %v, want %v", got, want)
	}
}

// TestRerankTieBreaksByIndex pins that equal scores fall back to request order
// so the ranking is total and reproducible.
func TestRerankTieBreaksByIndex(t *testing.T) {
	reranker := newTestReranker(t, jsonHandler(t, `{
		"data":[
			{"index":2,"relevance_score":0.5},
			{"index":0,"relevance_score":0.5},
			{"index":1,"relevance_score":0.5}
		],
		"model":"rerank-2.5",
		"usage":{"total_tokens":3}
	}`, nil))

	resp, err := reranker.Rerank(context.Background(), &core.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha", "bravo", "charlie"},
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	for i, res := range resp.Results {
		if res.Index != i {
			t.Errorf("results[%d].Index = %d, want %d", i, res.Index, i)
		}
	}
}

// TestRerankModelFallback pins that a response omitting the model name reports
// the configured one rather than an empty string.
func TestRerankModelFallback(t *testing.T) {
	reranker := newTestReranker(t, jsonHandler(t, `{
		"data":[{"index":0,"relevance_score":0.4}],
		"usage":{"total_tokens":2}
	}`, nil))

	resp, err := reranker.Rerank(context.Background(), &core.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha"},
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if resp.Model != "rerank-2.5" {
		t.Errorf("model = %q, want the configured rerank-2.5", resp.Model)
	}
}

func TestRerankMalformedResponses(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "not json", body: "not json", want: "decode response"},
		{
			name: "no results",
			body: `{"data":[],"model":"rerank-2.5","usage":{"total_tokens":1}}`,
			want: "no results",
		},
		{
			name: "more results than documents",
			body: `{"data":[
				{"index":0,"relevance_score":0.1},
				{"index":1,"relevance_score":0.2},
				{"index":2,"relevance_score":0.3},
				{"index":0,"relevance_score":0.4}
			],"model":"rerank-2.5","usage":{"total_tokens":1}}`,
			want: "got 4 results for 3 documents",
		},
		{
			name: "duplicate index",
			body: `{"data":[
				{"index":1,"relevance_score":0.1},
				{"index":1,"relevance_score":0.2}
			],"model":"rerank-2.5","usage":{"total_tokens":1}}`,
			want: "duplicate result index 1",
		},
		{
			name: "negative index",
			body: `{"data":[{"index":-1,"relevance_score":0.1}],"model":"rerank-2.5","usage":{"total_tokens":1}}`,
			want: "out of range",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reranker := newTestReranker(t, jsonHandler(t, tt.body, nil))

			_, err := reranker.Rerank(context.Background(), &core.RerankRequest{
				Query:     "q",
				Documents: []string{"alpha", "bravo", "charlie"},
			})
			if err == nil {
				t.Fatalf("Rerank should fail on %s", tt.name)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want it to mention %q", err, tt.want)
			}
		})
	}
}

func TestRerankRejectsInvalidRequest(t *testing.T) {
	reranker, err := NewReranker("rerank-2.5", WithRerankerAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	if _, err := reranker.Rerank(context.Background(), &core.RerankRequest{}); err == nil {
		t.Fatal("Rerank should reject an empty request")
	}
}

func TestRerankRetriesTransientStatus(t *testing.T) {
	var calls int
	reranker := newTestReranker(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			http.Error(w, "temporary upstream failure", http.StatusServiceUnavailable)
			return
		}
		jsonHandler(t, rerankFixture, nil)(w, r)
	})

	if _, err := reranker.Rerank(context.Background(), &core.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha", "bravo", "charlie"},
	}); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2", calls)
	}
}

func TestRerankDoesNotRetryClientError(t *testing.T) {
	var calls int
	reranker := newTestReranker(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		http.Error(w, `{"detail":"model not found"}`, http.StatusBadRequest)
	})

	_, err := reranker.Rerank(context.Background(), &core.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha"},
	})
	if err == nil {
		t.Fatal("Rerank should fail on 400")
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1 (400 is not retryable)", calls)
	}
	if !strings.Contains(err.Error(), "/rerank") {
		t.Errorf("error = %v, want it to name the rerank path", err)
	}
}

func TestNewRerankerRequiresAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "")
	if _, err := NewReranker("rerank-2.5"); err == nil {
		t.Fatal("NewReranker should fail without an API key")
	}
}

func TestNewRerankerReadsEnvAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-key")
	reranker, err := NewReranker("rerank-2.5")
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	if reranker.apiKey != "env-key" {
		t.Errorf("apiKey = %q, want env-key", reranker.apiKey)
	}
}

func TestNewRerankerDefaults(t *testing.T) {
	reranker, err := NewReranker("rerank-2.5", WithRerankerAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	if reranker.baseURL != defaultBaseURL {
		t.Errorf("baseURL = %q, want %q", reranker.baseURL, defaultBaseURL)
	}
	if reranker.httpClient == nil {
		t.Error("httpClient should default to a non-nil client")
	}
	if reranker.maxRetries != 2 {
		t.Errorf("maxRetries = %d, want 2", reranker.maxRetries)
	}
	if reranker.truncation != nil {
		t.Errorf("truncation = %v, want unset", *reranker.truncation)
	}
}

func TestNewRerankerRejectsNegativeMaxRetries(t *testing.T) {
	if _, err := NewReranker("rerank-2.5", WithRerankerAPIKey("test-key"), WithRerankerMaxRetries(-1)); err == nil {
		t.Fatal("NewReranker should reject negative max retries")
	}
}

func TestNewRerankerHonorsMaxRetries(t *testing.T) {
	var calls int
	reranker := newTestReranker(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		http.Error(w, "temporary upstream failure", http.StatusServiceUnavailable)
	}, WithRerankerMaxRetries(0))

	if _, err := reranker.Rerank(context.Background(), &core.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha"},
	}); err == nil {
		t.Fatal("Rerank should fail on 503 with zero retries")
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1", calls)
	}
}

func TestRerankerName(t *testing.T) {
	reranker, err := NewReranker("rerank-2.5", WithRerankerAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	if reranker.Name() != "rerank-2.5" {
		t.Errorf("Name = %q, want rerank-2.5", reranker.Name())
	}
}

func TestMustNewReranker(t *testing.T) {
	reranker := MustNewReranker("rerank-2.5", WithRerankerAPIKey("test-key"))
	if reranker.Name() != "rerank-2.5" {
		t.Errorf("Name = %q, want rerank-2.5", reranker.Name())
	}
}

func TestMustNewRerankerPanicsWithoutAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "")
	defer func() {
		if recover() == nil {
			t.Fatal("MustNewReranker should panic without an API key")
		}
	}()
	MustNewReranker("rerank-2.5")
}
