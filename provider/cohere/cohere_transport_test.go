package cohere_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/provider/cohere"
)

func ptr[T any](v T) *T { return &v }

// newEmbedServer starts a server that replies with body, records the decoded
// request into *got, and returns an Embedder pointed at it.
func newEmbedServer(t *testing.T, body string, got *map[string]any, opts ...cohere.Option) *cohere.Embedder {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/embed" {
			t.Errorf("path = %q, want /v2/embed", r.URL.Path)
		}
		if got != nil {
			if err := json.NewDecoder(r.Body).Decode(got); err != nil {
				t.Errorf("decode request body: %v", err)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)

	opts = append([]cohere.Option{
		cohere.WithAPIKey("test-key"),
		cohere.WithBaseURL(server.URL),
		cohere.WithHTTPClient(server.Client()),
	}, opts...)

	embedder, err := cohere.New(cohere.DefaultEmbeddingModel, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return embedder
}

// newRerankServer starts a server that replies with body, records the decoded
// request into *got, and returns a Reranker pointed at it.
func newRerankServer(t *testing.T, body string, got *map[string]any, opts ...cohere.RerankerOption) *cohere.Reranker {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/rerank" {
			t.Errorf("path = %q, want /v2/rerank", r.URL.Path)
		}
		if got != nil {
			if err := json.NewDecoder(r.Body).Decode(got); err != nil {
				t.Errorf("decode request body: %v", err)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)

	opts = append([]cohere.RerankerOption{
		cohere.WithRerankerAPIKey("test-key"),
		cohere.WithRerankerBaseURL(server.URL),
		cohere.WithRerankerHTTPClient(server.Client()),
	}, opts...)

	reranker, err := cohere.NewReranker(cohere.DefaultRerankModel, opts...)
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	return reranker
}

// threeVectorResponse holds three distinguishable vectors in a fixed order.
const threeVectorResponse = `{
	"id":"abc",
	"embeddings":{"float":[[1,0,0],[0,1,0],[0,0,1]]},
	"meta":{"billed_units":{"input_tokens":9},"tokens":{"input_tokens":9}}
}`

// TestEmbedPreservesPositionalOrder pins the shape difference that matters most
// against OpenAI and Voyage AI: Cohere's response has no per-vector index
// field, so vectors are aligned with inputs by position and nothing else. A
// reorder here would silently mis-join every vector to the wrong source text.
func TestEmbedPreservesPositionalOrder(t *testing.T) {
	embedder := newEmbedServer(t, threeVectorResponse, nil)

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input: []string{"first", "second", "third"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	want := [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	if len(resp.Vectors) != len(want) {
		t.Fatalf("vectors = %v, want %v", resp.Vectors, want)
	}
	for i := range want {
		for j := range want[i] {
			if resp.Vectors[i][j] != want[i][j] {
				t.Errorf("vectors[%d][%d] = %v, want %v", i, j, resp.Vectors[i][j], want[i][j])
			}
		}
	}
	if resp.Model != cohere.DefaultEmbeddingModel {
		t.Errorf("model = %q, want %q", resp.Model, cohere.DefaultEmbeddingModel)
	}
	if resp.Usage.PromptTokens != 9 || resp.Usage.TotalTokens != 9 {
		t.Errorf("usage = %+v, want 9 prompt and 9 total tokens", resp.Usage)
	}
}

// TestEmbedCountMismatch pins that a response carrying the wrong number of
// vectors is an error. With no index field there is no way to detect a
// misalignment other than by count, so this check is the only thing standing
// between a short reply and a corrupted vector-to-text join.
func TestEmbedCountMismatch(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "too few", body: `{"embeddings":{"float":[[1,0]]},"meta":{}}`},
		{name: "too many", body: `{"embeddings":{"float":[[1,0],[0,1],[1,1]]},"meta":{}}`},
		{name: "float key absent", body: `{"embeddings":{"int8":[[1,0],[0,1]]},"meta":{}}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedder := newEmbedServer(t, tt.body, nil)

			_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
				Input: []string{"a", "b"},
			})
			if err == nil {
				t.Fatal("Embed should fail when the vector count does not match the input count")
			}
			if !strings.Contains(err.Error(), "for 2 inputs") {
				t.Errorf("error = %v, want it to report the input count", err)
			}
		})
	}
}

// TestEmbedInputTypeMapping pins the mapping onto Cohere's required input_type,
// including the choice made for EmbeddingInputNone, which has no natural answer.
func TestEmbedInputTypeMapping(t *testing.T) {
	tests := []struct {
		name      string
		inputType retrieval.EmbeddingInputType
		opts      []cohere.Option
		want      string
	}{
		{
			name:      "query",
			inputType: retrieval.EmbeddingInputQuery,
			want:      "search_query",
		},
		{
			name:      "document",
			inputType: retrieval.EmbeddingInputDocument,
			want:      "search_document",
		},
		{
			name:      "none defaults to search_document",
			inputType: retrieval.EmbeddingInputNone,
			want:      "search_document",
		},
		{
			name:      "none honors the configured default",
			inputType: retrieval.EmbeddingInputNone,
			opts:      []cohere.Option{cohere.WithDefaultInputType(cohere.InputTypeClustering)},
			want:      "clustering",
		},
		{
			name:      "an explicit type ignores the configured default",
			inputType: retrieval.EmbeddingInputQuery,
			opts:      []cohere.Option{cohere.WithDefaultInputType(cohere.InputTypeClassification)},
			want:      "search_query",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBody map[string]any
			embedder := newEmbedServer(t, `{"embeddings":{"float":[[1]]},"meta":{}}`, &gotBody, tt.opts...)

			if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
				Input:     []string{"text"},
				InputType: tt.inputType,
			}); err != nil {
				t.Fatalf("Embed: %v", err)
			}

			if got := gotBody["input_type"]; got != tt.want {
				t.Errorf("input_type = %v, want %q", got, tt.want)
			}
		})
	}
}

// TestEmbedAlwaysRequestsFloatEmbeddings pins that embedding_types is sent and
// pinned to float, which is what keeps the response shape to the single key
// this package reads.
func TestEmbedAlwaysRequestsFloatEmbeddings(t *testing.T) {
	var gotBody map[string]any
	embedder := newEmbedServer(t, `{"embeddings":{"float":[[1]]},"meta":{}}`, &gotBody)

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	types, ok := gotBody["embedding_types"].([]any)
	if !ok || len(types) != 1 || types[0] != "float" {
		t.Errorf("embedding_types = %v, want [float]", gotBody["embedding_types"])
	}
}

// TestEmbedSendsDimensionsAndTruncate pins the optional wire fields, including
// that a per-request Truncate overrides the constructor default and that
// omitting both leaves the field out so the API default applies.
func TestEmbedSendsDimensionsAndTruncate(t *testing.T) {
	tests := []struct {
		name        string
		opts        []cohere.Option
		dimensions  int
		truncate    *bool
		wantDims    any
		wantPresent bool
		wantTrunc   string
	}{
		{name: "neither set omits both", wantDims: nil, wantPresent: false},
		{name: "dimensions", dimensions: 512, wantDims: float64(512), wantPresent: false},
		{name: "constructor truncation", opts: []cohere.Option{cohere.WithTruncation(false)}, wantPresent: true, wantTrunc: "NONE"},
		{name: "request truncation true", truncate: ptr(true), wantPresent: true, wantTrunc: "END"},
		{
			name:        "request false overrides constructor true",
			opts:        []cohere.Option{cohere.WithTruncation(true)},
			truncate:    ptr(false),
			wantPresent: true,
			wantTrunc:   "NONE",
		},
		{
			name:        "request true overrides constructor false",
			opts:        []cohere.Option{cohere.WithTruncation(false)},
			truncate:    ptr(true),
			wantPresent: true,
			wantTrunc:   "END",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBody map[string]any
			embedder := newEmbedServer(t, `{"embeddings":{"float":[[1]]},"meta":{}}`, &gotBody, tt.opts...)

			if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
				Input:      []string{"text"},
				Dimensions: tt.dimensions,
				Truncate:   tt.truncate,
			}); err != nil {
				t.Fatalf("Embed: %v", err)
			}

			if got := gotBody["output_dimension"]; got != tt.wantDims {
				t.Errorf("output_dimension = %v, want %v", got, tt.wantDims)
			}
			got, present := gotBody["truncate"]
			if present != tt.wantPresent {
				t.Fatalf("truncate present = %v (%v), want %v", present, got, tt.wantPresent)
			}
			if tt.wantPresent && got != tt.wantTrunc {
				t.Errorf("truncate = %v, want %q", got, tt.wantTrunc)
			}
		})
	}
}

// TestEmbedUsageFallsBackToTokens pins that a response reporting only the
// tokens block (no billed_units) still yields usage rather than zero.
func TestEmbedUsageFallsBackToTokens(t *testing.T) {
	embedder := newEmbedServer(t, `{"embeddings":{"float":[[1]]},"meta":{"tokens":{"input_tokens":7}}}`, nil)

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Usage.PromptTokens != 7 || resp.Usage.TotalTokens != 7 {
		t.Errorf("usage = %+v, want 7 tokens from the fallback", resp.Usage)
	}
}

// TestErrorEnvelopeSurfacesMessage pins that Cohere's {"id","message"} error
// body is surfaced through the message field, and that a body which is not that
// shape still reaches the caller verbatim.
func TestErrorEnvelopeSurfacesMessage(t *testing.T) {
	tests := []struct {
		name    string
		status  int
		body    string
		want    string
		notWant string
	}{
		{
			name:    "documented envelope",
			status:  http.StatusBadRequest,
			body:    `{"id":"req-1","message":"invalid request: too many texts"}`,
			want:    "invalid request: too many texts",
			notWant: "req-1",
		},
		{
			name:   "envelope without a message",
			status: http.StatusBadRequest,
			body:   `{"id":"req-2"}`,
			want:   "req-2",
		},
		{
			name:   "non-JSON body from a proxy",
			status: http.StatusBadGateway,
			body:   "<html>bad gateway</html>",
			want:   "bad gateway",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = io.WriteString(w, tt.body)
			}))
			t.Cleanup(server.Close)

			embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
				cohere.WithAPIKey("test-key"),
				cohere.WithBaseURL(server.URL),
				cohere.WithHTTPClient(server.Client()),
				cohere.WithMaxRetries(0),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			_, err = embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}})
			if err == nil {
				t.Fatal("Embed should fail on a non-200 response")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want it to mention %q", err, tt.want)
			}
			if tt.notWant != "" && strings.Contains(err.Error(), tt.notWant) {
				t.Errorf("error = %v, should not carry %q", err, tt.notWant)
			}
		})
	}
}

// TestEmbedRetriesRateLimit pins that a 429 is retried and that a later
// attempt's success is what the caller sees.
func TestEmbedRetriesRateLimit(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"id":"req-3","message":"rate limit exceeded"}`)
			return
		}
		_, _ = io.WriteString(w, `{"embeddings":{"float":[[0.5,0.25]]},"meta":{"billed_units":{"input_tokens":2}}}`)
	}))
	t.Cleanup(server.Close)

	embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
		cohere.WithAPIKey("test-key"),
		cohere.WithBaseURL(server.URL),
		cohere.WithHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2 (the 429 must be retried)", calls)
	}
	if len(resp.Vectors) != 1 || resp.Vectors[0][0] != 0.5 {
		t.Errorf("vectors = %v, want the retried attempt's body", resp.Vectors)
	}
}

// TestEmbedDoesNotRetryClientError pins that a 400 is returned immediately: the
// request is malformed and resending it verbatim can only fail again.
func TestEmbedDoesNotRetryClientError(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"id":"req-4","message":"bad input"}`)
	}))
	t.Cleanup(server.Close)

	embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
		cohere.WithAPIKey("test-key"),
		cohere.WithBaseURL(server.URL),
		cohere.WithHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}}); err == nil {
		t.Fatal("Embed should fail on a 400")
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1 (a 400 must not be retried)", calls)
	}
}

// TestEmbedRetriesExhausted pins that the last error is returned once the retry
// budget runs out.
func TestEmbedRetriesExhausted(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"id":"req-5","message":"upstream unavailable"}`)
	}))
	t.Cleanup(server.Close)

	embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
		cohere.WithAPIKey("test-key"),
		cohere.WithBaseURL(server.URL),
		cohere.WithHTTPClient(server.Client()),
		cohere.WithMaxRetries(1),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}})
	if err == nil || !strings.Contains(err.Error(), "upstream unavailable") {
		t.Fatalf("error = %v, want the last attempt's message", err)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2 (one attempt plus one retry)", calls)
	}
}

// TestEmbedMalformedJSON pins that a body which is not JSON at all fails to
// decode rather than yielding an empty vector list.
func TestEmbedMalformedJSON(t *testing.T) {
	embedder := newEmbedServer(t, `{"embeddings":`, nil)

	_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}})
	if err == nil || !strings.Contains(err.Error(), "decode response") {
		t.Fatalf("error = %v, want a decode failure", err)
	}
}

// TestEmbedValidatesRequest pins that Validate runs before any HTTP call.
func TestEmbedValidatesRequest(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
	}))
	t.Cleanup(server.Close)

	embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
		cohere.WithAPIKey("test-key"),
		cohere.WithBaseURL(server.URL),
		cohere.WithHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{}); err == nil {
		t.Fatal("Embed should reject an empty input list")
	}
	if calls != 0 {
		t.Errorf("calls = %d, want 0 (validation must precede the request)", calls)
	}
}

// TestRerankSortsByDescendingScore pins that results are ordered by score
// regardless of the order the API returned them in, and that Document is filled
// from the request slice.
func TestRerankSortsByDescendingScore(t *testing.T) {
	body := `{
		"id":"r1",
		"results":[
			{"index":1,"relevance_score":0.10},
			{"index":2,"relevance_score":0.90},
			{"index":0,"relevance_score":0.50}
		],
		"meta":{"billed_units":{"search_units":1}}
	}`
	reranker := newRerankServer(t, body, nil)

	docs := []string{"alpha", "bravo", "charlie"}
	resp, err := reranker.Rerank(context.Background(), &retrieval.RerankRequest{
		Query:     "which one",
		Documents: docs,
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}

	wantIndexes := []int{2, 0, 1}
	if len(resp.Results) != len(wantIndexes) {
		t.Fatalf("results = %+v, want %d entries", resp.Results, len(wantIndexes))
	}
	for i, want := range wantIndexes {
		got := resp.Results[i]
		if got.Index != want {
			t.Errorf("results[%d].Index = %d, want %d", i, got.Index, want)
		}
		if got.Document != docs[want] {
			t.Errorf("results[%d].Document = %q, want %q from the request slice", i, got.Document, docs[want])
		}
	}
	if resp.Results[0].Score != 0.90 {
		t.Errorf("top score = %v, want 0.90", resp.Results[0].Score)
	}
	if resp.Model != cohere.DefaultRerankModel {
		t.Errorf("model = %q, want %q", resp.Model, cohere.DefaultRerankModel)
	}
}

// TestRerankUsageIsSearchUnits pins that Cohere's billing unit is mapped to
// SearchUnits and that TotalTokens is left zero: reranking is not metered in
// tokens, and reporting a token count the bill does not reflect would mislead
// anyone summing usage across providers.
func TestRerankUsageIsSearchUnits(t *testing.T) {
	body := `{
		"id":"r2",
		"results":[{"index":0,"relevance_score":0.4}],
		"meta":{"billed_units":{"search_units":3},"tokens":{"input_tokens":250}}
	}`
	reranker := newRerankServer(t, body, nil)

	resp, err := reranker.Rerank(context.Background(), &retrieval.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha"},
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if resp.Usage.SearchUnits != 3 {
		t.Errorf("SearchUnits = %d, want 3", resp.Usage.SearchUnits)
	}
	if resp.Usage.TotalTokens != 0 {
		t.Errorf("TotalTokens = %d, want 0 (reranking is not metered in tokens)", resp.Usage.TotalTokens)
	}
}

// TestRerankBoundsChecksIndex pins that an index the API had no business
// returning produces an error rather than panicking inside the caller.
func TestRerankBoundsChecksIndex(t *testing.T) {
	tests := []struct {
		name  string
		index string
	}{
		{name: "past the end", index: "7"},
		{name: "equal to the length", index: "2"},
		{name: "negative", index: "-1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"id":"r3","results":[{"index":` + tt.index + `,"relevance_score":0.5}],"meta":{}}`
			reranker := newRerankServer(t, body, nil)

			_, err := reranker.Rerank(context.Background(), &retrieval.RerankRequest{
				Query:     "q",
				Documents: []string{"alpha", "bravo"},
			})
			if err == nil {
				t.Fatal("Rerank should fail on an out-of-range result index")
			}
			if !strings.Contains(err.Error(), "out of range for 2 documents") {
				t.Errorf("error = %v, want it to report the document count", err)
			}
		})
	}
}

// TestRerankWireBody pins the request fields, including that return_documents
// is never sent and that TopN is clamped to the document count.
func TestRerankWireBody(t *testing.T) {
	tests := []struct {
		name        string
		topN        int
		wantPresent bool
		want        float64
	}{
		{name: "zero omits top_n", topN: 0, wantPresent: false},
		{name: "within range", topN: 2, wantPresent: true, want: 2},
		{name: "larger than the document count is clamped", topN: 99, wantPresent: true, want: 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBody map[string]any
			reranker := newRerankServer(t, `{"results":[],"meta":{}}`, &gotBody)

			if _, err := reranker.Rerank(context.Background(), &retrieval.RerankRequest{
				Query:     "which one",
				Documents: []string{"alpha", "bravo", "charlie"},
				TopN:      tt.topN,
			}); err != nil {
				t.Fatalf("Rerank: %v", err)
			}

			if _, present := gotBody["return_documents"]; present {
				t.Error("return_documents must not be sent")
			}
			if got := gotBody["model"]; got != cohere.DefaultRerankModel {
				t.Errorf("model = %v, want %q", got, cohere.DefaultRerankModel)
			}
			if got := gotBody["query"]; got != "which one" {
				t.Errorf("query = %v, want the request query", got)
			}
			got, present := gotBody["top_n"]
			if present != tt.wantPresent {
				t.Fatalf("top_n present = %v (%v), want %v", present, got, tt.wantPresent)
			}
			if tt.wantPresent && got != tt.want {
				t.Errorf("top_n = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestRerankRetriesRateLimit pins that the reranker shares the embedder's retry
// behavior, which is the point of the shared client.
func TestRerankRetriesRateLimit(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"id":"r4","message":"rate limit exceeded"}`)
			return
		}
		_, _ = io.WriteString(w, `{"results":[{"index":0,"relevance_score":0.8}],"meta":{"billed_units":{"search_units":1}}}`)
	}))
	t.Cleanup(server.Close)

	reranker, err := cohere.NewReranker(cohere.DefaultRerankModel,
		cohere.WithRerankerAPIKey("test-key"),
		cohere.WithRerankerBaseURL(server.URL),
		cohere.WithRerankerHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	resp, err := reranker.Rerank(context.Background(), &retrieval.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha"},
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2 (the 429 must be retried)", calls)
	}
	if len(resp.Results) != 1 || resp.Results[0].Score != 0.8 {
		t.Errorf("results = %+v, want the retried attempt's body", resp.Results)
	}
}

// TestRerankSurfacesErrorEnvelope pins that a failed rerank call returns the
// API's message rather than an empty result set, which a caller ranking on the
// output would otherwise read as "nothing is relevant".
func TestRerankSurfacesErrorEnvelope(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"id":"r5","message":"too many documents"}`)
	}))
	t.Cleanup(server.Close)

	reranker, err := cohere.NewReranker(cohere.DefaultRerankModel,
		cohere.WithRerankerAPIKey("test-key"),
		cohere.WithRerankerBaseURL(server.URL),
		cohere.WithRerankerHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	resp, err := reranker.Rerank(context.Background(), &retrieval.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha"},
	})
	if err == nil {
		t.Fatal("Rerank should fail on a 400")
	}
	if resp != nil {
		t.Errorf("response = %+v, want nil alongside the error", resp)
	}
	if !strings.Contains(err.Error(), "too many documents") {
		t.Errorf("error = %v, want the API message", err)
	}
}

// TestRerankMalformedJSON pins that an undecodable body is an error.
func TestRerankMalformedJSON(t *testing.T) {
	reranker := newRerankServer(t, `{"results":`, nil)

	_, err := reranker.Rerank(context.Background(), &retrieval.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha"},
	})
	if err == nil || !strings.Contains(err.Error(), "decode response") {
		t.Fatalf("error = %v, want a decode failure", err)
	}
}

// TestRerankValidatesRequest pins that Validate runs before any HTTP call.
func TestRerankValidatesRequest(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
	}))
	t.Cleanup(server.Close)

	reranker, err := cohere.NewReranker(cohere.DefaultRerankModel,
		cohere.WithRerankerAPIKey("test-key"),
		cohere.WithRerankerBaseURL(server.URL),
		cohere.WithRerankerHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	if _, err := reranker.Rerank(context.Background(), &retrieval.RerankRequest{Documents: []string{"a"}}); err == nil {
		t.Fatal("Rerank should reject an empty query")
	}
	if calls != 0 {
		t.Errorf("calls = %d, want 0 (validation must precede the request)", calls)
	}
}

// TestAuthHeaderIsSent pins the bearer token on both endpoints.
func TestAuthHeaderIsSent(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if r.URL.Path == "/v2/rerank" {
			_, _ = io.WriteString(w, `{"results":[],"meta":{}}`)
			return
		}
		_, _ = io.WriteString(w, `{"embeddings":{"float":[[1]]},"meta":{}}`)
	}))
	t.Cleanup(server.Close)

	embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
		cohere.WithAPIKey("embed-key"),
		cohere.WithBaseURL(server.URL),
		cohere.WithHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if gotAuth != "Bearer embed-key" {
		t.Errorf("embed Authorization = %q, want %q", gotAuth, "Bearer embed-key")
	}

	reranker, err := cohere.NewReranker(cohere.DefaultRerankModel,
		cohere.WithRerankerAPIKey("rerank-key"),
		cohere.WithRerankerBaseURL(server.URL),
		cohere.WithRerankerHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	if _, err := reranker.Rerank(context.Background(), &retrieval.RerankRequest{
		Query:     "q",
		Documents: []string{"alpha"},
	}); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if gotAuth != "Bearer rerank-key" {
		t.Errorf("rerank Authorization = %q, want %q", gotAuth, "Bearer rerank-key")
	}
}

// roundTripFunc adapts a function to http.RoundTripper so a test can inject
// transport-level failures the httptest server cannot produce.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// errReadCloser fails on Read, Close, or both.
type errReadCloser struct {
	readErr  error
	closeErr error
}

func (e errReadCloser) Read([]byte) (int, error) {
	if e.readErr != nil {
		return 0, e.readErr
	}
	return 0, io.EOF
}

func (e errReadCloser) Close() error { return e.closeErr }

// TestEmbedInvalidBaseURL pins that a base URL that cannot form a request
// surfaces as an error rather than a panic.
func TestEmbedInvalidBaseURL(t *testing.T) {
	embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
		cohere.WithAPIKey("test-key"),
		cohere.WithBaseURL("http://exa\x7fmple.com"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}}); err == nil {
		t.Fatal("Embed should fail when the base URL cannot form a request")
	}
}

// TestEmbedRetriesNetworkError pins that a transport-level failure is retried.
func TestEmbedRetriesNetworkError(t *testing.T) {
	var calls int
	embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
		cohere.WithAPIKey("test-key"),
		cohere.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			calls++
			if calls == 1 {
				return nil, errors.New("dial tcp: connection reset")
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"embeddings":{"float":[[0.5]]},"meta":{}}`)),
				Header:     http.Header{},
			}, nil
		})}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2", calls)
	}
	if len(resp.Vectors) != 1 || resp.Vectors[0][0] != 0.5 {
		t.Errorf("vectors = %v", resp.Vectors)
	}
}

// TestEmbedNetworkErrorExhausted pins that a transport failure on the final
// attempt is returned to the caller.
func TestEmbedNetworkErrorExhausted(t *testing.T) {
	embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
		cohere.WithAPIKey("test-key"),
		cohere.WithMaxRetries(0),
		cohere.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return nil, errors.New("dial tcp: connection reset")
		})}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}})
	if err == nil || !strings.Contains(err.Error(), "connection reset") {
		t.Fatalf("error = %v, want the transport failure", err)
	}
}

// TestEmbedBodyFailures pins that a response body that cannot be read or closed
// surfaces as an error instead of being decoded as an empty payload.
func TestEmbedBodyFailures(t *testing.T) {
	tests := []struct {
		name string
		body errReadCloser
		want string
	}{
		{name: "read error", body: errReadCloser{readErr: errors.New("unexpected EOF reading body")}, want: "unexpected EOF"},
		{name: "close error", body: errReadCloser{closeErr: errors.New("closing body failed")}, want: "closing body failed"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
				cohere.WithAPIKey("test-key"),
				cohere.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: http.StatusOK, Body: tt.body, Header: http.Header{}}, nil
				})}),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			_, err = embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}})
			if err == nil {
				t.Fatalf("Embed should fail on a %s", tt.name)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want it to mention %q", err, tt.want)
			}
		})
	}
}

// TestEmbedDeadlineDuringBackoff pins that a deadline expiring while the retry
// loop waits out its backoff aborts promptly with the context error, for both
// the transport-failure and the retryable-status paths. The deadline is far
// shorter than the one-second first backoff, so the wait is what trips it.
func TestEmbedDeadlineDuringBackoff(t *testing.T) {
	tests := []struct {
		name  string
		reply func() (*http.Response, error)
	}{
		{
			name:  "after a transport failure",
			reply: func() (*http.Response, error) { return nil, errors.New("dial tcp: connection reset") },
		},
		{
			name: "after a retryable status",
			reply: func() (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusTooManyRequests,
					Body:       io.NopCloser(strings.NewReader(`{"id":"x","message":"slow down"}`)),
					Header:     http.Header{},
				}, nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()

			var calls int
			embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
				cohere.WithAPIKey("test-key"),
				cohere.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
					calls++
					return tt.reply()
				})}),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			_, err = embedder.Embed(ctx, &retrieval.EmbeddingRequest{Input: []string{"text"}})
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("error = %v, want context.DeadlineExceeded", err)
			}
			if calls != 1 {
				t.Errorf("calls = %d, want 1 (the expired deadline must stop the retry loop)", calls)
			}
		})
	}
}
