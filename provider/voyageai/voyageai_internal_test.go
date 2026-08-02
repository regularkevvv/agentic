package voyageai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/retrieval"
)

const embedFixture = `{
	"object":"list",
	"data":[
		{"object":"embedding","index":1,"embedding":[0.3,0.4]},
		{"object":"embedding","index":0,"embedding":[0.1,0.2]}
	],
	"model":"voyage-3.5",
	"usage":{"total_tokens":7}
}`

func newTestEmbedder(t *testing.T, handler http.HandlerFunc, opts ...Option) *Embedder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]Option{
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
	}, opts...)

	embedder, err := New("voyage-3.5", opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return embedder
}

func TestEmbedRequestAndResponse(t *testing.T) {
	var gotAuth, gotPath string
	var gotBody map[string]any
	embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, embedFixture)
	})

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:     []string{"first", "second"},
		InputType: retrieval.EmbeddingInputQuery,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if gotAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", gotAuth)
	}
	if gotPath != "/embeddings" {
		t.Errorf("path = %q, want /embeddings", gotPath)
	}
	if got, want := gotBody["model"], "voyage-3.5"; got != want {
		t.Errorf("model = %v, want %v", got, want)
	}
	if got, want := gotBody["input_type"], "query"; got != want {
		t.Errorf("input_type = %v, want %v", got, want)
	}
	if _, ok := gotBody["truncation"]; ok {
		t.Errorf("truncation should be omitted unless set, got %v", gotBody["truncation"])
	}
	if _, ok := gotBody["output_dimension"]; ok {
		t.Errorf("output_dimension should be omitted when zero, got %v", gotBody["output_dimension"])
	}

	// Fixture returns vectors out of order: placement must follow index.
	if resp.Vectors[0][0] != 0.1 || resp.Vectors[1][0] != 0.3 {
		t.Errorf("vectors = %v, want index-based placement", resp.Vectors)
	}
	if resp.Model != "voyage-3.5" {
		t.Errorf("model = %q, want voyage-3.5", resp.Model)
	}
	if resp.Usage.TotalTokens != 7 {
		t.Errorf("total tokens = %d, want 7", resp.Usage.TotalTokens)
	}
	if resp.Usage.PromptTokens != 0 {
		t.Errorf("prompt tokens = %d, want 0 (Voyage does not report them)", resp.Usage.PromptTokens)
	}
}

func TestEmbedInputTypeMapping(t *testing.T) {
	tests := []struct {
		name      string
		inputType retrieval.EmbeddingInputType
		want      string // "" means the field must be absent
	}{
		{name: "query", inputType: retrieval.EmbeddingInputQuery, want: "query"},
		{name: "document", inputType: retrieval.EmbeddingInputDocument, want: "document"},
		{name: "none omits the field", inputType: retrieval.EmbeddingInputNone, want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBody map[string]any
			embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
					t.Errorf("decode request body: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{
					"object":"list",
					"data":[{"object":"embedding","index":0,"embedding":[0.1]},{"object":"embedding","index":1,"embedding":[0.2]}],
					"model":"voyage-3.5",
					"usage":{"total_tokens":2}
				}`)
			})

			if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
				Input:     []string{"a", "b"},
				InputType: tt.inputType,
			}); err != nil {
				t.Fatalf("Embed: %v", err)
			}

			got, present := gotBody["input_type"]
			if tt.want == "" {
				if present {
					t.Errorf("input_type = %v, want absent", got)
				}
				return
			}
			if got != tt.want {
				t.Errorf("input_type = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEmbedSendsTruncationAndDimensions(t *testing.T) {
	var gotBody map[string]any
	embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[{"object":"embedding","index":0,"embedding":[0.1]}],
			"model":"voyage-3.5",
			"usage":{"total_tokens":1}
		}`)
	}, WithTruncation(false))

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:      []string{"text"},
		Dimensions: 512,
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if got, want := gotBody["truncation"], false; got != want {
		t.Errorf("truncation = %v, want %v", got, want)
	}
	if got, want := gotBody["output_dimension"], float64(512); got != want {
		t.Errorf("output_dimension = %v, want %v", got, want)
	}
}

func TestEmbedRetriesTransientStatus(t *testing.T) {
	var calls int
	embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			http.Error(w, "temporary upstream failure", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],
			"model":"voyage-3.5",
			"usage":{"total_tokens":1}
		}`)
	})

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2", calls)
	}
	if len(resp.Vectors) != 1 || resp.Vectors[0][0] != 0.1 {
		t.Fatalf("vectors = %v", resp.Vectors)
	}
}

func TestEmbedDoesNotRetryClientError(t *testing.T) {
	var calls int
	embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		http.Error(w, `{"detail":"model not found"}`, http.StatusBadRequest)
	})

	_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}})
	if err == nil {
		t.Fatal("Embed should fail on 400")
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1 (400 is not retryable)", calls)
	}
	if !strings.Contains(err.Error(), "status 400") {
		t.Fatalf("error = %v, want status 400 mentioned", err)
	}
}

func TestEmbedVectorCountMismatch(t *testing.T) {
	embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[{"object":"embedding","index":0,"embedding":[0.1]}],
			"model":"voyage-3.5",
			"usage":{"total_tokens":2}
		}`)
	})

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input: []string{"first", "second"},
	}); err == nil {
		t.Fatal("Embed should fail when the vector count does not match the input count")
	}
}

func TestNewRequiresAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "")
	if _, err := New("voyage-3.5"); err == nil {
		t.Fatal("New should fail without an API key")
	}
}

func TestNewReadsEnvAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-key")
	embedder, err := New("voyage-3.5")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if embedder.apiKey != "env-key" {
		t.Errorf("apiKey = %q, want env-key", embedder.apiKey)
	}
}

func TestNewDefaults(t *testing.T) {
	embedder, err := New("voyage-3.5", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if embedder.baseURL != defaultBaseURL {
		t.Errorf("baseURL = %q, want %q", embedder.baseURL, defaultBaseURL)
	}
	if embedder.httpClient == nil {
		t.Error("httpClient should default to a non-nil client")
	}
	if embedder.maxRetries != 2 {
		t.Errorf("maxRetries = %d, want 2", embedder.maxRetries)
	}
	if embedder.truncation != nil {
		t.Errorf("truncation = %v, want unset", *embedder.truncation)
	}
}

func TestNewRejectsNegativeMaxRetries(t *testing.T) {
	if _, err := New("voyage-3.5", WithAPIKey("test-key"), WithMaxRetries(-1)); err == nil {
		t.Fatal("New should reject negative max retries")
	}
}

func TestName(t *testing.T) {
	embedder, err := New("voyage-3.5", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if embedder.Name() != "voyage-3.5" {
		t.Errorf("Name = %q, want voyage-3.5", embedder.Name())
	}
}

func TestMustNew(t *testing.T) {
	embedder := MustNew("voyage-3.5", WithAPIKey("test-key"))
	if embedder.Name() != "voyage-3.5" {
		t.Errorf("Name = %q, want voyage-3.5", embedder.Name())
	}
}

func TestMustNewPanicsWithoutAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "")
	defer func() {
		if recover() == nil {
			t.Fatal("MustNew should panic without an API key")
		}
	}()
	MustNew("voyage-3.5")
}

func TestEmbedZeroRetriesFailsImmediately(t *testing.T) {
	var calls int
	embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		http.Error(w, "temporary upstream failure", http.StatusServiceUnavailable)
	}, WithMaxRetries(0))

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}}); err == nil {
		t.Fatal("Embed should fail on 503 with zero retries")
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1", calls)
	}
}

func TestEmbedNetworkErrorZeroRetries(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	serverURL := server.URL
	client := server.Client()
	server.Close() // connection refused from here on

	embedder, err := New("voyage-3.5",
		WithAPIKey("test-key"),
		WithBaseURL(serverURL),
		WithHTTPClient(client),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}}); err == nil {
		t.Fatal("Embed should surface the network error")
	}
}

func TestEmbedCanceledContext(t *testing.T) {
	embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, embedFixture)
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := embedder.Embed(ctx, &retrieval.EmbeddingRequest{Input: []string{"hello"}}); err == nil {
		t.Fatal("Embed should fail with a canceled context")
	}
}

func TestEmbedDecodeError(t *testing.T) {
	embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, "not json")
	})

	_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}})
	if err == nil || !strings.Contains(err.Error(), "decode response") {
		t.Fatalf("error = %v, want decode response error", err)
	}
}

func TestEmbedVectorIndexOutOfRange(t *testing.T) {
	embedder := newTestEmbedder(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[{"object":"embedding","index":5,"embedding":[0.1]}],
			"model":"voyage-3.5",
			"usage":{"total_tokens":1}
		}`)
	})

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}}); err == nil {
		t.Fatal("Embed should reject an out-of-range vector index")
	}
}

func TestEmbedRejectsInvalidRequest(t *testing.T) {
	embedder, err := New("voyage-3.5", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{}); err == nil {
		t.Fatal("Embed should reject an empty request")
	}
}

func TestRetryDelayCapped(t *testing.T) {
	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{attempt: 0, want: 1 * time.Second},
		{attempt: 1, want: 2 * time.Second},
		{attempt: 2, want: 4 * time.Second},
		{attempt: 5, want: 4 * time.Second},
	}
	for _, tt := range tests {
		if got := retryDelay(tt.attempt); got != tt.want {
			t.Errorf("retryDelay(%d) = %v, want %v", tt.attempt, got, tt.want)
		}
	}
}

func TestSleepRetryCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := sleepRetry(ctx, time.Minute); err == nil {
		t.Fatal("sleepRetry should return the context error")
	}
}
