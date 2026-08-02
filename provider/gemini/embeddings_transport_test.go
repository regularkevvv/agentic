package gemini

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/regularkevvv/agentic/internal/retrieval"
)

// embedServer starts an httptest server on the embeddings route and points the
// SDK at it, returning the decoded request bodies it received.
func embedServer(t *testing.T, handler func(w http.ResponseWriter, body map[string]any)) *[]map[string]any {
	t.Helper()

	var (
		mu     sync.Mutex
		bodies []map[string]any
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "embedContent") && !strings.Contains(r.URL.Path, "batchEmbedContents") {
			http.NotFound(w, r)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)

		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		handler(w, body)
	}))
	t.Cleanup(server.Close)

	t.Setenv("GOOGLE_GEMINI_BASE_URL", server.URL)
	return &bodies
}

func TestEmbedSuccess(t *testing.T) {
	bodies := embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]}]}`)
	})

	e, err := NewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	resp, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:     []string{"first", "second"},
		InputType: retrieval.EmbeddingInputDocument,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(resp.Vectors) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(resp.Vectors))
	}
	// Vectors must stay in input order so a caller can join them back to the
	// source texts by position.
	if resp.Vectors[0][0] != 0.1 || resp.Vectors[1][0] != 0.3 {
		t.Errorf("vectors out of input order: %v", resp.Vectors)
	}
	if resp.Model != "gemini-embedding-001" {
		t.Errorf("unexpected model %q", resp.Model)
	}
	// The Gemini API reports no token usage for embeddings, so nothing is
	// invented here.
	if resp.Usage.TotalTokens != 0 || resp.Usage.PromptTokens != 0 {
		t.Errorf("expected zero usage on the Gemini API, got %+v", resp.Usage)
	}

	if len(*bodies) != 1 {
		t.Fatalf("expected 1 request, got %d", len(*bodies))
	}
	assertTaskType(t, (*bodies)[0], TaskTypeRetrievalDocument)
}

func TestEmbedSendsQueryTaskTypeAndDimensions(t *testing.T) {
	bodies := embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"embeddings":[{"values":[1.0]}]}`)
	})

	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	if _, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:      []string{"a query"},
		InputType:  retrieval.EmbeddingInputQuery,
		Dimensions: 768,
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	body := (*bodies)[0]
	assertTaskType(t, body, TaskTypeRetrievalQuery)
	if !strings.Contains(mustJSON(t, body), "768") {
		t.Errorf("expected the dimensionality override on the wire, got %s", mustJSON(t, body))
	}
}

func TestEmbedOmitsTaskTypeForInputTypeNone(t *testing.T) {
	bodies := embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"embeddings":[{"values":[1.0]}]}`)
	})

	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	if _, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"plain"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if strings.Contains(mustJSON(t, (*bodies)[0]), "taskType") {
		t.Errorf("expected no taskType when InputType is none, got %s", mustJSON(t, (*bodies)[0]))
	}
}

func TestEmbedSendsConfiguredTaskType(t *testing.T) {
	bodies := embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"embeddings":[{"values":[1.0]}]}`)
	})

	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"), WithEmbeddingTaskType(TaskTypeClustering))
	if _, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"plain"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	assertTaskType(t, (*bodies)[0], TaskTypeClustering)
}

// A short response would misalign every vector against its source text, so it
// must be an error rather than a truncated result.
func TestEmbedRejectsVectorCountMismatch(t *testing.T) {
	embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"embeddings":[{"values":[0.1]}]}`)
	})

	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	_, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a", "b"}})
	if err == nil {
		t.Fatal("expected an error for a short response")
	}
	if !strings.Contains(err.Error(), "1 vectors for 2 inputs") {
		t.Errorf("unexpected error: %v", err)
	}
}

// A malformed or proxied response must produce an error, never a panic.
func TestEmbedRejectsMissingVectorValues(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"null entry", `{"embeddings":[{"values":[0.1]},null]}`},
		{"entry without values", `{"embeddings":[{"values":[0.1]},{}]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
				_, _ = io.WriteString(w, tt.body)
			})

			e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
			_, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a", "b"}})
			if err == nil {
				t.Fatal("expected an error for a missing vector")
			}
			if !strings.Contains(err.Error(), "missing vector at index 1") {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestEmbedPropagatesAPIError(t *testing.T) {
	embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"code":500,"message":"boom"}}`)
	})

	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	_, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a"}})
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "gemini embeddings") {
		t.Errorf("expected the error to be tagged with the provider, got %v", err)
	}
}

// Vertex AI is the only backend that reports per-embedding token statistics.
func TestEmbedSumsVertexTokenStatistics(t *testing.T) {
	embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"embeddings":[
			{"values":[0.1],"statistics":{"tokenCount":3}},
			{"values":[0.2],"statistics":{"tokenCount":4}}
		]}`)
	})

	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	resp, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a", "b"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Usage.TotalTokens != 7 {
		t.Errorf("expected 7 total tokens, got %d", resp.Usage.TotalTokens)
	}
	if resp.Usage.PromptTokens != 7 {
		t.Errorf("expected 7 prompt tokens, got %d", resp.Usage.PromptTokens)
	}
}

// Backends that cap a call at one input must be fanned out and reassembled in
// input order. singleInput is set directly so the fan-out path can be driven
// against the Gemini API test transport, which needs no GCP credentials.
func TestEmbedFansOutForSingleInputBackends(t *testing.T) {
	bodies := embedServer(t, func(w http.ResponseWriter, body map[string]any) {
		// Echo a vector derived from the input so ordering is verifiable.
		text := firstContentText(body)
		switch text {
		case "a":
			_, _ = io.WriteString(w, `{"embeddings":[{"values":[1]}]}`)
		case "b":
			_, _ = io.WriteString(w, `{"embeddings":[{"values":[2]}]}`)
		default:
			_, _ = io.WriteString(w, `{"embeddings":[{"values":[3]}]}`)
		}
	})

	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	e.singleInput = true

	resp, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a", "b", "c"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 3 {
		t.Fatalf("expected 3 vectors, got %d", len(resp.Vectors))
	}
	for i, want := range []float32{1, 2, 3} {
		if resp.Vectors[i][0] != want {
			t.Errorf("vector %d = %v, want %v (fan-out lost input order)", i, resp.Vectors[i][0], want)
		}
	}
	if len(*bodies) != 3 {
		t.Errorf("expected one request per input, got %d", len(*bodies))
	}
}

func TestEmbedFanOutPropagatesError(t *testing.T) {
	embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"code":500,"message":"boom"}}`)
	})

	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	e.singleInput = true

	if _, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a", "b"}}); err == nil {
		t.Fatal("expected the fan-out to surface the request error")
	}
}

// A single input on a capped backend takes the direct path, not the fan-out.
func TestEmbedSingleInputBackendWithOneInput(t *testing.T) {
	bodies := embedServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = io.WriteString(w, `{"embeddings":[{"values":[9]}]}`)
	})

	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	e.singleInput = true

	resp, err := e.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"only"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 1 || resp.Vectors[0][0] != 9 {
		t.Errorf("unexpected vectors %v", resp.Vectors)
	}
	if len(*bodies) != 1 {
		t.Errorf("expected 1 request, got %d", len(*bodies))
	}
}

func assertTaskType(t *testing.T, body map[string]any, want string) {
	t.Helper()
	if !strings.Contains(mustJSON(t, body), want) {
		t.Errorf("expected task type %q on the wire, got %s", want, mustJSON(t, body))
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(raw)
}

// firstContentText digs the input text out of a batchEmbedContents payload.
func firstContentText(body map[string]any) string {
	requests, ok := body["requests"].([]any)
	if !ok || len(requests) == 0 {
		return ""
	}
	req, ok := requests[0].(map[string]any)
	if !ok {
		return ""
	}
	content, ok := req["content"].(map[string]any)
	if !ok {
		return ""
	}
	parts, ok := content["parts"].([]any)
	if !ok || len(parts) == 0 {
		return ""
	}
	part, ok := parts[0].(map[string]any)
	if !ok {
		return ""
	}
	text, _ := part["text"].(string)
	return text
}
