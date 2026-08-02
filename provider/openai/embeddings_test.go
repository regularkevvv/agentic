package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/option"

	"github.com/regularkevvv/agentic/internal/retrieval"
)

func TestEmbedderWithLocalServer(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode request body: %v", err)
		}

		// Vectors returned out of order on purpose: placement must follow index.
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[
				{"object":"embedding","index":1,"embedding":[0.3,0.4]},
				{"object":"embedding","index":0,"embedding":[0.1,0.2]}
			],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":5,"total_tokens":5}
		}`)
	}))
	defer server.Close()

	embedder, err := NewEmbedder(
		"text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:     []string{"first", "second"},
		InputType: retrieval.EmbeddingInputQuery,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if got, want := gotBody["model"], "text-embedding-3-small"; got != want {
		t.Errorf("model = %v, want %v", got, want)
	}
	input, _ := gotBody["input"].([]any)
	if len(input) != 2 || input[0] != "first" || input[1] != "second" {
		t.Errorf("input = %v, want [first second]", gotBody["input"])
	}
	if _, ok := gotBody["dimensions"]; ok {
		t.Errorf("dimensions should be omitted when zero, got %v", gotBody["dimensions"])
	}
	if _, ok := gotBody["input_type"]; ok {
		t.Errorf("input_type must not be sent to OpenAI, got %v", gotBody["input_type"])
	}

	if len(resp.Vectors) != 2 {
		t.Fatalf("vectors = %d, want 2", len(resp.Vectors))
	}
	if resp.Vectors[0][0] != 0.1 || resp.Vectors[0][1] != 0.2 {
		t.Errorf("vector 0 = %v, want [0.1 0.2] (index-based placement)", resp.Vectors[0])
	}
	if resp.Vectors[1][0] != 0.3 || resp.Vectors[1][1] != 0.4 {
		t.Errorf("vector 1 = %v, want [0.3 0.4] (index-based placement)", resp.Vectors[1])
	}
	if resp.Model != "text-embedding-3-small" {
		t.Errorf("model = %q, want text-embedding-3-small", resp.Model)
	}
	if resp.Usage.PromptTokens != 5 || resp.Usage.TotalTokens != 5 {
		t.Errorf("usage = %+v, want prompt/total 5", resp.Usage)
	}
}

func TestEmbedderSendsDimensions(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[{"object":"embedding","index":0,"embedding":[0.1]}],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`)
	}))
	defer server.Close()

	embedder, err := NewEmbedder(
		"text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:      []string{"text"},
		Dimensions: 256,
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if got, want := gotBody["dimensions"], float64(256); got != want {
		t.Errorf("dimensions = %v, want %v", got, want)
	}
}

func TestEmbedderVectorCountMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[{"object":"embedding","index":0,"embedding":[0.1]}],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":2,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	embedder, err := NewEmbedder(
		"text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input: []string{"first", "second"},
	}); err == nil {
		t.Fatal("Embed should fail when the vector count does not match the input count")
	}
}

func TestEmbedderRejectsInvalidRequest(t *testing.T) {
	embedder, err := NewEmbedder("text-embedding-3-small", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{}); err == nil {
		t.Fatal("Embed should reject an empty request")
	}
}

func TestEmbedderName(t *testing.T) {
	embedder := MustNewEmbedder(
		"text-embedding-3-small",
		WithAPIKey("test-key"),
		WithOrganization("org-123"),
	)
	if embedder.Name() != "text-embedding-3-small" {
		t.Errorf("Name = %q, want text-embedding-3-small", embedder.Name())
	}
}

func TestEmbedderSurfacesAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"invalid model"}}`, http.StatusBadRequest)
	}))
	defer server.Close()

	embedder, err := NewEmbedder(
		"text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client()), option.WithMaxRetries(0)),
	)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}}); err == nil {
		t.Fatal("Embed should surface the API error")
	}
}

func TestEmbedderVectorIndexOutOfRange(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[{"object":"embedding","index":5,"embedding":[0.1]}],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`)
	}))
	defer server.Close()

	embedder, err := NewEmbedder(
		"text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"text"}}); err == nil {
		t.Fatal("Embed should reject an out-of-range vector index")
	}
}

// TestEmbedderVectorDuplicateIndex covers a malformed or proxied response that
// returns the right number of vectors but repeats an index. Without a
// duplicate check the count guard passes and another input's slot is left nil,
// so a caller joins a nil vector to its source text with no error. This path
// matters most for OpenAI-compatible servers (TEI, vLLM, LocalAI) reached
// through the Ollama embedder.
func TestEmbedderVectorDuplicateIndex(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[
				{"object":"embedding","index":0,"embedding":[0.1]},
				{"object":"embedding","index":0,"embedding":[0.2]}
			],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":2,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	embedder, err := NewEmbedder(
		"text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	_, err = embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a", "b"}})
	if err == nil {
		t.Fatal("Embed should reject a response with a duplicate vector index rather than return a nil slot")
	}
}

// TestEmbedderVectorElementType pins that vectors come back as float32. The
// wire format is float64 JSON numbers, but no provider transmits more than
// float32 of precision and the narrower type halves what an index costs.
func TestEmbedderVectorElementType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[{"object":"embedding","index":0,"embedding":[0.5,-0.25]}],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`)
	}))
	defer server.Close()

	embedder, err := NewEmbedder(
		"text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"one"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	vectors := resp.Vectors
	if len(vectors) != 1 || len(vectors[0]) != 2 {
		t.Fatalf("unexpected vectors %#v", vectors)
	}
	if vectors[0][0] != float32(0.5) || vectors[0][1] != float32(-0.25) {
		t.Errorf("vector = %v, want [0.5 -0.25]", vectors[0])
	}
}

// TestEmbedderTruncate pins how the new Truncate field is honored. The OpenAI
// API has no truncation parameter and rejects over-length input rather than
// truncating it, so an explicit request to truncate cannot be satisfied and
// must fail loudly instead of silently storing a partial vector.
func TestEmbedderTruncate(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		if _, ok := body["truncate"]; ok {
			t.Errorf("truncate is not an OpenAI parameter, got %v", body["truncate"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[{"object":"embedding","index":0,"embedding":[0.1]}],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`)
	}))
	defer server.Close()

	embedder, err := NewEmbedder(
		"text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	truncate, reject := true, false
	tests := []struct {
		name      string
		truncate  *bool
		wantError bool
	}{
		{"nil uses the provider default", nil, false},
		{"false matches the API's own behavior", &reject, false},
		{"true cannot be honored", &truncate, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := requests
			_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
				Input:    []string{"one"},
				Truncate: tt.truncate,
			})
			if tt.wantError {
				if err == nil {
					t.Fatal("expected Embed to refuse an unsupported truncation request")
				}
				if requests != before {
					t.Error("expected no request to be sent when truncation is refused")
				}
				return
			}
			if err != nil {
				t.Fatalf("Embed: %v", err)
			}
		})
	}
}
