package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/option"

	"github.com/regularkevvv/agentic/internal/core"
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

	resp, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input:     []string{"first", "second"},
		InputType: core.EmbeddingInputQuery,
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

	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
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

	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
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
	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{}); err == nil {
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

	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"text"}}); err == nil {
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

	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"text"}}); err == nil {
		t.Fatal("Embed should reject an out-of-range vector index")
	}
}
