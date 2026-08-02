package ollama_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/provider/ollama"
)

const embeddingsResponse = `{
	"object": "list",
	"model": "nomic-embed-text",
	"data": [
		{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
		{"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
	],
	"usage": {"prompt_tokens": 6, "total_tokens": 6}
}`

// TestEmbedderSchemelessHostReachesServer is the end-to-end guard that the
// embedder shares the chat model's base-URL normalization. Ollama's own
// documented OLLAMA_HOST format is schemeless, so "127.0.0.1:PORT" must survive
// to the wire with its port intact and land on /v1/embeddings.
func TestEmbedderSchemelessHostReachesServer(t *testing.T) {
	tests := []struct {
		name string
		// hostFn derives the schemeless host from the test server's URL.
		hostFn func(serverURL string) string
		// useEnv routes the host through OLLAMA_HOST instead of WithHost.
		useEnv bool
	}{
		{
			name:   "host and port",
			hostFn: func(u string) string { return strings.TrimPrefix(u, "http://") },
		},
		{
			name:   "host and port with trailing slash",
			hostFn: func(u string) string { return strings.TrimPrefix(u, "http://") + "/" },
		},
		{
			name:   "host and port from env",
			hostFn: func(u string) string { return strings.TrimPrefix(u, "http://") },
			useEnv: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotPath, gotHost string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				gotHost = r.Host
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(embeddingsResponse))
			}))
			defer srv.Close()

			host := tt.hostFn(srv.URL)
			if strings.Contains(host, "://") {
				t.Fatalf("test setup: host %q is not schemeless", host)
			}

			var opts []ollama.Option
			if tt.useEnv {
				t.Setenv("OLLAMA_HOST", host)
			} else {
				t.Setenv("OLLAMA_HOST", "")
				opts = append(opts, ollama.WithHost(host))
			}

			embedder, err := ollama.NewEmbedder("", opts...)
			if err != nil {
				t.Fatalf("NewEmbedder(%q) unexpected error: %v", host, err)
			}

			resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
				Input: []string{"hello", "world"},
			})
			if err != nil {
				t.Fatalf("Embed unexpected error: %v", err)
			}

			if gotPath != "/v1/embeddings" {
				t.Errorf("server received path %q, want %q", gotPath, "/v1/embeddings")
			}
			wantHost := strings.TrimPrefix(srv.URL, "http://")
			if gotHost != wantHost {
				t.Errorf("server received Host %q, want %q (port must not be lost)", gotHost, wantHost)
			}

			if len(resp.Vectors) != 2 {
				t.Fatalf("got %d vectors, want 2", len(resp.Vectors))
			}
			if resp.Vectors[0][0] != 0.1 || resp.Vectors[1][0] != 0.3 {
				t.Errorf("vectors out of input order: %v", resp.Vectors)
			}
			if resp.Model != "nomic-embed-text" {
				t.Errorf("model = %q, want %q", resp.Model, "nomic-embed-text")
			}
			if resp.Usage.TotalTokens != 6 {
				t.Errorf("total tokens = %d, want 6", resp.Usage.TotalTokens)
			}
		})
	}
}

// TestEmbedderSendsModelAndDimensions asserts the default model and the
// Dimensions field reach Ollama's OpenAI-compatible surface.
func TestEmbedderSendsModelAndDimensions(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(embeddingsResponse))
	}))
	defer srv.Close()

	t.Setenv("OLLAMA_HOST", "")

	embedder, err := ollama.NewEmbedder("", ollama.WithHost(srv.URL))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:      []string{"hello", "world"},
		Dimensions: 256,
	}); err != nil {
		t.Fatalf("Embed unexpected error: %v", err)
	}

	if body["model"] != "nomic-embed-text" {
		t.Errorf("request model = %v, want %q", body["model"], "nomic-embed-text")
	}
	if body["dimensions"] != float64(256) {
		t.Errorf("request dimensions = %v, want 256", body["dimensions"])
	}
}

// TestEmbedderOpenAICompatibleServerOverride documents the TEI/llama.cpp path:
// any server speaking the OpenAI embeddings API is reachable through WithHost,
// including on a non-Ollama port and with a non-Ollama model name.
func TestEmbedderOpenAICompatibleServerOverride(t *testing.T) {
	var gotPath, gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotModel = body.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(embeddingsResponse))
	}))
	defer srv.Close()

	t.Setenv("OLLAMA_HOST", "")

	embedder, err := ollama.NewEmbedder("BAAI/bge-large-en-v1.5", ollama.WithHost(srv.URL))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input: []string{"hello", "world"},
	}); err != nil {
		t.Fatalf("Embed unexpected error: %v", err)
	}

	if gotPath != "/v1/embeddings" {
		t.Errorf("server received path %q, want %q", gotPath, "/v1/embeddings")
	}
	if gotModel != "BAAI/bge-large-en-v1.5" {
		t.Errorf("request model = %q, want %q", gotModel, "BAAI/bge-large-en-v1.5")
	}
}

// TestEmbedderSendsAuthorizationHeader pins the API-key resolution to the wire:
// Ollama's OpenAI-compatible endpoint receives "Authorization: Bearer <key>" for
// every resolution path. Ollama needs no real key, so an unset key must still
// send the "ollama" placeholder; OLLAMA_API_KEY supplies the key otherwise; and
// an explicit WithAPIKey outranks the environment. Each case fails if its rung
// of the precedence ladder regresses — a deleted default sends no header, a
// dropped env read falls back to the placeholder, and a broken precedence sends
// the env key instead of the explicit one.
func TestEmbedderSendsAuthorizationHeader(t *testing.T) {
	tests := []struct {
		name    string
		envKey  string // OLLAMA_API_KEY value; "" leaves it unset
		opts    []ollama.Option
		wantKey string // expected token in "Bearer <token>"
	}{
		{
			name:    "default placeholder key",
			wantKey: "ollama",
		},
		{
			name:    "from OLLAMA_API_KEY env",
			envKey:  "env-secret",
			wantKey: "env-secret",
		},
		{
			name:    "explicit WithAPIKey wins over env",
			envKey:  "env-secret",
			opts:    []ollama.Option{ollama.WithAPIKey("explicit-secret")},
			wantKey: "explicit-secret",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotAuth string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotAuth = r.Header.Get("Authorization")
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(embeddingsResponse))
			}))
			defer srv.Close()

			t.Setenv("OLLAMA_HOST", "")
			t.Setenv("OLLAMA_API_KEY", tt.envKey)

			opts := append([]ollama.Option{ollama.WithHost(srv.URL)}, tt.opts...)
			embedder, err := ollama.NewEmbedder("nomic-embed-text", opts...)
			if err != nil {
				t.Fatalf("NewEmbedder unexpected error: %v", err)
			}

			if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
				Input: []string{"hello", "world"},
			}); err != nil {
				t.Fatalf("Embed unexpected error: %v", err)
			}

			want := "Bearer " + tt.wantKey
			if gotAuth != want {
				t.Errorf("server received Authorization %q, want %q", gotAuth, want)
			}
		})
	}
}
