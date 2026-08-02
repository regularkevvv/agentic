package ollama_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/provider/ollama"
)

func TestNewEmbedderDefaultsToNomicEmbedText(t *testing.T) {
	embedder, err := ollama.NewEmbedder("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if embedder.Name() != ollama.DefaultEmbeddingModel {
		t.Errorf("expected name %q, got %q", ollama.DefaultEmbeddingModel, embedder.Name())
	}
	if ollama.DefaultEmbeddingModel != "nomic-embed-text" {
		t.Errorf("DefaultEmbeddingModel = %q, want %q", ollama.DefaultEmbeddingModel, "nomic-embed-text")
	}
}

func TestNewEmbedderExplicitModel(t *testing.T) {
	embedder, err := ollama.NewEmbedder("mxbai-embed-large")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if embedder.Name() != "mxbai-embed-large" {
		t.Errorf("expected name %q, got %q", "mxbai-embed-large", embedder.Name())
	}
}

func TestNewEmbedderWithHost(t *testing.T) {
	embedder, err := ollama.NewEmbedder("nomic-embed-text", ollama.WithHost("http://gpu-server:11434"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if embedder.Name() != "nomic-embed-text" {
		t.Errorf("expected name %q, got %q", "nomic-embed-text", embedder.Name())
	}
}

// TestNewEmbedderSchemelessHost proves the embedder reuses the chat model's
// base-URL normalization: Ollama's documented OLLAMA_HOST format is schemeless,
// so a "host:port" value must be accepted here exactly as New accepts it AND
// survive to the wire with its port intact, landing on /v1/embeddings. Each case
// derives a schemeless host from the test server and asserts what reached the
// server, so it fails if normalization drops the port, mangles the host, or
// routes to the wrong path — not merely if the constructor returns nil.
func TestNewEmbedderSchemelessHost(t *testing.T) {
	tests := []struct {
		name string
		// hostFn derives a schemeless host from the test server's URL.
		hostFn func(serverURL string) string
		// useEnv routes the host through OLLAMA_HOST instead of WithHost.
		useEnv bool
	}{
		{
			name:   "explicit ipv4",
			hostFn: func(u string) string { return strings.TrimPrefix(u, "http://") },
		},
		{
			name: "explicit named",
			hostFn: func(u string) string {
				return strings.Replace(strings.TrimPrefix(u, "http://"), "127.0.0.1", "localhost", 1)
			},
		},
		{
			name:   "explicit trailing slash",
			hostFn: func(u string) string { return strings.TrimPrefix(u, "http://") + "/" },
		},
		{
			name:   "from env",
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

			embedder, err := ollama.NewEmbedder("nomic-embed-text", opts...)
			if err != nil {
				t.Fatalf("NewEmbedder with host %q (useEnv=%v): unexpected error: %v", host, tt.useEnv, err)
			}

			if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
				Input: []string{"hello", "world"},
			}); err != nil {
				t.Fatalf("Embed unexpected error: %v", err)
			}

			if gotPath != "/v1/embeddings" {
				t.Errorf("server received path %q, want %q", gotPath, "/v1/embeddings")
			}
			wantHost := strings.TrimSuffix(host, "/")
			if gotHost != wantHost {
				t.Errorf("server received Host %q, want %q (schemeless host must normalize with its port intact)", gotHost, wantHost)
			}
		})
	}
}

func TestNewEmbedderWithAPIKey(t *testing.T) {
	embedder, err := ollama.NewEmbedder("nomic-embed-text", ollama.WithAPIKey("secret"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if embedder.Name() != "nomic-embed-text" {
		t.Errorf("expected name %q, got %q", "nomic-embed-text", embedder.Name())
	}
}

func TestNewEmbedderFromEnvAPIKey(t *testing.T) {
	t.Setenv("OLLAMA_API_KEY", "env-secret")

	embedder, err := ollama.NewEmbedder("nomic-embed-text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if embedder.Name() != "nomic-embed-text" {
		t.Errorf("expected name %q, got %q", "nomic-embed-text", embedder.Name())
	}
}

// TestNewEmbedderRejectsInvalidHost checks the host error surfaces from
// NewEmbedder rather than being deferred to the first request.
func TestNewEmbedderRejectsInvalidHost(t *testing.T) {
	tests := []struct {
		name    string
		host    string
		wantErr string
	}{
		{"unsupported scheme", "ftp://gpu:11434", "scheme must be http or https"},
		{"missing host", "http://", "missing host"},
		{"unparseable escape", "http://exa%zzmple.com:11434", "invalid host"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", "")

			embedder, err := ollama.NewEmbedder("nomic-embed-text", ollama.WithHost(tt.host))
			if err == nil {
				t.Fatalf("expected error, got embedder %v", embedder)
			}
			if embedder != nil {
				t.Errorf("expected nil embedder on error, got %v", embedder)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want it to contain %q", err.Error(), tt.wantErr)
			}
			if !strings.HasPrefix(err.Error(), "ollama: ") {
				t.Errorf("error = %q, want it to be prefixed with %q", err.Error(), "ollama: ")
			}
		})
	}
}

func TestMustNewEmbedder(t *testing.T) {
	embedder := ollama.MustNewEmbedder("nomic-embed-text")
	if embedder.Name() != "nomic-embed-text" {
		t.Errorf("expected name %q, got %q", "nomic-embed-text", embedder.Name())
	}
}

func TestMustNewEmbedderPanicsOnInvalidHost(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic, got none")
		}
		err, ok := r.(error)
		if !ok {
			t.Fatalf("expected panic value to be an error, got %T", r)
		}
		if !strings.Contains(err.Error(), "scheme must be http or https") {
			t.Errorf("panic error = %q, want it to mention the invalid scheme", err.Error())
		}
	}()

	ollama.MustNewEmbedder("nomic-embed-text", ollama.WithHost("ftp://gpu:11434"))
}

func TestEmbedderImplementsInterface(t *testing.T) {
	embedder, err := ollama.NewEmbedder("nomic-embed-text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var _ retrieval.Embedder = embedder
}
