package ollama

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

// TestSchemelessHostReachesServer is the end-to-end guard for Ollama's own
// documented OLLAMA_HOST format. A schemeless "127.0.0.1:PORT" previously
// produced a base URL that the OpenAI SDK could not resolve, so the request
// never left the process; the host and port must now survive to the wire.
func TestSchemelessHostReachesServer(t *testing.T) {
	tests := []struct {
		name string
		// hostFn derives the schemeless host from the test server's URL.
		hostFn func(serverURL string) string
	}{
		{
			name:   "host and port",
			hostFn: func(u string) string { return strings.TrimPrefix(u, "http://") },
		},
		{
			name:   "host and port with trailing slash",
			hostFn: func(u string) string { return strings.TrimPrefix(u, "http://") + "/" },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", "")

			var gotPath, gotHost string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				gotHost = r.Host
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{
					"id": "chatcmpl-1",
					"model": "llama3.2",
					"created": 1700000000,
					"choices": [{
						"index": 0,
						"message": {"role": "assistant", "content": "hi"},
						"finish_reason": "stop"
					}],
					"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
				}`))
			}))
			defer srv.Close()

			host := tt.hostFn(srv.URL)
			if strings.Contains(host, "://") {
				t.Fatalf("test setup: host %q is not schemeless", host)
			}

			model, err := New("llama3.2", WithHost(host))
			if err != nil {
				t.Fatalf("New(%q) unexpected error: %v", host, err)
			}

			// The response conversion is the OpenAI provider's concern; this
			// test only asserts the request was routed to the right endpoint.
			_, _ = model.Request(context.Background(), &core.ChatRequest{
				Model:    "llama3.2",
				Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
			})

			if gotPath != "/v1/chat/completions" {
				t.Errorf("server received path %q, want %q", gotPath, "/v1/chat/completions")
			}
			wantHost := strings.TrimPrefix(srv.URL, "http://")
			if gotHost != wantHost {
				t.Errorf("server received Host %q, want %q (port must not be lost)", gotHost, wantHost)
			}
		})
	}
}
