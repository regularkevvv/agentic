package azure

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go/option"

	"github.com/regularkevvv/agentic/internal/core"
)

// capturedRequest records the wire-level details of one inbound HTTP request.
type capturedRequest struct {
	path       string
	rawPath    string
	apiVersion string
	apiKey     string
	authHeader string
}

// newCapturingServer returns an httptest server that records every request it
// receives and answers with a minimal chat completion (or SSE stream when the
// body asks for one).
func newCapturingServer(t *testing.T) (*httptest.Server, func() []capturedRequest) {
	t.Helper()

	var (
		mu   sync.Mutex
		seen []capturedRequest
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}

		mu.Lock()
		seen = append(seen, capturedRequest{
			path:       r.URL.Path,
			rawPath:    r.URL.EscapedPath(),
			apiVersion: r.URL.Query().Get("api-version"),
			apiKey:     r.Header.Get("api-key"),
			authHeader: r.Header.Get("Authorization"),
		})
		mu.Unlock()

		if !strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{
				"id":"chatcmpl_azure",
				"object":"chat.completion",
				"created":123,
				"model":"gpt-4o",
				"choices":[{"index":0,"message":{"role":"assistant","content":"hi from azure"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
			}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"id":"chatcmpl_azure","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
			``,
			`data: {"id":"chatcmpl_azure","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			``,
			`data: [DONE]`,
			``,
		}, "\n"))
	}))

	return server, func() []capturedRequest {
		mu.Lock()
		defer mu.Unlock()
		out := make([]capturedRequest, len(seen))
		copy(out, seen)
		return out
	}
}

// TestAzureNeverSendsAPIVersion asserts on the wire URL rather than the base
// URL string, which is the only way to catch this class of bug: the api-version
// parameter this package used to encode into the base URL was silently
// discarded by BaseURL.Parse, and the v1 API it now targets rejects the
// parameter outright. Both Request and RequestStream are exercised.
func TestAzureNeverSendsAPIVersion(t *testing.T) {
	// A stray OPENAI_API_KEY must not leak an Authorization header.
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-sent")

	server, captured := newCapturingServer(t)
	defer server.Close()

	model, err := New("my-gpt4o-deployment",
		WithEndpoint(server.URL),
		WithAPIKey("azure-key"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := &core.ChatRequest{
		Model:    "my-gpt4o-deployment",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	}

	if _, err := model.Request(context.Background(), req); err != nil {
		t.Fatalf("Request: %v", err)
	}

	stream, err := model.RequestStream(context.Background(), req)
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}
	if _, err := stream.Text(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	got := captured()
	if len(got) != 2 {
		t.Fatalf("expected 2 requests (Request + RequestStream), got %d", len(got))
	}

	for i, name := range []string{"Request", "RequestStream"} {
		t.Run(name, func(t *testing.T) {
			c := got[i]
			if c.apiVersion != "" {
				t.Errorf("api-version = %q, want it absent — the v1 API rejects it", c.apiVersion)
			}
			if want := "/openai/v1/chat/completions"; c.path != want {
				t.Errorf("path = %q, want %q", c.path, want)
			}
			if strings.Contains(c.path, "/deployments/") {
				t.Errorf("path %q must not contain a deployment segment", c.path)
			}
			if c.apiKey != "azure-key" {
				t.Errorf("api-key header = %q, want %q", c.apiKey, "azure-key")
			}
			if c.authHeader != "" {
				t.Errorf("Authorization header must not be sent, got %q", c.authHeader)
			}
		})
	}
}

// TestAzureOpenAICompatibleEndpointsOmitAPIVersion covers the two endpoint
// shapes that serve the OpenAI-compatible API. Both REJECT api-version, and
// both address the model by name instead of by deployment path — so sending
// what classic Azure requires would break them.
func TestAzureOpenAICompatibleEndpointsOmitAPIVersion(t *testing.T) {
	tests := []struct {
		name     string
		endpoint func(serverURL string) string
		wantPath string
	}{
		{
			name:     "v1 GA API (path ends in /v1)",
			endpoint: func(u string) string { return u + "/openai/v1" },
			wantPath: "/openai/v1/chat/completions",
		},
		{
			name:     "v1 GA API with trailing slash",
			endpoint: func(u string) string { return u + "/openai/v1/" },
			wantPath: "/openai/v1/chat/completions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, captured := newCapturingServer(t)
			defer server.Close()

			model, err := New("gpt-4o",
				WithEndpoint(tt.endpoint(server.URL)),
				WithAPIKey("azure-key"),
				WithRequestOptions(option.WithHTTPClient(server.Client())),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			_, err = model.Request(context.Background(), &core.ChatRequest{
				Model:    "gpt-4o",
				Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
			})
			if err != nil {
				t.Fatalf("Request: %v", err)
			}

			got := captured()
			if len(got) != 1 {
				t.Fatalf("expected 1 request, got %d", len(got))
			}
			if got[0].apiVersion != "" {
				t.Errorf("api-version = %q, want it absent — this endpoint rejects it", got[0].apiVersion)
			}
			if got[0].path != tt.wantPath {
				t.Errorf("path = %q, want %q (no /openai/deployments segment)", got[0].path, tt.wantPath)
			}
		})
	}
}

// TestAzureRequestOptionsOverrideDefaults proves WithRequestOptions is applied
// after the Azure defaults, so AAD/Entra callers can supply a bearer token.
func TestAzureRequestOptionsOverrideDefaults(t *testing.T) {
	server, captured := newCapturingServer(t)
	defer server.Close()

	model, err := New("gpt-4o",
		WithEndpoint(server.URL),
		WithAPIKey("azure-key"),
		WithRequestOptions(
			option.WithHTTPClient(server.Client()),
			option.WithHeader("Authorization", "Bearer aad-token"),
		),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := &core.ChatRequest{
		Model:    "gpt-4o",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	}
	if _, err := model.Request(context.Background(), req); err != nil {
		t.Fatalf("Request: %v", err)
	}

	got := captured()
	if len(got) != 1 {
		t.Fatalf("expected 1 request, got %d", len(got))
	}
	if got[0].authHeader != "Bearer aad-token" {
		t.Errorf("Authorization = %q, want %q", got[0].authHeader, "Bearer aad-token")
	}
	// The api-key default still applies; the caller option only replaced the
	// header it named.
	if got[0].apiKey != "azure-key" {
		t.Errorf("api-key header = %q, want %q", got[0].apiKey, "azure-key")
	}
}
