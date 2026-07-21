package together_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/together"
)

// capture records what a stub Together endpoint saw on the wire.
type capture struct {
	path    string
	auth    string
	body    map[string]any
	rawBody string
}

// newServer starts a stub Together endpoint that records the request and
// replies with the given JSON body.
func newServer(t *testing.T, responseBody string, got *capture) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.path = r.URL.Path
		got.auth = r.Header.Get("Authorization")

		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
			return
		}
		got.rawBody = string(raw)
		if err := json.Unmarshal(raw, &got.body); err != nil {
			t.Errorf("decoding request body %q: %v", raw, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(responseBody))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// okResponse is a minimal successful Together completion.
const okResponse = `{
	"id": "abc-123",
	"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
	"created": 1700000000,
	"choices": [{
		"index": 0,
		"message": {"role": "assistant", "content": "hi"},
		"finish_reason": "stop"
	}],
	"usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
}`

// newModel points a Together model at the stub server. WithBaseURL takes the
// full base including the "/v1" suffix, mirroring DefaultBaseURL.
func newModel(t *testing.T, srv *httptest.Server, model, apiKey string) *together.Model {
	t.Helper()
	m, err := together.New(model,
		together.WithAPIKey(apiKey),
		together.WithBaseURL(srv.URL+"/v1"),
	)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	return m
}

// TestRequestReachesEndpoint is the end-to-end guard that WithBaseURL is
// actually wired through to the OpenAI client. Asserting only on Name() would
// pass even if the option were dropped and every request went to api.openai.com.
func TestRequestReachesEndpoint(t *testing.T) {
	var got capture
	srv := newServer(t, okResponse, &got)
	model := newModel(t, srv, "meta-llama/Llama-3.3-70B-Instruct-Turbo", "test-key")

	resp, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("Request() unexpected error: %v", err)
	}

	if got.path != "/v1/chat/completions" {
		t.Errorf("server received path %q, want %q", got.path, "/v1/chat/completions")
	}
	if resp.FinishReason != core.FinishReasonStop {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, core.FinishReasonStop)
	}
}

// TestAuthorizationHeader pins which environment variable supplies the bearer
// token, observed where it actually matters — on the wire.
func TestAuthorizationHeader(t *testing.T) {
	tests := []struct {
		name      string
		primary   string
		secondary string
		opt       string
		wantAuth  string
	}{
		{
			name:     "explicit option",
			opt:      "opt-key",
			wantAuth: "Bearer opt-key",
		},
		{
			name:     "TOGETHER_API_KEY",
			primary:  "primary-key",
			wantAuth: "Bearer primary-key",
		},
		{
			name:      "TOGETHER_API_KEY wins over TOGETHER_AI_API_KEY",
			primary:   "primary-key",
			secondary: "secondary-key",
			wantAuth:  "Bearer primary-key",
		},
		{
			name:      "TOGETHER_AI_API_KEY when primary unset",
			secondary: "secondary-key",
			wantAuth:  "Bearer secondary-key",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("TOGETHER_API_KEY", tt.primary)
			t.Setenv("TOGETHER_AI_API_KEY", tt.secondary)

			var got capture
			srv := newServer(t, okResponse, &got)

			opts := []together.Option{together.WithBaseURL(srv.URL + "/v1")}
			if tt.opt != "" {
				opts = append(opts, together.WithAPIKey(tt.opt))
			}
			model, err := together.New("test-model", opts...)
			if err != nil {
				t.Fatalf("New() unexpected error: %v", err)
			}

			_, err = model.Request(context.Background(), &core.ChatRequest{
				Model:    "test-model",
				Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
			})
			if err != nil {
				t.Fatalf("Request() unexpected error: %v", err)
			}

			if got.auth != tt.wantAuth {
				t.Errorf("Authorization = %q, want %q", got.auth, tt.wantAuth)
			}
		})
	}
}

// TestRequestBody pins the fields Together receives.
//
// The model name carries a vendor namespace ("meta-llama/…"), Together's
// documented naming scheme, and must survive into the body verbatim rather
// than being split or escaped on the way.
//
// max_tokens maps to max_completion_tokens, not the legacy max_tokens:
// pydantic-ai's TogetherProvider sets no
// openai_chat_supports_max_completion_tokens override, leaving it at the
// default of true (pydantic_ai/profiles/openai.py:263), whereas the OpenRouter
// provider must set it false (pydantic_ai/providers/openrouter.py:182). This
// test fails if the wrapper is ever switched to OpenRouter's legacy field.
func TestRequestBody(t *testing.T) {
	var got capture
	srv := newServer(t, okResponse, &got)
	model := newModel(t, srv, "meta-llama/Llama-3.3-70B-Instruct-Turbo", "test-key")

	maxTokens := 256
	_, err := model.Request(context.Background(), &core.ChatRequest{
		Model:         "meta-llama/Llama-3.3-70B-Instruct-Turbo",
		Messages:      []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		MaxTokens:     &maxTokens,
		StopSequences: []string{"END", "STOP"},
	})
	if err != nil {
		t.Fatalf("Request() unexpected error: %v", err)
	}

	if model := got.body["model"]; model != "meta-llama/Llama-3.3-70B-Instruct-Turbo" {
		t.Errorf("body model = %v, want %q", model, "meta-llama/Llama-3.3-70B-Instruct-Turbo")
	}

	if v, ok := got.body["max_completion_tokens"]; !ok {
		t.Errorf("body is missing max_completion_tokens; got %s", got.rawBody)
	} else if v != float64(256) {
		t.Errorf("max_completion_tokens = %v, want 256", v)
	}
	if _, ok := got.body["max_tokens"]; ok {
		t.Errorf("body sent the legacy max_tokens field; Together takes max_completion_tokens: %s", got.rawBody)
	}

	// StopSequences is a universal core field; it must survive delegation.
	stop, ok := got.body["stop"].([]any)
	if !ok {
		t.Fatalf("body stop = %v (%T), want a JSON array; got %s", got.body["stop"], got.body["stop"], got.rawBody)
	}
	if len(stop) != 2 || stop[0] != "END" || stop[1] != "STOP" {
		t.Errorf("stop = %v, want [END STOP]", stop)
	}
}

// TestFinishReasonMapping pins that a stop reason this library does not
// recognize is reported as Unknown rather than as a clean Stop, and that the
// provider's original string always survives on RawFinishReason.
func TestFinishReasonMapping(t *testing.T) {
	tests := []struct {
		name   string
		raw    string
		want   core.FinishReason
		wantEq string
	}{
		{name: "stop", raw: "stop", want: core.FinishReasonStop, wantEq: "stop"},
		{name: "length", raw: "length", want: core.FinishReasonLength, wantEq: "length"},
		{name: "tool calls", raw: "tool_calls", want: core.FinishReasonToolCalls, wantEq: "tool_calls"},
		{name: "content filter", raw: "content_filter", want: core.FinishReasonContentFilter, wantEq: "content_filter"},
		{
			// Together's docs list eos and length for its own models; eos is
			// not an OpenAI reason, so it must surface as Unknown with the
			// original string intact rather than as a successful stop.
			name: "unrecognized reason is not a clean stop",
			raw:  "eos",
			want: core.FinishReasonUnknown, wantEq: "eos",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{
				"id": "abc-123",
				"model": "test-model",
				"created": 1700000000,
				"choices": [{
					"index": 0,
					"message": {"role": "assistant", "content": "hi"},
					"finish_reason": "` + tt.raw + `"
				}],
				"usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
			}`

			var got capture
			srv := newServer(t, body, &got)
			model := newModel(t, srv, "test-model", "test-key")

			resp, err := model.Request(context.Background(), &core.ChatRequest{
				Model:    "test-model",
				Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
			})
			if err != nil {
				t.Fatalf("Request() unexpected error: %v", err)
			}

			if resp.FinishReason != tt.want {
				t.Errorf("FinishReason = %q, want %q", resp.FinishReason, tt.want)
			}
			if resp.RawFinishReason != tt.wantEq {
				t.Errorf("RawFinishReason = %q, want %q", resp.RawFinishReason, tt.wantEq)
			}
		})
	}
}

// TestDefaultBaseURL pins the endpoint used when WithBaseURL is not given,
// against the base_url of pydantic-ai's TogetherProvider.
func TestDefaultBaseURL(t *testing.T) {
	if together.DefaultBaseURL != "https://api.together.xyz/v1" {
		t.Errorf("DefaultBaseURL = %q, want %q", together.DefaultBaseURL, "https://api.together.xyz/v1")
	}
}
