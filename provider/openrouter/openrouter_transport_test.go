package openrouter_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go/option"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/openrouter"
)

// recordingServer serves a fixed body and captures the decoded request body of
// every call, so tests can assert on what actually went over the wire.
type recordingServer struct {
	*httptest.Server

	mu     sync.Mutex
	bodies []map[string]any
}

func newRecordingServer(t *testing.T, handler func(w http.ResponseWriter, body map[string]any)) *recordingServer {
	t.Helper()

	rec := &recordingServer{}
	rec.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
			http.Error(w, "bad body", http.StatusBadRequest)
			return
		}

		rec.mu.Lock()
		rec.bodies = append(rec.bodies, body)
		rec.mu.Unlock()

		handler(w, body)
	}))
	t.Cleanup(rec.Close)
	return rec
}

func (rec *recordingServer) lastBody(t *testing.T) map[string]any {
	t.Helper()
	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.bodies) == 0 {
		t.Fatal("no request was recorded")
	}
	return rec.bodies[len(rec.bodies)-1]
}

func (rec *recordingServer) model(t *testing.T, name string, opts ...openrouter.Option) *openrouter.Model {
	t.Helper()
	opts = append([]openrouter.Option{
		openrouter.WithAPIKey("sk-or-test"),
		openrouter.WithBaseURL(rec.URL + "/api/v1"),
		openrouter.WithRequestOptions(option.WithHTTPClient(rec.Client())),
	}, opts...)

	model, err := openrouter.New(name, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return model
}

const okCompletion = `{
	"id":"gen-1",
	"object":"chat.completion",
	"created":1700000000,
	"model":"anthropic/claude-sonnet-4",
	"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
	"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
}`

func serveOK(w http.ResponseWriter, _ map[string]any) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, okCompletion)
}

// TestMaxTokensUsesLegacyField pins the field name for the token cap.
// OpenRouter accepts only the legacy max_tokens; max_completion_tokens is
// accepted syntactically but ignored, so the cap would silently not apply.
func TestMaxTokensUsesLegacyField(t *testing.T) {
	rec := newRecordingServer(t, serveOK)
	model := rec.model(t, "anthropic/claude-sonnet-4")

	maxTokens := 256
	_, err := model.Request(context.Background(), &core.ChatRequest{
		Model:     "anthropic/claude-sonnet-4",
		Messages:  []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		MaxTokens: &maxTokens,
	})
	if err != nil {
		t.Fatalf("Request: %v", err)
	}

	body := rec.lastBody(t)
	if _, present := body["max_completion_tokens"]; present {
		t.Errorf("max_completion_tokens must not be sent to OpenRouter, body was %#v", body)
	}
	got, present := body["max_tokens"]
	if !present {
		t.Fatalf("max_tokens missing from request body %#v", body)
	}
	if got != float64(256) {
		t.Errorf("max_tokens = %v, want 256", got)
	}
}

func TestStopSequencesAreSent(t *testing.T) {
	rec := newRecordingServer(t, serveOK)
	model := rec.model(t, "anthropic/claude-sonnet-4")

	_, err := model.Request(context.Background(), &core.ChatRequest{
		Model:         "anthropic/claude-sonnet-4",
		Messages:      []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		StopSequences: []string{"END", "STOP"},
	})
	if err != nil {
		t.Fatalf("Request: %v", err)
	}

	stop, ok := rec.lastBody(t)["stop"].([]any)
	if !ok {
		t.Fatalf("stop missing or not an array in %#v", rec.lastBody(t))
	}
	if len(stop) != 2 || stop[0] != "END" || stop[1] != "STOP" {
		t.Errorf("stop = %#v, want [END STOP]", stop)
	}
}

func TestProviderRoutingIsSent(t *testing.T) {
	rec := newRecordingServer(t, serveOK)
	model := rec.model(t, "anthropic/claude-sonnet-4")

	allow := false
	_, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		ProviderOptions: map[string]any{
			"openrouter": openrouter.Options{
				Provider: &openrouter.ProviderRouting{
					Order:          []string{"anthropic", "openai"},
					AllowFallbacks: &allow,
					Sort:           "throughput",
				},
			},
			// Another provider's settings must be ignored, not rejected.
			"anthropic": struct{ TopK int }{TopK: 40},
		},
	})
	if err != nil {
		t.Fatalf("Request: %v", err)
	}

	provider, ok := rec.lastBody(t)["provider"].(map[string]any)
	if !ok {
		t.Fatalf("provider missing or not an object in %#v", rec.lastBody(t))
	}
	order, _ := provider["order"].([]any)
	if len(order) != 2 || order[0] != "anthropic" || order[1] != "openai" {
		t.Errorf("provider.order = %#v, want [anthropic openai]", provider["order"])
	}
	if provider["allow_fallbacks"] != false {
		t.Errorf("provider.allow_fallbacks = %#v, want false", provider["allow_fallbacks"])
	}
	if provider["sort"] != "throughput" {
		t.Errorf("provider.sort = %#v, want throughput", provider["sort"])
	}
	if _, present := provider["only"]; present {
		t.Errorf("unset routing fields must be omitted, got %#v", provider)
	}
}

func TestProviderRoutingOmittedForMistypedOptions(t *testing.T) {
	rec := newRecordingServer(t, serveOK)
	model := rec.model(t, "anthropic/claude-sonnet-4")

	_, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		// Wrong type under our own key: ignore it silently.
		ProviderOptions: map[string]any{"openrouter": "order=anthropic"},
	})
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if _, present := rec.lastBody(t)["provider"]; present {
		t.Errorf("provider must be omitted for a mistyped option, body %#v", rec.lastBody(t))
	}
}

func TestThinkingIsSentAsReasoningObject(t *testing.T) {
	rec := newRecordingServer(t, serveOK)
	model := rec.model(t, "anthropic/claude-sonnet-4")

	_, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		Thinking: &core.ThinkingConfig{Enabled: true, BudgetTokens: 4096},
	})
	if err != nil {
		t.Fatalf("Request: %v", err)
	}

	reasoning, ok := rec.lastBody(t)["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning missing or not an object in %#v", rec.lastBody(t))
	}
	if reasoning["max_tokens"] != float64(4096) {
		t.Errorf("reasoning.max_tokens = %#v, want 4096", reasoning["max_tokens"])
	}
	if reasoning["enabled"] != true {
		t.Errorf("reasoning.enabled = %#v, want true", reasoning["enabled"])
	}
	if _, present := reasoning["effort"]; present {
		t.Errorf("effort and max_tokens are mutually exclusive, got %#v", reasoning)
	}
}

func TestAppAttributionHeaders(t *testing.T) {
	var gotReferer, gotTitle string
	rec := newRecordingServer(t, serveOK)
	rec.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotReferer = r.Header.Get("HTTP-Referer")
		gotTitle = r.Header.Get("X-Title")
		serveOK(w, nil)
	})

	t.Setenv("OPENROUTER_APP_URL", "https://env.example")
	t.Setenv("OPENROUTER_APP_TITLE", "Env App")

	model := rec.model(t, "anthropic/claude-sonnet-4")
	if _, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	}); err != nil {
		t.Fatalf("Request: %v", err)
	}

	if gotReferer != "https://env.example" {
		t.Errorf("HTTP-Referer = %q, want the OPENROUTER_APP_URL value", gotReferer)
	}
	if gotTitle != "Env App" {
		t.Errorf("X-Title = %q, want the OPENROUTER_APP_TITLE value", gotTitle)
	}
}

// TestInBandErrorYieldsTypedError pins OpenRouter's in-band failure shape: an
// HTTP 200 whose body carries an error object and a null choices array. That
// must surface as a typed error, not as an empty successful turn.
func TestInBandErrorYieldsTypedError(t *testing.T) {
	rec := newRecordingServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{
			"id":"gen-err",
			"model":"anthropic/claude-sonnet-4",
			"choices":null,
			"error":{"code":429,"message":"rate limited by upstream","metadata":{"provider_name":"Anthropic"}}
		}`)
	})
	model := rec.model(t, "anthropic/claude-sonnet-4")

	resp, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err == nil {
		t.Fatalf("expected an error for an in-band failure, got response %#v", resp)
	}

	var apiErr *openrouter.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *openrouter.APIError, got %T: %v", err, err)
	}
	if apiErr.Code != 429 {
		t.Errorf("Code = %d, want 429", apiErr.Code)
	}
	if apiErr.Message != "rate limited by upstream" {
		t.Errorf("Message = %q", apiErr.Message)
	}
	if apiErr.Metadata["provider_name"] != "Anthropic" {
		t.Errorf("Metadata = %#v", apiErr.Metadata)
	}
	if !strings.Contains(apiErr.Error(), "429") {
		t.Errorf("Error() = %q, want it to mention the code", apiErr.Error())
	}

	if resp == nil {
		t.Fatal("expected the response to carry the finish reason alongside the error")
	}
	if resp.FinishReason != core.FinishReasonError {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, core.FinishReasonError)
	}
	if resp.RawFinishReason != "error" {
		t.Errorf("RawFinishReason = %q, want %q", resp.RawFinishReason, "error")
	}
}

func TestEmptyChoicesWithoutErrorObject(t *testing.T) {
	rec := newRecordingServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"gen-empty","model":"m","choices":[]}`)
	})
	model := rec.model(t, "m")

	_, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "m",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if !errors.Is(err, openrouter.ErrNoChoices) {
		t.Fatalf("expected ErrNoChoices, got %v", err)
	}
}

func TestReasoningBecomesThinkingContent(t *testing.T) {
	tests := []struct {
		name  string
		field string
	}{
		{name: "reasoning", field: "reasoning"},
		{name: "reasoning_content", field: "reasoning_content"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := newRecordingServer(t, func(w http.ResponseWriter, _ map[string]any) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{
					"id":"gen-think","object":"chat.completion","created":1,"model":"m",
					"choices":[{"index":0,"message":{"role":"assistant","content":"answer","`+tt.field+`":"let me think"},"finish_reason":"stop"}]
				}`)
			})
			model := rec.model(t, "m")

			resp, err := model.Request(context.Background(), &core.ChatRequest{
				Model:    "m",
				Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
			})
			if err != nil {
				t.Fatalf("Request: %v", err)
			}

			if got := resp.Message.GetThinkingContent(); got != "let me think" {
				t.Errorf("thinking content = %q, want %q", got, "let me think")
			}
			if got := resp.Message.GetTextContent(); got != "answer" {
				t.Errorf("text content = %q, want %q", got, "answer")
			}
			if resp.Message.Content[0].Type != core.ContentThinking {
				t.Errorf("thinking must come first, got %v", resp.Message.Content[0].Type)
			}
			if pn := resp.Message.Content[0].Thinking.ProviderName; pn != openrouter.ProviderName {
				t.Errorf("ProviderName = %q, want %q", pn, openrouter.ProviderName)
			}
			if id := resp.Message.Content[0].Thinking.ID; id != tt.field {
				t.Errorf("thinking ID = %q, want the source field %q", id, tt.field)
			}
		})
	}
}

func TestFinishReasonPassthrough(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want core.FinishReason
	}{
		{name: "stop", raw: "stop", want: core.FinishReasonStop},
		{name: "length", raw: "length", want: core.FinishReasonLength},
		{name: "tool calls", raw: "tool_calls", want: core.FinishReasonToolCalls},
		{name: "content filter", raw: "content_filter", want: core.FinishReasonContentFilter},
		{name: "error", raw: "error", want: core.FinishReasonError},
		{name: "legacy function call", raw: "function_call", want: core.FinishReasonToolCalls},
		{name: "unrecognized", raw: "guardrail_intervened", want: core.FinishReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := newRecordingServer(t, func(w http.ResponseWriter, _ map[string]any) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{
					"id":"gen-fr","object":"chat.completion","created":1,"model":"m",
					"choices":[{"index":0,"message":{"role":"assistant","content":"x"},"finish_reason":"`+tt.raw+`"}]
				}`)
			})
			model := rec.model(t, "m")

			resp, err := model.Request(context.Background(), &core.ChatRequest{
				Model:    "m",
				Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
			})
			if err != nil {
				t.Fatalf("Request: %v", err)
			}
			if resp.FinishReason != tt.want {
				t.Errorf("FinishReason = %q, want %q", resp.FinishReason, tt.want)
			}
			if resp.RawFinishReason != tt.raw {
				t.Errorf("RawFinishReason = %q, want the provider's own %q", resp.RawFinishReason, tt.raw)
			}
		})
	}
}

func TestStreamEmitsThinkingAndFinishReason(t *testing.T) {
	rec := newRecordingServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"id":"s","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"reasoning":"thinking..."}}]}`,
			``,
			`data: {"id":"s","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hello"}}]}`,
			``,
			`data: {"id":"s","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
			``,
			`data: {"id":"s","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}`,
			``,
			`data: [DONE]`,
			``,
		}, "\n"))
	})
	model := rec.model(t, "m")

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "m",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d: %#v", len(events), events)
	}
	if events[0].Type != core.StreamEventThinkingDelta || events[0].Delta != "thinking..." {
		t.Errorf("unexpected thinking event %#v", events[0])
	}
	if events[0].ProviderName != openrouter.ProviderName || events[0].ThinkingID != "reasoning" {
		t.Errorf("thinking event missing provenance: %#v", events[0])
	}
	if events[1].Type != core.StreamEventTextDelta || events[1].Delta != "hello" {
		t.Errorf("unexpected text event %#v", events[1])
	}
	if events[2].Type != core.StreamEventDone {
		t.Fatalf("unexpected final event %#v", events[2])
	}
	if events[2].FinishReason != core.FinishReasonLength {
		t.Errorf("done FinishReason = %q, want %q", events[2].FinishReason, core.FinishReasonLength)
	}
	if events[2].Usage == nil || events[2].Usage.TotalTokens != 11 {
		t.Errorf("unexpected usage %#v", events[2].Usage)
	}

	// Usage on a streamed turn is only reported when explicitly requested.
	if opts, ok := rec.lastBody(t)["stream_options"].(map[string]any); !ok || opts["include_usage"] != true {
		t.Errorf("stream_options.include_usage must be set, body %#v", rec.lastBody(t))
	}
}

func TestStreamInBandErrorYieldsTypedError(t *testing.T) {
	rec := newRecordingServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"id":"s","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"par"}}]}`,
			``,
			`data: {"error":{"code":502,"message":"upstream went away"}}`,
			``,
			`data: [DONE]`,
			``,
		}, "\n"))
	})
	model := rec.model(t, "m")

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "m",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	text, err := stream.Text()
	if err == nil {
		t.Fatalf("expected a stream error, got text %q", text)
	}
	var apiErr *openrouter.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *openrouter.APIError, got %T: %v", err, err)
	}
	if apiErr.Code != 502 || apiErr.Message != "upstream went away" {
		t.Errorf("unexpected error %#v", apiErr)
	}
}

// An ordinary HTTP failure is distinct from OpenRouter's in-band errors and is
// surfaced as the SDK's own error, not as an *APIError.
func TestRequestPropagatesHTTPError(t *testing.T) {
	rec := newRecordingServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"code":500,"message":"boom"}}`)
	})
	model := rec.model(t, "m", openrouter.WithRequestOptions(option.WithMaxRetries(0)))

	resp, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "m",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err == nil {
		t.Fatalf("expected an error for a 500, got %#v", resp)
	}
	if resp != nil {
		t.Errorf("expected no response on a transport failure, got %#v", resp)
	}

	var apiErr *openrouter.APIError
	if errors.As(err, &apiErr) {
		t.Errorf("a non-200 must not be reported as an in-band error: %v", err)
	}
}

func TestStreamToolCalls(t *testing.T) {
	rec := newRecordingServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"id":"s","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\""}}]}}]}`,
			``,
			`data: {"id":"s","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Lima\"}"}}]}}]}`,
			``,
			`data: {"id":"s","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			``,
			`data: [DONE]`,
			``,
		}, "\n"))
	})
	model := rec.model(t, "m")

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "m",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %d: %#v", len(events), events)
	}
	if events[0].Type != core.StreamEventToolCallStart || events[0].ToolUse.Name != "lookup" {
		t.Errorf("unexpected start event %#v", events[0])
	}
	if events[1].Type != core.StreamEventToolCallDelta || events[1].Delta != `{"city":"` {
		t.Errorf("unexpected first delta %#v", events[1])
	}
	// The ID appears only on the first chunk; later deltas must still carry it.
	if events[2].Type != core.StreamEventToolCallDelta || events[2].ToolCallID != "call_1" {
		t.Errorf("later delta lost its tool call ID: %#v", events[2])
	}
	if events[2].Delta != `Lima"}` {
		t.Errorf("unexpected second delta %#v", events[2])
	}
	if events[3].Type != core.StreamEventDone || events[3].FinishReason != core.FinishReasonToolCalls {
		t.Errorf("unexpected done event %#v", events[3])
	}
}

func TestStreamTransportErrorIsWrapped(t *testing.T) {
	rec := newRecordingServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		// A chunk that is not valid JSON aborts the stream.
		_, _ = io.WriteString(w, "data: {not json}\n\n")
	})
	model := rec.model(t, "m")

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "m",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	if _, err := stream.Text(); err == nil {
		t.Fatal("expected a stream error")
	} else if !strings.Contains(err.Error(), "openrouter stream") {
		t.Errorf("expected the error to name the provider, got %v", err)
	}
}

func TestRequestValidatesBeforeCalling(t *testing.T) {
	rec := newRecordingServer(t, serveOK)
	model := rec.model(t, "m")

	if _, err := model.Request(context.Background(), &core.ChatRequest{Model: "m"}); err == nil {
		t.Error("expected a validation error for a request with no messages")
	}
	if _, err := model.RequestStream(context.Background(), &core.ChatRequest{Model: "m"}); err == nil {
		t.Error("expected a validation error for a stream request with no messages")
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.bodies) != 0 {
		t.Errorf("an invalid request must not reach the network, got %d calls", len(rec.bodies))
	}
}
