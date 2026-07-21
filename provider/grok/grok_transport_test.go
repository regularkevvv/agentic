package grok_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/grok"

	"github.com/openai/openai-go/option"
)

// newServer starts a stub xAI endpoint that records the decoded request body
// and replies with the given JSON.
func newServer(t *testing.T, responseBody string, captured *map[string]any) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
			return
		}
		if captured != nil {
			decoded := map[string]any{}
			if err := json.Unmarshal(raw, &decoded); err != nil {
				t.Errorf("decoding request body %q: %v", raw, err)
				return
			}
			*captured = decoded
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, responseBody)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// newStreamServer starts a stub xAI endpoint that replies with the given
// server-sent-event chunks.
func newStreamServer(t *testing.T, chunks []string) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range chunks {
			_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv
}

const okResponse = `{
	"id": "resp-1",
	"model": "grok-4.3",
	"created": 1700000000,
	"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
	"usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
}`

func userRequest(model string) *core.ChatRequest {
	return &core.ChatRequest{
		Model:    model,
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	}
}

// TestReasoningEffortOnTheWire pins which models may carry reasoning_effort and
// with what value. xAI rejects an unsupported model or value with HTTP 400
// rather than ignoring the field.
func TestReasoningEffortOnTheWire(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		thinking *core.ThinkingConfig
		opts     map[string]any
		want     any // nil means the field must be absent
	}{
		{
			name:     "non-reasoning model never carries the field",
			model:    "grok-2-1212",
			thinking: &core.ThinkingConfig{Enabled: true, BudgetTokens: 30000},
			want:     nil,
		},
		{
			name:     "code model never carries the field",
			model:    "grok-code-fast-1",
			thinking: &core.ThinkingConfig{Enabled: true},
			want:     nil,
		},
		{
			name:     "grok 4.3 carries medium",
			model:    "grok-4.3",
			thinking: &core.ThinkingConfig{Enabled: true},
			want:     "medium",
		},
		{
			name:     "grok 4.3 is told not to reason with none",
			model:    "grok-4.3",
			thinking: &core.ThinkingConfig{},
			want:     "none",
		},
		{
			name:     "grok 4.5 rejects none so the field is dropped",
			model:    "grok-4.5",
			thinking: &core.ThinkingConfig{},
			want:     nil,
		},
		{
			name:     "grok 3 mini has no medium so it is raised to high",
			model:    "grok-3-mini",
			thinking: &core.ThinkingConfig{Enabled: true},
			want:     "high",
		},
		{
			name:     "no thinking config leaves the field out",
			model:    "grok-4.3",
			thinking: nil,
			want:     nil,
		},
		{
			name:  "provider options override the derived value",
			model: "grok-4.5",
			opts:  map[string]any{grok.ProviderName: grok.Options{ReasoningEffort: "low"}},
			want:  "low",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var body map[string]any
			srv := newServer(t, okResponse, &body)

			model := grok.MustNew(tt.model, grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))
			req := userRequest(tt.model)
			req.Thinking = tt.thinking
			req.ProviderOptions = tt.opts

			if _, err := model.Request(t.Context(), req); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			got, present := body["reasoning_effort"]
			if tt.want == nil {
				if present {
					t.Errorf("reasoning_effort was sent as %v for %q, want absent", got, tt.model)
				}
				return
			}
			if !present {
				t.Fatalf("reasoning_effort absent for %q, want %v", tt.model, tt.want)
			}
			if got != tt.want {
				t.Errorf("reasoning_effort = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestStopSequencesOnTheWire(t *testing.T) {
	var body map[string]any
	srv := newServer(t, okResponse, &body)

	model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))
	req := userRequest("grok-4.3")
	req.StopSequences = []string{"END", "STOP"}

	if _, err := model.Request(t.Context(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	stop, ok := body["stop"].([]any)
	if !ok {
		t.Fatalf("stop = %v (%T), want a JSON array", body["stop"], body["stop"])
	}
	if len(stop) != 2 || stop[0] != "END" || stop[1] != "STOP" {
		t.Errorf("stop = %v, want [END STOP]", stop)
	}
}

// TestReasoningInResponse covers the non-standard field xAI uses for thinking
// text, which the OpenAI-typed response drops into ExtraFields.
func TestReasoningInResponse(t *testing.T) {
	tests := []struct {
		name     string
		message  string
		wantText string
	}{
		{
			name:     "reasoning field",
			message:  `{"role": "assistant", "content": "hi", "reasoning": "thought hard"}`,
			wantText: "thought hard",
		},
		{
			name:     "reasoning_content field",
			message:  `{"role": "assistant", "content": "hi", "reasoning_content": "thought hard"}`,
			wantText: "thought hard",
		},
		{
			name:     "no reasoning at all",
			message:  `{"role": "assistant", "content": "hi"}`,
			wantText: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"id":"r","model":"grok-4.3","created":1700000000,"choices":[{"index":0,"message":` +
				tt.message + `,"finish_reason":"stop"}]}`
			srv := newServer(t, body, nil)

			model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))
			resp, err := model.Request(t.Context(), userRequest("grok-4.3"))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var thinking *core.ThinkingBlock
			for _, part := range resp.Message.Content {
				if part.Type == core.ContentThinking {
					thinking = part.Thinking
					break
				}
			}

			if tt.wantText == "" {
				if thinking != nil {
					t.Fatalf("got thinking part %+v, want none", thinking)
				}
				return
			}
			if thinking == nil {
				t.Fatal("no thinking part in response")
			}
			if thinking.Text != tt.wantText {
				t.Errorf("thinking text = %q, want %q", thinking.Text, tt.wantText)
			}
			if thinking.ProviderName != grok.ProviderName {
				t.Errorf("thinking provider = %q, want %q", thinking.ProviderName, grok.ProviderName)
			}
			// Thinking precedes the answer it explains.
			if resp.Message.Content[0].Type != core.ContentThinking {
				t.Errorf("first part is %q, want %q", resp.Message.Content[0].Type, core.ContentThinking)
			}
		})
	}
}

func TestFinishReasonFromResponse(t *testing.T) {
	tests := []struct {
		raw  string
		want core.FinishReason
	}{
		{"stop", core.FinishReasonStop},
		{"max_output_tokens", core.FinishReasonLength},
		{"failed", core.FinishReasonError},
		{"cancelled", core.FinishReasonError},
		{"a_reason_we_do_not_know", core.FinishReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.raw, func(t *testing.T) {
			body := `{"id":"r","model":"grok-4.3","created":1700000000,"choices":[{"index":0,` +
				`"message":{"role":"assistant","content":"hi"},"finish_reason":"` + tt.raw + `"}]}`
			srv := newServer(t, body, nil)

			model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))
			resp, err := model.Request(t.Context(), userRequest("grok-4.3"))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if resp.FinishReason != tt.want {
				t.Errorf("FinishReason = %q, want %q", resp.FinishReason, tt.want)
			}
			if resp.RawFinishReason != tt.raw {
				t.Errorf("RawFinishReason = %q, want %q", resp.RawFinishReason, tt.raw)
			}
		})
	}
}

func TestRequestResponseFields(t *testing.T) {
	body := `{"id":"resp-9","model":"grok-4.3","created":1700000000,` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"hi","tool_calls":[` +
		`{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}}]},` +
		`"finish_reason":"tool_calls"}],` +
		`"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7,` +
		`"completion_tokens_details":{"reasoning_tokens":2},"prompt_tokens_details":{"cached_tokens":1}}}`
	srv := newServer(t, body, nil)

	model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))
	resp, err := model.Request(t.Context(), userRequest("grok-4.3"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.ID != "resp-9" {
		t.Errorf("ID = %q, want %q", resp.ID, "resp-9")
	}
	if resp.Model != "grok-4.3" {
		t.Errorf("Model = %q, want %q", resp.Model, "grok-4.3")
	}
	if resp.Usage.ReasoningTokens != 2 {
		t.Errorf("ReasoningTokens = %d, want 2", resp.Usage.ReasoningTokens)
	}
	if resp.Usage.CacheReadTokens != 1 {
		t.Errorf("CacheReadTokens = %d, want 1", resp.Usage.CacheReadTokens)
	}

	toolUses := resp.Message.GetToolUses()
	if len(toolUses) != 1 {
		t.Fatalf("got %d tool uses, want 1", len(toolUses))
	}
	if toolUses[0].Name != "lookup" || toolUses[0].Input["q"] != "go" {
		t.Errorf("tool use = %+v, want lookup{q: go}", toolUses[0])
	}
}

func TestRequestNoChoices(t *testing.T) {
	srv := newServer(t, `{"id":"r","model":"grok-4.3","created":1700000000,"choices":[]}`, nil)

	model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))
	_, err := model.Request(t.Context(), userRequest("grok-4.3"))
	if !errors.Is(err, grok.ErrNoChoices) {
		t.Errorf("err = %v, want %v", err, grok.ErrNoChoices)
	}
}

func TestRequestStreamEvents(t *testing.T) {
	srv := newStreamServer(t, []string{
		`{"id":"c","model":"grok-4.3","choices":[{"index":0,"delta":{"reasoning":"first thought"}}]}`,
		`{"id":"c","model":"grok-4.3","choices":[{"index":0,"delta":{"reasoning_content":"second thought"}}]}`,
		`{"id":"c","model":"grok-4.3","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		`{"id":"c","model":"grok-4.3","choices":[{"index":0,"delta":{},"finish_reason":"failed"}]}`,
		`{"id":"c","model":"grok-4.3","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`,
	})

	model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))
	stream, err := model.RequestStream(t.Context(), userRequest("grok-4.3"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var (
		thinking []string
		text     strings.Builder
		done     *core.StreamEvent
	)
	for event := range stream.Events {
		switch event.Type {
		case core.StreamEventThinkingDelta:
			if event.ProviderName != grok.ProviderName {
				t.Errorf("thinking provider = %q, want %q", event.ProviderName, grok.ProviderName)
			}
			thinking = append(thinking, event.Delta)
		case core.StreamEventTextDelta:
			text.WriteString(event.Delta)
		case core.StreamEventDone:
			e := event
			done = &e
		case core.StreamEventError:
			t.Fatalf("unexpected stream error: %v", event.Error)
		}
	}

	if len(thinking) != 2 || thinking[0] != "first thought" || thinking[1] != "second thought" {
		t.Errorf("thinking deltas = %v, want [first thought second thought]", thinking)
	}
	if text.String() != "hi" {
		t.Errorf("text = %q, want %q", text.String(), "hi")
	}
	if done == nil {
		t.Fatal("no done event")
	}
	if done.FinishReason != core.FinishReasonError {
		t.Errorf("done finish reason = %q, want %q", done.FinishReason, core.FinishReasonError)
	}
	if done.Usage == nil || done.Usage.TotalTokens != 7 {
		t.Errorf("done usage = %+v, want 7 total tokens", done.Usage)
	}
}

func TestRequestStreamToolCalls(t *testing.T) {
	srv := newStreamServer(t, []string{
		`{"id":"c","model":"grok-4.3","choices":[{"index":0,"delta":{"tool_calls":[` +
			`{"index":0,"id":"call-1","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}`,
		`{"id":"c","model":"grok-4.3","choices":[{"index":0,"delta":{"tool_calls":[` +
			`{"index":0,"function":{"arguments":"\"go\"}"}}]}}]}`,
		`{"id":"c","model":"grok-4.3","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	})

	model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))
	stream, err := model.RequestStream(t.Context(), userRequest("grok-4.3"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var (
		starts []string
		args   strings.Builder
		callID string
	)
	for event := range stream.Events {
		switch event.Type {
		case core.StreamEventToolCallStart:
			starts = append(starts, event.ToolUse.Name)
		case core.StreamEventToolCallDelta:
			args.WriteString(event.Delta)
			callID = event.ToolCallID
		case core.StreamEventError:
			t.Fatalf("unexpected stream error: %v", event.Error)
		}
	}

	if len(starts) != 1 || starts[0] != "lookup" {
		t.Errorf("tool call starts = %v, want [lookup]", starts)
	}
	if args.String() != `{"q":"go"}` {
		t.Errorf("arguments = %q, want %q", args.String(), `{"q":"go"}`)
	}
	if callID != "call-1" {
		t.Errorf("tool call id on later delta = %q, want %q", callID, "call-1")
	}
}

func TestRequestStreamAsksForUsage(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))
	stream, err := model.RequestStream(t.Context(), userRequest("grok-4.3"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := stream.Wait(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}

	opts, ok := body["stream_options"].(map[string]any)
	if !ok || opts["include_usage"] != true {
		t.Errorf("stream_options = %v, want include_usage true", body["stream_options"])
	}
}

func TestRequestParamsOnTheWire(t *testing.T) {
	var body map[string]any
	srv := newServer(t, okResponse, &body)

	model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))

	temperature, topP := 0.25, 0.9
	maxTokens := 512
	toolChoice := core.ToolChoiceRequired
	strict := true

	req := userRequest("grok-4.3")
	req.Temperature = &temperature
	req.TopP = &topP
	req.MaxTokens = &maxTokens
	req.ToolChoice = &toolChoice
	req.Tools = []core.Tool{{
		Type: core.ToolTypeFunction,
		Function: core.Function{
			Name:        "lookup",
			Description: "look something up",
			Parameters:  map[string]any{"type": "object"},
		},
	}}
	req.ResponseFormat = &core.ResponseFormat{
		Type: "json_schema",
		JSONSchema: &core.JSONSchemaFormat{
			Name:        "answer",
			Description: "the answer",
			Schema:      map[string]any{"type": "object"},
			Strict:      &strict,
		},
	}

	if _, err := model.Request(t.Context(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if body["temperature"] != 0.25 {
		t.Errorf("temperature = %v, want 0.25", body["temperature"])
	}
	if body["top_p"] != 0.9 {
		t.Errorf("top_p = %v, want 0.9", body["top_p"])
	}
	if body["max_completion_tokens"] != float64(512) {
		t.Errorf("max_completion_tokens = %v, want 512", body["max_completion_tokens"])
	}
	if body["tool_choice"] != "required" {
		t.Errorf("tool_choice = %v, want required", body["tool_choice"])
	}

	tools, ok := body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v, want one entry", body["tools"])
	}
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "lookup" || fn["description"] != "look something up" {
		t.Errorf("function = %v, want lookup/look something up", fn)
	}

	format, ok := body["response_format"].(map[string]any)
	if !ok || format["type"] != "json_schema" {
		t.Fatalf("response_format = %v, want a json_schema object", body["response_format"])
	}
	schema := format["json_schema"].(map[string]any)
	if schema["name"] != "answer" || schema["strict"] != true {
		t.Errorf("json_schema = %v, want name answer and strict true", schema)
	}
}

// TestMessageConversionOnTheWire covers every message shape the converter
// handles, including the multimodal parts that only appear on an array-form
// user message.
func TestMessageConversionOnTheWire(t *testing.T) {
	var body map[string]any
	srv := newServer(t, okResponse, &body)

	model := grok.MustNew("grok-4.3", grok.WithAPIKey("k"), grok.WithBaseURL(srv.URL))

	req := &core.ChatRequest{
		Model: "grok-4.3",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "be brief"),
			{Role: core.RoleUser, Content: []core.Part{
				{Type: core.ContentText, Text: "look at this"},
				{Type: core.ContentImageURL, ImageURL: &core.ImageURL{URL: "https://x.test/a.png", Detail: "high"}},
				{Type: core.ContentImageData, ImageData: &core.ImageData{
					Data:           "QUJD",
					MediaType:      "image/png",
					VendorMetadata: map[string]any{"detail": "low"},
				}},
				{Type: core.ContentCachePoint},
			}},
			core.NewToolUseMessage(core.ToolUse{ID: "call-1", Name: "lookup", Input: map[string]any{"q": "go"}}),
			core.NewToolResultMessageFor("call-1", "lookup", "result text", false),
			core.NewTextMessage(core.RoleAssistant, "plain answer"),
		},
	}

	if _, err := model.Request(t.Context(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages, ok := body["messages"].([]any)
	if !ok || len(messages) != 5 {
		t.Fatalf("messages = %v, want five entries", body["messages"])
	}

	system := messages[0].(map[string]any)
	if system["role"] != "system" || system["content"] != "be brief" {
		t.Errorf("system message = %v", system)
	}

	parts, ok := messages[1].(map[string]any)["content"].([]any)
	if !ok || len(parts) != 4 {
		t.Fatalf("user content = %v, want four parts", messages[1])
	}
	imageURL := parts[1].(map[string]any)["image_url"].(map[string]any)
	if imageURL["url"] != "https://x.test/a.png" || imageURL["detail"] != "high" {
		t.Errorf("image_url part = %v", imageURL)
	}
	imageData := parts[2].(map[string]any)["image_url"].(map[string]any)
	if imageData["url"] != "data:image/png;base64,QUJD" || imageData["detail"] != "low" {
		t.Errorf("image_data part = %v", imageData)
	}
	if parts[3].(map[string]any)["text"] != "" {
		t.Errorf("cache point part = %v, want empty text", parts[3])
	}

	toolCalls := messages[2].(map[string]any)["tool_calls"].([]any)
	call := toolCalls[0].(map[string]any)["function"].(map[string]any)
	if call["name"] != "lookup" || call["arguments"] != `{"q":"go"}` {
		t.Errorf("tool call = %v", call)
	}

	toolResult := messages[3].(map[string]any)
	if toolResult["role"] != "tool" || toolResult["tool_call_id"] != "call-1" {
		t.Errorf("tool result message = %v", toolResult)
	}

	assistant := messages[4].(map[string]any)
	if assistant["role"] != "assistant" || assistant["content"] != "plain answer" {
		t.Errorf("assistant message = %v", assistant)
	}
}

func TestWithRequestOptions(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("X-Custom")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, okResponse)
	}))
	t.Cleanup(srv.Close)

	model := grok.MustNew("grok-4.3",
		grok.WithAPIKey("k"),
		grok.WithBaseURL(srv.URL),
		grok.WithRequestOptions(option.WithHeader("X-Custom", "set")),
	)
	if _, err := model.Request(t.Context(), userRequest("grok-4.3")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "set" {
		t.Errorf("X-Custom header = %q, want %q", got, "set")
	}
}

func TestRequestAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"bad request"}}`)
	}))
	t.Cleanup(srv.Close)

	model := grok.MustNew("grok-4.3",
		grok.WithAPIKey("k"),
		grok.WithBaseURL(srv.URL),
		grok.WithRequestOptions(option.WithMaxRetries(0)),
	)
	if _, err := model.Request(t.Context(), userRequest("grok-4.3")); err == nil {
		t.Error("expected an error for a 400 response, got nil")
	}
}

func TestRequestStreamAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"bad request"}}`)
	}))
	t.Cleanup(srv.Close)

	model := grok.MustNew("grok-4.3",
		grok.WithAPIKey("k"),
		grok.WithBaseURL(srv.URL),
		grok.WithRequestOptions(option.WithMaxRetries(0)),
	)
	stream, err := model.RequestStream(t.Context(), userRequest("grok-4.3"))
	if err != nil {
		t.Fatalf("unexpected error opening stream: %v", err)
	}

	var streamErr error
	for event := range stream.Events {
		if event.Type == core.StreamEventError {
			streamErr = event.Error
		}
		if event.Type == core.StreamEventDone {
			t.Error("got a done event after a failed stream")
		}
	}
	if streamErr == nil {
		t.Fatal("expected a stream error event, got none")
	}
	if !strings.Contains(streamErr.Error(), "grok stream:") {
		t.Errorf("stream error = %v, want it wrapped with %q", streamErr, "grok stream:")
	}
}
