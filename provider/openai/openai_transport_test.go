package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/option"
	sdkopenairesponses "github.com/openai/openai-go/responses"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestOpenAIChatRequestAndStreamWithLocalServer(t *testing.T) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}

		stream, _ := body["stream"].(bool)
		if !stream {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{
				"id":"chatcmpl_req",
				"object":"chat.completion",
				"created":123,
				"model":"gpt-4o",
				"choices":[{"index":0,"message":{"role":"assistant","content":"hello from chat"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
			}`)
			return
		}

		// The live API sends a usage chunk only when the request asked for
		// one. Emitting it unconditionally would let a client that never
		// sends stream_options pass here and report zero usage in
		// production, so assert the wire format before emitting.
		streamOptions, ok := body["stream_options"].(map[string]any)
		if !ok {
			t.Errorf("streaming request omitted stream_options: %#v", body)
			http.Error(w, "missing stream_options", http.StatusBadRequest)
			return
		}
		if includeUsage, _ := streamOptions["include_usage"].(bool); !includeUsage {
			t.Errorf("streaming request did not set stream_options.include_usage: %#v", streamOptions)
			http.Error(w, "missing include_usage", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hello "}}]}`,
			``,
			`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\""}}]}}]}`,
			``,
			`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Lima\"}"}}]}}]}`,
			``,
			`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			``,
			`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}`,
			``,
			`data: [DONE]`,
			``,
		}, "\n"))
	}))
	defer server.Close()

	model, err := New(
		"gpt-4o",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := &core.ChatRequest{
		Model:    "gpt-4o",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	}

	resp, err := model.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if resp.Message.GetTextContent() != "hello from chat" {
		t.Fatalf("unexpected request output %q", resp.Message.GetTextContent())
	}
	if resp.Usage.TotalTokens != 5 {
		t.Fatalf("unexpected request usage %#v", resp.Usage)
	}
	if resp.FinishReason != core.FinishReasonStop || resp.RawFinishReason != "stop" {
		t.Fatalf("unexpected finish reason %q/%q", resp.FinishReason, resp.RawFinishReason)
	}

	stream, err := model.RequestStream(context.Background(), req)
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 5 {
		t.Fatalf("expected 5 stream events, got %d: %#v", len(events), events)
	}
	if events[0].Type != core.StreamEventTextDelta || events[0].Delta != "hello " {
		t.Fatalf("unexpected text delta %#v", events[0])
	}
	if events[1].Type != core.StreamEventToolCallStart || events[1].ToolUse == nil || events[1].ToolUse.Name != "lookup" {
		t.Fatalf("unexpected tool-call start %#v", events[1])
	}
	if events[2].Type != core.StreamEventToolCallDelta || events[2].ToolCallID != "call_1" || events[2].Delta != `{"city":"` {
		t.Fatalf("unexpected first tool-call delta %#v", events[2])
	}
	if events[3].Type != core.StreamEventToolCallDelta || events[3].ToolCallID != "call_1" || events[3].Delta != `Lima"}` {
		t.Fatalf("unexpected second tool-call delta %#v", events[3])
	}
	if events[4].Type != core.StreamEventDone || events[4].Usage == nil {
		t.Fatalf("unexpected done event %#v", events[4])
	}
	// Usage only arrives because the request asked for it; a zero here means
	// stream_options.include_usage never reached the wire.
	if events[4].Usage.TotalTokens != 11 || events[4].Usage.PromptTokens != 7 {
		t.Fatalf("expected the streamed usage chunk to be reported, got %#v", events[4].Usage)
	}
	if events[4].FinishReason != core.FinishReasonToolCalls {
		t.Fatalf("expected tool_calls finish reason on done, got %q", events[4].FinishReason)
	}
}

func TestOpenAIResponsesRequestAndStreamWithLocalServer(t *testing.T) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			http.NotFound(w, r)
			return
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}

		stream, _ := body["stream"].(bool)
		if !stream {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{
				"id":"resp_req",
				"object":"response",
				"created_at":123,
				"status":"completed",
				"model":"gpt-4.1",
				"output":[
					{
						"id":"msg_1",
						"type":"message",
						"role":"assistant",
						"status":"completed",
						"content":[{"type":"output_text","text":"hello from responses"}]
					}
				],
				"usage":{"input_tokens":5,"output_tokens":4,"total_tokens":9}
			}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","status":"in_progress"}}`,
			``,
			`data: {"type":"response.function_call_arguments.delta","sequence_number":2,"output_index":0,"item_id":"fc_1","delta":"{\"city\":\"Lima\"}"}`,
			``,
			`data: {"type":"response.output_text.delta","sequence_number":3,"output_index":1,"item_id":"msg_1","content_index":0,"delta":"hello "}`,
			``,
			// The live API carries delta events' payload in "delta"; "text"
			// and "refusal" are only present on the matching .done events.
			`data: {"type":"response.reasoning_summary_text.delta","sequence_number":4,"output_index":2,"item_id":"rs_1","summary_index":0,"delta":"think"}`,
			``,
			`data: {"type":"response.output_item.done","sequence_number":5,"output_index":2,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"think"}],"encrypted_content":"enc_abc"}}`,
			``,
			`data: {"type":"response.refusal.delta","sequence_number":6,"output_index":1,"item_id":"msg_1","content_index":1,"delta":"no"}`,
			``,
			`data: {"type":"response.completed","sequence_number":6,"response":{"id":"resp_stream","object":"response","created_at":123,"status":"completed","model":"gpt-4.1","output":[],"usage":{"input_tokens":7,"output_tokens":4,"total_tokens":11}}}`,
			``,
			`data: [DONE]`,
			``,
		}, "\n"))
	}))
	defer server.Close()

	model, err := NewResponses(
		"gpt-4.1",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewResponses: %v", err)
	}

	req := &core.ChatRequest{
		Model:    "gpt-4.1",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	}

	resp, err := model.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if resp.Message.GetTextContent() != "hello from responses" {
		t.Fatalf("unexpected request output %q", resp.Message.GetTextContent())
	}
	if resp.Usage.TotalTokens != 9 {
		t.Fatalf("unexpected request usage %#v", resp.Usage)
	}
	if resp.FinishReason != core.FinishReasonStop || resp.RawFinishReason != "completed" {
		t.Fatalf("unexpected finish reason %q/%q", resp.FinishReason, resp.RawFinishReason)
	}

	stream, err := model.RequestStream(context.Background(), req)
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 7 {
		t.Fatalf("expected 7 stream events, got %d: %#v", len(events), events)
	}
	if events[0].Type != core.StreamEventToolCallStart || events[0].ToolUse == nil || events[0].ToolUse.ID != "call_1" {
		t.Fatalf("unexpected tool-call start %#v", events[0])
	}
	if events[1].Type != core.StreamEventToolCallDelta || events[1].ToolCallID != "call_1" || events[1].Delta != `{"city":"Lima"}` {
		t.Fatalf("unexpected tool-call delta %#v", events[1])
	}
	if events[2].Type != core.StreamEventTextDelta || events[2].Delta != "hello " {
		t.Fatalf("unexpected text delta %#v", events[2])
	}
	// The reasoning delta text lives in the event's "delta" field, not "text".
	if events[3].Type != core.StreamEventThinkingDelta || events[3].Delta != "think" {
		t.Fatalf("unexpected thinking delta %#v", events[3])
	}
	if events[3].ThinkingID != "rs_1" || events[3].ProviderName != "openai" {
		t.Fatalf("expected thinking delta to carry its item id and provider, got %#v", events[3])
	}
	// The encrypted content arrives only once the reasoning item completes,
	// and is delivered as a terminal event on that block.
	if events[4].Type != core.StreamEventThinkingDelta || events[4].Signature != "enc_abc" || events[4].ThinkingID != "rs_1" {
		t.Fatalf("expected a terminal thinking event carrying the signature, got %#v", events[4])
	}
	// The refusal delta text also lives in "delta", not "refusal".
	if events[5].Type != core.StreamEventTextDelta || events[5].Delta != "no" {
		t.Fatalf("unexpected refusal delta %#v", events[5])
	}
	if events[6].Type != core.StreamEventDone || events[6].Usage == nil || events[6].Usage.TotalTokens != 11 {
		t.Fatalf("unexpected done event %#v", events[6])
	}
	// A refusal settles the finish reason even though the response completed.
	if events[6].FinishReason != core.FinishReasonContentFilter {
		t.Fatalf("expected content_filter finish reason on done, got %q", events[6].FinishReason)
	}
}

// TestOpenAIResponsesStreamReportsInBandFailure covers a response that ends
// incomplete on a 200 stream: usage must still be reported and the failure
// surfaced rather than presented as a clean stop.
func TestOpenAIResponsesStreamReportsInBandFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"response.output_text.delta","sequence_number":1,"output_index":0,"item_id":"msg_1","content_index":0,"delta":"partial"}`,
			``,
			`data: {"type":"response.incomplete","sequence_number":2,"response":{"id":"resp_1","object":"response","created_at":123,"status":"incomplete","model":"gpt-4.1","output":[],"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":7,"output_tokens":4,"total_tokens":11}}}`,
			``,
			`data: [DONE]`,
			``,
		}, "\n"))
	}))
	defer server.Close()

	model, err := NewResponses(
		"gpt-4.1",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewResponses: %v", err)
	}

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "gpt-4.1",
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
		t.Fatalf("expected 3 stream events, got %d: %#v", len(events), events)
	}
	if events[1].Type != core.StreamEventError || events[1].Error == nil {
		t.Fatalf("expected an error event for the incomplete response, got %#v", events[1])
	}
	if !strings.Contains(events[1].Error.Error(), "max_output_tokens") {
		t.Fatalf("expected the reason in the error, got %v", events[1].Error)
	}
	if events[2].Type != core.StreamEventDone || events[2].Usage == nil || events[2].Usage.TotalTokens != 11 {
		t.Fatalf("expected usage to survive the failure, got %#v", events[2])
	}
	if events[2].FinishReason != core.FinishReasonLength {
		t.Fatalf("expected length finish reason, got %q", events[2].FinishReason)
	}
}

// TestOpenAIResponsesStreamSurfacesErrorEvent covers an upstream failure
// reported in-band on a 200 stream.
func TestOpenAIResponsesStreamSurfacesErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"error","sequence_number":1,"code":"server_error","message":"upstream exploded","param":""}`,
			``,
			`data: [DONE]`,
			``,
		}, "\n"))
	}))
	defer server.Close()

	model, err := NewResponses(
		"gpt-4.1",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewResponses: %v", err)
	}

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "gpt-4.1",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 2 {
		t.Fatalf("expected 2 stream events, got %d: %#v", len(events), events)
	}
	if events[0].Type != core.StreamEventError || !strings.Contains(events[0].Error.Error(), "upstream exploded") {
		t.Fatalf("unexpected error event %#v", events[0])
	}
	if events[1].Type != core.StreamEventDone || events[1].FinishReason != core.FinishReasonError {
		t.Fatalf("expected an error finish reason on done, got %#v", events[1])
	}
}

// TestOpenAIResponsesRequestsEncryptedReasoning pins the include[] parameter:
// without it the API omits the encrypted content and a streamed reasoning turn
// cannot be replayed.
func TestOpenAIResponsesRequestsEncryptedReasoning(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","created_at":123,"status":"completed","model":"o3","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()

	model, err := NewResponses(
		"o3",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("NewResponses: %v", err)
	}

	if _, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "o3",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		Thinking: &core.ThinkingConfig{Enabled: true},
	}); err != nil {
		t.Fatalf("Request: %v", err)
	}

	include, _ := captured["include"].([]any)
	if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("expected include=[reasoning.encrypted_content], got %#v", captured["include"])
	}
	// o3 is a reasoning model; sampling params must not be on the wire.
	if _, ok := captured["temperature"]; ok {
		t.Fatalf("expected no temperature for a reasoning model, got %#v", captured)
	}
}

func TestResponsesHelpersCoverRemainingBranches(t *testing.T) {
	schema := ensureAdditionalPropertiesFalse(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"tags": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"value": map[string]any{"type": "string"},
					},
				},
			},
		},
	})

	if schema["additionalProperties"] != false {
		t.Fatalf("expected top-level additionalProperties=false, got %#v", schema)
	}
	required, _ := schema["required"].([]string)
	if len(required) != 2 {
		t.Fatalf("expected both properties to become required, got %#v", schema["required"])
	}
	tags := schema["properties"].(map[string]any)["tags"].(map[string]any)
	items := tags["items"].(map[string]any)
	if items["additionalProperties"] != false {
		t.Fatalf("expected nested object additionalProperties=false, got %#v", items)
	}

	// A failed generation is not a clean stop.
	if got, raw := convertResponsesFinishReason(&sdkopenairesponses.Response{Status: "failed"}); got != core.FinishReasonError || raw != "failed" {
		t.Fatalf("expected failed status to map to error, got %q/%q", got, raw)
	}
	if got, raw := convertResponsesFinishReason(&sdkopenairesponses.Response{Status: "cancelled"}); got != core.FinishReasonError || raw != "cancelled" {
		t.Fatalf("expected cancelled status to map to error, got %q/%q", got, raw)
	}
	if got := convertTextConfig(&core.ResponseFormat{Type: "json_schema"}); got.Format.OfText != nil || got.Format.OfJSONSchema != nil {
		t.Fatalf("expected empty text config when schema details are missing, got %#v", got)
	}
}

func TestOpenAIChatRequestPropagatesTransportErrors(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	baseURL := server.URL + "/v1"
	server.Close()

	model, err := New(
		"gpt-4o",
		WithAPIKey("test-key"),
		WithBaseURL(baseURL),
		WithRequestOptions(option.WithHTTPClient(&http.Client{Timeout: time.Second})),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = model.Request(context.Background(), &core.ChatRequest{
		Model:    "gpt-4o",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err == nil {
		t.Fatal("expected Request to propagate the transport error")
	}
}

func TestOpenAIChatStreamEmitsErrorEventOnTransportFailure(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	baseURL := server.URL + "/v1"
	server.Close()

	model, err := New(
		"gpt-4o",
		WithAPIKey("test-key"),
		WithBaseURL(baseURL),
		WithRequestOptions(option.WithHTTPClient(&http.Client{Timeout: time.Second})),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "gpt-4o",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 1 {
		t.Fatalf("expected 1 stream event, got %d: %#v", len(events), events)
	}
	if events[0].Type != core.StreamEventError || events[0].Error == nil {
		t.Fatalf("expected a stream error event, got %#v", events[0])
	}
	if !strings.Contains(events[0].Error.Error(), "openai stream:") {
		t.Fatalf("expected openai stream prefix, got %v", events[0].Error)
	}
}

func TestOpenAIResponsesRequestPropagatesTransportErrors(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	baseURL := server.URL + "/v1"
	server.Close()

	model, err := NewResponses(
		"gpt-4.1",
		WithAPIKey("test-key"),
		WithBaseURL(baseURL),
		WithRequestOptions(option.WithHTTPClient(&http.Client{Timeout: time.Second})),
	)
	if err != nil {
		t.Fatalf("NewResponses: %v", err)
	}

	_, err = model.Request(context.Background(), &core.ChatRequest{
		Model:    "gpt-4.1",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err == nil {
		t.Fatal("expected Request to propagate the transport error")
	}
	if !strings.Contains(err.Error(), "openai responses:") {
		t.Fatalf("expected wrapped responses error, got %v", err)
	}
}

func TestOpenAIResponsesStreamEmitsErrorEventOnTransportFailure(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	baseURL := server.URL + "/v1"
	server.Close()

	model, err := NewResponses(
		"gpt-4.1",
		WithAPIKey("test-key"),
		WithBaseURL(baseURL),
		WithRequestOptions(option.WithHTTPClient(&http.Client{Timeout: time.Second})),
	)
	if err != nil {
		t.Fatalf("NewResponses: %v", err)
	}

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "gpt-4.1",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 1 {
		t.Fatalf("expected 1 stream event, got %d: %#v", len(events), events)
	}
	if events[0].Type != core.StreamEventError || events[0].Error == nil {
		t.Fatalf("expected a stream error event, got %#v", events[0])
	}
	if !strings.Contains(events[0].Error.Error(), "openai responses stream:") {
		t.Fatalf("expected openai responses stream prefix, got %v", events[0].Error)
	}
}

// TestOpenAIChatStreamSurfacesRefusal pins that a streamed safety refusal
// reaches the caller as text and settles the finish reason, rather than
// arriving as an empty, clean stop.
func TestOpenAIChatStreamSurfacesRefusal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"refusal":"I can't help with that."}}]}`,
			``,
			`data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			``,
			`data: [DONE]`,
			``,
		}, "\n"))
	}))
	defer server.Close()

	model, err := New(
		"gpt-4o",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL+"/v1"),
		WithRequestOptions(option.WithHTTPClient(server.Client())),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "gpt-4o",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 2 {
		t.Fatalf("expected 2 stream events, got %d: %#v", len(events), events)
	}
	if events[0].Type != core.StreamEventTextDelta || events[0].Delta != "I can't help with that." {
		t.Fatalf("expected the refusal text to be streamed, got %#v", events[0])
	}
	// The refusal wins over the "stop" the final chunk reports.
	if events[1].Type != core.StreamEventDone || events[1].FinishReason != core.FinishReasonContentFilter {
		t.Fatalf("expected content_filter finish reason on done, got %#v", events[1])
	}
}
