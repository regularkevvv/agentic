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

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hello "}}]}`,
			``,
			`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\""}}]}}]}`,
			``,
			`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Lima\"}"}}]}}]}`,
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
	if resp.Choices[0].Message.GetTextContent() != "hello from chat" {
		t.Fatalf("unexpected request output %q", resp.Choices[0].Message.GetTextContent())
	}
	if resp.Usage.TotalTokens != 5 {
		t.Fatalf("unexpected request usage %#v", resp.Usage)
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
	if events[4].Type != core.StreamEventDone || events[4].Usage == nil || events[4].Usage.TotalTokens != 11 {
		t.Fatalf("unexpected done event %#v", events[4])
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
			`data: {"type":"response.reasoning_summary_text.delta","sequence_number":4,"output_index":2,"item_id":"rs_1","summary_index":0,"text":"think"}`,
			``,
			`data: {"type":"response.refusal.delta","sequence_number":5,"output_index":1,"item_id":"msg_1","content_index":1,"refusal":"no"}`,
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
	if resp.Choices[0].Message.GetTextContent() != "hello from responses" {
		t.Fatalf("unexpected request output %q", resp.Choices[0].Message.GetTextContent())
	}
	if resp.Usage.TotalTokens != 9 {
		t.Fatalf("unexpected request usage %#v", resp.Usage)
	}

	stream, err := model.RequestStream(context.Background(), req)
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 6 {
		t.Fatalf("expected 6 stream events, got %d: %#v", len(events), events)
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
	if events[3].Type != core.StreamEventThinkingDelta || events[3].Delta != "think" {
		t.Fatalf("unexpected thinking delta %#v", events[3])
	}
	if events[4].Type != core.StreamEventTextDelta || events[4].Delta != "no" {
		t.Fatalf("unexpected refusal delta %#v", events[4])
	}
	if events[5].Type != core.StreamEventDone || events[5].Usage == nil || events[5].Usage.TotalTokens != 11 {
		t.Fatalf("unexpected done event %#v", events[5])
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

	if got := convertResponsesFinishReason(&sdkopenairesponses.Response{Status: "failed"}); got != core.FinishReasonStop {
		t.Fatalf("expected failed status to map to stop, got %q", got)
	}
	if got := convertResponsesFinishReason(&sdkopenairesponses.Response{Status: "canceled"}); got != core.FinishReasonStop {
		t.Fatalf("expected canceled status to map to stop, got %q", got)
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
