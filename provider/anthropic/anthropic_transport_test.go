package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestAnthropicRequestAndStreamWithLocalServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
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
				"id":"msg_req",
				"type":"message",
				"role":"assistant",
				"model":"claude-sonnet",
				"content":[{"type":"text","text":"hello from claude"}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":4,"output_tokens":2,"cache_read_input_tokens":1,"cache_creation_input_tokens":3}
			}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"msg_stream","type":"message","role":"assistant","content":[],"model":"claude-sonnet","usage":{"input_tokens":7,"cache_read_input_tokens":1,"cache_creation_input_tokens":2}}}`,
			``,
			`event: content_block_start`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"lookup","input":{}}}`,
			``,
			`event: content_block_delta`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Lima\"}"}}`,
			``,
			`event: content_block_start`,
			`data: {"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}`,
			``,
			`event: content_block_delta`,
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"think"}}`,
			``,
			`event: content_block_start`,
			`data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
			``,
			`event: content_block_delta`,
			`data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"hello"}}`,
			``,
			`event: message_delta`,
			`data: {"type":"message_delta","usage":{"output_tokens":5}}`,
			``,
			`event: message_stop`,
			`data: {"type":"message_stop"}`,
			``,
		}, "\n"))
	}))
	defer server.Close()

	model, err := New("claude-sonnet", WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := &core.ChatRequest{
		Model:    "claude-sonnet",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	}

	resp, err := model.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if resp.Choices[0].Message.GetTextContent() != "hello from claude" {
		t.Fatalf("unexpected request output %q", resp.Choices[0].Message.GetTextContent())
	}
	if resp.Usage.CacheReadTokens != 1 || resp.Usage.CacheCreationTokens != 3 {
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
	if events[0].Type != core.StreamEventToolCallStart || events[0].ToolUse == nil || events[0].ToolUse.Name != "lookup" {
		t.Fatalf("unexpected tool-call start %#v", events[0])
	}
	if events[1].Type != core.StreamEventToolCallDelta || events[1].ToolCallID != "call_1" || events[1].Delta != `{"city":"Lima"}` {
		t.Fatalf("unexpected tool-call delta %#v", events[1])
	}
	if events[2].Type != core.StreamEventThinkingDelta || events[2].Delta != "think" {
		t.Fatalf("unexpected thinking delta %#v", events[2])
	}
	if events[3].Type != core.StreamEventTextDelta || events[3].Delta != "hello" {
		t.Fatalf("unexpected text delta %#v", events[3])
	}
	if events[4].Type != core.StreamEventDone || events[4].Usage == nil || events[4].Usage.TotalTokens != 12 {
		t.Fatalf("unexpected done event %#v", events[4])
	}
}

func TestAnthropicConversionHelpersCoverAdditionalBranches(t *testing.T) {
	param := convertMessage(core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{
				Type:     core.ContentThinking,
				Thinking: &core.ThinkingBlock{ID: "redacted_thinking", Signature: "secret"},
			},
			{Type: core.ContentText, Text: "cached text"},
			{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{TTL: "1h"}},
			{
				Type:      core.ContentImageData,
				ImageData: &core.ImageData{Data: "AQID", MediaType: "image/png"},
			},
			{
				Type:     core.ContentImageURL,
				ImageURL: &core.ImageURL{URL: "https://example.com/image.png"},
			},
			{
				Type:        core.ContentDocumentURL,
				DocumentURL: &core.DocumentURL{URL: "https://example.com/file.pdf"},
			},
			{
				Type:    core.ContentToolUse,
				ToolUse: &core.ToolUse{ID: "call_1", Name: "lookup", Input: map[string]any{"city": "Lima"}},
			},
			{
				Type:       core.ContentToolResult,
				ToolResult: &core.ToolResult{ToolUseID: "call_1", Content: `{"ok":true}`, IsError: true},
			},
			{
				Type:         core.ContentUploadedFile,
				UploadedFile: &core.UploadedFile{FileID: "file_123"},
			},
		},
	})

	if len(param.Content) != 7 {
		t.Fatalf("expected 7 content blocks, got %#v", param.Content)
	}
	if cc := param.Content[1].GetCacheControl(); cc == nil || cc.TTL != anthropic.CacheControlEphemeralTTLTTL1h {
		t.Fatalf("expected cache control to be attached to previous block, got %#v", cc)
	}

	jsonSchema := convertResponseFormat(&core.ResponseFormat{
		Type:       "json_schema",
		JSONSchema: &core.JSONSchemaFormat{Schema: map[string]any{"type": "object"}},
	})
	if jsonSchema.Format.Schema["type"] != "object" {
		t.Fatalf("expected JSON schema response format, got %#v", jsonSchema)
	}

	unsupported := convertResponseFormat(&core.ResponseFormat{Type: "json_object"})
	if unsupported.Format.Schema != nil {
		t.Fatalf("expected unsupported format to be empty, got %#v", unsupported)
	}

	mustBlock := func(raw string) anthropic.ContentBlockUnion {
		t.Helper()
		var block anthropic.ContentBlockUnion
		if err := json.Unmarshal([]byte(raw), &block); err != nil {
			t.Fatalf("unmarshal content block: %v", err)
		}
		return block
	}

	msg := convertResponseMessage([]anthropic.ContentBlockUnion{
		mustBlock(`{"type":"thinking","thinking":"reasoning","signature":"sig"}`),
		mustBlock(`{"type":"redacted_thinking","data":"encrypted"}`),
		mustBlock(`{"type":"text","text":"hello"}`),
		mustBlock(`{"type":"tool_use","id":"call_1","name":"lookup","input":{"city":"Lima"}}`),
	}, "assistant")

	if msg.GetThinkingContent() != "reasoning" {
		t.Fatalf("unexpected thinking content %q", msg.GetThinkingContent())
	}
	toolUses := msg.GetToolUses()
	if len(toolUses) != 1 || toolUses[0].Input["city"] != "Lima" {
		t.Fatalf("unexpected tool uses %#v", toolUses)
	}
	if msg.Content[1].Thinking == nil || !msg.Content[1].Thinking.IsRedacted() || msg.Content[1].Thinking.Signature != "encrypted" {
		t.Fatalf("unexpected redacted thinking block %#v", msg.Content[1])
	}
}
