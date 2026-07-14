package gemini

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/genai"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestGeminiNewRequestAndStreamWithLocalServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, ":generateContent"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{
				"candidates":[
					{
						"content":{"role":"model","parts":[{"text":"hello from gemini"}]},
						"finishReason":"STOP"
					}
				],
				"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2}
			}`)
		case strings.HasSuffix(r.URL.Path, ":streamGenerateContent"):
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, strings.Join([]string{
				`data:{"candidates":[{"content":{"role":"model","parts":[{"text":"hello "},{"thought":true,"text":"think"},{"functionCall":{"name":"lookup","args":{"city":"Lima"}}}]}}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":4,"cachedContentTokenCount":1}}`,
				``,
			}, "\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	t.Setenv("GOOGLE_GEMINI_BASE_URL", server.URL)

	model, err := New("gemini-2.5-pro", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := &core.ChatRequest{
		Model:    "gemini-2.5-pro",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	}

	resp, err := model.Request(context.Background(), req)
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if resp.Choices[0].Message.GetTextContent() != "hello from gemini" {
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
	if events[1].Type != core.StreamEventThinkingDelta || events[1].Delta != "think" {
		t.Fatalf("unexpected thinking delta %#v", events[1])
	}
	if events[2].Type != core.StreamEventToolCallStart || events[2].ToolUse == nil || events[2].ToolUse.Name != "lookup" {
		t.Fatalf("unexpected tool-call start %#v", events[2])
	}
	if events[3].Type != core.StreamEventToolCallDelta || events[3].ToolCallID == "" || events[3].Delta != `{"city":"Lima"}` {
		t.Fatalf("unexpected tool-call delta %#v", events[3])
	}
	if events[4].Type != core.StreamEventDone || events[4].Usage == nil || events[4].Usage.TotalTokens != 11 {
		t.Fatalf("unexpected done event %#v", events[4])
	}
}

func TestGeminiNewVertexAIWithCustomBaseURL(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()

	t.Setenv("GOOGLE_VERTEX_BASE_URL", server.URL)

	model, err := New("gemini-2.5-pro", WithVertexAI("", ""))
	if err != nil {
		t.Fatalf("expected Vertex AI client creation to succeed with custom base URL, got %v", err)
	}
	if model.Name() != "gemini-2.5-pro" {
		t.Fatalf("unexpected model name %q", model.Name())
	}
}

func TestGeminiSchemaAndCandidateHelpersCoverRemainingBranches(t *testing.T) {
	schema := convertSchema(map[string]any{
		"type":        "array",
		"description": "outer",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"kind": map[string]any{
					"type": "string",
					"enum": []any{"a", "b"},
				},
			},
			"required": []any{"kind"},
		},
	})

	if schema.Items == nil || len(schema.Items.Required) != 1 {
		t.Fatalf("unexpected converted schema %#v", schema)
	}
	if len(schema.Items.Properties["kind"].Enum) != 2 {
		t.Fatalf("expected enum values on nested property, got %#v", schema.Items.Properties["kind"])
	}

	msg := convertCandidateMessage(&genai.Candidate{})
	if msg.Role != core.RoleAssistant || len(msg.Content) != 0 {
		t.Fatalf("expected empty assistant message for nil content, got %#v", msg)
	}
}
