package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
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
				"responseId":"resp-1",
				"modelVersion":"gemini-2.5-pro-002",
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
				`data:{"candidates":[{"content":{"role":"model","parts":[{"text":"hello "},{"thought":true,"text":"think","thoughtSignature":"c2lnLWJ5dGVz"},{"functionCall":{"id":"fc-1","name":"lookup","args":{"city":"Lima"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":4,"cachedContentTokenCount":1}}`,
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
	if resp.Message.GetTextContent() != "hello from gemini" {
		t.Fatalf("unexpected request output %q", resp.Message.GetTextContent())
	}
	if resp.ID != "resp-1" || resp.Model != "gemini-2.5-pro-002" {
		t.Fatalf("expected response id and resolved model, got %q / %q", resp.ID, resp.Model)
	}
	if resp.FinishReason != core.FinishReasonStop || resp.RawFinishReason != "STOP" {
		t.Fatalf("unexpected finish reason %q / %q", resp.FinishReason, resp.RawFinishReason)
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
	if events[1].Signature != base64.StdEncoding.EncodeToString([]byte("sig-bytes")) {
		t.Fatalf("expected the thought signature on the thinking delta, got %q", events[1].Signature)
	}
	if events[1].ProviderName != providerName {
		t.Fatalf("unexpected thinking provider %q", events[1].ProviderName)
	}
	if events[2].Type != core.StreamEventToolCallStart || events[2].ToolUse == nil || events[2].ToolUse.Name != "lookup" {
		t.Fatalf("unexpected tool-call start %#v", events[2])
	}
	if events[2].ToolUse.ID != "fc-1" {
		t.Fatalf("expected the provider-issued call id, got %q", events[2].ToolUse.ID)
	}
	if events[3].Type != core.StreamEventToolCallDelta || events[3].ToolCallID != "fc-1" || events[3].Delta != `{"city":"Lima"}` {
		t.Fatalf("unexpected tool-call delta %#v", events[3])
	}
	if events[4].Type != core.StreamEventDone || events[4].Usage == nil || events[4].Usage.TotalTokens != 11 {
		t.Fatalf("unexpected done event %#v", events[4])
	}
	if events[4].FinishReason != core.FinishReasonStop {
		t.Fatalf("expected the finish reason on the done event, got %q", events[4].FinishReason)
	}
}

// TestGeminiToolResultWireFormat pins the request body: Gemini correlates a
// functionResponse to its functionCall by name, so the name must be the tool's
// own name and never the tool-call id.
func TestGeminiToolResultWireFormat(t *testing.T) {
	var body []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
	}))
	defer server.Close()

	t.Setenv("GOOGLE_GEMINI_BASE_URL", server.URL)

	model, err := New("gemini-2.5-pro", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = model.Request(context.Background(), &core.ChatRequest{
		Model: "gemini-2.5-pro",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "weather in Lima?"),
			core.NewToolUseMessage(core.ToolUse{ID: "fc-1", Name: "get_weather", Input: map[string]any{"city": "Lima"}}),
			core.NewToolResultMessageFor("fc-1", "get_weather", `{"temp":72}`, false),
		},
		StopSequences: []string{"HALT"},
	})
	if err != nil {
		t.Fatalf("Request: %v", err)
	}

	var sent struct {
		Contents []struct {
			Parts []struct {
				FunctionCall *struct {
					ID   string `json:"id"`
					Name string `json:"name"`
				} `json:"functionCall"`
				FunctionResponse *struct {
					ID   string `json:"id"`
					Name string `json:"name"`
				} `json:"functionResponse"`
				ThoughtSignature string `json:"thoughtSignature"`
			} `json:"parts"`
		} `json:"contents"`
		GenerationConfig struct {
			StopSequences []string `json:"stopSequences"`
		} `json:"generationConfig"`
	}
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatalf("unmarshal request body: %v (body %s)", err, body)
	}

	if len(sent.Contents) != 3 {
		t.Fatalf("expected 3 contents on the wire, got %d: %s", len(sent.Contents), body)
	}
	call := sent.Contents[1].Parts[0].FunctionCall
	if call == nil || call.Name != "get_weather" || call.ID != "fc-1" {
		t.Fatalf("unexpected functionCall %+v: %s", call, body)
	}
	if sent.Contents[1].Parts[0].ThoughtSignature == "" {
		t.Fatalf("expected the first functionCall to carry a thought signature: %s", body)
	}
	resp := sent.Contents[2].Parts[0].FunctionResponse
	if resp == nil {
		t.Fatalf("expected a functionResponse part: %s", body)
	}
	if resp.Name != "get_weather" {
		t.Fatalf("expected functionResponse.name to be the tool name, got %q: %s", resp.Name, body)
	}
	if resp.ID != "fc-1" {
		t.Fatalf("expected functionResponse.id to echo the call id, got %q: %s", resp.ID, body)
	}
	if len(sent.GenerationConfig.StopSequences) != 1 || sent.GenerationConfig.StopSequences[0] != "HALT" {
		t.Fatalf("expected stop sequences on the wire, got %+v: %s", sent.GenerationConfig.StopSequences, body)
	}
}

// TestGeminiSchemaWireFormat pins that a JSON Schema reaches the API through
// the SDK passthrough fields with $ref, anyOf, format and constraints intact.
func TestGeminiSchemaWireFormat(t *testing.T) {
	var body []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"{}"}]},"finishReason":"STOP"}]}`)
	}))
	defer server.Close()

	t.Setenv("GOOGLE_GEMINI_BASE_URL", server.URL)

	model, err := New("gemini-2.5-pro", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	schema := map[string]any{
		"type":  "object",
		"$defs": map[string]any{"Leg": map[string]any{"type": "string", "format": "date-time"}},
		"properties": map[string]any{
			"leg":  map[string]any{"$ref": "#/$defs/Leg"},
			"n":    map[string]any{"type": "integer", "minimum": 1},
			"opt":  map[string]any{"anyOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "null"}}},
			"note": map[string]any{"type": "string", "maxLength": 20},
		},
	}

	_, err = model.Request(context.Background(), &core.ChatRequest{
		Model:    "gemini-2.5-pro",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "go")},
		Tools: []core.Tool{{
			Type:     core.ToolTypeFunction,
			Function: core.Function{Name: "plan", Parameters: schema},
		}},
		ResponseFormat: &core.ResponseFormat{
			Type:       "json_schema",
			JSONSchema: &core.JSONSchemaFormat{Name: "plan", Schema: schema},
		},
	})
	if err != nil {
		t.Fatalf("Request: %v", err)
	}

	sent := string(body)
	if !strings.Contains(sent, "parametersJsonSchema") {
		t.Fatalf("expected tool parameters sent as parametersJsonSchema: %s", sent)
	}
	if !strings.Contains(sent, "responseJsonSchema") {
		t.Fatalf("expected response schema sent as responseJsonSchema: %s", sent)
	}
	for _, keyword := range []string{`"$ref"`, `"$defs"`, `"anyOf"`, `"format"`, `"minimum"`, `"maxLength"`} {
		if !strings.Contains(sent, keyword) {
			t.Errorf("expected %s to reach the wire, got %s", keyword, sent)
		}
	}
}

// TestGeminiBlockedPromptIsAnError pins that a safety-blocked prompt — which
// the API reports with HTTP 200 and no candidate — surfaces as an error rather
// than an empty, successful-looking response.
func TestGeminiBlockedPromptIsAnError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasSuffix(r.URL.Path, ":streamGenerateContent"):
			_, _ = io.WriteString(w, strings.Join([]string{
				`data:{"promptFeedback":{"blockReason":"SAFETY"}}`,
				``,
			}, "\n"))
		default:
			_, _ = io.WriteString(w, `{"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"blocked"}}`)
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

	if _, err := model.Request(context.Background(), req); !errors.Is(err, ErrPromptBlocked) {
		t.Fatalf("expected ErrPromptBlocked from Request, got %v", err)
	}

	stream, err := model.RequestStream(context.Background(), req)
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}
	if err := stream.Wait(); !errors.Is(err, ErrPromptBlocked) {
		t.Fatalf("expected ErrPromptBlocked from the stream, got %v", err)
	}
}

func TestGeminiStreamErrorAndEmptyChunks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data:{"candidates":[]}`,
			``,
			`data:{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL"}]}`,
			``,
		}, "\n"))
	}))
	defer server.Close()

	t.Setenv("GOOGLE_GEMINI_BASE_URL", server.URL)

	model, err := New("gemini-2.5-pro", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "gemini-2.5-pro",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) != 1 || events[0].Type != core.StreamEventDone {
		t.Fatalf("expected a single done event, got %#v", events)
	}
	if events[0].FinishReason != core.FinishReasonError {
		t.Fatalf("expected a malformed call reported as an error finish, got %q", events[0].FinishReason)
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

func TestGeminiCandidateHelpersCoverRemainingBranches(t *testing.T) {
	msg := convertCandidateMessage(&genai.Candidate{})
	if msg.Role != core.RoleAssistant || len(msg.Content) != 0 {
		t.Fatalf("expected empty assistant message for nil content, got %#v", msg)
	}

	if got := encodeSignature(nil); got != "" {
		t.Fatalf("expected an empty signature for no bytes, got %q", got)
	}
}
