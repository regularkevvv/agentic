package anthropic

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestToStringSlice(t *testing.T) {
	tests := []struct {
		name  string
		input interface{}
		want  []string
	}{
		{"string slice", []string{"a", "b"}, []string{"a", "b"}},
		{"any slice", []interface{}{"a", "b"}, []string{"a", "b"}},
		{"any slice with non-strings", []interface{}{"a", 7, nil}, []string{"a"}},
		{"any slice of only non-strings", []interface{}{1, 2}, nil},
		{"empty any slice", []interface{}{}, nil},
		{"wrong type", "required", nil},
		{"nil", nil, nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := toStringSlice(tt.input); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("toStringSlice(%#v) = %#v, want %#v", tt.input, got, tt.want)
			}
		})
	}
}

func TestConvertToolSchemaNilAndNilProperties(t *testing.T) {
	// No parameters at all still yields a valid empty object schema.
	schema := convertToolSchema(nil)
	props, ok := schema.Properties.(map[string]interface{})
	if !ok || len(props) != 0 {
		t.Errorf("expected empty properties, got %#v", schema.Properties)
	}
	if schema.ExtraFields != nil {
		t.Errorf("expected no extra fields, got %#v", schema.ExtraFields)
	}

	// An explicit null "properties" must not blank out the emitted object.
	schema = convertToolSchema(map[string]interface{}{"properties": nil})
	if props, ok := schema.Properties.(map[string]interface{}); !ok || len(props) != 0 {
		t.Errorf("expected empty properties, got %#v", schema.Properties)
	}
}

func TestApplyToolCacheControlSkipsNonCustomTools(t *testing.T) {
	// A union entry that carries no OfTool (a server-side tool) has nowhere to
	// hang a breakpoint and must be left alone rather than panicking.
	tools := []anthropic.ToolUnionParam{{}}
	applyToolCacheControl(tools, "5m")
	if tools[0].OfTool != nil {
		t.Errorf("expected the union to be untouched, got %#v", tools[0])
	}
}

func TestConvertMessageSkipsEmptyThinkingParts(t *testing.T) {
	param := convertMessage(core.Message{
		Role: core.RoleAssistant,
		Content: []core.Part{
			{Type: core.ContentThinking, Thinking: nil},
			// Unsigned and textless: nothing worth degrading to text.
			{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{ProviderName: providerName}},
		},
	})
	if len(param.Content) != 0 {
		t.Errorf("expected no content blocks, got %#v", param.Content)
	}
}

func TestUsesAdaptiveThinking(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"claude-opus-4-7", true},
		{"claude-opus-4-8-20260101", true},
		{"claude-sonnet-5", true},
		{"claude-fable-5", true},
		{"claude-mythos-5", true},
		{"claude-sonnet-4-20250514", false},
		{"claude-opus-4-6", false},
		{"", false},
	}

	for _, tt := range tests {
		if got := usesAdaptiveThinking(tt.model); got != tt.want {
			t.Errorf("usesAdaptiveThinking(%q) = %v, want %v", tt.model, got, tt.want)
		}
	}
}

func TestRequestStreamSurfacesTransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// A malformed event terminates the stream with an error.
		_, _ = io.WriteString(w, "event: message_start\ndata: {not json}\n\n")
	}))
	defer server.Close()

	model, err := New("claude-sonnet", WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stream, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "claude-sonnet",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}
	if len(events) != 1 {
		t.Fatalf("expected a single error event, got %#v", events)
	}
	if events[0].Type != core.StreamEventError || events[0].Error == nil {
		t.Fatalf("expected a stream error, got %#v", events[0])
	}
}

func TestRequestStreamValidationError(t *testing.T) {
	model, _ := New("claude-sonnet", WithAPIKey("test-key"))
	if _, err := model.RequestStream(context.Background(), &core.ChatRequest{}); err == nil {
		t.Error("expected validation error")
	}
}
