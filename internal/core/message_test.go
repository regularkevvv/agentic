package core

import "testing"

func TestNewTextMessage(t *testing.T) {
	msg := NewTextMessage(RoleUser, "hello")
	if msg.Role != RoleUser {
		t.Errorf("expected role %q, got %q", RoleUser, msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("expected 1 part, got %d", len(msg.Content))
	}
	if msg.Content[0].Type != ContentText {
		t.Errorf("expected type %q, got %q", ContentText, msg.Content[0].Type)
	}
	if msg.Content[0].Text != "hello" {
		t.Errorf("expected text %q, got %q", "hello", msg.Content[0].Text)
	}
}

func TestNewToolUseMessage(t *testing.T) {
	tu := ToolUse{ID: "1", Name: "test", Input: map[string]interface{}{"key": "val"}}
	msg := NewToolUseMessage(tu)
	if msg.Role != RoleAssistant {
		t.Errorf("expected role %q, got %q", RoleAssistant, msg.Role)
	}
	uses := msg.GetToolUses()
	if len(uses) != 1 {
		t.Fatalf("expected 1 tool use, got %d", len(uses))
	}
	if uses[0].Name != "test" {
		t.Errorf("expected name %q, got %q", "test", uses[0].Name)
	}
}

func TestNewToolResultMessage(t *testing.T) {
	msg := NewToolResultMessage("call_1", "result text", false)
	if msg.Role != RoleTool {
		t.Errorf("expected role %q, got %q", RoleTool, msg.Role)
	}
	results := msg.GetToolResults()
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Content != "result text" {
		t.Errorf("expected content %q, got %q", "result text", results[0].Content)
	}
	if results[0].IsError {
		t.Error("expected IsError to be false")
	}
}

func TestGetTextContent(t *testing.T) {
	msg := Message{
		Role: RoleAssistant,
		Content: []Part{
			{Type: ContentText, Text: "Hello "},
			{Type: ContentText, Text: "world"},
		},
	}
	if got := msg.GetTextContent(); got != "Hello world" {
		t.Errorf("expected %q, got %q", "Hello world", got)
	}
}

func TestGetTextContentEmpty(t *testing.T) {
	msg := Message{Role: RoleAssistant, Content: []Part{}}
	if got := msg.GetTextContent(); got != "" {
		t.Errorf("expected empty string, got %q", got)
	}
}

func TestThinkingHelpers(t *testing.T) {
	redacted := &ThinkingBlock{ID: "redacted_thinking"}
	if !redacted.IsRedacted() {
		t.Fatal("expected redacted thinking block to be detected")
	}

	visible := &ThinkingBlock{ID: "visible"}
	if visible.IsRedacted() {
		t.Fatal("expected non-redacted thinking block")
	}

	msg := Message{
		Role: RoleAssistant,
		Content: []Part{
			{Type: ContentThinking, Thinking: &ThinkingBlock{Text: "first "}},
			{Type: ContentText, Text: "ignored"},
			{Type: ContentThinking, Thinking: &ThinkingBlock{Text: "second"}},
		},
	}

	if got := msg.GetThinkingContent(); got != "first second" {
		t.Fatalf("expected combined thinking content, got %q", got)
	}
}
