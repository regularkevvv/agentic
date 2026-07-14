package agentic

import (
	"context"
	"strings"
	"testing"
)

type outputCoverageValue struct {
	Value string `json:"value"`
}

func TestTextOutputSpec(t *testing.T) {
	spec := &TextOutputSpec{}

	tools := spec.Tools()
	if tools != nil {
		t.Errorf("expected nil tools, got %v", tools)
	}

	msg := NewTextMessage(RoleAssistant, "hello world")
	parsed, err := spec.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if parsed != "hello world" {
		t.Errorf("expected %q, got %q", "hello world", parsed)
	}
}

func TestToolOutputSpecTools(t *testing.T) {
	type Result struct {
		Value int `json:"value"`
	}

	spec := NewToolOutput[Result]("test output")
	tools := spec.Tools()

	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].Function.Name != "__output__" {
		t.Errorf("expected name %q, got %q", "__output__", tools[0].Function.Name)
	}
}

func TestToolOutputSpecParseNoOutput(t *testing.T) {
	type Result struct {
		Value int `json:"value"`
	}

	spec := NewToolOutput[Result]("test")

	// Empty message — no tool call, no text
	msg := Message{Role: RoleAssistant, Content: []Part{}}
	_, err := spec.Parse(msg)
	if err == nil {
		t.Error("expected error when no output found")
	}
}

func TestToolOutputSpecParseInvalidJSON(t *testing.T) {
	type Result struct {
		Value int `json:"value"`
	}

	spec := NewToolOutput[Result]("test")

	// Text that is not valid JSON
	msg := NewTextMessage(RoleAssistant, "not json at all")
	_, err := spec.Parse(msg)
	if err == nil {
		t.Error("expected error for invalid JSON text")
	}
}

func TestToolOutputSpecParseWrongToolName(t *testing.T) {
	type Result struct {
		Value int `json:"value"`
	}

	spec := NewToolOutput[Result]("test")

	// Tool call with wrong name — should fall through to text parsing
	msg := NewToolUseMessage(ToolUse{
		ID:    "1",
		Name:  "wrong_name",
		Input: map[string]interface{}{"value": float64(42)},
	})
	_, err := spec.Parse(msg)
	if err == nil {
		t.Error("expected error when tool name doesn't match and no text content")
	}
}

func TestToolOutputSpecToolsPanic(t *testing.T) {
	// ToolOutputSpec.Tools() should work with a valid struct
	type Good struct {
		X int `json:"x"`
	}
	spec := NewToolOutput[Good]("test")
	tools := spec.Tools()
	if len(tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(tools))
	}
}

func TestNoopToolHandler(t *testing.T) {
	h := &noopToolHandler{name: "test_output"}
	if h.Name() != "test_output" {
		t.Errorf("expected name %q, got %q", "test_output", h.Name())
	}

	input := map[string]interface{}{"key": "value"}
	result, err := h.Execute(context.TODO(), input, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resultMap, ok := result.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map, got %T", result)
	}
	if resultMap["key"] != "value" {
		t.Errorf("expected key=value, got %v", resultMap["key"])
	}
}

func TestToolOutputSpecErrors(t *testing.T) {
	t.Run("panics when output tool schema is invalid", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic for empty output tool description")
			}
		}()
		NewToolOutput[outputCoverageValue]("").Tools()
	})

	t.Run("returns no output found", func(t *testing.T) {
		spec := NewToolOutput[outputCoverageValue]("desc")
		_, err := spec.Parse(Message{Role: RoleAssistant})
		if err == nil || !strings.Contains(err.Error(), "no output found") {
			t.Fatalf("expected no output error, got %v", err)
		}
	})

	t.Run("returns invalid json error", func(t *testing.T) {
		spec := NewToolOutput[outputCoverageValue]("desc")
		_, err := spec.Parse(NewTextMessage(RoleAssistant, "not json"))
		if err == nil || !strings.Contains(err.Error(), "not valid JSON") {
			t.Fatalf("expected invalid JSON error, got %v", err)
		}
	})

	t.Run("returns marshal error for non-json tool input", func(t *testing.T) {
		spec := NewToolOutput[outputCoverageValue]("desc")
		msg := NewToolUseMessage(ToolUse{
			ID:   "call_1",
			Name: "__output__",
			Input: map[string]interface{}{
				"value": func() {},
			},
		})

		_, err := spec.Parse(msg)
		if err == nil || !strings.Contains(err.Error(), "marshal output tool input") {
			t.Fatalf("expected marshal error, got %v", err)
		}
	})

	t.Run("returns unmarshal error for wrong tool input shape", func(t *testing.T) {
		type numericOutput struct {
			Value int `json:"value"`
		}

		spec := NewToolOutput[numericOutput]("desc")
		msg := NewToolUseMessage(ToolUse{
			ID:   "call_1",
			Name: "__output__",
			Input: map[string]interface{}{
				"value": "wrong",
			},
		})

		_, err := spec.Parse(msg)
		if err == nil || !strings.Contains(err.Error(), "unmarshal to") {
			t.Fatalf("expected unmarshal error, got %v", err)
		}
	})
}
