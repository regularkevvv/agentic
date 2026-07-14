package tool

import (
	"context"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestMustNewToolFromStruct(t *testing.T) {
	type Input struct {
		Query string `json:"query"`
	}

	tool := MustNewToolFromStruct("search", "Search for things", Input{})
	if tool.Function.Name != "search" {
		t.Errorf("expected name %q, got %q", "search", tool.Function.Name)
	}
}

func TestMustNewToolFromStructPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty name")
		}
	}()
	type Input struct{}
	MustNewToolFromStruct("", "desc", Input{})
}

func TestMustToolPlainPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty name")
		}
	}()
	type Input struct{}
	MustToolPlain("", "desc", func(in Input) (string, error) { return "", nil })
}

func TestMustToolWithDepsPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty name")
		}
	}()
	type Input struct{}
	MustToolWithDeps[Input, string, struct{}]("", "desc", func(ctx core.RunContext[struct{}], in Input) (string, error) { return "", nil })
}

func TestToolPlainError(t *testing.T) {
	type Input struct{}
	_, _, err := ToolPlain("", "desc", func(in Input) (string, error) { return "", nil })
	if err == nil {
		t.Error("expected error for empty name")
	}
}

func TestToolWithDepsError(t *testing.T) {
	type Input struct{}
	_, _, err := ToolWithDeps[Input, string, struct{}]("", "desc", func(ctx core.RunContext[struct{}], in Input) (string, error) { return "", nil })
	if err == nil {
		t.Error("expected error for empty name")
	}
}

func TestPlainToolHandlerMarshalError(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}

	_, handler := MustToolPlain("test", "test", func(in In) (string, error) { return "ok", nil })

	// Valid input should work
	result, err := handler.Execute(context.TODO(), map[string]interface{}{"x": float64(5)}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "ok" {
		t.Errorf("expected %q, got %q", "ok", result)
	}

	// Invalid input type should fail unmarshal
	_, err = handler.Execute(context.TODO(), map[string]interface{}{"x": "not_a_number"}, nil)
	if err == nil {
		t.Error("expected error for invalid input type")
	}

	_, err = handler.Execute(context.TODO(), map[string]interface{}{"x": func() {}}, nil)
	if err == nil || !strings.Contains(err.Error(), "marshal input") {
		t.Fatalf("expected marshal input error, got %v", err)
	}
}

func TestDepsToolHandlerMarshalError(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}

	_, handler := MustToolWithDeps("test", "test", func(ctx core.RunContext[struct{}], in In) (string, error) {
		return "ok", nil
	})

	// Invalid input type should fail unmarshal
	_, err := handler.Execute(context.TODO(), map[string]interface{}{"x": "not_a_number"}, nil)
	if err == nil {
		t.Error("expected error for invalid input type")
	}

	_, err = handler.Execute(context.TODO(), map[string]interface{}{"x": func() {}}, nil)
	if err == nil || !strings.Contains(err.Error(), "marshal input") {
		t.Fatalf("expected marshal input error, got %v", err)
	}
}

func TestRegisterToolsetError(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}
	type Out struct {
		Y int `json:"y"`
	}

	t1, h1 := MustToolPlain("dup", "dup", func(in In) (Out, error) { return Out{}, nil })

	ts := NewToolset().Add(t1, h1)

	reg := NewRegistry()
	_ = RegisterToolset(reg, ts)

	// Second registration should fail
	err := RegisterToolset(reg, ts)
	if err == nil {
		t.Error("expected error for duplicate registration via toolset")
	}
}

func TestFormatToolResultJSON(t *testing.T) {
	// Test with a complex type that marshals to JSON
	type result struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}
	got := core.FormatToolResult(result{Name: "test", Value: 42})
	if got != `{"name":"test","value":42}` {
		t.Errorf("unexpected result: %q", got)
	}
}

func TestNewToolFromStructAdditionalCases(t *testing.T) {
	t.Run("empty description", func(t *testing.T) {
		_, err := NewToolFromStruct("search", "", struct{}{})
		if err == nil || err.Error() != "tool description cannot be empty" {
			t.Fatalf("expected empty description error, got %v", err)
		}
	})

	t.Run("primitive input still generates schema", func(t *testing.T) {
		tool, err := NewToolFromStruct("count", "Count a value", 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if tool.Function.Parameters["type"] == nil {
			t.Fatalf("expected schema type to be present, got %#v", tool.Function.Parameters)
		}
	})
}
