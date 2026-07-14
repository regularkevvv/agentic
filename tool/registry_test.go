package tool

import (
	"context"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestRegistryExecuteBatch(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}
	type Out struct {
		Y int `json:"y"`
	}

	tool1, handler1 := MustToolPlain("double", "Double", func(in In) (Out, error) {
		return Out{Y: in.X * 2}, nil
	})
	tool2, handler2 := MustToolPlain("triple", "Triple", func(in In) (Out, error) {
		return Out{Y: in.X * 3}, nil
	})

	reg := NewRegistry()
	_ = reg.Register(tool1, handler1)
	_ = reg.Register(tool2, handler2)

	results, err := reg.ExecuteBatch(context.Background(), []core.ToolUse{
		{ID: "c1", Name: "double", Input: map[string]interface{}{"x": float64(5)}},
		{ID: "c2", Name: "triple", Input: map[string]interface{}{"x": float64(3)}},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	out1, ok := results[0].Content.(Out)
	if !ok {
		t.Fatalf("expected Out, got %T", results[0].Content)
	}
	if out1.Y != 10 {
		t.Errorf("expected 10, got %d", out1.Y)
	}

	out2, ok := results[1].Content.(Out)
	if !ok {
		t.Fatalf("expected Out, got %T", results[1].Content)
	}
	if out2.Y != 9 {
		t.Errorf("expected 9, got %d", out2.Y)
	}
}

func TestRegistryBasic(t *testing.T) {
	reg := NewRegistry()

	type Input struct {
		X int `json:"x"`
	}
	type Output struct {
		Y int `json:"y"`
	}

	tool, handler := MustToolPlain("double", "Double a number", func(input Input) (Output, error) {
		return Output{Y: input.X * 2}, nil
	})

	if err := reg.Register(tool, handler); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !reg.Has("double") {
		t.Error("expected registry to have 'double'")
	}
	if reg.Count() != 1 {
		t.Errorf("expected count 1, got %d", reg.Count())
	}

	result, err := reg.Execute(context.Background(), core.ToolUse{
		ID: "call_1", Name: "double", Input: map[string]interface{}{"x": float64(5)},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("unexpected tool error: %v", result.Error)
	}

	output, ok := result.Content.(Output)
	if !ok {
		t.Fatalf("expected Output, got %T", result.Content)
	}
	if output.Y != 10 {
		t.Errorf("expected 10, got %d", output.Y)
	}
}

func TestRegistryDuplicate(t *testing.T) {
	reg := NewRegistry()

	type Input struct{}
	type Output struct{}

	tool, handler := MustToolPlain("test", "test tool", func(input Input) (Output, error) {
		return Output{}, nil
	})

	if err := reg.Register(tool, handler); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := reg.Register(tool, handler); err == nil {
		t.Error("expected error for duplicate registration")
	}
}

func TestRegistryUnknownTool(t *testing.T) {
	reg := NewRegistry()

	result, err := reg.Execute(context.Background(), core.ToolUse{
		ID: "call_1", Name: "unknown", Input: map[string]interface{}{},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected error for unknown tool")
	}
}

func TestRegistryExecuteBatchUnknownTool(t *testing.T) {
	reg := NewRegistry()

	// Execute batch with unknown tool — should still return results (not fatal error)
	results, err := reg.ExecuteBatch(context.Background(), []core.ToolUse{
		{ID: "c1", Name: "unknown", Input: map[string]interface{}{}},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if !results[0].IsError {
		t.Error("expected error result for unknown tool")
	}
}

func TestRegistryGet(t *testing.T) {
	reg := NewRegistry()

	_, ok := reg.Get("nonexistent")
	if ok {
		t.Error("expected Get to return false for nonexistent tool")
	}

	type In struct{}
	type Out struct{}
	tool, handler := MustToolPlain("test", "test", func(in In) (Out, error) { return Out{}, nil })
	_ = reg.Register(tool, handler)

	h, ok := reg.Get("test")
	if !ok {
		t.Error("expected Get to return true for registered tool")
	}
	if h.Name() != "test" {
		t.Errorf("expected handler name %q, got %q", "test", h.Name())
	}
}

func TestRegistryHandlerNameMismatch(t *testing.T) {
	reg := NewRegistry()

	type In struct{}
	type Out struct{}
	_, handler := MustToolPlain("actual_name", "test", func(in In) (Out, error) { return Out{}, nil })

	// Create a tool with a different name than the handler
	tool := core.Tool{
		Type: core.ToolTypeFunction,
		Function: core.Function{
			Name:        "different_name",
			Description: "test",
			Parameters:  map[string]interface{}{"type": "object"},
		},
	}

	err := reg.Register(tool, handler)
	if err == nil {
		t.Error("expected error for handler name mismatch")
	}
}

func TestRegistryExecuteToolError(t *testing.T) {
	reg := NewRegistry()

	type In struct {
		X int `json:"x"`
	}

	tool, handler := MustToolPlain("fail", "always fails", func(in In) (string, error) {
		return "", errors.New("bad input")
	})
	_ = reg.Register(tool, handler)

	result, err := reg.Execute(context.Background(), core.ToolUse{
		ID: "c1", Name: "fail", Input: map[string]interface{}{"x": float64(1)},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected error result")
	}
	if result.Error == nil {
		t.Error("expected non-nil Error field")
	}
}

func TestRegistryTools(t *testing.T) {
	type input struct {
		Value int `json:"value"`
	}

	plainTool, plainHandler, err := ToolPlain("plain", "plain tool", func(input input) (string, error) {
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("ToolPlain: %v", err)
	}

	depsTool, depsHandler, err := ToolWithDeps[input, string, int]("deps", "deps tool", func(ctx core.RunContext[int], input input) (string, error) {
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("ToolWithDeps: %v", err)
	}

	reg := NewRegistry()
	if err := reg.Register(plainTool, plainHandler); err != nil {
		t.Fatalf("Register plain tool: %v", err)
	}
	if err := reg.Register(depsTool, depsHandler); err != nil {
		t.Fatalf("Register deps tool: %v", err)
	}

	if !reg.Has("plain") || !reg.Has("deps") || reg.Count() != 2 {
		t.Fatalf("unexpected registry state count=%d", reg.Count())
	}
	if got := reg.Tools(); len(got) != 2 {
		t.Fatalf("expected 2 registered tools, got %#v", got)
	}
}
