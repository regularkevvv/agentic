package tool

import (
	"context"
	"testing"
)

func TestFuncToolset(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}
	type Out struct {
		Y int `json:"y"`
	}

	t1, h1 := MustToolPlain("a", "tool a", func(in In) (Out, error) { return Out{Y: 1}, nil })
	t2, h2 := MustToolPlain("b", "tool b", func(in In) (Out, error) { return Out{Y: 2}, nil })

	ts := NewToolset().Add(t1, h1).Add(t2, h2)
	tools, handlers := ts.ToolsAndHandlers()

	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
	if len(handlers) != 2 {
		t.Fatalf("expected 2 handlers, got %d", len(handlers))
	}
}

func TestCombineToolsets(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}
	type Out struct {
		Y int `json:"y"`
	}

	t1, h1 := MustToolPlain("a", "tool a", func(in In) (Out, error) { return Out{}, nil })
	t2, h2 := MustToolPlain("b", "tool b", func(in In) (Out, error) { return Out{}, nil })

	ts1 := NewToolset().Add(t1, h1)
	ts2 := NewToolset().Add(t2, h2)

	combined := CombineToolsets(ts1, ts2)
	tools, _ := combined.ToolsAndHandlers()

	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
}

func TestFilterToolset(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}
	type Out struct {
		Y int `json:"y"`
	}

	t1, h1 := MustToolPlain("math_add", "add", func(in In) (Out, error) { return Out{}, nil })
	t2, h2 := MustToolPlain("math_sub", "sub", func(in In) (Out, error) { return Out{}, nil })
	t3, h3 := MustToolPlain("io_read", "read", func(in In) (Out, error) { return Out{}, nil })

	ts := NewToolset().Add(t1, h1).Add(t2, h2).Add(t3, h3)

	filtered := FilterToolset(ts, func(name string) bool {
		return len(name) >= 4 && name[:4] == "math"
	})
	tools, _ := filtered.ToolsAndHandlers()

	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
	for _, tool := range tools {
		if len(tool.Function.Name) < 4 || tool.Function.Name[:4] != "math" {
			t.Errorf("unexpected tool: %s", tool.Function.Name)
		}
	}
}

func TestPrefixToolset(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}
	type Out struct {
		Y int `json:"y"`
	}

	t1, h1 := MustToolPlain("add", "add numbers", func(in In) (Out, error) { return Out{}, nil })

	ts := NewToolset().Add(t1, h1)
	prefixed := PrefixToolset(ts, "math")

	tools, handlers := prefixed.ToolsAndHandlers()
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].Function.Name != "math__add" {
		t.Errorf("expected %q, got %q", "math__add", tools[0].Function.Name)
	}
	if handlers[0].Name() != "math__add" {
		t.Errorf("expected handler name %q, got %q", "math__add", handlers[0].Name())
	}
}

func TestPrefixToolsetExecutesWithRenamedHandler(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}
	type Out struct {
		Y int `json:"y"`
	}

	tool, handler := MustToolPlain("original", "test", func(in In) (Out, error) {
		return Out{Y: in.X * 2}, nil
	})

	prefixed := PrefixToolset(NewToolset().Add(tool, handler), "prefixed")
	tools, handlers := prefixed.ToolsAndHandlers()
	if len(tools) != 1 || len(handlers) != 1 {
		t.Fatalf("expected 1 tool and handler, got %d and %d", len(tools), len(handlers))
	}
	if tools[0].Function.Name != "prefixed__original" {
		t.Fatalf("expected prefixed tool name, got %q", tools[0].Function.Name)
	}
	if handlers[0].Name() != "prefixed__original" {
		t.Fatalf("expected handler name %q, got %q", "prefixed__original", handlers[0].Name())
	}

	result, err := handlers[0].Execute(context.Background(), map[string]interface{}{"x": float64(5)}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	out, ok := result.(Out)
	if !ok {
		t.Fatalf("expected Out, got %T", result)
	}
	if out.Y != 10 {
		t.Errorf("expected 10, got %d", out.Y)
	}
}

func TestRegisterToolset(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}
	type Out struct {
		Y int `json:"y"`
	}

	t1, h1 := MustToolPlain("foo", "foo tool", func(in In) (Out, error) { return Out{}, nil })
	ts := NewToolset().Add(t1, h1)

	reg := NewRegistry()
	if err := RegisterToolset(reg, ts); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !reg.Has("foo") {
		t.Error("expected registry to have 'foo'")
	}
}
