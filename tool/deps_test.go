package tool

import (
	"context"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestDepsToolHandlerInvalidDeps(t *testing.T) {
	type MyDeps struct {
		Value string
	}
	type In struct {
		X string `json:"x"`
	}

	_, handler, err := ToolWithDeps("test", "test",
		func(ctx core.RunContext[MyDeps], in In) (string, error) {
			return ctx.Deps.Value + in.X, nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Pass wrong deps type
	wrongDeps := "not a dependency envelope"
	_, execErr := handler.Execute(context.Background(), map[string]interface{}{"x": "hello"}, wrongDeps)
	if execErr == nil {
		t.Error("expected error for invalid deps type")
	}
}

func TestDepsToolHandlerExactDeps(t *testing.T) {
	type MyDeps struct {
		Value string
	}
	type In struct {
		X string `json:"x"`
	}

	_, handler, err := ToolWithDeps("test", "test",
		func(ctx core.RunContext[MyDeps], in In) (string, error) {
			return ctx.Deps.Value + in.X, nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result, execErr := handler.Execute(context.Background(), map[string]interface{}{"x": "hello"}, core.NewDependencyEnvelope(MyDeps{Value: "hi "}))
	if execErr != nil {
		t.Fatalf("unexpected error: %v", execErr)
	}
	if result != "hi hello" {
		t.Errorf("expected %q, got %q", "hi hello", result)
	}
}
