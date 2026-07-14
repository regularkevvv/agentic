package tool

import (
	"context"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestToolPlain(t *testing.T) {
	type Input struct {
		Name string `json:"name" description:"Person name"`
	}
	type Output struct {
		Greeting string `json:"greeting"`
	}

	tool, handler, err := ToolPlain("greet", "Greet a person", func(input Input) (Output, error) {
		return Output{Greeting: "Hello " + input.Name}, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "greet" {
		t.Errorf("expected name %q, got %q", "greet", tool.Function.Name)
	}
	if tool.Function.Description != "Greet a person" {
		t.Errorf("expected description %q, got %q", "Greet a person", tool.Function.Description)
	}
	if tool.Type != core.ToolTypeFunction {
		t.Errorf("expected type %q, got %q", core.ToolTypeFunction, tool.Type)
	}
	if handler.Name() != "greet" {
		t.Errorf("expected handler name %q, got %q", "greet", handler.Name())
	}

	// Execute the handler
	result, err := handler.Execute(context.Background(), map[string]interface{}{"name": "World"}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	output, ok := result.(Output)
	if !ok {
		t.Fatalf("expected Output, got %T", result)
	}
	if output.Greeting != "Hello World" {
		t.Errorf("expected %q, got %q", "Hello World", output.Greeting)
	}
}

func TestToolWithDeps(t *testing.T) {
	type MyDeps struct {
		Prefix string
	}
	type Input struct {
		Text string `json:"text"`
	}
	type Output struct {
		Result string `json:"result"`
	}

	tool, handler, err := ToolWithDeps("prefix", "Add prefix",
		func(ctx core.RunContext[*MyDeps], input Input) (Output, error) {
			return Output{Result: ctx.Deps.Prefix + input.Text}, nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "prefix" {
		t.Errorf("expected name %q, got %q", "prefix", tool.Function.Name)
	}

	deps := &MyDeps{Prefix: ">> "}
	result, err := handler.Execute(context.Background(), map[string]interface{}{"text": "hello"}, core.NewDependencyEnvelope(deps))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	output, ok := result.(Output)
	if !ok {
		t.Fatalf("expected Output, got %T", result)
	}
	if output.Result != ">> hello" {
		t.Errorf("expected %q, got %q", ">> hello", output.Result)
	}
}

func TestNewToolFromStruct(t *testing.T) {
	type Input struct {
		Query string `json:"query" description:"Search query"`
		Limit int    `json:"limit"`
	}

	tool, err := NewToolFromStruct("search", "Search for items", Input{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "search" {
		t.Errorf("expected name %q, got %q", "search", tool.Function.Name)
	}
	params := tool.Function.Parameters
	if params["type"] != "object" {
		t.Errorf("expected type 'object', got %v", params["type"])
	}
	props, ok := params["properties"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected properties map, got %T", params["properties"])
	}
	if _, ok := props["query"]; !ok {
		t.Error("expected 'query' property")
	}
	if _, ok := props["limit"]; !ok {
		t.Error("expected 'limit' property")
	}
}

func TestNewToolFromStructValidation(t *testing.T) {
	type Input struct{}

	_, err := NewToolFromStruct("", "desc", Input{})
	if err == nil {
		t.Error("expected error for empty name")
	}

	_, err = NewToolFromStruct("name", "", Input{})
	if err == nil {
		t.Error("expected error for empty description")
	}
}

func TestFormatToolResult(t *testing.T) {
	if got := core.FormatToolResult(nil); got != "" {
		t.Errorf("expected empty, got %q", got)
	}
	if got := core.FormatToolResult("hello"); got != "hello" {
		t.Errorf("expected 'hello', got %q", got)
	}
	if got := core.FormatToolResult(map[string]int{"x": 1}); got != `{"x":1}` {
		t.Errorf("expected JSON, got %q", got)
	}
}

func TestHandlersExposeToolConfig(t *testing.T) {
	type input struct {
		Value int `json:"value"`
	}

	_, plainHandler, err := ToolPlain("plain", "plain tool", func(input input) (string, error) {
		return "ok", nil
	}, WithToolMaxRetries(2))
	if err != nil {
		t.Fatalf("ToolPlain: %v", err)
	}

	_, depsHandler, err := ToolWithDeps[input, string, int]("deps", "deps tool", func(ctx core.RunContext[int], input input) (string, error) {
		return "ok", nil
	}, WithToolMaxRetries(3))
	if err != nil {
		t.Fatalf("ToolWithDeps: %v", err)
	}

	if cfg := plainHandler.(*PlainToolHandler[input, string]).ToolConfig(); cfg == nil || cfg.MaxRetries == nil || *cfg.MaxRetries != 2 {
		t.Fatalf("unexpected plain tool config %#v", cfg)
	}
	if cfg := depsHandler.(*DepsToolHandler[input, string, int]).ToolConfig(); cfg == nil || cfg.MaxRetries == nil || *cfg.MaxRetries != 3 {
		t.Fatalf("unexpected deps tool config %#v", cfg)
	}
}
