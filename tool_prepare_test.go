package agentic_test

import (
	"context"
	"fmt"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestToolPrepare_FilterTools(t *testing.T) {
	type InputA struct {
		X string `json:"x"`
	}
	type InputB struct {
		Y string `json:"y"`
	}

	toolA, handlerA := agentic.MustToolPlain("tool_a", "Tool A", func(input InputA) (string, error) {
		return "a", nil
	})
	toolB, handlerB := agentic.MustToolPlain("tool_b", "Tool B", func(input InputB) (string, error) {
		return "b", nil
	})

	model := test.NewTestModel(test.ModelResponse{Text: "ok"})

	// Filter out tool_b
	agent := agentic.NewAgent("test", model,
		agentic.WithToolPrepare(func(ctx context.Context, tools []agentic.Tool) ([]agentic.Tool, error) {
			var filtered []agentic.Tool
			for _, tool := range tools {
				if tool.Function.Name != "tool_b" {
					filtered = append(filtered, tool)
				}
			}
			return filtered, nil
		}),
	).AddTool(toolA, handlerA).AddTool(toolB, handlerB)

	_, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify only tool_a was sent to model
	calls := model.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if len(calls[0].Tools) != 1 {
		t.Fatalf("expected 1 tool in request, got %d", len(calls[0].Tools))
	}
	if calls[0].Tools[0].Function.Name != "tool_a" {
		t.Errorf("expected tool_a, got %q", calls[0].Tools[0].Function.Name)
	}
}

func TestToolPrepare_EmptyToolList(t *testing.T) {
	type Input struct {
		V string `json:"v"`
	}
	tool, handler := agentic.MustToolPlain("my_tool", "My tool", func(input Input) (string, error) {
		return "result", nil
	})

	model := test.NewTestModel(test.ModelResponse{Text: "no tools available"})

	// Return empty tool list
	agent := agentic.NewAgent("test", model,
		agentic.WithToolPrepare(func(ctx context.Context, tools []agentic.Tool) ([]agentic.Tool, error) {
			return nil, nil
		}),
	).AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "no tools available" {
		t.Errorf("expected 'no tools available', got %q", result.Output)
	}

	// No tools should be in the request
	calls := model.Calls()
	if len(calls[0].Tools) != 0 {
		t.Errorf("expected 0 tools in request, got %d", len(calls[0].Tools))
	}
}

func TestToolPrepare_ErrorPropagation(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "ok"})

	type Input struct {
		V string `json:"v"`
	}
	tool, handler := agentic.MustToolPlain("t", "t", func(input Input) (string, error) {
		return "", nil
	})

	agent := agentic.NewAgent("test", model,
		agentic.WithToolPrepare(func(ctx context.Context, tools []agentic.Tool) ([]agentic.Tool, error) {
			return nil, fmt.Errorf("permission denied")
		}),
	).AddTool(tool, handler)

	_, err := agent.Run(context.Background(), "go")
	if err == nil {
		t.Fatal("expected error from tool prepare")
	}
	if err.Error() != "tool prepare: permission denied" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestToolPrepare_ModifyDescription(t *testing.T) {
	type Input struct {
		V string `json:"v"`
	}
	tool, handler := agentic.MustToolPlain("my_tool", "Original desc", func(input Input) (string, error) {
		return "ok", nil
	})

	model := test.NewTestModel(test.ModelResponse{Text: "ok"})

	agent := agentic.NewAgent("test", model,
		agentic.WithToolPrepare(func(ctx context.Context, tools []agentic.Tool) ([]agentic.Tool, error) {
			modified := make([]agentic.Tool, len(tools))
			for i, t := range tools {
				t.Function.Description = "Modified: " + t.Function.Description
				modified[i] = t
			}
			return modified, nil
		}),
	).AddTool(tool, handler)

	_, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	calls := model.Calls()
	desc := calls[0].Tools[0].Function.Description
	if desc != "Modified: Original desc" {
		t.Errorf("expected modified description, got %q", desc)
	}
}

func TestToolPrepare_DepsAware(t *testing.T) {
	type MyDeps struct {
		IsAdmin bool
	}
	type Input struct {
		V string `json:"v"`
	}

	tool, handler := agentic.MustToolPlain("admin_tool", "Admin only", func(input Input) (string, error) {
		return "admin result", nil
	})

	model := test.NewTestModel(test.ModelResponse{Text: "ok"})

	agent := agentic.NewAgentWithDeps[*MyDeps]("test", model).SetToolPrepare(
		func(ctx agentic.RunContext[*MyDeps], tools []agentic.Tool) ([]agentic.Tool, error) {
			if !ctx.Deps.IsAdmin {
				// Filter out admin tools for non-admins
				var filtered []agentic.Tool
				for _, t := range tools {
					if t.Function.Name != "admin_tool" {
						filtered = append(filtered, t)
					}
				}
				return filtered, nil
			}
			return tools, nil
		},
	).AddTool(tool, handler)

	// Non-admin: tool should be filtered
	deps := &MyDeps{IsAdmin: false}
	_, err := agent.Run(context.Background(), "go", deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	calls := model.Calls()
	if len(calls[0].Tools) != 0 {
		t.Errorf("expected 0 tools for non-admin, got %d", len(calls[0].Tools))
	}

	// Admin: tool should be present
	model.Reset()
	deps = &MyDeps{IsAdmin: true}
	_, err = agent.Run(context.Background(), "go", deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	calls = model.Calls()
	if len(calls[0].Tools) != 1 {
		t.Errorf("expected 1 tool for admin, got %d", len(calls[0].Tools))
	}
}

func TestToolPrepare_CalledEachIteration(t *testing.T) {
	type Input struct {
		V string `json:"v"`
	}

	tool, handler := agentic.MustToolPlain("my_tool", "Tool", func(input Input) (string, error) {
		return "ok", nil
	})

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "my_tool", Input: map[string]interface{}{"v": "x"}},
			},
		},
		test.ModelResponse{Text: "done"},
	)

	prepareCallCount := 0
	agent := agentic.NewAgent("test", model,
		agentic.WithToolPrepare(func(ctx context.Context, tools []agentic.Tool) ([]agentic.Tool, error) {
			prepareCallCount++
			return tools, nil
		}),
	).AddTool(tool, handler)

	_, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should be called once per LLM request (2 iterations: tool call + final)
	if prepareCallCount != 2 {
		t.Errorf("expected prepare called 2 times (once per iteration), got %d", prepareCallCount)
	}
}
