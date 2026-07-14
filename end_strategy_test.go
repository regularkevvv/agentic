package agentic_test

import (
	"context"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestEndStrategyExhaustive_AllToolsExecute(t *testing.T) {
	// Setup: two regular tools
	type InputA struct {
		X string `json:"x"`
	}
	type InputB struct {
		Y string `json:"y"`
	}

	toolA, handlerA := agentic.MustToolPlain("tool_a", "Tool A", func(input InputA) (string, error) {
		return "result_a", nil
	})
	toolB, handlerB := agentic.MustToolPlain("tool_b", "Tool B", func(input InputB) (string, error) {
		return "result_b", nil
	})

	// Model returns both tool calls, then a final response
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "tool_a", Input: map[string]interface{}{"x": "hello"}},
				{ID: "c2", Name: "tool_b", Input: map[string]interface{}{"y": "world"}},
			},
		},
		test.ModelResponse{Text: "Done with both"},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithEndStrategy(agentic.EndStrategyExhaustive),
	).AddTool(toolA, handlerA).AddTool(toolB, handlerB)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "Done with both" {
		t.Errorf("expected 'Done with both', got %q", result.Output)
	}
	if len(result.ToolCalls) != 2 {
		t.Errorf("expected 2 tool calls, got %d", len(result.ToolCalls))
	}
	if len(result.ToolResults) != 2 {
		t.Errorf("expected 2 tool results, got %d", len(result.ToolResults))
	}
}

func TestEndStrategyExhaustive_IsDefault(t *testing.T) {
	// Verify that the default behavior is exhaustive (all tool calls processed)
	type Input struct {
		V string `json:"v"`
	}

	var callCount atomic.Int32
	tool, handler := agentic.MustToolPlain("counter", "Count", func(input Input) (string, error) {
		callCount.Add(1)
		return "ok", nil
	})

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "counter", Input: map[string]interface{}{"v": "1"}},
				{ID: "c2", Name: "counter", Input: map[string]interface{}{"v": "2"}},
				{ID: "c3", Name: "counter", Input: map[string]interface{}{"v": "3"}},
			},
		},
		test.ModelResponse{Text: "done"},
	)

	agent := agentic.NewAgent("test", model).AddTool(tool, handler)
	_, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if callCount.Load() != 3 {
		t.Errorf("expected all 3 tools to execute (exhaustive default), got %d", callCount.Load())
	}
}

func TestEndStrategyEarly_StopsAtOutputTool(t *testing.T) {
	// Create a TypedAgent that uses output tools, and verify early strategy
	// stops at the first output tool without executing remaining tool calls.
	type Input struct {
		V string `json:"v"`
	}

	sideEffectCalled := false
	tool, handler := agentic.MustToolPlain("side_effect", "Side effect", func(input Input) (string, error) {
		sideEffectCalled = true
		return "side", nil
	})

	// Model returns output tool + regular tool in same response
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				// Output tool first, side_effect second
				{ID: "c1", Name: "final_result", Input: map[string]interface{}{"answer": "42"}},
				{ID: "c2", Name: "side_effect", Input: map[string]interface{}{"v": "boom"}},
			},
		},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithEndStrategy(agentic.EndStrategyEarly),
	).AddTool(tool, handler)

	// Manually mark "final_result" as an output tool
	agent.SetOutputToolNames(map[string]bool{"final_result": true})

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Side effect should NOT have been called
	if sideEffectCalled {
		t.Error("expected side_effect tool NOT to be called with EndStrategyEarly")
	}

	// Should have returned with the output tool call recorded
	if len(result.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call (only the output tool), got %d", len(result.ToolCalls))
	}
}

func TestEndStrategyEarly_NoOutputTool_ExecutesAll(t *testing.T) {
	// With EndStrategyEarly but no output tools, all regular tools still execute
	type Input struct {
		V string `json:"v"`
	}

	var callCount atomic.Int32
	tool, handler := agentic.MustToolPlain("regular", "Regular", func(input Input) (string, error) {
		callCount.Add(1)
		return "ok", nil
	})

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "regular", Input: map[string]interface{}{"v": "1"}},
				{ID: "c2", Name: "regular", Input: map[string]interface{}{"v": "2"}},
			},
		},
		test.ModelResponse{Text: "done"},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithEndStrategy(agentic.EndStrategyEarly),
	).AddTool(tool, handler)

	_, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if callCount.Load() != 2 {
		t.Errorf("expected all 2 regular tools to execute, got %d", callCount.Load())
	}
}

func TestWithRunEndStrategy_OverridesAgent(t *testing.T) {
	type Input struct {
		V string `json:"v"`
	}

	sideEffectCalled := false
	tool, handler := agentic.MustToolPlain("side_effect", "Side", func(input Input) (string, error) {
		sideEffectCalled = true
		return "ok", nil
	})

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "output_tool", Input: map[string]interface{}{}},
				{ID: "c2", Name: "side_effect", Input: map[string]interface{}{"v": "x"}},
			},
		},
	)

	// Agent default is Exhaustive, but run overrides to Early
	agent := agentic.NewAgent("test", model,
		agentic.WithEndStrategy(agentic.EndStrategyExhaustive),
	).AddTool(tool, handler)
	agent.SetOutputToolNames(map[string]bool{"output_tool": true})

	_, err := agent.Run(context.Background(), "go", agentic.WithRunEndStrategy(agentic.EndStrategyEarly))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if sideEffectCalled {
		t.Error("expected side_effect NOT called when run overrides to Early")
	}
}
