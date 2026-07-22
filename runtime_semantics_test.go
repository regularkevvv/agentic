package agentic_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

type semanticToolInput struct {
	Value string `json:"value"`
}

type semanticOutput struct {
	Answer string `json:"answer" validate:"required"`
}

func TestTypedAgentExhaustiveExecutesOrdinaryToolsAlongsideOutput(t *testing.T) {
	var calls atomic.Int32
	before, beforeHandler := agentic.MustToolPlain("before", "before output", func(input semanticToolInput) (string, error) {
		calls.Add(1)
		return input.Value, nil
	})
	after, afterHandler := agentic.MustToolPlain("after", "after output", func(input semanticToolInput) (string, error) {
		calls.Add(1)
		return input.Value, nil
	})

	model := test.NewTestModel(test.ModelResponse{ToolCalls: []agentic.ToolUse{
		{ID: "before-1", Name: "before", Input: map[string]interface{}{"value": "a"}},
		{ID: "output-1", Name: "__output__", Input: map[string]interface{}{"answer": "done"}},
		{ID: "after-1", Name: "after", Input: map[string]interface{}{"value": "b"}},
	}})
	agent := agentic.NewTypedAgent[semanticOutput](
		"system",
		model,
		"final answer",
		agentic.WithEndStrategy(agentic.EndStrategyExhaustive),
	).AddTool(before, beforeHandler).AddTool(after, afterHandler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Output.Answer != "done" {
		t.Fatalf("unexpected output %#v", result.Output)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("expected both ordinary tools to execute, got %d calls", got)
	}
	if len(result.ToolCalls) != 3 || len(result.ToolResults) != 3 {
		t.Fatalf("unexpected calls/results: %d/%d", len(result.ToolCalls), len(result.ToolResults))
	}
}

func TestTypedAgentEarlySkipsOrdinaryToolsWhenOutputPresent(t *testing.T) {
	var calls atomic.Int32
	ordinary, handler := agentic.MustToolPlain("ordinary", "ordinary tool", func(input semanticToolInput) (string, error) {
		calls.Add(1)
		return input.Value, nil
	})
	model := test.NewTestModel(test.ModelResponse{ToolCalls: []agentic.ToolUse{
		{ID: "ordinary-1", Name: "ordinary", Input: map[string]interface{}{"value": "a"}},
		{ID: "output-1", Name: "__output__", Input: map[string]interface{}{"answer": "done"}},
	}})
	agent := agentic.NewTypedAgent[semanticOutput](
		"system",
		model,
		"final answer",
		agentic.WithEndStrategy(agentic.EndStrategyEarly),
	).AddTool(ordinary, handler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Output.Answer != "done" || calls.Load() != 0 {
		t.Fatalf("expected output without ordinary execution, output=%#v calls=%d", result.Output, calls.Load())
	}
}

func TestTypedOutputValidatorRetriesWithRunDeps(t *testing.T) {
	type deps struct{ Prefix string }
	model := test.NewTestModel(
		test.ModelResponse{ToolCalls: []agentic.ToolUse{{
			ID: "output-1", Name: "__output__", Input: map[string]interface{}{"answer": "wrong"},
		}}},
		test.ModelResponse{ToolCalls: []agentic.ToolUse{{
			ID: "output-2", Name: "__output__", Input: map[string]interface{}{"answer": "approved"},
		}}},
	)
	agent := agentic.NewTypedAgentWithDeps[semanticOutput, *deps]("system", model, "final answer")
	agent.AddOutputValidatorWithDeps(agentic.TypedOutputValidatorWithDepsFunc[*deps, semanticOutput](
		func(ctx agentic.RunContext[*deps], output semanticOutput) error {
			if output.Answer != ctx.Deps.Prefix {
				return errors.New("answer was not approved")
			}
			return nil
		},
	))

	result, err := agent.Run(context.Background(), "go", &deps{Prefix: "approved"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Output.Answer != "approved" || model.CallCount() != 2 {
		t.Fatalf("expected validated retry, output=%#v calls=%d", result.Output, model.CallCount())
	}
}

func TestRetryBudgetsAreIndependentPerTool(t *testing.T) {
	var firstCalls atomic.Int32
	var secondCalls atomic.Int32
	first, firstHandler := agentic.MustToolPlain("first", "first tool", func(input semanticToolInput) (string, error) {
		if firstCalls.Add(1) == 1 {
			return "", agentic.Retry("retry first")
		}
		return "first ok", nil
	}, agentic.WithToolMaxRetries(1))
	second, secondHandler := agentic.MustToolPlain("second", "second tool", func(input semanticToolInput) (string, error) {
		if secondCalls.Add(1) == 1 {
			return "", agentic.Retry("retry second")
		}
		return "second ok", nil
	}, agentic.WithToolMaxRetries(1))

	calls := []agentic.ToolUse{
		{ID: "first", Name: "first", Input: map[string]interface{}{"value": "x"}},
		{ID: "second", Name: "second", Input: map[string]interface{}{"value": "x"}},
	}
	model := test.NewTestModel(
		test.ModelResponse{ToolCalls: calls},
		test.ModelResponse{ToolCalls: calls},
		test.ModelResponse{Text: "done"},
	)
	agent := agentic.NewAgent("system", model).AddTool(first, firstHandler).AddTool(second, secondHandler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Retries != 2 || firstCalls.Load() != 2 || secondCalls.Load() != 2 {
		t.Fatalf("unexpected retry accounting: retries=%d first=%d second=%d", result.Retries, firstCalls.Load(), secondCalls.Load())
	}
}

func TestAgentSendsToolsInRegistrationOrder(t *testing.T) {
	b, bh := agentic.MustToolPlain("b", "tool b", func(input semanticToolInput) (string, error) { return "b", nil })
	a, ah := agentic.MustToolPlain("a", "tool a", func(input semanticToolInput) (string, error) { return "a", nil })
	model := test.NewTestModel(test.ModelResponse{Text: "done"})
	agent := agentic.NewAgent("system", model).AddTool(b, bh).AddTool(a, ah)
	if _, err := agent.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	tools := model.Calls()[0].Tools
	if len(tools) != 2 || tools[0].Function.Name != "b" || tools[1].Function.Name != "a" {
		t.Fatalf("tools were not stable: %#v", tools)
	}
}

func TestContextAwareToolReceivesRunContext(t *testing.T) {
	type contextKey struct{}
	tool, handler := agentic.MustToolWithContext("context_tool", "context-aware tool", func(ctx context.Context, input semanticToolInput) (string, error) {
		value, _ := ctx.Value(contextKey{}).(string)
		return value + input.Value, nil
	})
	model := test.NewTestModel(
		test.ModelResponse{ToolCalls: []agentic.ToolUse{{
			ID: "context-1", Name: "context_tool", Input: map[string]interface{}{"value": "-value"},
		}}},
		test.ModelResponse{Text: "done"},
	)
	agent := agentic.NewAgent("system", model).AddTool(tool, handler)
	ctx := context.WithValue(context.Background(), contextKey{}, "run")
	result, err := agent.Run(ctx, "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := result.ToolResults[0].Content; got != "run-value" {
		t.Fatalf("unexpected context-aware result %#v", got)
	}
}

func TestApprovalToolRejectionUsesAgentRetryBudget(t *testing.T) {
	var approvals atomic.Int32
	tool, handler := agentic.MustApprovalTool(
		"guarded",
		"guarded tool",
		func(ctx context.Context, input semanticToolInput) (string, error) {
			return "should not run", nil
		},
		func(ctx context.Context, call agentic.ToolUse) (bool, error) {
			approvals.Add(1)
			return false, nil
		},
	)
	toolCall := test.ModelResponse{ToolCalls: []agentic.ToolUse{{
		ID: "guarded-1", Name: "guarded", Input: map[string]interface{}{"value": "x"},
	}}}
	model := test.NewTestModel(toolCall, toolCall, test.ModelResponse{Text: "not approved"})
	agent := agentic.NewAgent("system", model,
		agentic.WithRetries(agentic.RetryConfig{MaxRetries: 1}),
	).AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Retries != 1 || approvals.Load() != 2 || result.Output != "not approved" {
		t.Fatalf("unexpected approval retry behavior: retries=%d approvals=%d output=%q", result.Retries, approvals.Load(), result.Output)
	}
}
