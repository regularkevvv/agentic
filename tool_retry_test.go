package agentic_test

import (
	"context"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestPerToolRetry_OverridesGlobal(t *testing.T) {
	type Input struct {
		V string `json:"v"`
	}

	retryCount := 0
	// Tool with MaxRetries=3 (agent global is 1)
	tool, handler := agentic.MustToolPlain("flaky", "Flaky tool", func(input Input) (string, error) {
		retryCount++
		if retryCount <= 3 {
			return "", agentic.Retry("try again")
		}
		return "success", nil
	}, agentic.WithToolMaxRetries(3))

	// Model calls tool, then gets retries, eventually gets success and final text
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "flaky", Input: map[string]interface{}{"v": "x"}},
			},
		},
		// After retry 1 - model calls tool again
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c2", Name: "flaky", Input: map[string]interface{}{"v": "x"}},
			},
		},
		// After retry 2 - model calls tool again
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c3", Name: "flaky", Input: map[string]interface{}{"v": "x"}},
			},
		},
		// After retry 3 - model calls tool again (this time it succeeds)
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c4", Name: "flaky", Input: map[string]interface{}{"v": "x"}},
			},
		},
		test.ModelResponse{Text: "done"},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithRetries(agentic.RetryConfig{MaxRetries: 1}), // global = 1
	).AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "done" {
		t.Errorf("expected 'done', got %q", result.Output)
	}
	// Tool was called 4 times (3 retries + 1 success)
	if retryCount != 4 {
		t.Errorf("expected 4 calls (3 retries + 1 success), got %d", retryCount)
	}
}

func TestPerToolRetry_ZeroMeansNoRetries(t *testing.T) {
	type Input struct {
		V string `json:"v"`
	}

	callCount := 0
	// Tool with MaxRetries=0 — should not retry even though global allows 1
	tool, handler := agentic.MustToolPlain("no_retry", "No retry", func(input Input) (string, error) {
		callCount++
		return "", agentic.Retry("fail")
	}, agentic.WithToolMaxRetries(0))

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "no_retry", Input: map[string]interface{}{"v": "x"}},
			},
		},
		test.ModelResponse{Text: "gave up"},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithRetries(agentic.RetryConfig{MaxRetries: 5}), // global allows 5
	).AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// With 0 retries, the tool is called once, retry fails immediately,
	// and the error message is sent back to model
	if callCount != 1 {
		t.Errorf("expected 1 call (no retries), got %d", callCount)
	}
	if result.Output != "gave up" {
		t.Errorf("expected 'gave up', got %q", result.Output)
	}
}

func TestPerToolRetry_NoConfig_FallsBackToGlobal(t *testing.T) {
	type Input struct {
		V string `json:"v"`
	}

	callCount := 0
	// Tool without per-tool config — should use global default of 2
	tool, handler := agentic.MustToolPlain("default_tool", "Default", func(input Input) (string, error) {
		callCount++
		if callCount <= 2 {
			return "", agentic.Retry("retry")
		}
		return "ok", nil
	})

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "default_tool", Input: map[string]interface{}{"v": "x"}},
			},
		},
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c2", Name: "default_tool", Input: map[string]interface{}{"v": "x"}},
			},
		},
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c3", Name: "default_tool", Input: map[string]interface{}{"v": "x"}},
			},
		},
		test.ModelResponse{Text: "done"},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithRetries(agentic.RetryConfig{MaxRetries: 2}), // global = 2
	).AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "done" {
		t.Errorf("expected 'done', got %q", result.Output)
	}
	if callCount != 3 {
		t.Errorf("expected 3 calls (2 retries + 1 success), got %d", callCount)
	}
}

func TestPerToolRetry_DifferentToolsDifferentLimits(t *testing.T) {
	type Input struct {
		V string `json:"v"`
	}

	flakyCount := 0
	// flaky tool: 3 retries
	toolFlaky, handlerFlaky := agentic.MustToolPlain("flaky", "Flaky", func(input Input) (string, error) {
		flakyCount++
		return "", agentic.Retry("fail")
	}, agentic.WithToolMaxRetries(3))

	strictCount := 0
	// strict tool: 0 retries
	toolStrict, handlerStrict := agentic.MustToolPlain("strict", "Strict", func(input Input) (string, error) {
		strictCount++
		return "", agentic.Retry("fail")
	}, agentic.WithToolMaxRetries(0))

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "flaky", Input: map[string]interface{}{"v": "x"}},
				{ID: "c2", Name: "strict", Input: map[string]interface{}{"v": "x"}},
			},
		},
		test.ModelResponse{Text: "done"},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithRetries(agentic.RetryConfig{MaxRetries: 1}),
	).AddTool(toolFlaky, handlerFlaky).AddTool(toolStrict, handlerStrict)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// flaky: called once, retry allowed (count < 3), error sent back
	if flakyCount != 1 {
		t.Errorf("expected flaky called 1 time, got %d", flakyCount)
	}
	// strict: called once, no retries (MaxRetries=0), error appears in result
	if strictCount != 1 {
		t.Errorf("expected strict called 1 time, got %d", strictCount)
	}
	if result.Output != "done" {
		t.Errorf("expected 'done', got %q", result.Output)
	}
}
