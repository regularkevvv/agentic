package agentic_test

import (
	"context"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestUsageLimitsRequestLimit(t *testing.T) {
	// Model always returns tool calls, so agent loops forever.
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "get_weather", Input: map[string]interface{}{"location": "NYC"}},
			},
		},
	)

	tool, handler := agentic.MustToolPlain("get_weather", "Get weather", func(input struct {
		Location string `json:"location"`
	}) (string, error) {
		return "sunny", nil
	})

	agent := agentic.NewAgent("test", model,
		agentic.WithUsageLimits(agentic.UsageLimits{
			MaxRequests: agentic.IntPtr(3),
		}),
		agentic.WithMaxIterations(100),
	)
	agent.AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "weather?")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !agentic.IsUsageLimitExceeded(err) {
		t.Fatalf("expected UsageLimitExceededError, got %T: %v", err, err)
	}

	var ule *agentic.UsageLimitExceededError
	if !errors.As(err, &ule) {
		t.Fatalf("expected UsageLimitExceededError, got %T", err)
	}
	if ule.LimitName != "requests" {
		t.Errorf("expected limit name 'requests', got %q", ule.LimitName)
	}
	if ule.Max != 3 {
		t.Errorf("expected max 3, got %d", ule.Max)
	}
	// Should have made exactly 3 requests before hitting the limit
	if result.Usage.Requests != 3 {
		t.Errorf("expected 3 requests in usage, got %d", result.Usage.Requests)
	}
}

func TestUsageLimitsResponseTokenLimit(t *testing.T) {
	// Each response uses 100 completion tokens.
	model := test.NewTestModel(
		test.ModelResponse{
			Text: "first",
			Usage: &agentic.Usage{
				PromptTokens:     50,
				CompletionTokens: 100,
				TotalTokens:      150,
			},
		},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithUsageLimits(agentic.UsageLimits{
			MaxResponseTokens: agentic.IntPtr(50), // Lower than the 100 tokens the model uses
		}),
	)

	_, err := agent.Run(context.Background(), "go")
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var ule *agentic.UsageLimitExceededError
	if !errors.As(err, &ule) {
		t.Fatalf("expected UsageLimitExceededError, got %T: %v", err, err)
	}
	if ule.LimitName != "response_tokens" {
		t.Errorf("expected limit name 'response_tokens', got %q", ule.LimitName)
	}
}

func TestUsageLimitsTotalTokenLimit(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "ping", Input: map[string]interface{}{}},
			},
			Usage: &agentic.Usage{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
			},
		},
		test.ModelResponse{
			Text: "done",
			Usage: &agentic.Usage{
				PromptTokens:     200,
				CompletionTokens: 50,
				TotalTokens:      250,
			},
		},
	)

	tool, handler := agentic.MustToolPlain("ping", "Ping", func(input struct{}) (string, error) {
		return "pong", nil
	})

	agent := agentic.NewAgent("test", model,
		agentic.WithUsageLimits(agentic.UsageLimits{
			MaxTotalTokens: agentic.IntPtr(200), // Exceeded after second request (150+250=400)
		}),
	)
	agent.AddTool(tool, handler)

	_, err := agent.Run(context.Background(), "go")
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var ule *agentic.UsageLimitExceededError
	if !errors.As(err, &ule) {
		t.Fatalf("expected UsageLimitExceededError, got %T: %v", err, err)
	}
	if ule.LimitName != "total_tokens" {
		t.Errorf("expected limit name 'total_tokens', got %q", ule.LimitName)
	}
}

func TestUsageLimitsToolCallLimit(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "ping", Input: map[string]interface{}{}},
				{ID: "c2", Name: "ping", Input: map[string]interface{}{}},
				{ID: "c3", Name: "ping", Input: map[string]interface{}{}},
			},
		},
	)

	tool, handler := agentic.MustToolPlain("ping", "Ping", func(input struct{}) (string, error) {
		return "pong", nil
	})

	agent := agentic.NewAgent("test", model,
		agentic.WithUsageLimits(agentic.UsageLimits{
			MaxToolCalls: agentic.IntPtr(2), // 3 tool calls requested, limit is 2
		}),
		agentic.WithMaxIterations(100),
	)
	agent.AddTool(tool, handler)

	_, err := agent.Run(context.Background(), "go")
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var ule *agentic.UsageLimitExceededError
	if !errors.As(err, &ule) {
		t.Fatalf("expected UsageLimitExceededError, got %T: %v", err, err)
	}
	if ule.LimitName != "tool_calls" {
		t.Errorf("expected limit name 'tool_calls', got %q", ule.LimitName)
	}
}

func TestUsageLimitsNoLimitNoError(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "hello"})
	agent := agentic.NewAgent("test", model)

	result, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "hello" {
		t.Errorf("expected 'hello', got %q", result.Output)
	}
	if result.Usage.Requests != 1 {
		t.Errorf("expected 1 request, got %d", result.Usage.Requests)
	}
}

func TestUsageLimitsRunOptionOverridesAgent(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "ping", Input: map[string]interface{}{}},
			},
		},
	)

	tool, handler := agentic.MustToolPlain("ping", "Ping", func(input struct{}) (string, error) {
		return "pong", nil
	})

	// Agent has generous limit
	agent := agentic.NewAgent("test", model,
		agentic.WithUsageLimits(agentic.UsageLimits{
			MaxRequests: agentic.IntPtr(100),
		}),
		agentic.WithMaxIterations(100),
	)
	agent.AddTool(tool, handler)

	// Run overrides with strict limit
	_, err := agent.Run(context.Background(), "go",
		agentic.WithRunUsageLimits(agentic.UsageLimits{
			MaxRequests: agentic.IntPtr(2),
		}),
	)
	if !agentic.IsUsageLimitExceeded(err) {
		t.Fatalf("expected UsageLimitExceededError from run option override, got %v", err)
	}
}

func TestUsageLimitsRequestTokenLimit(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{
			Text: "first",
			Usage: &agentic.Usage{
				PromptTokens:     500,
				CompletionTokens: 10,
				TotalTokens:      510,
			},
		},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithUsageLimits(agentic.UsageLimits{
			MaxRequestTokens: agentic.IntPtr(100), // Lower than 500 prompt tokens
		}),
	)

	_, err := agent.Run(context.Background(), "go")
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var ule *agentic.UsageLimitExceededError
	if !errors.As(err, &ule) {
		t.Fatalf("expected UsageLimitExceededError, got %T: %v", err, err)
	}
	if ule.LimitName != "request_tokens" {
		t.Errorf("expected limit name 'request_tokens', got %q", ule.LimitName)
	}
}

func TestUsageTrackingAccumulation(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "ping", Input: map[string]interface{}{}},
			},
			Usage: &agentic.Usage{
				PromptTokens:     100,
				CompletionTokens: 20,
				TotalTokens:      120,
			},
		},
		test.ModelResponse{
			Text: "done",
			Usage: &agentic.Usage{
				PromptTokens:     200,
				CompletionTokens: 30,
				TotalTokens:      230,
			},
		},
	)

	tool, handler := agentic.MustToolPlain("ping", "Ping", func(input struct{}) (string, error) {
		return "pong", nil
	})

	agent := agentic.NewAgent("test", model)
	agent.AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check accumulated usage
	if result.Usage.PromptTokens != 300 {
		t.Errorf("expected 300 prompt tokens, got %d", result.Usage.PromptTokens)
	}
	if result.Usage.CompletionTokens != 50 {
		t.Errorf("expected 50 completion tokens, got %d", result.Usage.CompletionTokens)
	}
	if result.Usage.TotalTokens != 350 {
		t.Errorf("expected 350 total tokens, got %d", result.Usage.TotalTokens)
	}
	if result.Usage.Requests != 2 {
		t.Errorf("expected 2 requests, got %d", result.Usage.Requests)
	}
	if result.Usage.ToolCalls != 1 {
		t.Errorf("expected 1 tool call, got %d", result.Usage.ToolCalls)
	}

	// Check per-request breakdown
	if len(result.Usage.RequestUsages) != 2 {
		t.Fatalf("expected 2 request usages, got %d", len(result.Usage.RequestUsages))
	}
	if result.Usage.RequestUsages[0].PromptTokens != 100 {
		t.Errorf("expected first request 100 prompt tokens, got %d", result.Usage.RequestUsages[0].PromptTokens)
	}
	if result.Usage.RequestUsages[1].PromptTokens != 200 {
		t.Errorf("expected second request 200 prompt tokens, got %d", result.Usage.RequestUsages[1].PromptTokens)
	}
}

func TestDefaultUsageLimits(t *testing.T) {
	limits := agentic.DefaultUsageLimits()
	if limits.MaxRequests == nil {
		t.Fatal("expected MaxRequests to be set")
	}
	if *limits.MaxRequests != 50 {
		t.Errorf("expected MaxRequests=50, got %d", *limits.MaxRequests)
	}
	if limits.MaxTotalTokens != nil {
		t.Error("expected MaxTotalTokens to be nil")
	}
}

func TestUsageLimitExceededErrorMessage(t *testing.T) {
	err := &agentic.UsageLimitExceededError{
		LimitName: "requests",
		Current:   10,
		Max:       5,
	}
	expected := "usage limit exceeded: requests (10 > 5)"
	if err.Error() != expected {
		t.Errorf("expected %q, got %q", expected, err.Error())
	}
}

func TestIntPtr(t *testing.T) {
	p := agentic.IntPtr(42)
	if p == nil {
		t.Fatal("expected non-nil pointer")
	}
	if *p != 42 {
		t.Errorf("expected 42, got %d", *p)
	}
}
