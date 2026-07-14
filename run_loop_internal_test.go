package agentic

import (
	"context"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/internal/testutil"
)

func TestPrepareLoopAndBuildRequestAppliesOverrides(t *testing.T) {
	type input struct {
		Value int `json:"value"`
	}

	toolA, handlerA := MustToolPlain("tool_a", "tool a", func(input input) (string, error) {
		return "a", nil
	})
	toolB, handlerB := MustToolPlain("tool_b", "tool b", func(input input) (string, error) {
		return "b", nil
	})

	seenToolCount := 0
	limits := UsageLimits{MaxRequests: IntPtr(2)}
	agent := NewAgent(
		"base system",
		&testutil.StubModel{NameValue: "coverage-model"},
		WithUsageLimits(limits),
		WithToolPrepare(func(ctx context.Context, tools []Tool) ([]Tool, error) {
			seenToolCount = len(tools)
			return tools[:1], nil
		}),
	).AddTool(toolA, handlerA).AddTool(toolB, handlerB)

	agent.core.systemPromptSuffix = "schema suffix"
	agent.core.responseFormat = &ResponseFormat{Type: "json_object"}
	agent.core.config.thinking = &ThinkingConfig{Enabled: true, BudgetTokens: 9}

	ls, err := agent.core.prepareLoop(
		context.Background(),
		"current prompt",
		dependencyEnvelope{},
		WithMessages(NewTextMessage(RoleUser, "history")),
		WithRunHistoryProcessor(HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
			return messages[1:], nil
		})),
		WithRunTemperature(0.3),
		WithRunMaxTokens(55),
		WithRunMaxIterations(7),
		WithRunEndStrategy(EndStrategyEarly),
	)
	if err != nil {
		t.Fatalf("prepareLoop: %v", err)
	}

	if ls.maxIterations != 7 {
		t.Fatalf("expected max iterations override, got %d", ls.maxIterations)
	}
	if ls.endStrategy != EndStrategyEarly {
		t.Fatalf("expected end strategy override, got %v", ls.endStrategy)
	}
	if ls.usageLimits == nil || ls.usageLimits.MaxRequests == nil || *ls.usageLimits.MaxRequests != 2 {
		t.Fatalf("expected inherited usage limits, got %#v", ls.usageLimits)
	}
	if len(ls.messages) != 3 {
		t.Fatalf("expected system + history + prompt, got %#v", ls.messages)
	}
	if got := ls.messages[0].GetTextContent(); got != "base system\n\nschema suffix" {
		t.Fatalf("unexpected system prompt %q", got)
	}

	req, err := agent.core.buildRequest(ls, true)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if !req.Stream {
		t.Fatal("expected streaming request")
	}
	if req.Temperature == nil || *req.Temperature != 0.3 {
		t.Fatalf("expected run temperature override, got %#v", req.Temperature)
	}
	if req.MaxTokens == nil || *req.MaxTokens != 55 {
		t.Fatalf("expected run max tokens override, got %#v", req.MaxTokens)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("expected history processor to trim one message, got %#v", req.Messages)
	}
	if req.Messages[0].GetTextContent() != "history" || req.Messages[1].GetTextContent() != "current prompt" {
		t.Fatalf("unexpected processed messages %#v", req.Messages)
	}
	if seenToolCount != 2 {
		t.Fatalf("expected tool prepare to see both tools, got %d", seenToolCount)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("expected tool prepare to filter tools, got %#v", req.Tools)
	}
	if req.ResponseFormat == nil || req.ResponseFormat.Type != "json_object" {
		t.Fatalf("expected response format on request, got %#v", req.ResponseFormat)
	}
	if req.Thinking == nil || req.Thinking.BudgetTokens != 9 {
		t.Fatalf("expected thinking config on request, got %#v", req.Thinking)
	}
}

func TestBuildRequestWrapsProcessorAndToolPrepareErrors(t *testing.T) {
	t.Run("history processor error", func(t *testing.T) {
		agent := NewAgent("system", &testutil.StubModel{NameValue: "coverage-model"})
		ls, err := agent.core.prepareLoop(
			context.Background(),
			"prompt",
			dependencyEnvelope{},
			WithRunHistoryProcessor(HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
				return nil, errors.New("boom")
			})),
		)
		if err != nil {
			t.Fatalf("prepareLoop: %v", err)
		}

		_, err = agent.core.buildRequest(ls, false)
		if err == nil || err.Error() != "history processor: boom" {
			t.Fatalf("expected wrapped history processor error, got %v", err)
		}
	})

	t.Run("tool prepare error", func(t *testing.T) {
		type input struct {
			Value int `json:"value"`
		}

		toolDef, handler := MustToolPlain("tool_a", "tool a", func(input input) (string, error) {
			return "ok", nil
		})
		agent := NewAgent(
			"system",
			&testutil.StubModel{NameValue: "coverage-model"},
			WithToolPrepare(func(ctx context.Context, tools []Tool) ([]Tool, error) {
				return nil, errors.New("no tools today")
			}),
		).AddTool(toolDef, handler)

		ls, err := agent.core.prepareLoop(context.Background(), "prompt", dependencyEnvelope{})
		if err != nil {
			t.Fatalf("prepareLoop: %v", err)
		}

		_, err = agent.core.buildRequest(ls, false)
		if err == nil || err.Error() != "tool prepare: no tools today" {
			t.Fatalf("expected wrapped tool prepare error, got %v", err)
		}
	})
}
