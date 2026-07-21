package agentic

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/testutil"
	testprovider "github.com/regularkevvv/agentic/provider/test"
)

func TestNewAgentDynamicRegistersHandoffs(t *testing.T) {
	child := NewAgent("child", testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"}))
	h := NewHandoff("delegate", "delegate work", child)

	agent := NewAgentDynamic(
		func(ctx context.Context) (string, error) { return "dynamic", nil },
		&testutil.StubModel{NameValue: "dynamic-model"},
	).AddHandoff(h)

	if agent.core.registry == nil || !agent.core.registry.Has("delegate") {
		t.Fatalf("expected dynamic agent to register handoff, got %#v", agent.core.registry)
	}
}

func TestAgentRunAdditionalErrorPaths(t *testing.T) {
	t.Run("empty response", func(t *testing.T) {
		agent := NewAgent("system", &testutil.StubModel{
			NameValue: "empty-model",
			Response:  &ChatResponse{},
		})

		_, err := agent.Run(context.Background(), "prompt")
		if err == nil || !IsProviderError(err) {
			t.Fatalf("expected provider error for empty response, got %v", err)
		}
	})

	t.Run("tool call without registered tools", func(t *testing.T) {
		agent := NewAgent("system", &testutil.StubModel{
			NameValue: "tool-model",
			Response: &ChatResponse{
				Message: NewToolUseMessage(ToolUse{
					ID:    "call_1",
					Name:  "lookup",
					Input: map[string]interface{}{"city": "Lima"},
				}),
				FinishReason: FinishReasonToolCalls,
			},
		})

		_, err := agent.Run(context.Background(), "prompt")
		if err == nil || !strings.Contains(err.Error(), "no tools are registered") {
			t.Fatalf("expected missing registry error, got %v", err)
		}
	})

	t.Run("model error is wrapped", func(t *testing.T) {
		agent := NewAgent("system", &testutil.StubModel{
			NameValue: "error-model",
			Err:       errors.New("boom"),
		})

		_, err := agent.Run(context.Background(), "prompt")
		if err == nil || err.Error() != "model request: boom" {
			t.Fatalf("expected wrapped model error, got %v", err)
		}
	})
}
