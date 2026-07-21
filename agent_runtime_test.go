package agentic_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/internal/testutil"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestAgentSetRegistry(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "ok"})
	agent := agentic.NewAgent("test", model)

	reg := agentic.NewRegistry()
	agent.SetRegistry(reg)

	result, err := agent.Run(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "ok" {
		t.Errorf("expected %q, got %q", "ok", result.Output)
	}
}

func TestAgentDynamicPromptError(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "ok"})

	agent := agentic.NewAgentDynamic(
		func(ctx context.Context) (string, error) {
			return "", fmt.Errorf("prompt generation failed")
		},
		model,
	)

	_, err := agent.Run(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error from dynamic prompt")
	}
	if !strings.Contains(err.Error(), "system prompt") {
		t.Errorf("expected 'system prompt' in error, got %q", err.Error())
	}
}

func TestAgentNoToolsButToolCallRequested(t *testing.T) {
	// Model returns tool calls but no tools are registered
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "missing", Input: map[string]interface{}{}},
			},
		},
	)

	agent := agentic.NewAgent("test", model)

	_, err := agent.Run(context.Background(), "go")
	if err == nil {
		t.Fatal("expected error when model requests tools but none registered")
	}
}

func TestAgentWithRunOptions(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "ok"})
	agent := agentic.NewAgent("test", model)

	result, err := agent.Run(context.Background(), "hello",
		agentic.WithRunTemperature(0.5),
		agentic.WithRunMaxTokens(200),
		agentic.WithRunMaxIterations(3),
		agentic.WithMessages(agentic.NewTextMessage(agentic.RoleUser, "previous message")),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "ok" {
		t.Errorf("expected %q, got %q", "ok", result.Output)
	}
}

func TestAgentWithMessagesContainingSystemPrompt(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "ok"})
	agent := agentic.NewAgent("test", model)

	// If messages already contain a system prompt, don't add another
	result, err := agent.Run(context.Background(), "hello",
		agentic.WithMessages(
			agentic.NewTextMessage(agentic.RoleSystem, "custom system prompt"),
			agentic.NewTextMessage(agentic.RoleUser, "previous"),
		),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "ok" {
		t.Errorf("expected %q, got %q", "ok", result.Output)
	}

	// Verify only one system message in the call
	calls := model.Calls()
	systemCount := 0
	for _, msg := range calls[0].Messages {
		if msg.Role == agentic.RoleSystem {
			systemCount++
		}
	}
	if systemCount != 1 {
		t.Errorf("expected 1 system message, got %d", systemCount)
	}
}

func TestAgentModelRequestError(t *testing.T) {
	model := &testutil.StubModel{NameValue: "error-model", Err: fmt.Errorf("api error")}
	agent := agentic.NewAgent("test", model)

	_, err := agent.Run(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error from model request")
	}
	if !strings.Contains(err.Error(), "model request") {
		t.Errorf("expected 'model request' in error, got %q", err.Error())
	}
}

// TestAgentEmptyResponseIsRejected covers both ways a provider can complete the
// transport successfully without producing a usable turn. Neither may be
// mistaken for a real answer, so both must surface as *agentic.ProviderError.
func TestAgentEmptyResponseIsRejected(t *testing.T) {
	t.Run("message has no content parts", func(t *testing.T) {
		model := &testutil.StubModel{
			NameValue: "empty-model",
			Response: &agentic.ChatResponse{
				Message:      agentic.Message{Role: agentic.RoleAssistant},
				FinishReason: agentic.FinishReasonStop,
			},
		}
		agent := agentic.NewAgent("test", model)

		_, err := agent.Run(context.Background(), "hello")
		if err == nil {
			t.Fatal("expected error for response with no content parts")
		}
		if !agentic.IsProviderError(err) {
			t.Fatalf("expected *agentic.ProviderError, got %T: %v", err, err)
		}
		if !strings.Contains(err.Error(), "empty response from provider") {
			t.Errorf("expected empty-response reason, got %q", err.Error())
		}
	})

	// Content being present must not rescue the turn: an in-band failure means
	// whatever text arrived is partial, not a complete answer.
	t.Run("finish reason reports an in-band failure", func(t *testing.T) {
		model := &testutil.StubModel{
			NameValue: "failed-model",
			Response: &agentic.ChatResponse{
				Message:         agentic.NewTextMessage(agentic.RoleAssistant, "partial answ"),
				FinishReason:    agentic.FinishReasonError,
				RawFinishReason: "upstream_overloaded",
			},
		}
		agent := agentic.NewAgent("test", model)

		_, err := agent.Run(context.Background(), "hello")
		if err == nil {
			t.Fatal("expected error for FinishReasonError response")
		}
		if !agentic.IsProviderError(err) {
			t.Fatalf("expected *agentic.ProviderError, got %T: %v", err, err)
		}
		// The provider's own stop reason is passed through so callers can act
		// on provider-specific values.
		if !strings.Contains(err.Error(), "upstream_overloaded") {
			t.Errorf("expected raw finish reason in error, got %q", err.Error())
		}
	})
}

func TestAgentWithTemperatureAndTopP(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "ok"})
	agent := agentic.NewAgent("test", model,
		agentic.WithTemperature(0.8),
		agentic.WithMaxTokens(500),
		agentic.WithTopP(0.9),
		agentic.WithToolChoice(agentic.ToolChoiceAuto),
	)

	result, err := agent.Run(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "ok" {
		t.Errorf("expected %q, got %q", "ok", result.Output)
	}
}

func TestAgentAddToolPanicsOnDuplicate(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on duplicate tool registration")
		}
	}()

	model := test.NewTestModel(test.ModelResponse{Text: "ok"})
	type In struct{}
	type Out struct{}
	tool, handler := agentic.MustToolPlain("dup", "duplicate", func(in In) (Out, error) { return Out{}, nil })

	agent := agentic.NewAgent("test", model)
	agent.AddTool(tool, handler)
	agent.AddTool(tool, handler) // should panic
}

func TestFirstNonNil(t *testing.T) {
	// Tested indirectly through agent options, but let's verify behavior
	// by using run options that override agent options
	model := test.NewTestModel(test.ModelResponse{Text: "ok"})
	agent := agentic.NewAgent("test", model,
		agentic.WithTemperature(0.3),
	)

	// Run with temperature override
	_, err := agent.Run(context.Background(), "hi",
		agentic.WithRunTemperature(0.9),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	calls := model.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Temperature == nil {
		t.Fatal("expected temperature to be set")
	}
	if *calls[0].Temperature != 0.9 {
		t.Errorf("expected run temperature 0.9 to override agent 0.3, got %f", *calls[0].Temperature)
	}
}

func TestAgentNewAgentDynamicWithOptions(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "ok"})

	agent := agentic.NewAgentDynamic(
		func(ctx context.Context) (string, error) {
			return "dynamic prompt", nil
		},
		model,
		agentic.WithMaxIterations(5),
		agentic.WithRetries(agentic.RetryConfig{MaxRetries: 3}),
	)

	result, err := agent.Run(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "ok" {
		t.Errorf("expected %q, got %q", "ok", result.Output)
	}
}
