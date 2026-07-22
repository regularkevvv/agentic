package agentic

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/regularkevvv/agentic/internal/testutil"
)

type typedAgentCoverageValue struct {
	Value string `json:"value"`
}

type invalidModeOutputSpec struct {
	*ToolOutputSpec[typedAgentCoverageValue]
}

func (s invalidModeOutputSpec) Mode() OutputMode {
	return OutputMode("invalid")
}

type failingOutputSpec struct {
	err error
}

func (s failingOutputSpec) Tools() []Tool {
	return nil
}

func (s failingOutputSpec) Parse(Message) (typedAgentCoverageValue, error) {
	return typedAgentCoverageValue{}, s.err
}

type failingResponseFormatSpec struct {
	err error
}

type concurrentTypedModel struct{}

func (concurrentTypedModel) Name() string { return "concurrent-typed" }

func (concurrentTypedModel) Request(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	return &ChatResponse{
		Message:      NewTextMessage(RoleAssistant, `{"value":"ok"}`),
		FinishReason: FinishReasonStop,
	}, nil
}

func TestTypedAgentResponseConfigurationIsImmutableAcrossConcurrentRuns(t *testing.T) {
	agent := NewTypedAgentWithMode[typedAgentCoverageValue](
		"system",
		concurrentTypedModel{},
		NewNativeOutput[typedAgentCoverageValue]("value", "typed value"),
	)
	const runs = 32
	var wg sync.WaitGroup
	errs := make(chan error, runs)
	wg.Add(runs)
	for range runs {
		go func() {
			defer wg.Done()
			result, err := agent.Run(context.Background(), "prompt")
			if err == nil && result.Output.Value != "ok" {
				err = errors.New("unexpected typed output")
			}
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Run: %v", err)
		}
	}
}

func (s failingResponseFormatSpec) Tools() []Tool {
	return nil
}

func (s failingResponseFormatSpec) Parse(Message) (typedAgentCoverageValue, error) {
	return typedAgentCoverageValue{}, s.err
}

func (s failingResponseFormatSpec) ResponseFormat() *ResponseFormat {
	return &ResponseFormat{Type: "json_object"}
}

func (s failingResponseFormatSpec) Mode() OutputMode {
	return OutputModeNative
}

func TestTypedAgentHelpersAndTextProcessorError(t *testing.T) {
	type input struct {
		Value int `json:"value"`
	}

	auxTool, auxHandler := MustToolPlain("aux", "aux tool", func(input input) (string, error) {
		return "ok", nil
	})

	ta := NewTypedAgentWithMode[typedAgentCoverageValue](
		"system",
		&testutil.StubModel{NameValue: "typed-model"},
		NewToolOutput[typedAgentCoverageValue]("desc"),
	)

	if got := ta.AddToolset(NewToolset().Add(auxTool, auxHandler)); got != ta {
		t.Fatal("expected AddToolset to return the same typed agent")
	}
	if ta.runtime.core.registry == nil || !ta.runtime.core.registry.Has("aux") || !ta.runtime.core.registry.Has("__output__") {
		t.Fatalf("expected registry to contain toolset and output tools, got %#v", ta.runtime.core.registry)
	}

	newRegistry := NewRegistry()
	if got := ta.SetRegistry(newRegistry); got != ta {
		t.Fatal("expected SetRegistry to return the same typed agent")
	}
	if !newRegistry.Has("__output__") {
		t.Fatal("expected output tool to be re-registered on replacement registry")
	}
	if newRegistry.Has("aux") {
		t.Fatal("expected replacement registry to contain only re-registered output tools")
	}

	model := &testutil.StubModel{
		NameValue: "typed-model",
		Response: &ChatResponse{
			Message:      NewTextMessage(RoleAssistant, "bad"),
			FinishReason: FinishReasonStop,
		},
	}
	textAgent := NewTypedAgentWithMode[int](
		"system",
		model,
		NewTextProcessorOutput(func(text string) (int, error) {
			return 0, errors.New("cannot convert")
		}),
	)

	_, err := textAgent.Run(context.Background(), "prompt")
	if err == nil || !strings.Contains(err.Error(), "text processor: cannot convert") {
		t.Fatalf("expected text processor error, got %v", err)
	}
}

func TestTypedAgentRunFallsBackToToolModeForUnknownOutputMode(t *testing.T) {
	spec := invalidModeOutputSpec{
		ToolOutputSpec: NewToolOutput[typedAgentCoverageValue]("desc"),
	}

	model := &testutil.StubModel{
		NameValue: "typed-model",
		Response: &ChatResponse{
			Message: NewToolUseMessage(ToolUse{
				ID:   "call_1",
				Name: "__output__",
				Input: map[string]interface{}{
					"value": "fallback",
				},
			}),
			FinishReason: FinishReasonToolCalls,
		},
	}

	ta := NewTypedAgentWithMode[typedAgentCoverageValue]("system", model, spec)

	result, err := ta.Run(context.Background(), "prompt")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Output.Value != "fallback" {
		t.Fatalf("expected fallback output, got %#v", result.Output)
	}
}

func TestTypedAgentRunAdditionalErrorPaths(t *testing.T) {
	t.Run("tool output parse error is wrapped", func(t *testing.T) {
		ta := NewTypedAgentWithMode[typedAgentCoverageValue]("system", &testutil.StubModel{
			NameValue: "typed-model",
			Response: &ChatResponse{
				Message:      NewTextMessage(RoleAssistant, "plain text"),
				FinishReason: FinishReasonStop,
			},
		}, failingOutputSpec{err: errors.New("bad parse")}, WithMaxValidationRetries(0))

		_, err := ta.Run(context.Background(), "prompt")
		if err == nil || err.Error() != "output validation failed after 0 retries: bad parse" {
			t.Fatalf("expected validation error from shared fold, got %v", err)
		}
	})

	t.Run("response format parse error is wrapped", func(t *testing.T) {
		ta := NewTypedAgentWithMode[typedAgentCoverageValue]("system", &testutil.StubModel{
			NameValue: "typed-model",
			Response: &ChatResponse{
				Message:      NewTextMessage(RoleAssistant, `{}`),
				FinishReason: FinishReasonStop,
			},
		}, failingResponseFormatSpec{err: errors.New("bad parse")}, WithMaxValidationRetries(0))

		_, err := ta.Run(context.Background(), "prompt")
		if err == nil || err.Error() != "output validation failed after 0 retries: bad parse" {
			t.Fatalf("expected validation error from shared fold, got %v", err)
		}
	})

	t.Run("response format model error is returned", func(t *testing.T) {
		ta := NewTypedAgentWithMode[typedAgentCoverageValue]("system", &testutil.StubModel{
			NameValue: "typed-model",
			Err:       errors.New("boom"),
		}, failingResponseFormatSpec{err: errors.New("unused")})

		_, err := ta.Run(context.Background(), "prompt")
		if err == nil || err.Error() != "model request: boom" {
			t.Fatalf("expected model error, got %v", err)
		}
	})

	t.Run("text processor model error is returned", func(t *testing.T) {
		ta := NewTypedAgentWithMode[int](
			"system",
			&testutil.StubModel{
				NameValue: "typed-model",
				Err:       errors.New("boom"),
			},
			NewTextProcessorOutput(func(text string) (int, error) {
				return 0, nil
			}),
		)

		_, err := ta.Run(context.Background(), "prompt")
		if err == nil || err.Error() != "model request: boom" {
			t.Fatalf("expected model error, got %v", err)
		}
	})
}
