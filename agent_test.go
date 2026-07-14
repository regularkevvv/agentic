package agentic_test

import (
	"context"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestAgentSimpleRun(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "Hello!"})
	agent := agentic.NewAgent("You are helpful", model)

	result, err := agent.Run(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "Hello!" {
		t.Errorf("expected output %q, got %q", "Hello!", result.Output)
	}
	if result.FinishReason != agentic.FinishReasonStop {
		t.Errorf("expected finish reason %q, got %q", agentic.FinishReasonStop, result.FinishReason)
	}
	if model.CallCount() != 1 {
		t.Errorf("expected 1 call, got %d", model.CallCount())
	}
}

func TestAgentWithTools(t *testing.T) {
	type WeatherInput struct {
		Location string `json:"location" description:"City name"`
	}
	type WeatherOutput struct {
		Temperature float64 `json:"temperature"`
	}

	tool, handler := agentic.MustToolPlain("get_weather", "Get weather", func(input WeatherInput) (WeatherOutput, error) {
		return WeatherOutput{Temperature: 72.0}, nil
	})

	// First response: tool call, second response: final text
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "call_1", Name: "get_weather", Input: map[string]interface{}{"location": "Tokyo"}},
			},
		},
		test.ModelResponse{Text: "The weather in Tokyo is 72F"},
	)

	agent := agentic.NewAgent("You are a weather assistant", model).
		AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "What's the weather in Tokyo?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "The weather in Tokyo is 72F" {
		t.Errorf("expected weather response, got %q", result.Output)
	}
	if len(result.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(result.ToolCalls))
	}
	if model.CallCount() != 2 {
		t.Errorf("expected 2 model calls, got %d", model.CallCount())
	}
}

func TestAgentWithDeps(t *testing.T) {
	type MyDeps struct {
		Counter int
	}
	type CountInput struct {
		Amount int `json:"amount" description:"Amount to add"`
	}
	type CountOutput struct {
		Total int `json:"total"`
	}

	tool, handler := agentic.MustToolWithDeps("add_count", "Add to counter",
		func(ctx agentic.RunContext[*MyDeps], input CountInput) (CountOutput, error) {
			ctx.Deps.Counter += input.Amount
			return CountOutput{Total: ctx.Deps.Counter}, nil
		},
	)

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "call_1", Name: "add_count", Input: map[string]interface{}{"amount": float64(5)}},
			},
		},
		test.ModelResponse{Text: "Counter is now 5"},
	)

	agent := agentic.NewAgentWithDeps[*MyDeps]("You count things", model).
		AddTool(tool, handler)

	deps := &MyDeps{Counter: 0}
	result, err := agent.Run(context.Background(), "Add 5", deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "Counter is now 5" {
		t.Errorf("expected counter response, got %q", result.Output)
	}
	if deps.Counter != 5 {
		t.Errorf("expected counter to be 5, got %d", deps.Counter)
	}
}

func TestAgentMaxIterations(t *testing.T) {
	// Model always returns tool calls — should hit max iterations
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "call_1", Name: "noop", Input: map[string]interface{}{}},
			},
		},
	)

	type NoopInput struct{}
	type NoopOutput struct{ OK bool }

	tool, handler := agentic.MustToolPlain("noop", "Does nothing", func(input NoopInput) (NoopOutput, error) {
		return NoopOutput{OK: true}, nil
	})

	agent := agentic.NewAgent("test", model,
		agentic.WithMaxIterations(3),
	).AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "loop forever")
	if err == nil {
		t.Fatal("expected MaxIterationsError")
	}

	var maxIterErr *agentic.MaxIterationsError
	if !isMaxIterError(err, &maxIterErr) {
		t.Fatalf("expected MaxIterationsError, got %T: %v", err, err)
	}

	if result == nil {
		t.Fatal("expected result even on max iterations")
	}
}

func isMaxIterError(err error, target **agentic.MaxIterationsError) bool {
	switch e := err.(type) {
	case *agentic.MaxIterationsError:
		*target = e
		return true
	default:
		return false
	}
}

func TestAgentDynamicPrompt(t *testing.T) {
	type MyDeps struct {
		UserName string
	}

	model := test.NewTestModel(test.ModelResponse{Text: "Hi Kevin!"})

	agent := agentic.NewAgentWithDepsDynamic[*MyDeps](
		func(ctx agentic.RunContext[*MyDeps]) (string, error) {
			return "You are helping " + ctx.Deps.UserName, nil
		},
		model,
	)

	deps := &MyDeps{UserName: "Kevin"}
	result, err := agent.Run(context.Background(), "Hello", deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "Hi Kevin!" {
		t.Errorf("expected %q, got %q", "Hi Kevin!", result.Output)
	}

	// Verify the system prompt was included
	calls := model.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if len(calls[0].Messages) < 1 {
		t.Fatal("expected at least 1 message")
	}
	firstMsg := calls[0].Messages[0]
	if firstMsg.Role != agentic.RoleSystem {
		t.Errorf("expected first message to be system, got %q", firstMsg.Role)
	}
	if firstMsg.GetTextContent() != "You are helping Kevin" {
		t.Errorf("expected dynamic prompt, got %q", firstMsg.GetTextContent())
	}
}

func TestAgentModelRetry(t *testing.T) {
	callCount := 0
	type Input struct {
		Query string `json:"query"`
	}
	type Output struct {
		Result string `json:"result"`
	}

	tool, handler := agentic.MustToolPlain("search", "Search", func(input Input) (Output, error) {
		callCount++
		if callCount == 1 {
			return Output{}, agentic.Retry("No results, try different query")
		}
		return Output{Result: "found it"}, nil
	})

	model := test.NewTestModel(
		// First call: tool call
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "call_1", Name: "search", Input: map[string]interface{}{"query": "bad query"}},
			},
		},
		// After retry error, model tries again
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "call_2", Name: "search", Input: map[string]interface{}{"query": "better query"}},
			},
		},
		// Final response
		test.ModelResponse{Text: "Found the answer!"},
	)

	agent := agentic.NewAgent("search assistant", model,
		agentic.WithRetries(agentic.RetryConfig{MaxRetries: 2}),
	).AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "Find something")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "Found the answer!" {
		t.Errorf("expected final answer, got %q", result.Output)
	}
	if result.Retries != 1 {
		t.Errorf("expected 1 retry, got %d", result.Retries)
	}
}

func TestAgentUsageAccumulation(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "call_1", Name: "noop", Input: map[string]interface{}{}},
			},
		},
		test.ModelResponse{Text: "done"},
	)

	type NoopInput struct{}
	type NoopOutput struct{}

	tool, handler := agentic.MustToolPlain("noop", "noop", func(input NoopInput) (NoopOutput, error) {
		return NoopOutput{}, nil
	})

	agent := agentic.NewAgent("test", model).AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// TestModel returns 15 total tokens per call, 2 calls = 30
	if result.Usage.TotalTokens != 30 {
		t.Errorf("expected 30 total tokens, got %d", result.Usage.TotalTokens)
	}
}
