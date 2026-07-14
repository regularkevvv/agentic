package agentic_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

type MovieReview struct {
	Title   string  `json:"title" description:"Movie title"`
	Rating  float64 `json:"rating" description:"Rating from 1-10"`
	Summary string  `json:"summary" description:"Brief review"`
}

func TestTypedAgentBasic(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{
					ID:   "call_1",
					Name: "__output__",
					Input: map[string]interface{}{
						"title":   "The Matrix",
						"rating":  float64(9),
						"summary": "A mind-bending sci-fi classic",
					},
				},
			},
		},
	)

	agent := agentic.NewTypedAgent[MovieReview](
		"You review movies.",
		model,
		"Provide a structured movie review",
	)

	result, err := agent.Run(context.Background(), "Review The Matrix")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Title != "The Matrix" {
		t.Errorf("expected title %q, got %q", "The Matrix", result.Output.Title)
	}
	if result.Output.Rating != 9 {
		t.Errorf("expected rating 9, got %f", result.Output.Rating)
	}
	if result.Output.Summary != "A mind-bending sci-fi classic" {
		t.Errorf("expected summary, got %q", result.Output.Summary)
	}
}

func TestTypedAgentWithTools(t *testing.T) {
	type LookupInput struct {
		Query string `json:"query" description:"Search query"`
	}
	type LookupOutput struct {
		Info string `json:"info"`
	}
	type Summary struct {
		Answer string `json:"answer" description:"Final answer"`
	}

	tool, handler := agentic.MustToolPlain("lookup", "Look up information",
		func(input LookupInput) (LookupOutput, error) {
			return LookupOutput{Info: "The Matrix was released in 1999"}, nil
		},
	)

	model := test.NewTestModel(
		// First: model calls the lookup tool
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "lookup", Input: map[string]interface{}{"query": "Matrix release year"}},
			},
		},
		// Second: model calls the output tool with the answer
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{
					ID:   "c2",
					Name: "__output__",
					Input: map[string]interface{}{
						"answer": "The Matrix was released in 1999",
					},
				},
			},
		},
	)

	agent := agentic.NewTypedAgent[Summary](
		"You answer questions.",
		model,
		"Provide a structured summary",
	).AddTool(tool, handler)

	result, err := agent.Run(context.Background(), "When was The Matrix released?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Answer != "The Matrix was released in 1999" {
		t.Errorf("expected answer about 1999, got %q", result.Output.Answer)
	}
	if len(result.ToolCalls) == 0 {
		t.Error("expected tool calls")
	}
}

func TestTypedAgentDynamic(t *testing.T) {
	type Deps struct {
		UserName string
	}
	type Greeting struct {
		Message string `json:"message" description:"Greeting message"`
	}

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{
					ID:   "c1",
					Name: "__output__",
					Input: map[string]interface{}{
						"message": "Hello Alice!",
					},
				},
			},
		},
	)

	agent := agentic.NewTypedAgentWithDepsDynamic[Greeting, *Deps](
		func(ctx agentic.RunContext[*Deps]) (string, error) {
			return fmt.Sprintf("You greet users. The current user is %s.", ctx.Deps.UserName), nil
		},
		model,
		"Provide a greeting",
	)

	deps := &Deps{UserName: "Alice"}
	result, err := agent.Run(context.Background(), "Say hello", deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Message != "Hello Alice!" {
		t.Errorf("expected greeting, got %q", result.Output.Message)
	}
}

type ValidatedOutput struct {
	Name  string `json:"name" validate:"required,min=1" description:"Name"`
	Score int    `json:"score" validate:"required,min=1,max=100" description:"Score"`
}

func TestTypedAgentValidationRetry(t *testing.T) {
	model := test.NewTestModel(
		// First attempt: invalid (score=0 fails required)
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{
					ID:   "c1",
					Name: "__output__",
					Input: map[string]interface{}{
						"name":  "test",
						"score": float64(0),
					},
				},
			},
		},
		// Second attempt: valid
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{
					ID:   "c2",
					Name: "__output__",
					Input: map[string]interface{}{
						"name":  "test",
						"score": float64(85),
					},
				},
			},
		},
	)

	agent := agentic.NewTypedAgent[ValidatedOutput](
		"You provide scores.",
		model,
		"Provide a name and score",
	)

	result, err := agent.Run(context.Background(), "Score this")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Score != 85 {
		t.Errorf("expected score 85, got %d", result.Output.Score)
	}
}

func TestTypedAgentValidationExhausted(t *testing.T) {
	// Always returns invalid output (score=0)
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "__output__", Input: map[string]interface{}{"name": "x", "score": float64(0)}},
			},
		},
	)

	agent := agentic.NewTypedAgent[ValidatedOutput](
		"You provide scores.",
		model,
		"Provide a name and score",
		agentic.WithMaxValidationRetries(1),
	)

	_, err := agent.Run(context.Background(), "Score this")
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "validation") {
		t.Errorf("expected validation error, got: %v", err)
	}
}

func TestTypedAgentAddAutoTool(t *testing.T) {
	type CalcInput struct {
		_    struct{} `tool:"Calculate a math expression"`
		Expr string   `json:"expression" description:"Math expression"`
	}
	type CalcOutput struct {
		Result float64 `json:"result"`
	}
	type Answer struct {
		Value string `json:"value" description:"Final answer"`
	}

	calcTool, calcHandler := agentic.MustAutoTool(func(input CalcInput) (CalcOutput, error) {
		return CalcOutput{Result: 42}, nil
	})

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "calc", Input: map[string]interface{}{"expression": "6*7"}},
			},
		},
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c2", Name: "__output__", Input: map[string]interface{}{"value": "42"}},
			},
		},
	)

	agent := agentic.NewTypedAgent[Answer](
		"You are a calculator.",
		model,
		"Provide the answer",
	).AddAutoTool(calcTool, calcHandler)

	result, err := agent.Run(context.Background(), "What is 6*7?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Value != "42" {
		t.Errorf("expected '42', got %q", result.Output.Value)
	}
}
