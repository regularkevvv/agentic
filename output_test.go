package agentic_test

import (
	"context"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestTypedAgentStructured(t *testing.T) {
	type MovieReview struct {
		Title   string  `json:"title" description:"Movie title"`
		Rating  float64 `json:"rating" description:"Rating from 1-10"`
		Summary string  `json:"summary" description:"Brief review"`
	}

	// Model responds with a tool call using the output tool
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
		"You review movies",
		model,
		"Provide a structured movie review",
	)

	result, err := agent.Run(
		context.Background(), "Review The Matrix",
	)
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

func TestTypedAgentStructuredFull(t *testing.T) {
	type Result struct {
		Name  string `json:"name" description:"Name"`
		Score int    `json:"score" description:"Score"`
	}

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{
					ID:   "c1",
					Name: "__output__",
					Input: map[string]interface{}{
						"name":  "test",
						"score": float64(100),
					},
				},
			},
		},
	)

	agent := agentic.NewTypedAgent[Result](
		"test",
		model,
		"Provide result",
	)

	result, err := agent.Run(
		context.Background(), "go",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output.Name != "test" {
		t.Errorf("expected name %q, got %q", "test", result.Output.Name)
	}
	if result.Output.Score != 100 {
		t.Errorf("expected score 100, got %d", result.Output.Score)
	}
}

func TestToolOutputSpecParse(t *testing.T) {
	type Result struct {
		Value int    `json:"value"`
		Label string `json:"label"`
	}

	spec := agentic.NewToolOutput[Result]("test output")

	// Test parsing from tool call
	msg := agentic.NewToolUseMessage(agentic.ToolUse{
		ID:   "1",
		Name: "__output__",
		Input: map[string]interface{}{
			"value": float64(42),
			"label": "answer",
		},
	})

	parsed, err := spec.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := parsed
	if result.Value != 42 {
		t.Errorf("expected value 42, got %d", result.Value)
	}
	if result.Label != "answer" {
		t.Errorf("expected label %q, got %q", "answer", result.Label)
	}
}

func TestToolOutputSpecParseFromText(t *testing.T) {
	type Result struct {
		X int `json:"x"`
	}

	spec := agentic.NewToolOutput[Result]("test")

	// When there's no tool call, it should try to parse text as JSON
	msg := agentic.NewTextMessage(agentic.RoleAssistant, `{"x": 5}`)

	parsed, err := spec.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := parsed
	if result.X != 5 {
		t.Errorf("expected x=5, got %d", result.X)
	}
}
