package agentic_test

import (
	"context"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestRunStreamEvents(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "Hello streaming!"})
	agent := agentic.NewAgent("You are helpful", model)

	stream, err := agent.RunStream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Collect events by ranging over the channel
	var events []agentic.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(events) < 2 {
		t.Fatalf("expected at least 2 events, got %d", len(events))
	}

	found := false
	for _, e := range events {
		if e.Type == agentic.StreamEventTextDelta && e.Delta == "Hello streaming!" {
			found = true
		}
	}
	if !found {
		t.Error("expected text delta event with 'Hello streaming!'")
	}

	lastEvent := events[len(events)-1]
	if lastEvent.Type != agentic.StreamEventDone {
		t.Errorf("expected last event to be Done, got %d", lastEvent.Type)
	}
}

func TestRunStreamText(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "Hello streaming!"})
	agent := agentic.NewAgent("You are helpful", model)

	stream, err := agent.RunStream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Use Text() to consume the stream and get accumulated text
	text, err := stream.Text()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if text != "Hello streaming!" {
		t.Errorf("expected %q, got %q", "Hello streaming!", text)
	}
}

func TestRunStreamWithTools(t *testing.T) {
	type Input struct {
		X int `json:"x"`
	}
	type Output struct {
		Y int `json:"y"`
	}

	tool, handler := agentic.MustToolPlain("double", "Double", func(input Input) (Output, error) {
		return Output{Y: input.X * 2}, nil
	})

	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{ID: "c1", Name: "double", Input: map[string]interface{}{"x": float64(5)}},
			},
		},
		test.ModelResponse{Text: "Result is 10"},
	)

	agent := agentic.NewAgent("math helper", model).AddTool(tool, handler)

	stream, err := agent.RunStream(context.Background(), "double 5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text, err := stream.Text()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if text != "Result is 10" {
		t.Errorf("expected %q, got %q", "Result is 10", text)
	}
}

func TestRunStreamError(t *testing.T) {
	// Model always returns tool calls with no registered tools — should eventually error
	model := test.NewTestModel(test.ModelResponse{
		ToolCalls: []agentic.ToolUse{
			{ID: "c1", Name: "unknown", Input: map[string]interface{}{}},
		},
	})

	agent := agentic.NewAgent("test", model, agentic.WithMaxIterations(2))

	stream, err := agent.RunStream(context.Background(), "go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = stream.Wait()
	if err == nil {
		t.Error("expected error from stream")
	}
}
