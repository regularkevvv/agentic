package test

import (
	"context"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNewTestModelDefault(t *testing.T) {
	model := NewTestModel()
	if model.Name() != "test:mock" {
		t.Errorf("expected name %q, got %q", "test:mock", model.Name())
	}

	resp, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Message.Content) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(resp.Message.Content))
	}
	if resp.Message.GetTextContent() != "test response" {
		t.Errorf("expected default response, got %q", resp.Message.GetTextContent())
	}
}

func TestNewTestModelWithResponses(t *testing.T) {
	model := NewTestModel(
		ModelResponse{Text: "first"},
		ModelResponse{Text: "second"},
	)

	// First call
	resp1, _ := model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "1")},
	})
	if resp1.Message.GetTextContent() != "first" {
		t.Errorf("expected 'first', got %q", resp1.Message.GetTextContent())
	}

	// Second call
	resp2, _ := model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "2")},
	})
	if resp2.Message.GetTextContent() != "second" {
		t.Errorf("expected 'second', got %q", resp2.Message.GetTextContent())
	}

	// Third call — repeats last
	resp3, _ := model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "3")},
	})
	if resp3.Message.GetTextContent() != "second" {
		t.Errorf("expected 'second' (repeated), got %q", resp3.Message.GetTextContent())
	}
}

func TestTestModelWithToolCalls(t *testing.T) {
	model := NewTestModel(ModelResponse{
		ToolCalls: []core.ToolUse{
			{ID: "c1", Name: "test_tool", Input: map[string]interface{}{"key": "val"}},
		},
	})

	resp, _ := model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
	})

	if resp.FinishReason != core.FinishReasonToolCalls {
		t.Errorf("expected finish reason %q, got %q", core.FinishReasonToolCalls, resp.FinishReason)
	}

	uses := resp.Message.GetToolUses()
	if len(uses) != 1 {
		t.Fatalf("expected 1 tool use, got %d", len(uses))
	}
	if uses[0].Name != "test_tool" {
		t.Errorf("expected name %q, got %q", "test_tool", uses[0].Name)
	}
}

func TestTestModelToolCallAutoID(t *testing.T) {
	model := NewTestModel(ModelResponse{
		ToolCalls: []core.ToolUse{
			{Name: "test_tool", Input: map[string]interface{}{}}, // no ID set
		},
	})

	resp, _ := model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
	})

	uses := resp.Message.GetToolUses()
	if uses[0].ID == "" {
		t.Error("expected auto-generated ID")
	}
}

func TestTestModelCalls(t *testing.T) {
	model := NewTestModel(ModelResponse{Text: "ok"})

	if model.CallCount() != 0 {
		t.Errorf("expected 0 calls, got %d", model.CallCount())
	}

	model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "1")},
	})
	model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "2")},
	})

	if model.CallCount() != 2 {
		t.Errorf("expected 2 calls, got %d", model.CallCount())
	}

	calls := model.Calls()
	if len(calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(calls))
	}
}

func TestTestModelReset(t *testing.T) {
	model := NewTestModel(
		ModelResponse{Text: "first"},
		ModelResponse{Text: "second"},
	)

	model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "1")},
	})

	model.Reset()

	if model.CallCount() != 0 {
		t.Errorf("expected 0 calls after reset, got %d", model.CallCount())
	}

	// Should start from first response again
	resp, _ := model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "1")},
	})
	if resp.Message.GetTextContent() != "first" {
		t.Errorf("expected 'first' after reset, got %q", resp.Message.GetTextContent())
	}
}

func TestTestModelUsage(t *testing.T) {
	model := NewTestModel(ModelResponse{Text: "ok"})

	resp, _ := model.Request(context.Background(), &core.ChatRequest{
		Model:    "test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
	})

	if resp.Usage.PromptTokens != 10 {
		t.Errorf("expected 10 prompt tokens, got %d", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 5 {
		t.Errorf("expected 5 completion tokens, got %d", resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", resp.Usage.TotalTokens)
	}
}
