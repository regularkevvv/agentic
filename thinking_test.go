package agentic_test

import (
	"context"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestThinkingConfig_AgentOption(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{Text: "answer"},
	)

	agent := agentic.NewAgent("prompt", model,
		agentic.WithThinking(agentic.ThinkingConfig{
			Enabled:      true,
			BudgetTokens: 10000,
		}),
	)

	result, err := agent.Run(context.Background(), "think about this")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Output != "answer" {
		t.Errorf("expected 'answer', got %q", result.Output)
	}

	// Verify the thinking config was passed in the request
	calls := model.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Thinking == nil {
		t.Fatal("expected thinking config in request")
	}
	if !calls[0].Thinking.Enabled {
		t.Error("expected thinking to be enabled")
	}
	if calls[0].Thinking.BudgetTokens != 10000 {
		t.Errorf("expected 10000 budget tokens, got %d", calls[0].Thinking.BudgetTokens)
	}
}

func TestThinkingConfig_NotSet(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{Text: "answer"},
	)

	agent := agentic.NewAgent("prompt", model)

	_, err := agent.Run(context.Background(), "question")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	calls := model.Calls()
	if calls[0].Thinking != nil {
		t.Error("expected nil thinking config when not set")
	}
}

func TestThinkingBlock_Message(t *testing.T) {
	msg := agentic.Message{
		Role: agentic.RoleAssistant,
		Content: []agentic.Part{
			{
				Type:     agentic.ContentThinking,
				Thinking: &agentic.ThinkingBlock{Text: "reasoning..."},
			},
			{
				Type: agentic.ContentText,
				Text: "The answer is 42",
			},
		},
	}

	thinking := msg.GetThinkingContent()
	if thinking != "reasoning..." {
		t.Errorf("expected 'reasoning...', got %q", thinking)
	}

	text := msg.GetTextContent()
	if text != "The answer is 42" {
		t.Errorf("expected 'The answer is 42', got %q", text)
	}
}

func TestStreamEventThinkingDelta(t *testing.T) {
	// Verify the constant exists and has a distinct value
	if agentic.StreamEventThinkingDelta == agentic.StreamEventTextDelta {
		t.Error("StreamEventThinkingDelta should not equal StreamEventTextDelta")
	}
	if agentic.StreamEventThinkingDelta == agentic.StreamEventDone {
		t.Error("StreamEventThinkingDelta should not equal StreamEventDone")
	}
}

func TestThinkingBlock_FullFields(t *testing.T) {
	block := &agentic.ThinkingBlock{
		Text:         "deep reasoning here",
		ID:           "think_001",
		Signature:    "sig_abc",
		ProviderName: "anthropic",
		ProviderDetails: map[string]interface{}{
			"model_version": "3.5",
		},
	}

	if block.IsRedacted() {
		t.Error("non-redacted block should not be redacted")
	}
	if block.Text != "deep reasoning here" {
		t.Errorf("unexpected text: %q", block.Text)
	}
	if block.ProviderName != "anthropic" {
		t.Errorf("unexpected provider: %q", block.ProviderName)
	}
	if block.ProviderDetails["model_version"] != "3.5" {
		t.Error("provider details not preserved")
	}
}

func TestThinkingBlock_Redacted(t *testing.T) {
	block := &agentic.ThinkingBlock{
		Text:         "",
		ID:           "redacted_thinking",
		Signature:    "encrypted_data_here",
		ProviderName: "anthropic",
	}

	if !block.IsRedacted() {
		t.Error("redacted block should be redacted")
	}
	if block.Signature != "encrypted_data_here" {
		t.Errorf("signature not preserved: %q", block.Signature)
	}
}

func TestThinkingBlock_SerializeRoundtrip(t *testing.T) {
	msg := agentic.Message{
		Role: agentic.RoleAssistant,
		Content: []agentic.Part{
			{
				Type: agentic.ContentThinking,
				Thinking: &agentic.ThinkingBlock{
					Text:         "",
					ID:           "redacted_thinking",
					Signature:    "sig_data",
					ProviderName: "anthropic",
				},
			},
			{
				Type: agentic.ContentThinking,
				Thinking: &agentic.ThinkingBlock{
					Text:         "actual thinking",
					Signature:    "sig_real",
					ProviderName: "anthropic",
				},
			},
			{
				Type: agentic.ContentText,
				Text: "final answer",
			},
		},
	}

	history := agentic.NewHistory(msg)
	data, err := history.SaveJSON()
	if err != nil {
		t.Fatalf("SaveJSON: %v", err)
	}

	loaded, err := agentic.LoadHistory(data)
	if err != nil {
		t.Fatalf("LoadHistory: %v", err)
	}

	parts := loaded.Messages[0].Content
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts, got %d", len(parts))
	}

	// Redacted block
	if !parts[0].Thinking.IsRedacted() {
		t.Error("first thinking block should be redacted")
	}
	if parts[0].Thinking.Signature != "sig_data" {
		t.Error("redacted signature not preserved")
	}

	// Normal thinking block
	if parts[1].Thinking.IsRedacted() {
		t.Error("second thinking block should not be redacted")
	}
	if parts[1].Thinking.Text != "actual thinking" {
		t.Error("thinking text not preserved")
	}
	if parts[1].Thinking.ProviderName != "anthropic" {
		t.Error("provider name not preserved")
	}
}
