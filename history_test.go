package agentic_test

import (
	"context"
	"encoding/json"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestMessageHistory_SerializeDeserialize(t *testing.T) {
	history := agentic.NewHistory(
		agentic.NewTextMessage(agentic.RoleUser, "Hello"),
		agentic.NewTextMessage(agentic.RoleAssistant, "Hi there!"),
		agentic.NewTextMessage(agentic.RoleUser, "How are you?"),
	)

	data, err := history.SaveJSON()
	if err != nil {
		t.Fatalf("SaveJSON: %v", err)
	}

	loaded, err := agentic.LoadHistory(data)
	if err != nil {
		t.Fatalf("LoadHistory: %v", err)
	}

	if len(loaded.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(loaded.Messages))
	}
	if loaded.Messages[0].GetTextContent() != "Hello" {
		t.Errorf("expected 'Hello', got %q", loaded.Messages[0].GetTextContent())
	}
	if loaded.Messages[1].GetTextContent() != "Hi there!" {
		t.Errorf("expected 'Hi there!', got %q", loaded.Messages[1].GetTextContent())
	}
}

func TestMessageHistory_AllMessageTypes(t *testing.T) {
	// Test serialization of all message types including tool use/result
	toolUse := agentic.ToolUse{
		ID:    "call_123",
		Name:  "get_weather",
		Input: map[string]interface{}{"city": "SF"},
	}

	history := agentic.NewHistory(
		agentic.NewTextMessage(agentic.RoleSystem, "You are helpful"),
		agentic.NewTextMessage(agentic.RoleUser, "What's the weather?"),
		agentic.NewToolUseMessage(toolUse),
		agentic.NewToolResultMessage("call_123", "Sunny, 72F", false),
		agentic.NewTextMessage(agentic.RoleAssistant, "It's sunny and 72F!"),
	)

	data, err := history.SaveJSON()
	if err != nil {
		t.Fatalf("SaveJSON: %v", err)
	}

	loaded, err := agentic.LoadHistory(data)
	if err != nil {
		t.Fatalf("LoadHistory: %v", err)
	}

	if len(loaded.Messages) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(loaded.Messages))
	}

	// Verify tool use message
	toolUses := loaded.Messages[2].GetToolUses()
	if len(toolUses) != 1 || toolUses[0].Name != "get_weather" {
		t.Errorf("tool use not preserved correctly")
	}

	// Verify tool result message
	toolResults := loaded.Messages[3].GetToolResults()
	if len(toolResults) != 1 || toolResults[0].Content != "Sunny, 72F" {
		t.Errorf("tool result not preserved correctly")
	}
}

func TestMessageHistory_ThinkingBlocks(t *testing.T) {
	// Verify thinking blocks survive serialization
	msg := agentic.Message{
		Role: agentic.RoleAssistant,
		Content: []agentic.Part{
			{
				Type:     agentic.ContentThinking,
				Thinking: &agentic.ThinkingBlock{Text: "Let me think...", Signature: "sig123"},
			},
			{
				Type: agentic.ContentText,
				Text: "The answer is 42.",
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

	thinking := loaded.Messages[0].GetThinkingContent()
	if thinking != "Let me think..." {
		t.Errorf("expected thinking 'Let me think...', got %q", thinking)
	}

	if loaded.Messages[0].Content[0].Thinking.Signature != "sig123" {
		t.Errorf("signature not preserved")
	}
}

func TestRunResult_HistoryMethods(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{Text: "response"},
	)

	agent := agentic.NewAgent("system prompt", model)

	// Run with history
	historyMsgs := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "previous question"),
		agentic.NewTextMessage(agentic.RoleAssistant, "previous answer"),
	}

	result, err := agent.Run(context.Background(), "new question",
		agentic.WithMessages(historyMsgs...),
	)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// AllMessages should include history + system + user + assistant
	allMsgs := result.AllMessages()
	if len(allMsgs) < 4 {
		t.Fatalf("expected at least 4 messages in AllMessages, got %d", len(allMsgs))
	}

	// NewMessages should exclude the 2 history messages
	newMsgs := result.NewMessages()
	if len(newMsgs) < 2 {
		t.Fatalf("expected at least 2 new messages, got %d", len(newMsgs))
	}

	// History() should return a serializable history
	h := result.History()
	if len(h.Messages) != len(allMsgs) {
		t.Errorf("History().Messages length mismatch: %d vs %d", len(h.Messages), len(allMsgs))
	}
}

func TestRunResult_ResumeConversation(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{Text: "first response"},
		test.ModelResponse{Text: "second response"},
	)

	agent := agentic.NewAgent("system prompt", model)

	// First run
	result1, err := agent.Run(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Run 1: %v", err)
	}
	if result1.Output != "first response" {
		t.Fatalf("expected 'first response', got %q", result1.Output)
	}

	// Resume with history from first run
	history := result1.History()
	result2, err := agent.Run(context.Background(), "follow up",
		history.ToRunOption(),
	)
	if err != nil {
		t.Fatalf("Run 2: %v", err)
	}
	if result2.Output != "second response" {
		t.Fatalf("expected 'second response', got %q", result2.Output)
	}

	// Verify the second run included the first conversation
	allMsgs := result2.AllMessages()
	// Should have: system + user("hello") + assistant("first") + user("follow up") + assistant("second")
	// but system prompt only added once, etc.
	found := false
	for _, msg := range allMsgs {
		if msg.GetTextContent() == "first response" {
			found = true
			break
		}
	}
	if !found {
		t.Error("history from first run not found in second run's messages")
	}
}

func TestLoadHistory_InvalidJSON(t *testing.T) {
	_, err := agentic.LoadHistory([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestMessageHistory_MultiModal(t *testing.T) {
	// Verify multi-modal parts survive serialization
	msg := agentic.NewMultiPartMessage(
		agentic.TextPart("Look at this"),
		agentic.ImageURLPart("https://example.com/img.png", "high"),
		agentic.ImageDataPart([]byte{0x89, 0x50}, "image/png"),
	)

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

	if parts[1].Type != agentic.ContentImageURL || parts[1].ImageURL.URL != "https://example.com/img.png" {
		t.Errorf("image URL part not preserved")
	}

	if parts[2].Type != agentic.ContentImageData || parts[2].ImageData.MediaType != "image/png" {
		t.Errorf("image data part not preserved")
	}
}

func TestMessageHistory_SaveJSON_Roundtrip(t *testing.T) {
	// Ensure JSON round-trip produces identical data
	original := agentic.NewHistory(
		agentic.NewTextMessage(agentic.RoleUser, "test"),
	)

	data1, _ := original.SaveJSON()
	loaded, _ := agentic.LoadHistory(data1)
	data2, _ := loaded.SaveJSON()

	if !json.Valid(data1) || !json.Valid(data2) {
		t.Error("invalid JSON produced")
	}

	if string(data1) != string(data2) {
		t.Error("round-trip JSON mismatch")
	}
}
