package agentic

import "testing"

func TestRunResultNewMessagesNilWhenHistoryConsumesAll(t *testing.T) {
	result := &RunResult{
		Messages: []Message{
			NewTextMessage(RoleUser, "history"),
			NewTextMessage(RoleAssistant, "answer"),
		},
		historyLen: 2,
	}

	if got := result.NewMessages(); got != nil {
		t.Fatalf("expected nil new messages, got %#v", got)
	}
}
