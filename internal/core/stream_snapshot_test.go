package core

import "testing"

func TestStreamResultSnapshotIsOptionalAndDefensive(t *testing.T) {
	ch := make(chan StreamEvent)
	close(ch)
	stream := NewStreamResult(ch)
	if _, ok := stream.Snapshot(); ok {
		t.Fatal("new provider stream unexpectedly has a snapshot")
	}

	stream.SetSnapshot(ExecutionSnapshot{
		Status:      ExecutionSuspended,
		Messages:    []Message{NewTextMessage(RoleUser, "prompt")},
		ToolCalls:   []ToolUse{{ID: "call", Name: "tool", Input: map[string]interface{}{"value": "x"}}},
		ToolResults: []ToolExecutionResult{{ToolUseID: "call", ToolName: "tool", Content: "result"}},
		Usage:       Usage{TotalTokens: 3},
		Suspension:  &Suspension{ID: "s", Kind: "approval", FrontierHash: "hash", Payload: []byte("payload")},
	})

	snapshot, ok := stream.Snapshot()
	if !ok || snapshot.Status != ExecutionSuspended || snapshot.Suspension == nil {
		t.Fatalf("snapshot = %#v, %v", snapshot, ok)
	}
	snapshot.Messages[0].Content[0].Text = "mutated"
	snapshot.ToolCalls[0].Name = "mutated"
	snapshot.ToolResults[0].ToolName = "mutated"
	snapshot.Suspension.Payload[0] = 'X'

	again, ok := stream.Snapshot()
	if !ok || again.Messages[0].GetTextContent() != "prompt" || again.ToolCalls[0].Name != "tool" || again.ToolResults[0].ToolName != "tool" || string(again.Suspension.Payload) != "payload" {
		t.Fatalf("snapshot was not defensive: %#v", again)
	}
}
