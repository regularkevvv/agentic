package sessionloop_test

import (
	"encoding/json"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func TestEntryBlockCloneIsDeeplyIndependent(t *testing.T) {
	t.Parallel()
	entry := sessionloop.Entry{
		ID:        "entry-1",
		SessionID: "session-1",
		RunID:     "run-1",
		CommandID: "cmd-1",
		Position:  sessionloop.Position{Sequence: 3, Token: "tk-3"},
		Role:      sessionloop.RoleAssistant,
		Origin:    sessionloop.OriginAssistant,
		Blocks: []sessionloop.EntryBlock{
			{Kind: sessionloop.EntryBlockText, Text: "keep"},
			{Kind: sessionloop.EntryBlockData, MediaType: "application/json", Data: json.RawMessage(`{"a":1}`)},
			{Kind: sessionloop.EntryBlockToolCall, ToolCall: &sessionloop.EntryToolCall{CallID: "call-1", Name: "lookup", Data: json.RawMessage(`{"q":1}`)}},
			{Kind: sessionloop.EntryBlockToolResult, Text: "ok", ToolResult: &sessionloop.EntryToolResult{CallID: "call-1", Name: "lookup", Data: json.RawMessage(`{"r":2}`)}},
		},
	}
	clone := entry.Clone()
	clone.Blocks[0].Text = "mutated"
	clone.Blocks[1].Data[2] = 'X'
	clone.Blocks[2].ToolCall.Name = "mutated"
	clone.Blocks[2].ToolCall.Data[2] = 'X'
	clone.Blocks[3].ToolResult.IsError = true
	clone.Blocks[3].ToolResult.Data[2] = 'X'
	if entry.Blocks[0].Text != "keep" ||
		string(entry.Blocks[1].Data) != `{"a":1}` ||
		entry.Blocks[2].ToolCall.Name != "lookup" ||
		string(entry.Blocks[2].ToolCall.Data) != `{"q":1}` ||
		entry.Blocks[3].ToolResult.IsError ||
		string(entry.Blocks[3].ToolResult.Data) != `{"r":2}` {
		t.Fatalf("mutating the clone leaked into the original entry: %#v", entry)
	}

	bare := sessionloop.EntryBlock{Kind: sessionloop.EntryBlockText, Text: "no pointers"}
	bareClone := bare.Clone()
	if bareClone.Data != nil || bareClone.ToolCall != nil || bareClone.ToolResult != nil {
		t.Fatalf("cloning a bare block invented payloads: %#v", bareClone)
	}
}
