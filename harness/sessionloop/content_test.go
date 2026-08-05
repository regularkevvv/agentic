package sessionloop_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func TestValidateInputEnforcesPerKindBlockRules(t *testing.T) {
	t.Parallel()
	toolCall := &sessionloop.ToolCall{CallID: "call-1", Name: "lookup", Data: json.RawMessage(`{"q":1}`)}
	toolResult := &sessionloop.ToolResult{CallID: "call-1", Name: "lookup"}

	valid := []sessionloop.Input{
		{Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "hello"}}},
		{Content: []sessionloop.Block{{Kind: sessionloop.BlockData, MediaType: "application/json", Data: json.RawMessage(`{"a":1}`)}}},
		{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolCall, ToolCall: toolCall}}},
		{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolResult, Text: "done", ToolResult: toolResult}}},
	}
	for index, input := range valid {
		if err := sessionloop.ValidateInput(input); err != nil {
			t.Errorf("valid input %d rejected: %v", index, err)
		}
	}

	invalid := []struct {
		name  string
		input sessionloop.Input
	}{
		{"no blocks", sessionloop.Input{}},
		{"text without text", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockText}}}},
		{"text with tool payload", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "x", ToolCall: toolCall}}}},
		{"data without data", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockData}}}},
		{"data with tool payload", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockData, Data: json.RawMessage(`1`), ToolResult: toolResult}}}},
		{"truncated JSON data", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockData, Data: json.RawMessage(`{"a"`)}}}},
		{"two JSON values in one block", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockData, Data: json.RawMessage(`{} {}`)}}}},
		{"tool_call without payload", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolCall}}}},
		{"tool_call without identity", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolCall, ToolCall: &sessionloop.ToolCall{}}}}},
		{"tool_call with invalid data", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolCall, ToolCall: &sessionloop.ToolCall{CallID: "c", Name: "n", Data: json.RawMessage(`{`)}}}}},
		{"tool_call with a result attached", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolCall, ToolCall: toolCall, ToolResult: toolResult}}}},
		{"tool_result without payload", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolResult}}}},
		{"tool_result without call ID", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolResult, ToolResult: &sessionloop.ToolResult{}}}}},
		{"tool_result with invalid data", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolResult, ToolResult: &sessionloop.ToolResult{CallID: "c", Data: json.RawMessage(`[`)}}}}},
		{"tool_result with a call attached", sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockToolResult, ToolResult: toolResult, ToolCall: toolCall}}}},
		{"unknown block kind", sessionloop.Input{Content: []sessionloop.Block{{Kind: "hologram"}}}},
	}
	for _, violation := range invalid {
		if err := sessionloop.ValidateInput(violation.input); !errors.Is(err, sessionloop.ErrInvalidCommand) {
			t.Errorf("%s: ValidateInput = %v, want ErrInvalidCommand", violation.name, err)
		}
	}
}

func TestEntryBlockAndInputClonesAreDeeplyIndependent(t *testing.T) {
	t.Parallel()
	entry := sessionloop.Entry{
		ID:        "entry-1",
		SessionID: "session-1",
		RunID:     "run-1",
		CommandID: "cmd-1",
		Position:  sessionloop.Position{Sequence: 3, Token: "tk-3"},
		Role:      sessionloop.RoleAssistant,
		Origin:    sessionloop.OriginAssistant,
		Content: []sessionloop.Block{
			{Kind: sessionloop.BlockText, Text: "keep"},
			{Kind: sessionloop.BlockData, MediaType: "application/json", Data: json.RawMessage(`{"a":1}`)},
			{Kind: sessionloop.BlockToolCall, ToolCall: &sessionloop.ToolCall{CallID: "call-1", Name: "lookup", Data: json.RawMessage(`{"q":1}`)}},
			{Kind: sessionloop.BlockToolResult, Text: "ok", ToolResult: &sessionloop.ToolResult{CallID: "call-1", Name: "lookup", Data: json.RawMessage(`{"r":2}`)}},
		},
	}
	clone := entry.Clone()
	clone.Content[0].Text = "mutated"
	clone.Content[1].Data[2] = 'X'
	clone.Content[2].ToolCall.Name = "mutated"
	clone.Content[2].ToolCall.Data[2] = 'X'
	clone.Content[3].ToolResult.IsError = true
	clone.Content[3].ToolResult.Data[2] = 'X'
	if entry.Content[0].Text != "keep" ||
		string(entry.Content[1].Data) != `{"a":1}` ||
		entry.Content[2].ToolCall.Name != "lookup" ||
		string(entry.Content[2].ToolCall.Data) != `{"q":1}` ||
		entry.Content[3].ToolResult.IsError ||
		string(entry.Content[3].ToolResult.Data) != `{"r":2}` {
		t.Fatalf("mutating the clone leaked into the original entry: %#v", entry)
	}

	input := sessionloop.Input{
		Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "keep"}},
		Meta:    map[string]string{"k": "v"},
	}
	inputClone := input.Clone()
	inputClone.Content[0].Text = "mutated"
	inputClone.Meta["k"] = "mutated"
	if input.Content[0].Text != "keep" || input.Meta["k"] != "v" {
		t.Fatalf("mutating the cloned input leaked into the original: %#v", input)
	}

	var empty sessionloop.Input
	emptyClone := empty.Clone()
	if emptyClone.Content != nil || emptyClone.Meta != nil {
		t.Fatalf("cloning an empty input invented content: %#v", emptyClone)
	}

	bare := sessionloop.Block{Kind: sessionloop.BlockText, Text: "no pointers"}
	bareClone := bare.Clone()
	if bareClone.Data != nil || bareClone.ToolCall != nil || bareClone.ToolResult != nil {
		t.Fatalf("cloning a bare block invented payloads: %#v", bareClone)
	}
}
