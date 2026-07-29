package repair

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

func callMessage(calls ...agentic.ToolUse) agentic.Message {
	return agentic.NewToolUseMessage(calls...)
}

func TestRepairIsIdempotentByteStableAndNonMutating(t *testing.T) {
	t.Parallel()
	call := agentic.ToolUse{ID: "call-1", Name: "write", Input: map[string]any{"path": "file"}}
	history := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "do it"),
		{
			Role: agentic.RoleAssistant,
			Content: []agentic.Part{
				{Type: agentic.ContentThinking, Thinking: &agentic.ThinkingBlock{Text: "done", ID: "thinking", Signature: "sig", ProviderName: "provider"}},
				{Type: agentic.ContentToolUse, ToolUse: &call},
			},
		},
	}
	before, _ := json.Marshal(history)
	processor := Repair(CloseInterruptedFrontier, PendingCalls{Calls: []PendingCall{{ID: call.ID, State: PendingIndeterminate}}})
	first, err := processor.Process(context.Background(), history)
	if err != nil {
		t.Fatal(err)
	}
	second, err := processor.Process(context.Background(), first)
	if err != nil {
		t.Fatal(err)
	}
	firstBytes, _ := json.Marshal(first)
	secondBytes, _ := json.Marshal(second)
	after, _ := json.Marshal(history)
	if string(firstBytes) != string(secondBytes) {
		t.Fatalf("repair not byte stable:\n%s\n%s", firstBytes, secondBytes)
	}
	if string(before) != string(after) {
		t.Fatal("repair mutated input")
	}
	if len(first) != 3 || first[2].Role != agentic.RoleTool {
		t.Fatalf("repaired history = %#v", first)
	}
	result := first[2].GetToolResults()[0]
	if result.ToolUseID != call.ID || result.Name != call.Name || !result.IsError {
		t.Fatalf("synthetic result = %#v", result)
	}
	thinking := first[1].Content[0].Thinking
	if thinking.ID != "thinking" || thinking.Signature != "sig" || thinking.ProviderName != "provider" {
		t.Fatalf("thinking metadata changed: %#v", thinking)
	}
}

func TestRepairPreservesPairIntegrityAndRemovesOnlyOrphans(t *testing.T) {
	t.Parallel()
	one := agentic.ToolUse{ID: "one", Name: "tool"}
	two := agentic.ToolUse{ID: "two", Name: "tool"}
	history := []agentic.Message{
		agentic.NewToolResultMessageFor("orphan", "tool", "old", false),
		callMessage(one, two),
		agentic.NewToolResultMessageFor(one.ID, one.Name, "ok", false),
	}
	repaired, err := Process(history, CloseInterruptedFrontier, PendingCalls{})
	if err != nil {
		t.Fatal(err)
	}
	if len(repaired) != 3 {
		t.Fatalf("messages = %#v", repaired)
	}
	if repaired[0].Role != agentic.RoleAssistant {
		t.Fatalf("orphan was retained: %#v", repaired[0])
	}
	if got := repaired[1].GetToolResults()[0].ToolUseID; got != one.ID {
		t.Fatalf("first result = %q", got)
	}
	if got := repaired[2].GetToolResults()[0].ToolUseID; got != two.ID {
		t.Fatalf("synthetic result = %q", got)
	}
	if open, err := InspectFrontier(repaired); err != nil || len(open) != 0 {
		t.Fatalf("open frontier = %#v, %v", open, err)
	}
}

func TestRepairRejectsAmbiguousDuplicateFrontiers(t *testing.T) {
	t.Parallel()
	call := agentic.ToolUse{ID: "duplicate", Name: "tool"}
	cases := map[string][]agentic.Message{
		"duplicate calls":   {callMessage(call, call)},
		"duplicate results": {callMessage(call), agentic.NewToolResultMessageFor(call.ID, call.Name, "one", false), agentic.NewToolResultMessageFor(call.ID, call.Name, "two", false)},
		"wrong result name": {callMessage(call), agentic.NewToolResultMessageFor(call.ID, "other", "one", false)},
	}
	for name, history := range cases {
		history := history
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if _, err := Process(history, CloseInterruptedFrontier, PendingCalls{}); !errors.Is(err, agentic.ErrTranscriptInvalid) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestPreserveDeferredFrontierRequiresExactPendingSetAndOmitsIt(t *testing.T) {
	t.Parallel()
	one := agentic.ToolUse{ID: "one", Name: "first"}
	two := agentic.ToolUse{ID: "two", Name: "second"}
	history := []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "run"), callMessage(one, two)}
	pending := PendingCalls{Calls: []PendingCall{{ID: one.ID, Name: one.Name}, {ID: two.ID, Name: two.Name}}}
	projected, err := Process(history, PreserveDeferredFrontier, pending)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(projected, history[:1]) {
		t.Fatalf("projection = %#v", projected)
	}
	wrong := PendingCalls{Calls: []PendingCall{{ID: two.ID}, {ID: one.ID}}}
	if _, err := Process(history, PreserveDeferredFrontier, wrong); !errors.Is(err, agentic.ErrTranscriptInvalid) {
		t.Fatalf("wrong pending error = %v", err)
	}
}

func TestCompactionThenRepairPreservesOpenCallSet(t *testing.T) {
	t.Parallel()
	old := agentic.ToolUse{ID: "reused", Name: "old"}
	open := agentic.ToolUse{ID: "reused", Name: "new"}
	full := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "old"),
		callMessage(old),
		agentic.NewToolResultMessageFor(old.ID, old.Name, "done", false),
		agentic.NewTextMessage(agentic.RoleUser, "new"),
		callMessage(open),
	}
	compacted := append([]agentic.Message(nil), full[3:]...)
	before, err := InspectFrontier(compacted)
	if err != nil {
		t.Fatal(err)
	}
	repaired, err := Process(compacted, CloseInterruptedFrontier, PendingCalls{})
	if err != nil {
		t.Fatal(err)
	}
	if len(before) != 1 || before[0].ID != open.ID {
		t.Fatalf("before open = %#v", before)
	}
	if after, err := InspectFrontier(repaired); err != nil || len(after) != 0 {
		t.Fatalf("after open = %#v, %v", after, err)
	}
	result := repaired[len(repaired)-1].GetToolResults()[0]
	if result.Name != open.Name {
		t.Fatalf("paired against wrong reused frontier: %#v", result)
	}
}
