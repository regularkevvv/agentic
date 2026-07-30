package repair

import (
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

func TestProcessRejectsRemainingInvalidFrontierShapes(t *testing.T) {
	call := agentic.ToolUse{ID: "call", Name: "tool"}
	usePart := agentic.NewToolUseMessage(call).Content[0]
	resultPart := agentic.NewToolResultMessageFor(call.ID, call.Name, "done", false).Content[0]
	cases := map[string][]agentic.Message{
		"call in user message": {
			{Role: agentic.RoleUser, Content: []agentic.Part{usePart}},
		},
		"call and result together": {
			{Role: agentic.RoleAssistant, Content: []agentic.Part{usePart, resultPart}},
		},
		"empty call ID": {
			agentic.NewToolUseMessage(agentic.ToolUse{Name: "tool"}),
		},
		"result in assistant message": {
			agentic.NewToolUseMessage(call),
			{Role: agentic.RoleAssistant, Content: []agentic.Part{resultPart}},
		},
		"duplicate result in one message": {
			agentic.NewToolUseMessage(call),
			{Role: agentic.RoleTool, Content: []agentic.Part{resultPart, resultPart}},
		},
		"new preserved frontier": {
			agentic.NewToolUseMessage(call),
			agentic.NewToolUseMessage(agentic.ToolUse{ID: "next", Name: "tool"}),
		},
		"ordinary message after preserved frontier": {
			agentic.NewToolUseMessage(call),
			agentic.NewTextMessage(agentic.RoleUser, "next"),
		},
	}
	for name, history := range cases {
		t.Run(name, func(t *testing.T) {
			mode := CloseInterruptedFrontier
			pending := PendingCalls{}
			if strings.Contains(name, "preserved") {
				mode = PreserveDeferredFrontier
				pending = PendingCalls{Calls: []PendingCall{{ID: call.ID, Name: call.Name}}}
			}
			if _, err := Process(history, mode, pending); !errors.Is(err, agentic.ErrTranscriptInvalid) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestProcessClosesPriorFrontierAndPreservesNonResultParts(t *testing.T) {
	first := agentic.ToolUse{ID: "first", Name: "tool"}
	second := agentic.ToolUse{ID: "second", Name: "tool"}
	projected, err := Process([]agentic.Message{
		agentic.NewToolUseMessage(first),
		agentic.NewToolUseMessage(second),
	}, CloseInterruptedFrontier, PendingCalls{})
	if err != nil || len(projected) != 4 ||
		projected[1].GetToolResults()[0].ToolUseID != first.ID ||
		projected[3].GetToolResults()[0].ToolUseID != second.ID {
		t.Fatalf("closed adjacent frontiers = %#v, %v", projected, err)
	}

	textPart := agentic.NewTextMessage(agentic.RoleTool, "metadata").Content[0]
	resultPart := agentic.NewToolResultMessageFor(first.ID, first.Name, "done", false).Content[0]
	projected, err = Process([]agentic.Message{
		agentic.NewToolUseMessage(first),
		{Role: agentic.RoleTool, Content: []agentic.Part{textPart, resultPart}},
	}, CloseInterruptedFrontier, PendingCalls{})
	if err != nil || len(projected) != 2 || len(projected[1].Content) != 2 {
		t.Fatalf("mixed tool result message = %#v, %v", projected, err)
	}

	projected, err = Process([]agentic.Message{
		agentic.NewToolUseMessage(first),
		agentic.NewTextMessage(agentic.RoleUser, "continue"),
	}, CloseInterruptedFrontier, PendingCalls{})
	if err != nil || len(projected) != 3 || projected[1].Role != agentic.RoleTool {
		t.Fatalf("ordinary message close = %#v, %v", projected, err)
	}
}

func TestInspectRawFrontierAndSyntheticStateEdges(t *testing.T) {
	call := agentic.ToolUse{ID: "call", Name: "tool"}
	invalid := [][]agentic.Message{
		{
			agentic.NewToolUseMessage(call),
			agentic.NewToolUseMessage(agentic.ToolUse{ID: "next", Name: "tool"}),
		},
		{agentic.NewToolUseMessage(agentic.ToolUse{Name: "tool"})},
		{
			agentic.NewToolUseMessage(call),
			{Role: agentic.RoleTool, Content: []agentic.Part{
				agentic.NewToolResultMessageFor(call.ID, call.Name, "one", false).Content[0],
				agentic.NewToolResultMessageFor(call.ID, call.Name, "two", false).Content[0],
			}},
		},
		{
			agentic.NewToolUseMessage(call),
			agentic.NewToolResultMessageFor(call.ID, "other", "done", false),
		},
	}
	for index, history := range invalid {
		if _, err := rawFinalFrontier(history); !errors.Is(err, agentic.ErrTranscriptInvalid) {
			t.Fatalf("raw frontier %d error = %v", index, err)
		}
	}
	if _, err := InspectFrontier(invalid[0]); !errors.Is(err, agentic.ErrTranscriptInvalid) {
		t.Fatalf("inspect error = %v", err)
	}
	open, err := rawFinalFrontier([]agentic.Message{
		agentic.NewToolResultMessageFor("orphan", "tool", "ignored", false),
	})
	if err != nil || open != nil {
		t.Fatalf("orphan-only frontier = %#v, %v", open, err)
	}
	open, err = rawFinalFrontier([]agentic.Message{
		agentic.NewToolUseMessage(call),
		agentic.NewToolResultMessageFor("orphan", "tool", "ignored", false),
	})
	if err != nil || len(open) != 1 || open[0].ID != call.ID {
		t.Fatalf("orphan within active frontier = %#v, %v", open, err)
	}

	planned := syntheticResult(call, PendingCalls{
		Calls: []PendingCall{{ID: call.ID, State: PendingPlanned}},
	})
	if !strings.Contains(planned.GetToolResults()[0].Content, "abandoned before it started") {
		t.Fatalf("planned synthetic result = %#v", planned)
	}
	custom := syntheticResult(call, PendingCalls{Reason: "custom"})
	if !strings.Contains(custom.GetToolResults()[0].Content, "custom") {
		t.Fatalf("custom synthetic result = %#v", custom)
	}
	if samePending([]agentic.ToolUse{call}, nil) {
		t.Fatal("different pending lengths compared equal")
	}
	if samePending([]agentic.ToolUse{call}, []PendingCall{{ID: call.ID, Name: "other"}}) {
		t.Fatal("different pending names compared equal")
	}
}

func TestCloneTranscriptNilAndEncodingFailure(t *testing.T) {
	if cloned, err := cloneMessages(nil); err != nil || cloned != nil {
		t.Fatalf("nil clone = %#v, %v", cloned, err)
	}
	bad := agentic.NewToolUseMessage(agentic.ToolUse{
		ID: "bad", Name: "tool", Input: map[string]any{"channel": make(chan int)},
	})
	if _, err := Process([]agentic.Message{bad}, CloseInterruptedFrontier, PendingCalls{}); err == nil {
		t.Fatal("unencodable transcript was repaired")
	}
}
