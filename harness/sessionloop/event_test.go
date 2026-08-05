package sessionloop_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func fullEvent() sessionloop.Event {
	return sessionloop.Event{
		Position:  sessionloop.Position{Sequence: 9, Token: "tk-9"},
		Ordinal:   9,
		Nature:    sessionloop.EventAuthoritative,
		Kind:      sessionloop.EventEntryCommitted,
		SessionID: "session-1",
		RunID:     "run-1",
		CommandID: "cmd-1",
		State:     sessionloop.StateRunning,
		Entry: &sessionloop.Entry{
			ID:      "entry-1",
			Role:    sessionloop.RoleUser,
			Origin:  sessionloop.OriginStart,
			Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "keep"}},
		},
		Queue: &sessionloop.QueuedInput{
			ID:        "queue-1",
			Kind:      sessionloop.CommandNextTurn,
			CommandID: "cmd-2",
			Position:  sessionloop.Position{Sequence: 4, Token: "tk-4"},
			Content:   []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "queued"}},
		},
		Suspension: &sessionloop.Suspension{
			ID:          "susp-1",
			Kind:        "approval",
			Description: "safe display",
			Decisions:   []sessionloop.SuspensionDecision{{ID: "d-1", Name: "lookup", Capability: "tools", Action: "invoke", Resource: "db"}},
		},
		Outcome: &sessionloop.RunOutcome{RunID: "run-1", Kind: sessionloop.RunCompleted, Output: json.RawMessage(`{"a":1}`)},
		Usage:   &sessionloop.Usage{TotalTokens: 10, Requests: 1},
		Preview: &sessionloop.Preview{Kind: sessionloop.PreviewText, Text: "partial"},
		Dropped: 2,
	}
}

func TestEventCloneDeepCopiesEveryTypedPayload(t *testing.T) {
	t.Parallel()
	original := fullEvent()
	reference := fullEvent()
	clone := original.Clone()
	if !reflect.DeepEqual(clone, reference) {
		t.Fatalf("clone diverged from the original:\nclone    %#v\noriginal %#v", clone, reference)
	}

	clone.Entry.Content[0].Text = "mutated"
	clone.Queue.Content[0].Text = "mutated"
	clone.Suspension.Decisions[0].Name = "mutated"
	clone.Outcome.Output[2] = 'X'
	clone.Usage.TotalTokens = 999
	clone.Preview.Text = "mutated"
	if !reflect.DeepEqual(original, reference) {
		t.Fatalf("mutating the clone leaked into the original: %#v", original)
	}

	var zero sessionloop.Event
	zeroClone := zero.Clone()
	if !reflect.DeepEqual(zeroClone, zero) {
		t.Fatalf("cloning the zero event invented payloads: %#v", zeroClone)
	}
}

func TestPayloadClonesAreDeeplyIndependent(t *testing.T) {
	t.Parallel()
	queued := sessionloop.QueuedInput{ID: "queue-1", Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "keep"}}}
	queuedClone := queued.Clone()
	queuedClone.Content[0].Text = "mutated"
	if queued.Content[0].Text != "keep" {
		t.Fatalf("QueuedInput.Clone shares content: %#v", queued)
	}

	suspension := sessionloop.Suspension{ID: "susp-1", Decisions: []sessionloop.SuspensionDecision{{ID: "d-1", Name: "keep"}}}
	suspensionClone := suspension.Clone()
	suspensionClone.Decisions[0].Name = "mutated"
	if suspension.Decisions[0].Name != "keep" {
		t.Fatalf("Suspension.Clone shares decisions: %#v", suspension)
	}
	if bare := (sessionloop.Suspension{ID: "susp-2"}).Clone(); bare.Decisions != nil {
		t.Fatalf("cloning a suspension without decisions invented decisions: %#v", bare)
	}

	resolution := sessionloop.Resolution{
		SuspensionID: "susp-1",
		Decisions:    []sessionloop.ResolutionDecision{{ID: "d-1", Action: sessionloop.ResolutionApprove, Data: json.RawMessage(`{"x":1}`)}},
	}
	resolutionClone := resolution.Clone()
	resolutionClone.Decisions[0].Data[2] = 'X'
	resolutionClone.Decisions[0].ID = "mutated"
	if string(resolution.Decisions[0].Data) != `{"x":1}` || resolution.Decisions[0].ID != "d-1" {
		t.Fatalf("Resolution.Clone shares decisions: %#v", resolution)
	}
	if bare := (sessionloop.Resolution{SuspensionID: "susp-2"}).Clone(); bare.Decisions != nil {
		t.Fatalf("cloning a resolution without decisions invented decisions: %#v", bare)
	}

	decision := sessionloop.ResolutionDecision{ID: "d-1", Action: sessionloop.ResolutionDeny}
	if clone := decision.Clone(); clone.Data != nil {
		t.Fatalf("cloning a decision without data invented data: %#v", clone)
	}

	outcome := sessionloop.RunOutcome{RunID: "run-1", Kind: sessionloop.RunFailed, Failure: "boom", Output: json.RawMessage(`{"y":2}`)}
	outcomeClone := outcome.Clone()
	outcomeClone.Output[2] = 'X'
	if string(outcome.Output) != `{"y":2}` {
		t.Fatalf("RunOutcome.Clone shares output bytes: %#v", outcome)
	}
	if bare := (sessionloop.RunOutcome{RunID: "run-2", Kind: sessionloop.RunInterrupted}).Clone(); bare.Output != nil {
		t.Fatalf("cloning an outcome without output invented output: %#v", bare)
	}
}

func TestSnapshotCloneIsDeeplyIndependent(t *testing.T) {
	t.Parallel()
	suspension := &sessionloop.Suspension{ID: "susp-1", Decisions: []sessionloop.SuspensionDecision{{ID: "d-1", Name: "keep"}}}
	original := sessionloop.Snapshot{
		SessionID:    "session-1",
		Position:     sessionloop.Position{Sequence: 5, Token: "tk-5"},
		State:        sessionloop.StateSuspended,
		ActiveRunID:  "run-1",
		Entries:      []sessionloop.Entry{{ID: "entry-1", Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "keep"}}}},
		Pending:      []sessionloop.QueuedInput{{ID: "queue-1", Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "keep"}}}},
		Suspension:   suspension,
		Usage:        sessionloop.Usage{TotalTokens: 7},
		Capabilities: sessionloop.NewCapabilities(sessionloop.CapabilitySteer),
	}
	clone := original.Clone()
	clone.Entries[0].Content[0].Text = "mutated"
	clone.Pending[0].Content[0].Text = "mutated"
	clone.Suspension.Decisions[0].Name = "mutated"
	clone.Capabilities[0] = "mutated"
	if original.Entries[0].Content[0].Text != "keep" ||
		original.Pending[0].Content[0].Text != "keep" ||
		original.Suspension.Decisions[0].Name != "keep" ||
		original.Capabilities[0] != sessionloop.CapabilitySteer {
		t.Fatalf("mutating the snapshot clone leaked into the original: %#v", original)
	}

	var empty sessionloop.Snapshot
	emptyClone := empty.Clone()
	if emptyClone.Entries != nil || emptyClone.Pending != nil || emptyClone.Suspension != nil || emptyClone.Capabilities != nil {
		t.Fatalf("cloning an empty snapshot invented state: %#v", emptyClone)
	}
}
