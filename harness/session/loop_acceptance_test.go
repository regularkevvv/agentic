// Acceptance-reader tests verify actual journal replay and immutable lookup.
package session

import (
	"context"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestAcceptanceLookupSurvivesJournalReopenWithoutDispatch(t *testing.T) {
	config := sessionConfig(t, agentic.NewAgent("", &scriptedModel{steps: []modelStep{textStep("done")}}), storememory.New(), artifactmemory.New(), spill.Config{})
	s, err := New(t.Context(), config)
	if err != nil {
		t.Fatal(err)
	}
	v, err := NewLoopView(s, LoopConfig[string]{CloseRoot: s.Close})
	if err != nil {
		t.Fatal(err)
	}
	command := sessionloopStartCommand("hello")
	command.ID = "command"
	command.IdempotencyKey = "key"
	if _, found, err := v.Acceptance(t.Context(), command); err != nil || found {
		t.Fatalf("unsubmitted=%v %v", found, err)
	}
	receipt, err := v.Dispatch(t.Context(), command)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	if err := v.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, _, err := v.Acceptance(t.Context(), command); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatal(err)
	}
	v, err = reopenLoopView(t, config)
	if err != nil {
		t.Fatal(err)
	}
	restored, found, err := v.Acceptance(t.Context(), command)
	if err != nil || !found || restored != receipt {
		t.Fatalf("restored=%+v %v %v original=%+v", restored, found, err, receipt)
	}
	changed := command.Clone()
	changed.Input.Blocks[0].Text = "different"
	if _, _, err := v.Acceptance(t.Context(), changed); !errors.Is(err, sessionloop.ErrCommandConflict) {
		t.Fatal(err)
	}
	changed = command.Clone()
	changed.ID = "different-id"
	if _, _, err := v.Acceptance(t.Context(), changed); !errors.Is(err, sessionloop.ErrCommandConflict) {
		t.Fatal(err)
	}
	changed = command.Clone()
	changed.IdempotencyKey = ""
	if _, _, err := v.Acceptance(t.Context(), changed); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatal(err)
	}
	if _, _, err := v.Acceptance(t.Context(), sessionloop.Command{}); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, _, err := v.Acceptance(ctx, command); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestRejectionIsJournaledIdempotentAndSurvivesReopen(t *testing.T) {
	config := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	s, err := New(t.Context(), config)
	if err != nil {
		t.Fatal(err)
	}
	v, err := NewLoopView(s, LoopConfig[string]{CloseRoot: s.Close})
	if err != nil {
		t.Fatal(err)
	}
	c := sessionloop.Command{ID: "stale", IdempotencyKey: "stale", Kind: sessionloop.CommandInterrupt, RunID: "gone"}
	r, err := v.Reject(t.Context(), c, sessionloop.RejectionStaleRun)
	if err != nil || r.Rejection != sessionloop.RejectionStaleRun || r.Position.IsZero() {
		t.Fatalf("reject=%+v %v", r, err)
	}
	if again, err := v.Reject(t.Context(), c, sessionloop.RejectionNotRunning); err != nil || again != r {
		t.Fatalf("changed resolution=%+v %v", again, err)
	}
	changed := c
	changed.RunID = "another"
	if _, err := v.Reject(t.Context(), changed, sessionloop.RejectionStaleRun); !errors.Is(err, sessionloop.ErrCommandConflict) {
		t.Fatal(err)
	}
	changed = c
	changed.ID = "other"
	if _, err := v.Reject(t.Context(), changed, sessionloop.RejectionStaleRun); !errors.Is(err, sessionloop.ErrCommandConflict) {
		t.Fatal(err)
	}
	if err := v.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := v.Reject(t.Context(), c, sessionloop.RejectionStaleRun); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatal(err)
	}
	v, err = reopenLoopView(t, config)
	if err != nil {
		t.Fatal(err)
	}
	got, found, err := v.Acceptance(t.Context(), c)
	if err != nil || !found || got != r {
		t.Fatalf("restored=%+v %v %v", got, found, err)
	}
	if len(v.runCommands) != 0 || len(v.queueCommands) != 0 {
		t.Fatal("rejection invented a run or queue item")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := v.Reject(ctx, c, sessionloop.RejectionStaleRun); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := v.Reject(t.Context(), sessionloop.Command{}, sessionloop.RejectionStaleRun); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatal(err)
	}
	if _, err := v.Reject(t.Context(), c, "invalid reason"); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatal(err)
	}
	changed = c
	changed.ID = "new"
	changed.IdempotencyKey = "new"
	restore := failJournal(v.inner, nil, errors.New("storage unavailable"))
	if _, err := v.Reject(t.Context(), changed, sessionloop.RejectionStaleRun); err == nil {
		t.Fatal("failed persistence accepted")
	}
	restore()
	if _, found, err := v.Acceptance(t.Context(), changed); err != nil || found {
		t.Fatalf("failed append became receipt=%v %v", found, err)
	}
	v.inner.mu.Lock()
	v.inner.state = Faulted
	v.inner.mu.Unlock()
	if _, err := v.Reject(t.Context(), changed, sessionloop.RejectionStaleRun); !errors.Is(err, sessionloop.ErrSessionFaulted) {
		t.Fatal(err)
	}
}
