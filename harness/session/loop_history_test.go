// Finite replay is the native journal-to-observation boundary used during
// worker handoff; it must preserve positions, content and privacy.
package session

import (
	"context"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestFiniteLoopReplay(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	ctx := loopTestContext(t)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{})
	receipt := loopDispatch(t, view, sessionloopStartCommand("hello"))
	awaitLoopSettled(t, stream, receipt.RunID)
	snapshot, err := view.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	events, err := view.Replay(ctx, sessionloop.Position{}, snapshot.Position)
	if err != nil || len(events) == 0 {
		t.Fatal(events, err)
	}
	for _, event := range events {
		if event.Nature != sessionloop.EventAuthoritative || event.Position.IsZero() || event.Position.Sequence > snapshot.Position.Sequence {
			t.Fatal(event)
		}
	}
	if prefix, err := view.Replay(ctx, sessionloop.Position{}, sessionloop.Position{Sequence: 1}); err != nil {
		t.Fatal(err)
	} else {
		for _, event := range prefix {
			if event.Position.Sequence > 1 {
				t.Fatal("replay crossed requested upper bound", event)
			}
		}
	}
	if suffix, err := view.Replay(ctx, snapshot.Position, snapshot.Position); err != nil || len(suffix) != 0 {
		t.Fatal(suffix, err)
	}
	if _, err := view.Replay(ctx, sessionloop.Position{Sequence: 9999}, snapshot.Position); !errors.Is(err, sessionloop.ErrUnknownPosition) {
		t.Fatal(err)
	}
	if _, err := view.Replay(ctx, sessionloop.Position{}, sessionloop.Position{Sequence: 9999}); !errors.Is(err, sessionloop.ErrUnknownPosition) {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := view.Replay(canceled, sessionloop.Position{}, snapshot.Position); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := view.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := view.Replay(ctx, sessionloop.Position{}, snapshot.Position); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatal(err)
	}
}

func TestFiniteReplayRejectsCorruptJournal(t *testing.T) {
	for _, kind := range []string{kindRunOpened, "agentic.assistant_committed"} {
		t.Run(kind, func(t *testing.T) {
			view, inner := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
			commit, err := inner.journal.Append(t.Context(), inner.cursor, store.PendingEntry{Kind: kind, Payload: []byte("{")})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := view.Replay(t.Context(), sessionloop.Position{}, sessionloop.Position{Sequence: commit.Cursor.Seq}); err == nil {
				t.Fatal("corrupt payload accepted")
			}
		})
	}
}
