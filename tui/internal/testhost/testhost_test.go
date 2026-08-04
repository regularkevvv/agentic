package testhost

import (
	"context"
	"errors"
	"strings"
	"testing"

	uit "github.com/regularkevvv/agentic/tui"
)

func TestDefaultTaskPersistenceResumeAndExport(t *testing.T) {
	t.Parallel()
	host := New(nil)
	port, err := host.NewSession(context.Background(), uit.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	session := port.(*Session)
	sub := session.Subscribe(uit.SubscribeOptions{Buffer: 32, Preview: true})
	if err := session.Submit(context.Background(), uit.Input{Text: "do it"}); err != nil {
		t.Fatal(err)
	}
	snapshot, _ := session.Snapshot(context.Background())
	if snapshot.State != uit.StateIdle || len(snapshot.Transcript) != 2 || snapshot.Usage.CacheReadTokens != 75 || snapshot.Cursor == 0 {
		t.Fatalf("snapshot = %#v", snapshot)
	}
	var sawPreview, sawTool, sawCompleted bool
	for {
		select {
		case event := <-sub.Events():
			sawPreview = sawPreview || event.Kind == uit.EventTextDelta
			sawTool = sawTool || event.Kind == uit.EventToolStarted
			sawCompleted = sawCompleted || event.Kind == uit.EventRunCompleted
			if sawPreview && sawTool && sawCompleted {
				goto observed
			}
		default:
			goto observed
		}
	}
observed:
	if !sawPreview || !sawTool || !sawCompleted {
		t.Fatalf("events preview=%v tool=%v complete=%v", sawPreview, sawTool, sawCompleted)
	}
	sub.Close()
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	resumed, err := host.ResumeSession(context.Background(), session.ID())
	if err != nil {
		t.Fatal(err)
	}
	reopened, _ := resumed.Snapshot(context.Background())
	if reopened.State != uit.StateIdle || reopened.Cursor != snapshot.Cursor || len(reopened.Transcript) != 2 {
		t.Fatalf("reopened = %#v", reopened)
	}
	exported, err := host.ExportTranscript(context.Background(), session.ID())
	if err != nil || !strings.Contains(exported, "Offline task complete.") {
		t.Fatalf("export = %q, %v", exported, err)
	}
	listed, _ := host.ListSessions(context.Background())
	if len(listed) != 1 || listed[0].ID != session.ID() {
		t.Fatalf("listed = %#v", listed)
	}
}

func TestPermissionQueueInterruptAndValidation(t *testing.T) {
	t.Parallel()
	host := New(nil)
	port, _ := host.NewSession(context.Background(), uit.SessionOptions{})
	session := port.(*Session)
	if err := session.Submit(context.Background(), uit.Input{Text: "need permission"}); err != nil {
		t.Fatal(err)
	}
	snapshot, _ := session.Snapshot(context.Background())
	if snapshot.State != uit.StateSuspended || snapshot.Suspension == nil {
		t.Fatalf("snapshot = %#v", snapshot)
	}
	if err := session.Resolve(context.Background(), uit.Resolution{SuspensionID: "wrong"}); err == nil {
		t.Fatal("wrong suspension resolved")
	}
	if err := session.Resolve(context.Background(), uit.Resolution{SuspensionID: snapshot.Suspension.ID}); err == nil {
		t.Fatal("incomplete resolution succeeded")
	}
	if err := session.Resolve(context.Background(), uit.Resolution{SuspensionID: snapshot.Suspension.ID, Decisions: []uit.Decision{{CallID: "unknown", Action: uit.DecisionDeny}}}); err == nil {
		t.Fatal("unknown decision succeeded")
	}
	resolution := uit.Resolution{SuspensionID: snapshot.Suspension.ID, Decisions: []uit.Decision{{CallID: "call-write", Action: uit.DecisionDeny}}}
	if err := session.Resolve(context.Background(), resolution); err != nil {
		t.Fatal(err)
	}
	for _, invoke := range []func(context.Context, uit.Input) error{session.Steer, session.FollowUp, session.NextTurn} {
		if err := invoke(context.Background(), uit.Input{Text: "queued"}); err != nil {
			t.Fatal(err)
		}
	}
	session.mu.Lock()
	session.snapshot.State = uit.StateRunning
	session.mu.Unlock()
	if err := session.Interrupt(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := session.Interrupt(context.Background()); err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if session.Interrupt(cancelled) == nil || session.queue(cancelled, uit.QueueSteer, uit.Input{Text: "x"}) == nil || session.Resolve(cancelled, resolution) == nil {
		t.Fatal("cancelled operations succeeded")
	}
	if session.queue(context.Background(), uit.QueueSteer, uit.Input{}) == nil || session.Submit(context.Background(), uit.Input{}) == nil {
		t.Fatal("empty operations succeeded")
	}
}

func TestSubscriptionsLagDropAndErrors(t *testing.T) {
	t.Parallel()
	host := New(func(context.Context, *Session, uit.Input) error { return nil })
	port, _ := host.NewSession(context.Background(), uit.SessionOptions{})
	session := port.(*Session)
	preview := session.Subscribe(uit.SubscribeOptions{Buffer: 1, Preview: true})
	session.PreviewText("one")
	session.PreviewText("two")
	first := <-preview.Events()
	if first.TextDelta != "one" {
		t.Fatalf("first = %#v", first)
	}
	session.PreviewText("three")
	third := <-preview.Events()
	if third.Dropped != 1 {
		t.Fatalf("dropped = %d", third.Dropped)
	}
	preview.Close()

	durable := session.Subscribe(uit.SubscribeOptions{Buffer: 1})
	session.mu.Lock()
	session.emitLocked(uit.Event{Kind: uit.EventRunStarted, Durable: true})
	session.emitLocked(uit.Event{Kind: uit.EventRunEnded, Durable: true})
	session.mu.Unlock()
	if err := <-durable.Errors(); !errors.Is(err, ErrLagged) {
		t.Fatalf("lag error = %v", err)
	}
	durable.Close()

	replay := session.Subscribe(uit.SubscribeOptions{AfterCursor: 0, Buffer: 1})
	if err := <-replay.Errors(); !errors.Is(err, ErrLagged) {
		t.Fatalf("replay error = %v", err)
	}
	if _, err := host.ResumeSession(context.Background(), "missing"); err == nil {
		t.Fatal("missing resume succeeded")
	}
	if _, err := host.ExportTranscript(context.Background(), "missing"); err == nil {
		t.Fatal("missing export succeeded")
	}
}

func TestCustomScriptErrorAndClosedLease(t *testing.T) {
	t.Parallel()
	want := errors.New("script")
	host := New(func(context.Context, *Session, uit.Input) error { return want })
	port, _ := host.NewSession(context.Background(), uit.SessionOptions{})
	session := port.(*Session)
	if err := session.Submit(context.Background(), uit.Input{Text: "x"}); !errors.Is(err, want) {
		t.Fatalf("error = %v", err)
	}
	if err := session.Submit(context.Background(), uit.Input{Text: "again"}); err == nil {
		t.Fatal("running session accepted submit")
	}
	_ = session.Close(context.Background())
	if err := session.Submit(context.Background(), uit.Input{Text: "closed"}); err == nil {
		t.Fatal("closed session accepted submit")
	}
}
