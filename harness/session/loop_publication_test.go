// Command publication tests force readers to observe committed events before
// Dispatch returns. Attribution must already agree with snapshots and replay.
package session

import (
	"context"
	"reflect"
	"sync"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

// publicationGate models preemption just after the hub publishes a record. It
// is test-only: real hubs must never wait for subscribers. No sleeps are used.
type publicationGate struct {
	event.Hub
	kind    string
	release <-chan struct{}
}

func (g publicationGate) PublishDurable(record event.Record) {
	g.Hub.PublishDurable(record)
	if record.Name == g.kind {
		<-g.release
	}
}

func gatePublication(kind string) (event.Factory, func()) {
	release := make(chan struct{})
	var once sync.Once
	return event.FactoryFunc(func(_ context.Context, history []event.Record) (event.Hub, error) {
		return publicationGate{Hub: inproc.New(history), kind: kind, release: release}, nil
	}), func() { once.Do(func() { close(release) }) }
}

func dispatchAsync(ctx context.Context, view *LoopView[string], command sessionloop.Command) <-chan error {
	done := make(chan error, 1)
	go func() { _, err := view.Dispatch(ctx, command); done <- err }()
	return done
}

func assertPublishedCommand(t *testing.T, stream sessionloop.Stream, kind sessionloop.EventKind, command sessionloop.Command) sessionloop.Event {
	t.Helper()
	live, _ := awaitLoopKind(t, stream, kind)
	if live.CommandID != command.ID {
		t.Errorf("published %s command ID = %q, want %q before Dispatch returns", kind, live.CommandID, command.ID)
	}
	if live.Entry != nil && live.Entry.CommandID != command.ID {
		t.Errorf("published entry command ID = %q, want %q", live.Entry.CommandID, command.ID)
	}
	if live.Queue != nil && live.Queue.CommandID != command.ID {
		t.Errorf("published queue command ID = %q, want %q", live.Queue.CommandID, command.ID)
	}
	return live
}

func assertReplayEvent(t *testing.T, view *LoopView[string], live sessionloop.Event) {
	t.Helper()
	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	events, err := view.Replay(loopTestContext(t), sessionloop.Position{}, snapshot.Position)
	if err != nil {
		t.Fatal(err)
	}
	for _, replay := range events {
		if replay.Kind == live.Kind && replay.Position == live.Position {
			if !reflect.DeepEqual(live, replay) {
				t.Fatalf("live and replay differ:\nlive=%#v\nreplay=%#v", live, replay)
			}
			return
		}
	}
	t.Fatal("published event missing from replay")
}

func TestLoopRaceStartAttributionBeforePublication(t *testing.T) {
	for _, keyed := range []bool{false, true} {
		t.Run(map[bool]string{false: "unkeyed", true: "keyed"}[keyed], func(t *testing.T) {
			factory, release := gatePublication(kindMessage)
			var config Config[string]
			view, _ := newLoopViewForTest(t, agentic.NewAgent("", &scriptedModel{steps: []modelStep{textStep("ok")}}), storememory.New(), func(c *Config[string], _ *LoopConfig[string]) {
				c.Events = factory
				config = *c
			})
			defer release()
			stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64})
			command := sessionloopStartCommand("hello")
			command.ID = "cmd-start"
			if keyed {
				command.IdempotencyKey = "start-key"
			}
			done := dispatchAsync(loopTestContext(t), view, command)
			live := assertPublishedCommand(t, stream, sessionloop.EventEntryCommitted, command)
			// Snapshot and finite replay can also run while Dispatch is parked.
			assertReplayEvent(t, view, live)
			release()
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			awaitLoopSettled(t, stream, live.RunID)
			assertReplayEvent(t, view, live)
			if keyed {
				if err := view.Close(loopTestContext(t)); err != nil {
					t.Fatal(err)
				}
				reopened, err := reopenLoopView(t, config)
				if err != nil {
					t.Fatal(err)
				}
				assertReplayEvent(t, reopened, live)
			}
		})
	}
}

func TestLoopRaceQueueAttributionBeforePublication(t *testing.T) {
	factory, release := gatePublication(kindQueueAccepted)
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), func(c *Config[string], _ *LoopConfig[string]) { c.Events = factory })
	defer release()
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64})
	command := sessionloop.Command{ID: "cmd-next", IdempotencyKey: "next-key", Kind: sessionloop.CommandNextTurn, Input: sessionloopTextInput("later")}
	done := dispatchAsync(loopTestContext(t), view, command)
	live := assertPublishedCommand(t, stream, sessionloop.EventQueueAccepted, command)
	assertReplayEvent(t, view, live)
	// Another command may consume the queue while its Dispatch is still parked.
	// Attribution must precede queue visibility, not just event publication.
	start := loopDispatch(t, view, sessionloopStartCommand("consume queued turn"))
	entry := assertPublishedCommand(t, stream, sessionloop.EventEntryCommitted, command)
	if entry.Entry.Origin != sessionloop.OriginNextTurn {
		t.Fatalf("first input origin = %q, want next_turn", entry.Entry.Origin)
	}
	awaitLoopSettled(t, stream, start.RunID)
	assertReplayEvent(t, view, entry)
	release()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	assertReplayEvent(t, view, live)
}

func TestLoopRaceResolveAttributionBeforePublication(t *testing.T) {
	factory, release := gatePublication(kindResolutionAccepted)
	view, _ := newSuspendedGateView(t, func(c *Config[string], _ *LoopConfig[string]) { c.Events = factory })
	defer release()
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 256})
	start := loopDispatch(t, view, sessionloopStartCommand("begin"))
	suspended, _ := awaitLoopKind(t, stream, sessionloop.EventRunSuspended)
	command := sessionloop.Command{ID: "cmd-resolve", IdempotencyKey: "resolve-key", Kind: sessionloop.CommandResolve, RunID: start.RunID,
		Resolution: &sessionloop.Resolution{SuspensionID: suspended.Suspension.ID, Decisions: []sessionloop.ResolutionDecision{{ID: "gate-1", Action: sessionloop.ResolutionApprove}}}}
	done := dispatchAsync(loopTestContext(t), view, command)
	live := assertPublishedCommand(t, stream, sessionloop.EventCommandAccepted, command)
	release()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	awaitLoopSettled(t, stream, start.RunID)
	assertReplayEvent(t, view, live)
}

func TestLoopRaceRecoveryResolveAttributionBeforePublication(t *testing.T) {
	call := agentic.ToolUse{ID: "indeterminate-1", Name: "effect"}
	config, _, _ := crashedConfig(t, []agentic.ToolUse{call}, "started")
	factory, release := gatePublication(kindRunOpened)
	config.Events = factory
	view, err := reopenLoopView(t, config)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Suspension == nil {
		t.Fatal("missing recovery suspension")
	}
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{After: snapshot.Position, Buffer: 256})
	command := sessionloop.Command{ID: "cmd-recovery", IdempotencyKey: "recovery-key", Kind: sessionloop.CommandResolve, RunID: snapshot.ActiveRunID,
		Input:      sessionloopTextInput("continue"),
		Resolution: &sessionloop.Resolution{SuspensionID: snapshot.Suspension.ID, Decisions: []sessionloop.ResolutionDecision{{ID: call.ID, Action: sessionloop.ResolutionDeny}}}}
	done := dispatchAsync(loopTestContext(t), view, command)
	resolution := assertPublishedCommand(t, stream, sessionloop.EventCommandAccepted, command)
	opened := assertPublishedCommand(t, stream, sessionloop.EventRunStarted, command)
	if opened.RunID == snapshot.ActiveRunID {
		t.Fatal("recovery did not open a continuation run")
	}
	release()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	awaitLoopSettled(t, stream, opened.RunID)
	assertReplayEvent(t, view, resolution)
	assertReplayEvent(t, view, opened)
	if err := view.Close(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	reopened, err := reopenLoopView(t, config)
	if err != nil {
		t.Fatal(err)
	}
	assertReplayEvent(t, reopened, resolution)
	assertReplayEvent(t, reopened, opened)
}
