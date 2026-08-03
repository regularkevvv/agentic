package app

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	uit "github.com/regularkevvv/agentic/tui"
)

type bridgeSession struct {
	mu        sync.Mutex
	snapshot  uit.Snapshot
	snapshots int
	subs      []*bridgeSubscription
}

func (s *bridgeSession) ID() string { return "bridge" }
func (s *bridgeSession) Snapshot(context.Context) (uit.Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshots++
	return s.snapshot, nil
}
func (s *bridgeSession) Subscribe(options uit.SubscribeOptions) uit.Subscription {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub := &bridgeSubscription{events: make(chan uit.Event, 8), errors: make(chan error, 1)}
	s.subs = append(s.subs, sub)
	return sub
}
func (s *bridgeSession) Submit(context.Context, uit.Input) error       { return nil }
func (s *bridgeSession) Steer(context.Context, uit.Input) error        { return nil }
func (s *bridgeSession) FollowUp(context.Context, uit.Input) error     { return nil }
func (s *bridgeSession) NextTurn(context.Context, uit.Input) error     { return nil }
func (s *bridgeSession) Resolve(context.Context, uit.Resolution) error { return nil }
func (s *bridgeSession) Interrupt(context.Context) error               { return nil }
func (s *bridgeSession) Close(context.Context) error                   { return nil }

type bridgeSubscription struct {
	once   sync.Once
	events chan uit.Event
	errors chan error
}

func (s *bridgeSubscription) Events() <-chan uit.Event { return s.events }
func (s *bridgeSubscription) Errors() <-chan error     { return s.errors }
func (s *bridgeSubscription) Close() {
	s.once.Do(func() {
		close(s.events)
		close(s.errors)
	})
}

func waitSubscription(t *testing.T, session *bridgeSession, count int) *bridgeSubscription {
	t.Helper()
	deadline := time.After(time.Second)
	for {
		session.mu.Lock()
		if len(session.subs) >= count {
			result := session.subs[count-1]
			session.mu.Unlock()
			return result
		}
		session.mu.Unlock()
		select {
		case <-deadline:
			t.Fatalf("subscription %d not created", count)
		case <-time.After(time.Millisecond):
		}
	}
}

func TestBridgeBatchesAndResyncsDropsAndLag(t *testing.T) {
	t.Parallel()
	session := &bridgeSession{snapshot: uit.Snapshot{SessionID: "bridge", Cursor: 9, State: uit.StateIdle}}
	bridge := startBridge(context.Background(), session, uit.Snapshot{Cursor: 1}, 240, 8)
	defer bridge.Close()
	first := waitSubscription(t, session, 1)
	first.events <- uit.Event{Kind: uit.EventTextDelta, TextDelta: "a"}
	select {
	case update := <-bridge.updates:
		if len(update.events) != 1 || update.events[0].TextDelta != "a" {
			t.Fatalf("batch = %#v", update)
		}
	case <-time.After(time.Second):
		t.Fatal("batch not emitted")
	}
	first.events <- uit.Event{Kind: uit.EventTextDelta, Dropped: 2}
	select {
	case update := <-bridge.updates:
		if update.snapshot == nil || update.snapshot.Cursor != 9 {
			t.Fatalf("drop resync = %#v", update)
		}
	case <-time.After(time.Second):
		t.Fatal("drop did not resync")
	}
	second := waitSubscription(t, session, 2)
	second.errors <- errors.New("lag")
	select {
	case update := <-bridge.updates:
		if update.snapshot == nil {
			t.Fatalf("lag update = %#v", update)
		}
	case <-time.After(time.Second):
		t.Fatal("lag did not resync")
	}
	waitSubscription(t, session, 3)
	session.mu.Lock()
	if session.snapshots != 2 {
		t.Fatalf("snapshots = %d", session.snapshots)
	}
	session.mu.Unlock()
}

func TestBridgeTerminalEdges(t *testing.T) {
	t.Parallel()
	missing := startBridge(context.Background(), nil, uit.Snapshot{}, 0, 0)
	update := <-missing.updates
	if update.err == nil {
		t.Fatal("nil session bridge had no error")
	}
	missing.Close()

	nilSource := startBridge(context.Background(), nilSubscriptionSession{bridgeSession: &bridgeSession{}}, uit.Snapshot{}, 0, 0)
	if update := <-nilSource.updates; update.err == nil {
		t.Fatal("nil subscription had no error")
	}
	nilSource.Close()

	session := &bridgeSession{}
	bridge := startBridge(context.Background(), session, uit.Snapshot{}, 60, 1)
	sub := waitSubscription(t, session, 1)
	sub.Close()
	select {
	case _, ok := <-bridge.updates:
		if ok {
			t.Fatal("closed subscription emitted update")
		}
	case <-time.After(time.Second):
		t.Fatal("bridge did not stop on clean subscription close")
	}
	bridge.Close()
	bridge.Close()

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	direct := &eventBridge{updates: make(chan bridgeUpdate)}
	if direct.send(cancelled, bridgeUpdate{}) {
		t.Fatal("send on cancelled context succeeded")
	}
}

func TestBridgeSnapshotFailureIsTerminal(t *testing.T) {
	t.Parallel()
	want := errors.New("snapshot")
	session := &failingSnapshotSession{bridgeSession: bridgeSession{}}
	session.failure = want
	bridge := startBridge(context.Background(), session, uit.Snapshot{}, 60, 1)
	defer bridge.Close()
	sub := waitSubscription(t, &session.bridgeSession, 1)
	sub.errors <- errors.New("lag")
	update := <-bridge.updates
	if !errors.Is(update.err, want) {
		t.Fatalf("error = %v", update.err)
	}
}

type failingSnapshotSession struct {
	bridgeSession
	failure error
}

type nilSubscriptionSession struct{ *bridgeSession }

func (nilSubscriptionSession) Subscribe(uit.SubscribeOptions) uit.Subscription { return nil }

func (s *failingSnapshotSession) Snapshot(context.Context) (uit.Snapshot, error) {
	return uit.Snapshot{}, s.failure
}
