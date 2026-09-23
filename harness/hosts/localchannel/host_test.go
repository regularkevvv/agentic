// These tests exercise the concrete local flavor against a real Harness and
// disk journal, including lifecycle, rejection, replay and observation failures.
package localchannel

import (
	"context"
	"errors"
	"io"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type echoModel struct{ calls atomic.Int32 }

type testStream = stream

func (*echoModel) Name() string { return "test:localchannel" }
func (m *echoModel) Request(ctx context.Context, _ *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.calls.Add(1)
	return &agentic.ChatResponse{Message: agentic.NewTextMessage(agentic.RoleAssistant, "done"), FinishReason: agentic.FinishReasonStop}, nil
}

func fixture(t *testing.T) (*Host, Config, *echoModel, context.Context) {
	t.Helper()
	assembly, err := harness.AssembleDefault(harness.DefaultConfig{WorkspaceRoot: t.TempDir(), SessionDir: filepath.Join(t.TempDir(), "sessions"), ContextWindowTokens: 16000})
	if err != nil {
		t.Fatal(err)
	}
	model := &echoModel{}
	cfg := Config{Journals: assembly.Runtime.Sessions, Build: func(journals store.Repository) (sessionloop.Host, error) {
		owned := assembly.Runtime
		owned.Sessions = journals
		runtime, err := harness.New(agentic.NewAgent("test", model), harness.WithRuntime(owned), harness.WithCapabilities(assembly.Capabilities...)).Build()
		if err != nil {
			return nil, err
		}
		return harness.NewSessionLoopHost(runtime)
	}}
	h, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return h, cfg, model, ctx
}

func start(t *testing.T, h *Host) func() {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- h.Run(ctx) }()
	<-h.started
	var stopped bool
	stop := func() {
		if !stopped {
			stopped = true
			cancel()
			if err := <-done; !errors.Is(err, context.Canceled) {
				t.Errorf("worker: %v", err)
			}
		}
	}
	t.Cleanup(stop)
	return stop
}

func input(id string) sessionloop.Command {
	return sessionloop.Command{ID: sessionloop.CommandID(id), Kind: sessionloop.CommandStart, Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "hello"}}}}
}

func settle(t *testing.T, ctx context.Context, session sessionloop.Session, command sessionloop.Command) sessionloop.Receipt {
	t.Helper()
	snapshot, err := session.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{After: snapshot.Position})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	receipt, err := session.Dispatch(ctx, command)
	if err != nil {
		t.Fatal(err)
	}
	for {
		event, err := stream.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if event.Kind == sessionloop.EventRunSettled && event.RunID == receipt.RunID {
			break
		}
	}
	return receipt
}

func TestIndependentWorkerAndDiskReopen(t *testing.T) {
	h, cfg, model, ctx := fixture(t)
	beforeWorker, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
	defer cancel()
	if _, err := h.NewSession(beforeWorker, sessionloop.SessionOptions{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	if model.calls.Load() != 0 {
		t.Fatal("constructing/submitting must not run a model")
	}
	stop := start(t, h)
	if err := h.Run(ctx); !errors.Is(err, actor.ErrWorkerRunning) {
		t.Fatal(err)
	}
	s, err := h.NewSession(ctx, sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.OpenSession(ctx, s.ID()); !errors.Is(err, sessionloop.ErrSessionOpen) {
		t.Fatal(err)
	}
	first := settle(t, ctx, s, input("one"))
	if first.Guarantee != sessionloop.AcceptanceDurable {
		t.Fatal(first)
	}
	if retry, err := s.Dispatch(ctx, input("one")); err != nil || retry != first {
		t.Fatalf("retry: %+v %v", retry, err)
	}
	changed := input("one")
	changed.Input.Blocks[0].Text = "changed"
	if _, err := s.Dispatch(ctx, changed); !errors.Is(err, actor.ErrCommandConflict) {
		t.Fatal(err)
	}
	before, err := s.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Snapshot(ctx); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatal(err)
	}
	if _, err := s.Dispatch(ctx, input("two")); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatal(err)
	}
	if _, err := s.Subscribe(ctx, sessionloop.SubscribeOptions{}); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatal(err)
	}
	stop()

	// Fresh host, mailbox and projections. Only the existing disk journal remains.
	fresh, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	start(t, fresh)
	reopened, err := fresh.OpenSession(ctx, s.ID())
	if err != nil {
		t.Fatal(err)
	}
	after, err := reopened.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before.Entries, after.Entries) {
		t.Fatal("journal reconstruction changed transcript")
	}
	replay, err := reopened.Subscribe(ctx, sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for {
		event, err := replay.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if event.Kind == sessionloop.EventRunSettled && event.RunID == first.RunID {
			break
		}
	}
	_ = replay.Close()
	if retry, err := reopened.Dispatch(ctx, input("one")); err != nil || retry != first {
		t.Fatalf("restored receipt: %+v %v", retry, err)
	}
	settle(t, ctx, reopened, input("two"))
	if model.calls.Load() != 2 {
		t.Fatalf("duplicate execution: %d", model.calls.Load())
	}
}

func TestErrorsAndCancellation(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("empty configuration accepted")
	}
	h, cfg, _, ctx := fixture(t)
	start(t, h)
	if _, err := h.OpenSession(ctx, "../bad"); !errors.Is(err, store.ErrInvalidSessionID) {
		t.Fatal(err)
	}
	if _, err := h.OpenSession(ctx, "missing"); !errors.Is(err, store.ErrSessionNotFound) {
		t.Fatal(err)
	}
	s, err := h.NewSession(ctx, sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := s.Snapshot(canceled); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.Subscribe(canceled, sessionloop.SubscribeOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.Dispatch(canceled, input("cancel")); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.Dispatch(ctx, sessionloop.Command{}); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatal(err)
	}
	if _, err := s.Subscribe(ctx, sessionloop.SubscribeOptions{After: sessionloop.Position{Sequence: 9999}}); !errors.Is(err, sessionloop.ErrUnknownPosition) {
		t.Fatal(err)
	}
	want := errors.New("builder unavailable")
	cfg.Build = func(store.Repository) (sessionloop.Host, error) { return nil, want }
	broken, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := broken.NewSession(ctx, sessionloop.SessionOptions{}); !errors.Is(err, want) {
		t.Fatal(err)
	}
	if _, err := broken.openOwned(ctx, actor.Lease{}); !errors.Is(err, want) {
		t.Fatal(err)
	}
	for _, reason := range []sessionloop.Rejection{"", sessionloop.RejectionInvalidCommand, sessionloop.RejectionNotRunning, sessionloop.RejectionStaleRun, sessionloop.RejectionUnsupported, "unknown"} {
		if (rejectionError(reason) == nil) != (reason == "") {
			t.Fatal(reason)
		}
	}
}

func TestBoundedStreamsAndCopies(t *testing.T) {
	h, _, _, ctx := fixture(t)
	start(t, h)
	session, err := h.NewSession(ctx, sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	s := session.(*handle)
	snapshot, err := s.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	preview := sessionloop.Event{Nature: sessionloop.EventPreview, Kind: sessionloop.EventPreviewDelta, Preview: &sessionloop.Preview{Kind: sessionloop.PreviewText, Text: "one"}}
	sub, err := s.Subscribe(ctx, sessionloop.SubscribeOptions{After: snapshot.Position, Preview: true, Buffer: 1})
	if err != nil {
		t.Fatal(err)
	}
	stream := sub.(*stream)
	h.mu.Lock()
	stream.deliverLocked(preview)
	stream.deliverLocked(preview)
	h.mu.Unlock()
	first, err := stream.Next(ctx)
	if err != nil || first.Ordinal != 1 {
		t.Fatal(first, err)
	}
	first.Preview.Text = "mutated"
	h.mu.Lock()
	stream.deliverLocked(preview)
	h.mu.Unlock()
	next, err := stream.Next(ctx)
	if err != nil || next.Dropped != 1 || next.Preview.Text != "one" || next.Ordinal != 2 {
		t.Fatal(next, err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Next(ctx); !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}

	sub, _ = s.Subscribe(ctx, sessionloop.SubscribeOptions{After: snapshot.Position, Buffer: 1})
	stream = sub.(*testStream)
	h.mu.Lock()
	stream.deliverLocked(preview)                                                                                // filtered; consumes no capacity
	stream.deliverLocked(sessionloop.Event{Nature: sessionloop.EventAuthoritative, Position: snapshot.Position}) // already in the snapshot
	event := sessionloop.Event{Nature: sessionloop.EventAuthoritative, Kind: sessionloop.EventUsage, Position: sessionloop.Position{Sequence: 100}}
	stream.deliverLocked(event)
	stream.deliverLocked(event)
	h.mu.Unlock()
	if _, err := stream.Next(ctx); !errors.Is(err, sessionloop.ErrLagged) {
		t.Fatal(err)
	}
	_ = stream.Close()
	sub, _ = s.Subscribe(ctx, sessionloop.SubscribeOptions{After: snapshot.Position})
	_ = s.Close(ctx)
	if _, err := sub.Next(ctx); !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
}

type stubHost struct {
	session            sessionloop.Session
	createErr, openErr error
}

func (h stubHost) NewSession(context.Context, sessionloop.SessionOptions) (sessionloop.Session, error) {
	return h.session, h.createErr
}
func (h stubHost) OpenSession(context.Context, sessionloop.SessionID) (sessionloop.Session, error) {
	return h.session, h.openErr
}

type stubSession struct {
	actor.Session
	snapshot                         sessionloop.Snapshot
	snapshotErr, replayErr, closeErr error
}

func (s *stubSession) ID() sessionloop.SessionID { return "stub" }
func (s *stubSession) Snapshot(context.Context) (sessionloop.Snapshot, error) {
	return s.snapshot, s.snapshotErr
}
func (s *stubSession) Replay(context.Context, sessionloop.Position, sessionloop.Position) ([]sessionloop.Event, error) {
	return nil, s.replayErr
}
func (s *stubSession) Close(context.Context) error { return s.closeErr }

type plainSession struct{ sessionloop.Session }

func (plainSession) ID() sessionloop.SessionID   { return "stub" }
func (plainSession) Close(context.Context) error { return nil }

func TestFailureBoundaries(t *testing.T) {
	want := errors.New("injected")
	stub := &stubSession{snapshot: sessionloop.Snapshot{SessionID: "stub", State: sessionloop.StateIdle}}
	host := stubHost{session: stub}
	h, err := New(Config{Journals: storememory.New(), Build: func(store.Repository) (sessionloop.Host, error) { return host, nil }})
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	host.createErr = want
	if _, err := h.NewSession(ctx, sessionloop.SessionOptions{}); !errors.Is(err, want) {
		t.Fatal(err)
	}
	host.createErr, stub.closeErr = nil, want
	if _, err := h.NewSession(ctx, sessionloop.SessionOptions{}); !errors.Is(err, want) {
		t.Fatal(err)
	}
	stub.closeErr = nil
	host.session = plainSession{}
	if _, err := h.openOwned(ctx, actor.Lease{}); err == nil {
		t.Fatal("non-journal host accepted")
	}
	state := &state{snapshot: stub.snapshot, changed: make(chan struct{}), receipts: make(map[actor.CommandID]sessionloop.Receipt), published: make(map[eventKey]bool), streams: make(map[*stream]struct{})}
	h.sessions["stub"] = state
	owned := &observedSession{journalSession: stub, host: h, lease: actor.Lease{ActorID: "stub", Fence: 1}}
	state.current = owned
	stub.snapshotErr = want
	if _, err := owned.Snapshot(ctx); !errors.Is(err, want) {
		t.Fatal(err)
	}
	client := &handle{host: h, state: state, id: "stub"}
	if _, err := client.Snapshot(ctx); !errors.Is(err, want) {
		t.Fatal(err)
	}
	if _, err := client.Dispatch(ctx, input("refresh-fails")); !errors.Is(err, want) {
		t.Fatal(err)
	}
	stub.snapshotErr, stub.replayErr = nil, want
	if _, err := owned.Snapshot(ctx); !errors.Is(err, want) {
		t.Fatal(err)
	}
	stub.replayErr = nil
	state.current = nil
	if _, err := owned.Snapshot(ctx); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatal(err)
	}
	state.current = owned
	stub.closeErr = want
	if err := owned.Close(ctx); !errors.Is(err, want) {
		t.Fatal(err)
	}
	stub.closeErr = nil
	if err := owned.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if err := owned.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := owned.Snapshot(ctx); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatal(err)
	}
	if err := h.Observe(ctx, actor.Lease{ActorID: "stub", Fence: 2}, sessionloop.Event{}); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatal(err)
	}
	if err := h.Observe(ctx, actor.Lease{ActorID: "missing"}, sessionloop.Event{}); err != nil {
		t.Fatal(err)
	}
	if err := (&delivery{Store: h.mailbox, host: h}).Acknowledge(ctx, actor.Lease{}, "missing", sessionloop.Receipt{}); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatal(err)
	}
	h.mu.Lock()
	h.runErr = want
	if err := h.wait(ctx, state); !errors.Is(err, want) {
		t.Fatal(err)
	}
	h.runErr, state.err = nil, want
	if err := h.wait(ctx, state); !errors.Is(err, want) {
		t.Fatal(err)
	}
	h.mu.Unlock()
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	h.mu.Lock()
	if err := h.wait(canceled, state); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	h.mu.Unlock()
	if _, err := h.OpenSession(canceled, "canceled"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	h.mu.Lock()
	state.publish(sessionloop.Event{Nature: sessionloop.EventAuthoritative, Kind: sessionloop.EventQueueAccepted, Position: sessionloop.Position{Sequence: 2}, Queue: &sessionloop.QueuedInput{ID: "q"}})
	state.publish(sessionloop.Event{Nature: sessionloop.EventAuthoritative, Kind: sessionloop.EventQueueAccepted, Position: sessionloop.Position{Sequence: 2}, Queue: &sessionloop.QueuedInput{ID: "q"}})
	h.mu.Unlock()
}

func TestStartRefreshesLaggingBusyProjection(t *testing.T) {
	stub := &stubSession{snapshot: sessionloop.Snapshot{SessionID: "stub", State: sessionloop.StateIdle, Position: sessionloop.Position{Sequence: 2}}}
	h, err := New(Config{Journals: storememory.New(), Build: func(store.Repository) (sessionloop.Host, error) {
		return stubHost{session: stub}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	st := &state{snapshot: sessionloop.Snapshot{SessionID: "stub", State: sessionloop.StateRunning, Position: sessionloop.Position{Sequence: 1}},
		changed: make(chan struct{}), receipts: make(map[actor.CommandID]sessionloop.Receipt)}
	st.current = &observedSession{journalSession: stub, host: h, streaming: true}
	h.sessions["stub"] = st
	client := &handle{host: h, state: st, id: "stub"}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	received := make(chan error, 1)
	go func() {
		_, err := h.mailbox.Receive(ctx)
		received <- err
		cancel()
	}()
	// The journal is already idle while the projection still says running.
	// Submission must reach the mailbox, then wait for a worker's receipt.
	if _, err := client.Dispatch(ctx, input("next")); !errors.Is(err, context.Canceled) {
		t.Fatalf("Start used stale projection instead of enqueueing: %v", err)
	}
	if err := <-received; err != nil {
		t.Fatalf("command was not discoverable: %v", err)
	}
}

func TestGeneratedIdentityRejectionAndCanceledWait(t *testing.T) {
	h, _, _, ctx := fixture(t)
	stop := start(t, h)
	session, err := h.NewSession(ctx, sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	settle(t, ctx, session, input(""))
	keyed := input("")
	keyed.IdempotencyKey = "key"
	settle(t, ctx, session, keyed)
	keyed = input("explicit-id")
	keyed.IdempotencyKey = "different-stable-key"
	first := settle(t, ctx, session, keyed)
	if retry, err := session.Dispatch(ctx, keyed); err != nil || retry != first {
		t.Fatal("distinct command ID and idempotency key must survive retry", retry, err)
	}
	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "not-current"}); !errors.Is(err, sessionloop.ErrStaleRun) {
		t.Fatal(err)
	}
	s := session.(*handle)
	h.mu.Lock()
	s.state.snapshot.State = sessionloop.StateRunning
	s.state.startID = "pending"
	h.mu.Unlock()
	if _, err := s.Dispatch(ctx, input("busy")); !errors.Is(err, sessionloop.ErrSessionBusy) {
		t.Fatal(err)
	}
	h.mu.Lock()
	s.state.snapshot.State = sessionloop.StateIdle
	s.state.startID = ""
	h.mu.Unlock()
	sub, err := s.Subscribe(ctx, sessionloop.SubscribeOptions{After: s.state.snapshot.Position})
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := sub.Next(canceled); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	stop()
	if _, err := sub.Next(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.Dispatch(ctx, input("after-stop")); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	_ = sub.Close()
	_ = s.Close(ctx)
	if _, err := h.OpenSession(ctx, s.ID()); !errors.Is(err, context.Canceled) {
		t.Fatal("stopped worker returned a stale cached session", err)
	}
}

func TestOpenFailureNotifiesSupervisor(t *testing.T) {
	_, cfg, _, ctx := fixture(t)
	reported := make(chan error, 1)
	cfg.OnError = func(id actor.ActorID, err error) {
		if id != "missing" {
			t.Errorf("unexpected actor %s", id)
		}
		reported <- err
	}
	h, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	start(t, h)
	if _, err := h.OpenSession(ctx, "missing"); !errors.Is(err, store.ErrSessionNotFound) {
		t.Fatal(err)
	}
	select {
	case err := <-reported:
		if !errors.Is(err, store.ErrSessionNotFound) {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
}
