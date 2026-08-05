package sessionloop

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/session"
	sl "github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
	uit "github.com/regularkevvv/agentic/tui"
)

// streamStep is one scripted Next result.
type streamStep struct {
	event sl.Event
	err   error
}

// fakeStream serves a script and then blocks until ctx cancellation or
// Close (which yields io.EOF, the clean end of stream).
type fakeStream struct {
	mu     sync.Mutex
	script []streamStep
	closed chan struct{}
	closes int
}

func newFakeStream(steps ...streamStep) *fakeStream {
	return &fakeStream{script: steps, closed: make(chan struct{})}
}

func (s *fakeStream) Next(ctx context.Context) (sl.Event, error) {
	s.mu.Lock()
	if len(s.script) > 0 {
		step := s.script[0]
		s.script = s.script[1:]
		s.mu.Unlock()
		return step.event, step.err
	}
	s.mu.Unlock()
	select {
	case <-ctx.Done():
		return sl.Event{}, ctx.Err()
	case <-s.closed:
		return sl.Event{}, io.EOF
	}
}

func (s *fakeStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closes++
	if s.closes == 1 {
		close(s.closed)
	}
	return nil
}

type snapshotStep struct {
	snapshot sl.Snapshot
	err      error
}

type dispatchStep struct {
	receipt sl.Receipt
	err     error
}

type subscribeStep struct {
	stream sl.Stream
	err    error
}

// fakeSession scripts the sessionloop session surface and records calls.
type fakeSession struct {
	mu         sync.Mutex
	id         sl.SessionID
	snapshots  []snapshotStep
	dispatches []dispatchStep
	subscribes []subscribeStep
	commands   []sl.Command
	subOptions []sl.SubscribeOptions
	closeErr   error
	closes     int
}

func (s *fakeSession) ID() sl.SessionID              { return s.id }
func (s *fakeSession) Capabilities() sl.Capabilities { return nil }
func (s *fakeSession) Close(context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closes++
	return s.closeErr
}
func (s *fakeSession) dispatched() []sl.Command {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]sl.Command(nil), s.commands...)
}
func (s *fakeSession) subscribed() []sl.SubscribeOptions {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]sl.SubscribeOptions(nil), s.subOptions...)
}

func (s *fakeSession) Snapshot(context.Context) (sl.Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.snapshots) == 0 {
		return sl.Snapshot{SessionID: s.id}, nil
	}
	step := s.snapshots[0]
	if len(s.snapshots) > 1 {
		s.snapshots = s.snapshots[1:]
	}
	return step.snapshot, step.err
}

func (s *fakeSession) Dispatch(_ context.Context, command sl.Command) (sl.Receipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.commands = append(s.commands, command)
	if len(s.dispatches) == 0 {
		return sl.Receipt{}, nil
	}
	step := s.dispatches[0]
	s.dispatches = s.dispatches[1:]
	return step.receipt, step.err
}

func (s *fakeSession) Subscribe(_ context.Context, options sl.SubscribeOptions) (sl.Stream, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.subOptions = append(s.subOptions, options)
	if len(s.subscribes) == 0 {
		return newFakeStream(), nil
	}
	step := s.subscribes[0]
	s.subscribes = s.subscribes[1:]
	return step.stream, step.err
}

type fakeHost struct {
	session sl.Session
	err     error
	opened  []sl.SessionID
}

func (h *fakeHost) NewSession(context.Context, sl.SessionOptions) (sl.Session, error) {
	return h.session, h.err
}

func (h *fakeHost) OpenSession(_ context.Context, id sl.SessionID) (sl.Session, error) {
	h.opened = append(h.opened, id)
	return h.session, h.err
}

func settledEvent(runID string, kind sl.RunOutcomeKind, failure string) sl.Event {
	return sl.Event{
		Kind: sl.EventRunSettled, RunID: sl.RunID(runID),
		Outcome: &sl.RunOutcome{RunID: sl.RunID(runID), Kind: kind, Failure: failure},
	}
}

func runningSnapshot(runID string) sl.Snapshot {
	return sl.Snapshot{State: sl.StateRunning, ActiveRunID: sl.RunID(runID)}
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func TestConstructorAndHostValidation(t *testing.T) {
	t.Parallel()
	if _, err := New(nil); err == nil {
		t.Fatal("nil host succeeded")
	}
	if _, err := New(&fakeHost{}, nil); err == nil {
		t.Fatal("nil option succeeded")
	}
	inner := &fakeSession{id: "sess-1"}
	adapted, err := New(&fakeHost{session: inner}, WithProfileLabel("p"), WithWorkspace("w"), WithExecutionLabel("e"))
	if err != nil {
		t.Fatal(err)
	}
	created, err := adapted.NewSession(testContext(t), uit.SessionOptions{})
	if err != nil || created.ID() != "sess-1" {
		t.Fatalf("created = %v, %v", created, err)
	}
	if _, err := adapted.ResumeSession(testContext(t), ""); err == nil {
		t.Fatal("empty resume ID succeeded")
	}
	resumed, err := adapted.ResumeSession(testContext(t), "sess-1")
	if err != nil || resumed.ID() != "sess-1" {
		t.Fatalf("resumed = %v, %v", resumed, err)
	}
	if err := created.Close(testContext(t)); err != nil {
		t.Fatal(err)
	}

	wrapped := fmt.Errorf("%w: %w", sl.ErrSessionOpen, store.ErrSessionOpen)
	failing, err := New(&fakeHost{err: wrapped})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := failing.NewSession(testContext(t), uit.SessionOptions{}); !errors.Is(err, store.ErrSessionOpen) || err.Error() != store.ErrSessionOpen.Error() {
		t.Fatalf("new session error = %v", err)
	}
	if _, err := failing.ResumeSession(testContext(t), "other"); !errors.Is(err, store.ErrSessionOpen) {
		t.Fatalf("resume error = %v", err)
	}
}

func TestErrorUnwrapMappingTable(t *testing.T) {
	t.Parallel()
	if mapError(nil) != nil {
		t.Fatal("nil error mapped to non-nil")
	}
	plain := errors.New("plain")
	if mapError(plain) != plain {
		t.Fatal("plain error rewritten")
	}
	sentinels := []error{
		session.ErrSessionBusy, session.ErrNotRunning, session.ErrSessionSuspended,
		session.ErrRunClosing, session.ErrSessionClosed, session.ErrSessionFaulted,
		store.ErrSessionOpen,
	}
	protocol := []error{
		sl.ErrSessionBusy, sl.ErrNotRunning, sl.ErrSuspended,
		sl.ErrCommandConflict, sl.ErrSessionClosed, sl.ErrSessionFaulted,
		sl.ErrSessionOpen,
	}
	for index, sentinel := range sentinels {
		wrapped := fmt.Errorf("%w: %w", protocol[index], sentinel)
		if mapped := mapError(wrapped); mapped != sentinel {
			t.Fatalf("wrapped %v mapped to %v, want the bare sentinel", wrapped, mapped)
		}
	}
}

func TestSnapshotProjectionAndDecoration(t *testing.T) {
	t.Parallel()
	inner := &fakeSession{id: "sess-2", snapshots: []snapshotStep{{snapshot: sl.Snapshot{
		SessionID: "sess-2",
		Position:  sl.Position{Sequence: 42, Token: "opaque"},
		State:     sl.StateSuspended,
		Entries: []sl.Entry{
			{Role: sl.RoleUser, Origin: sl.OriginStart, Content: []sl.Block{
				{Kind: sl.BlockText, Text: "hi "}, {Kind: sl.BlockText, Text: "there"},
				{Kind: sl.BlockData, MediaType: "application/json", Data: []byte(`{"x":1}`)},
			}},
			{Role: sl.RoleAssistant, Origin: sl.OriginAssistant, Content: []sl.Block{
				{Kind: sl.BlockToolCall, ToolCall: &sl.ToolCall{CallID: "c1", Name: "lookup", Data: []byte(`{"secret":1}`)}},
				{Kind: sl.BlockToolCall},
			}},
			{Role: sl.RoleTool, Origin: sl.OriginTool, Content: []sl.Block{
				{Kind: sl.BlockToolResult, Text: "raw result", ToolResult: &sl.ToolResult{CallID: "c1", Name: "lookup", IsError: true}},
				{Kind: sl.BlockToolResult},
			}},
		},
		Pending: []sl.QueuedInput{
			{ID: "q1", Kind: sl.CommandNextTurn, Content: []sl.Block{{Kind: sl.BlockText, Text: "queued"}}},
			{ID: "q2", Kind: sl.CommandSteer, Content: []sl.Block{{Kind: sl.BlockData, Data: []byte(`1`)}}},
		},
		Suspension: &sl.Suspension{
			ID: "susp", Kind: "kind", Description: "paused",
			Decisions: []sl.SuspensionDecision{{ID: "c1", Name: "lookup", Capability: "tool", Action: "lookup", Resource: "display"}},
		},
		Usage: sl.Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3, CacheReadTokens: 4, CacheCreationTokens: 5, ReasoningTokens: 6, Requests: 7, ToolCalls: 8},
	}}}}
	presenter := uit.ToolPresenterFunc(func(tool uit.Tool) uit.ToolPresentation {
		return uit.ToolPresentation{Title: "safe " + tool.Name}
	})
	adapted, err := New(&fakeHost{session: inner},
		WithProfileLabel("profile"), WithWorkspace("/ws"), WithExecutionLabel("exec"), WithToolPresenter(presenter))
	if err != nil {
		t.Fatal(err)
	}
	created, err := adapted.NewSession(testContext(t), uit.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := created.Snapshot(testContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.SessionID != "sess-2" || snapshot.Cursor != 42 || snapshot.State != uit.StateSuspended ||
		snapshot.ProfileLabel != "profile" || snapshot.Workspace != "/ws" || snapshot.Execution != "exec" {
		t.Fatalf("snapshot header = %#v", snapshot)
	}
	if len(snapshot.Transcript) != 3 || snapshot.Transcript[0].Text != "hi there" || len(snapshot.Transcript[0].Tools) != 0 {
		t.Fatalf("transcript = %#v", snapshot.Transcript)
	}
	planned := snapshot.Transcript[1].Tools
	if len(planned) != 1 || planned[0].State != uit.ToolPlanned || planned[0].Summary != "lookup" ||
		planned[0].Presentation.Title != "safe lookup" {
		t.Fatalf("planned tools = %#v", planned)
	}
	results := snapshot.Transcript[2].Tools
	if len(results) != 1 || results[0].State != uit.ToolError || snapshot.Transcript[2].Text != "" {
		t.Fatalf("result tools = %#v text %q", results, snapshot.Transcript[2].Text)
	}
	if len(snapshot.Pending) != 2 || snapshot.Pending[0].Text != "queued" ||
		snapshot.Pending[0].Kind != uit.QueueNextTurn || snapshot.Pending[1].Text != "" {
		t.Fatalf("pending = %#v", snapshot.Pending)
	}
	if snapshot.Suspension == nil || !snapshot.Suspension.Supported || snapshot.Suspension.Description != "paused" ||
		snapshot.Suspension.Approvals[0].ResourceDisplay != "display" || snapshot.Suspension.Approvals[0].CanonicalResource != "" {
		t.Fatalf("suspension = %#v", snapshot.Suspension)
	}
	if snapshot.Usage.TotalTokens != 3 || snapshot.Usage.ToolCalls != 8 {
		t.Fatalf("usage = %#v", snapshot.Usage)
	}

	unsupported := suspension(sl.Suspension{ID: "x", Kind: "custom", Description: "handoff"})
	if unsupported.Supported || unsupported.Approvals != nil || unsupported.Description != "handoff" {
		t.Fatalf("unsupported = %#v", unsupported)
	}

	failing := &fakeSession{snapshots: []snapshotStep{{err: fmt.Errorf("%w: broke", sl.ErrSessionClosed)}}}
	if err := errors.New(""); err == nil {
		t.Fatal("unreachable")
	}
	closedSession := &bridgeSession{session: failing}
	if _, err := closedSession.Snapshot(testContext(t)); !errors.Is(err, sl.ErrSessionClosed) {
		t.Fatalf("snapshot error = %v", err)
	}
}

func TestSubmitWaitsForTheRightRunBoundary(t *testing.T) {
	t.Parallel()
	// A next-turn run settles immediately before ours: the wait must skip
	// the foreign settlement and suspension and finish on our run only.
	stream := newFakeStream(
		streamStep{event: sl.Event{Kind: sl.EventRunStarted, RunID: "run-old"}},
		streamStep{event: settledEvent("run-old", sl.RunCompleted, "")},
		streamStep{event: sl.Event{Kind: sl.EventRunSuspended, RunID: "run-old"}},
		streamStep{event: sl.Event{Kind: sl.EventRunSettled, RunID: "run-ours"}}, // no outcome: ignored
		streamStep{event: settledEvent("run-ours", sl.RunCompleted, "")},
	)
	inner := &fakeSession{
		snapshots:  []snapshotStep{{snapshot: sl.Snapshot{Position: sl.Position{Sequence: 7}}}},
		subscribes: []subscribeStep{{stream: stream}},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-ours"}}},
	}
	target := &bridgeSession{session: inner}
	if err := target.Submit(testContext(t), uit.Input{Text: "go"}); err != nil {
		t.Fatal(err)
	}
	options := inner.subscribed()
	if len(options) != 1 || options[0].After.Sequence != 7 || options[0].Preview {
		t.Fatalf("subscribe options = %#v", options)
	}
	commands := inner.dispatched()
	if len(commands) != 1 || commands[0].Kind != sl.CommandStart || commands[0].Input.Content[0].Text != "go" {
		t.Fatalf("commands = %#v", commands)
	}
	if err := target.Submit(testContext(t), uit.Input{}); err == nil {
		t.Fatal("empty submit succeeded")
	}
}

func TestSubmitOutcomeMapping(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		outcome sl.RunOutcomeKind
		failure string
		check   func(error) bool
		text    string
	}{
		{"completed", sl.RunCompleted, "", func(err error) bool { return err == nil }, ""},
		{"interrupted", sl.RunInterrupted, "", func(err error) bool { return errors.Is(err, context.Canceled) }, "context canceled"},
		{"failed", sl.RunFailed, "boom", func(err error) bool { return err != nil }, "boom"},
		{"failed-empty", sl.RunFailed, "", func(err error) bool { return err != nil }, "run failed"},
	}
	for _, current := range cases {
		inner := &fakeSession{
			subscribes: []subscribeStep{{stream: newFakeStream(streamStep{event: settledEvent("run-1", current.outcome, current.failure)})}},
			dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
		}
		err := (&bridgeSession{session: inner}).Submit(testContext(t), uit.Input{Text: "x"})
		if !current.check(err) {
			t.Fatalf("%s: err = %v", current.name, err)
		}
		if err != nil && err.Error() != current.text {
			t.Fatalf("%s: text = %q", current.name, err.Error())
		}
	}
}

func TestSubmitSuspensionConfirmsDurableState(t *testing.T) {
	t.Parallel()
	// The suspension event precedes the durable suspended transition; the
	// wait confirms through snapshots until the state lands.
	inner := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{Position: sl.Position{Sequence: 1}}},
			{snapshot: runningSnapshot("run-1")},
			{snapshot: sl.Snapshot{State: sl.StateSuspended, ActiveRunID: "run-1"}},
		},
		subscribes: []subscribeStep{{stream: newFakeStream(streamStep{event: sl.Event{Kind: sl.EventRunSuspended, RunID: "run-1"}})}},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: inner}).Submit(testContext(t), uit.Input{Text: "pause"}); err != nil {
		t.Fatal(err)
	}

	faulted := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{}},
			{snapshot: sl.Snapshot{State: sl.StateFaulted, ActiveRunID: "run-1"}},
		},
		subscribes: []subscribeStep{{stream: newFakeStream(streamStep{event: sl.Event{Kind: sl.EventRunSuspended, RunID: "run-1"}})}},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: faulted}).Submit(testContext(t), uit.Input{Text: "pause"}); !errors.Is(err, session.ErrSessionFaulted) {
		t.Fatalf("faulted confirm err = %v", err)
	}

	// A settlement racing the confirmation recovers the outcome from replay.
	settledDuringConfirm := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{Position: sl.Position{Sequence: 3}}},
			{snapshot: sl.Snapshot{State: sl.StateIdle}},
		},
		subscribes: []subscribeStep{
			{stream: newFakeStream(streamStep{event: sl.Event{Kind: sl.EventRunSuspended, RunID: "run-1"}})},
			{stream: newFakeStream(streamStep{event: settledEvent("run-1", sl.RunFailed, "late failure")})},
		},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	err := (&bridgeSession{session: settledDuringConfirm}).Submit(testContext(t), uit.Input{Text: "pause"})
	if err == nil || err.Error() != "late failure" {
		t.Fatalf("settled-during-confirm err = %v", err)
	}
	confirmErr := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{}},
			{err: fmt.Errorf("%w: gone", sl.ErrSessionClosed)},
		},
		subscribes: []subscribeStep{{stream: newFakeStream(streamStep{event: sl.Event{Kind: sl.EventRunSuspended, RunID: "run-1"}})}},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: confirmErr}).Submit(testContext(t), uit.Input{Text: "pause"}); !errors.Is(err, sl.ErrSessionClosed) {
		t.Fatalf("confirm snapshot err = %v", err)
	}
}

func TestSubmitEntryPathErrors(t *testing.T) {
	t.Parallel()
	snapshotErr := &fakeSession{snapshots: []snapshotStep{{err: errors.New("no snapshot")}}}
	if err := (&bridgeSession{session: snapshotErr}).Submit(testContext(t), uit.Input{Text: "x"}); err == nil {
		t.Fatal("snapshot error ignored")
	}
	subscribeErr := &fakeSession{subscribes: []subscribeStep{{err: errors.New("no stream")}}}
	if err := (&bridgeSession{session: subscribeErr}).Submit(testContext(t), uit.Input{Text: "x"}); err == nil {
		t.Fatal("subscribe error ignored")
	}
	stream := newFakeStream()
	dispatchErr := &fakeSession{
		subscribes: []subscribeStep{{stream: stream}},
		dispatches: []dispatchStep{{err: fmt.Errorf("%w: %w", sl.ErrSessionBusy, session.ErrSessionBusy)}},
	}
	err := (&bridgeSession{session: dispatchErr}).Submit(testContext(t), uit.Input{Text: "x"})
	if err != session.ErrSessionBusy {
		t.Fatalf("dispatch err = %v", err)
	}
	if stream.closes == 0 {
		t.Fatal("dispatch failure left the wait stream open")
	}
}

func TestAwaitRunStreamFailureFallbacks(t *testing.T) {
	t.Parallel()
	lag := fmt.Errorf("%w: too slow", sl.ErrLagged)

	// Still ours and running: resubscribe from the snapshot position.
	resubscribed := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{Position: sl.Position{Sequence: 1}}},
			{snapshot: sl.Snapshot{State: sl.StateRunning, ActiveRunID: "run-1", Position: sl.Position{Sequence: 9}}},
		},
		subscribes: []subscribeStep{
			{stream: newFakeStream(streamStep{err: lag})},
			{stream: newFakeStream(streamStep{event: settledEvent("run-1", sl.RunCompleted, "")})},
		},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: resubscribed}).Submit(testContext(t), uit.Input{Text: "x"}); err != nil {
		t.Fatal(err)
	}
	options := resubscribed.subscribed()
	if len(options) != 2 || options[1].After.Sequence != 9 {
		t.Fatalf("resubscribe options = %#v", options)
	}

	// Suspended with our run during the fallback resolves the wait.
	suspendedFallback := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{}},
			{snapshot: sl.Snapshot{State: sl.StateSuspended, ActiveRunID: "run-1"}},
		},
		subscribes: []subscribeStep{{stream: newFakeStream(streamStep{err: lag})}},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: suspendedFallback}).Submit(testContext(t), uit.Input{Text: "x"}); err != nil {
		t.Fatal(err)
	}

	// Faulted with our run surfaces the legacy fault identity.
	faultedFallback := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{}},
			{snapshot: sl.Snapshot{State: sl.StateFaulted, ActiveRunID: "run-1"}},
		},
		subscribes: []subscribeStep{{stream: newFakeStream(streamStep{err: lag})}},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: faultedFallback}).Submit(testContext(t), uit.Input{Text: "x"}); !errors.Is(err, session.ErrSessionFaulted) {
		t.Fatalf("faulted fallback err = %v", err)
	}

	// Run already settled: the outcome is recovered from durable replay.
	replayed := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{Position: sl.Position{Sequence: 2}}},
			{snapshot: sl.Snapshot{State: sl.StateIdle}},
		},
		subscribes: []subscribeStep{
			{stream: newFakeStream(streamStep{err: lag})},
			{stream: newFakeStream(
				streamStep{event: sl.Event{Kind: sl.EventQueueAccepted}},
				streamStep{event: settledEvent("run-1", sl.RunInterrupted, "")},
			)},
		},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: replayed}).Submit(testContext(t), uit.Input{Text: "x"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("replayed outcome err = %v", err)
	}

	// Snapshot failure during the fallback is terminal.
	snapshotFail := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{}},
			{err: errors.New("snapshot down")},
		},
		subscribes: []subscribeStep{{stream: newFakeStream(streamStep{err: lag})}},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: snapshotFail}).Submit(testContext(t), uit.Input{Text: "x"}); err == nil {
		t.Fatal("fallback snapshot error ignored")
	}

	// Resubscribe failure during the fallback is terminal.
	resubscribeFail := &fakeSession{
		snapshots: []snapshotStep{
			{snapshot: sl.Snapshot{}},
			{snapshot: runningSnapshot("run-1")},
		},
		subscribes: []subscribeStep{
			{stream: newFakeStream(streamStep{err: lag})},
			{err: errors.New("no more streams")},
		},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: resubscribeFail}).Submit(testContext(t), uit.Input{Text: "x"}); err == nil {
		t.Fatal("fallback resubscribe error ignored")
	}
}

func TestSettledOutcomeReplayEdges(t *testing.T) {
	t.Parallel()
	target := &bridgeSession{session: &fakeSession{subscribes: []subscribeStep{{err: errors.New("no replay")}}}}
	if err := target.settledOutcome(testContext(t), sl.Position{}, "run-1", false); err == nil {
		t.Fatal("replay subscribe error ignored")
	}

	// Replay stream failure without the record: an interrupt wait treats the
	// already-idle session as its satisfied boundary; other waits surface it.
	eof := &fakeSession{subscribes: []subscribeStep{{stream: newFakeStream(streamStep{err: io.EOF})}}}
	if err := (&bridgeSession{session: eof}).settledOutcome(testContext(t), sl.Position{}, "run-1", true); err != nil {
		t.Fatal(err)
	}
	failed := &fakeSession{subscribes: []subscribeStep{{stream: newFakeStream(streamStep{err: io.EOF})}}}
	if err := (&bridgeSession{session: failed}).settledOutcome(testContext(t), sl.Position{}, "run-1", false); err == nil {
		t.Fatal("missing settlement ignored")
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	blocked := &bridgeSession{session: &fakeSession{subscribes: []subscribeStep{{stream: newFakeStream()}}}}
	if err := blocked.settledOutcome(cancelled, sl.Position{}, "run-1", false); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled replay err = %v", err)
	}
}

func TestSubmitContextCancellationWhileWaiting(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	inner := &fakeSession{
		subscribes: []subscribeStep{{stream: newFakeStream()}},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	done := make(chan error, 1)
	go func() { done <- (&bridgeSession{session: inner}).Submit(ctx, uit.Input{Text: "x"}) }()
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled submit err = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancelled submit never returned")
	}
}

func TestQueueOperations(t *testing.T) {
	t.Parallel()
	running := &fakeSession{snapshots: []snapshotStep{{snapshot: runningSnapshot("run-1")}}}
	target := &bridgeSession{session: running}
	if err := target.Steer(testContext(t), uit.Input{}); err == nil {
		t.Fatal("empty steer succeeded")
	}
	if err := target.Steer(testContext(t), uit.Input{Text: "left"}); err != nil {
		t.Fatal(err)
	}
	if err := target.FollowUp(testContext(t), uit.Input{Text: "and then"}); err != nil {
		t.Fatal(err)
	}
	if err := target.NextTurn(testContext(t), uit.Input{Text: "later"}); err != nil {
		t.Fatal(err)
	}
	commands := running.dispatched()
	if len(commands) != 3 || commands[0].Kind != sl.CommandSteer || commands[0].RunID != "run-1" ||
		commands[1].Kind != sl.CommandFollowUp || commands[2].Kind != sl.CommandNextTurn || commands[2].RunID != "" {
		t.Fatalf("commands = %#v", commands)
	}

	idle := &bridgeSession{session: &fakeSession{}}
	if err := idle.Steer(testContext(t), uit.Input{Text: "x"}); err != session.ErrNotRunning {
		t.Fatalf("idle steer err = %v", err)
	}
	closed := &bridgeSession{session: &fakeSession{snapshots: []snapshotStep{{snapshot: sl.Snapshot{State: sl.StateClosed}}}}}
	if err := closed.FollowUp(testContext(t), uit.Input{Text: "x"}); err != session.ErrSessionClosed {
		t.Fatalf("closed follow-up err = %v", err)
	}
	faulted := &bridgeSession{session: &fakeSession{snapshots: []snapshotStep{{snapshot: sl.Snapshot{State: sl.StateFaulted}}}}}
	if err := faulted.Steer(testContext(t), uit.Input{Text: "x"}); err != session.ErrSessionFaulted {
		t.Fatalf("faulted steer err = %v", err)
	}
	snapshotErr := &bridgeSession{session: &fakeSession{snapshots: []snapshotStep{{err: errors.New("down")}}}}
	if err := snapshotErr.Steer(testContext(t), uit.Input{Text: "x"}); err == nil {
		t.Fatal("steer snapshot error ignored")
	}
	stale := &bridgeSession{session: &fakeSession{
		snapshots:  []snapshotStep{{snapshot: runningSnapshot("run-1")}},
		dispatches: []dispatchStep{{err: fmt.Errorf("gone: %w", sl.ErrStaleRun)}},
	}}
	if err := stale.Steer(testContext(t), uit.Input{Text: "x"}); err != session.ErrNotRunning {
		t.Fatalf("stale steer err = %v", err)
	}
	suspendedErr := &bridgeSession{session: &fakeSession{
		snapshots:  []snapshotStep{{snapshot: sl.Snapshot{State: sl.StateSuspended, ActiveRunID: "run-1"}}},
		dispatches: []dispatchStep{{err: fmt.Errorf("%w: %w", sl.ErrSuspended, session.ErrSessionSuspended)}},
	}}
	if err := suspendedErr.Steer(testContext(t), uit.Input{Text: "x"}); err != session.ErrSessionSuspended {
		t.Fatalf("suspended steer err = %v", err)
	}
}

func TestResolveValidationAndDispatch(t *testing.T) {
	t.Parallel()
	suspendedSnapshot := sl.Snapshot{
		State: sl.StateSuspended, ActiveRunID: "run-1", Position: sl.Position{Sequence: 5},
		Suspension: &sl.Suspension{ID: "susp-1", Kind: "harness.permission.v1", Decisions: []sl.SuspensionDecision{
			{ID: "call-a", Name: "danger"}, {ID: "call-b", Name: "danger"},
		}},
	}
	build := func(dispatchErr error, stream *fakeStream) *fakeSession {
		return &fakeSession{
			snapshots:  []snapshotStep{{snapshot: suspendedSnapshot}},
			subscribes: []subscribeStep{{stream: stream}},
			dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}, err: dispatchErr}},
		}
	}

	target := &bridgeSession{session: build(nil, newFakeStream())}
	if err := target.Resolve(testContext(t), uit.Resolution{}); err == nil {
		t.Fatal("invalid resolution succeeded")
	}
	if err := target.Resolve(testContext(t), uit.Resolution{SuspensionID: "other"}); err == nil ||
		err.Error() != "resolution does not match the current suspension" {
		t.Fatalf("mismatch err = %v", err)
	}
	unknown := uit.Resolution{SuspensionID: "susp-1", Decisions: []uit.Decision{{CallID: "nope", Action: uit.DecisionApprove}}}
	if err := target.Resolve(testContext(t), unknown); err == nil ||
		err.Error() != `resolution contains unknown approval "nope"` {
		t.Fatalf("unknown err = %v", err)
	}
	incomplete := uit.Resolution{SuspensionID: "susp-1", Decisions: []uit.Decision{{CallID: "call-a", Action: uit.DecisionApprove}}}
	if err := target.Resolve(testContext(t), incomplete); err == nil ||
		err.Error() != "resolution has 1 decisions for 2 approvals" {
		t.Fatalf("incomplete err = %v", err)
	}

	complete := uit.Resolution{
		SuspensionID: "susp-1",
		Decisions: []uit.Decision{
			{CallID: "call-b", Action: uit.DecisionDeny, Reason: "no"},
			{CallID: "call-a", Action: uit.DecisionApprove, Reason: "yes"},
		},
		Prompt: &uit.Input{Text: "carry on"},
	}
	resumed := build(nil, newFakeStream(streamStep{event: settledEvent("run-1", sl.RunCompleted, "")}))
	if err := (&bridgeSession{session: resumed}).Resolve(testContext(t), complete); err != nil {
		t.Fatal(err)
	}
	commands := resumed.dispatched()
	if len(commands) != 1 || commands[0].Kind != sl.CommandResolve || commands[0].RunID != "run-1" {
		t.Fatalf("commands = %#v", commands)
	}
	decisions := commands[0].Resolution.Decisions
	if len(decisions) != 2 || decisions[0].ID != "call-a" || decisions[0].Action != sl.ResolutionApprove ||
		decisions[1].ID != "call-b" || decisions[1].Action != sl.ResolutionDeny || decisions[1].Reason != "no" {
		t.Fatalf("decisions = %#v", decisions)
	}
	if commands[0].Input == nil || commands[0].Input.Content[0].Text != "carry on" {
		t.Fatalf("continuation input = %#v", commands[0].Input)
	}

	snapshotErr := &bridgeSession{session: &fakeSession{snapshots: []snapshotStep{{err: errors.New("down")}}}}
	if err := snapshotErr.Resolve(testContext(t), uit.Resolution{SuspensionID: "susp-1"}); err == nil {
		t.Fatal("resolve snapshot error ignored")
	}
	subscribeErr := &fakeSession{
		snapshots:  []snapshotStep{{snapshot: suspendedSnapshot}},
		subscribes: []subscribeStep{{err: errors.New("no stream")}},
	}
	full := uit.Resolution{SuspensionID: "susp-1", Decisions: []uit.Decision{
		{CallID: "call-a", Action: uit.DecisionApprove}, {CallID: "call-b", Action: uit.DecisionApprove},
	}}
	if err := (&bridgeSession{session: subscribeErr}).Resolve(testContext(t), full); err == nil {
		t.Fatal("resolve subscribe error ignored")
	}
	stream := newFakeStream()
	dispatchErr := build(fmt.Errorf("%w: %w", sl.ErrCommandConflict, session.ErrRunClosing), stream)
	if err := (&bridgeSession{session: dispatchErr}).Resolve(testContext(t), full); err != session.ErrRunClosing {
		t.Fatalf("resolve dispatch err = %v", err)
	}
	if stream.closes == 0 {
		t.Fatal("resolve dispatch failure left the wait stream open")
	}
}

func TestInterruptBoundaries(t *testing.T) {
	t.Parallel()
	idle := &bridgeSession{session: &fakeSession{}}
	if err := idle.Interrupt(testContext(t)); err != session.ErrNotRunning {
		t.Fatalf("idle interrupt err = %v", err)
	}
	snapshotErr := &bridgeSession{session: &fakeSession{snapshots: []snapshotStep{{err: errors.New("down")}}}}
	if err := snapshotErr.Interrupt(testContext(t)); err == nil {
		t.Fatal("interrupt snapshot error ignored")
	}
	subscribeErr := &bridgeSession{session: &fakeSession{
		snapshots:  []snapshotStep{{snapshot: runningSnapshot("run-1")}},
		subscribes: []subscribeStep{{err: errors.New("no stream")}},
	}}
	if err := subscribeErr.Interrupt(testContext(t)); err == nil {
		t.Fatal("interrupt subscribe error ignored")
	}
	stale := &bridgeSession{session: &fakeSession{
		snapshots:  []snapshotStep{{snapshot: runningSnapshot("run-1")}},
		subscribes: []subscribeStep{{stream: newFakeStream()}},
		dispatches: []dispatchStep{{err: fmt.Errorf("gone: %w", sl.ErrStaleRun)}},
	}}
	if err := stale.Interrupt(testContext(t)); err != session.ErrNotRunning {
		t.Fatalf("stale interrupt err = %v", err)
	}
	failing := &bridgeSession{session: &fakeSession{
		snapshots:  []snapshotStep{{snapshot: runningSnapshot("run-1")}},
		subscribes: []subscribeStep{{stream: newFakeStream()}},
		dispatches: []dispatchStep{{err: errors.New("refused")}},
	}}
	if err := failing.Interrupt(testContext(t)); err == nil || err.Error() != "refused" {
		t.Fatalf("interrupt dispatch err = %v", err)
	}
	// A suspension of the targeted run never satisfies an interrupt wait;
	// only settlement does.
	waits := &fakeSession{
		snapshots: []snapshotStep{{snapshot: runningSnapshot("run-1")}},
		subscribes: []subscribeStep{{stream: newFakeStream(
			streamStep{event: sl.Event{Kind: sl.EventRunSuspended, RunID: "run-1"}},
			streamStep{event: settledEvent("run-1", sl.RunInterrupted, "context canceled")},
		)}},
		dispatches: []dispatchStep{{receipt: sl.Receipt{RunID: "run-1"}}},
	}
	if err := (&bridgeSession{session: waits}).Interrupt(testContext(t)); err != nil {
		t.Fatal(err)
	}
}

func TestCloseMapsWrappedSentinels(t *testing.T) {
	t.Parallel()
	inner := &fakeSession{closeErr: fmt.Errorf("%w: %w", sl.ErrSessionClosed, session.ErrSessionClosed)}
	target := &bridgeSession{session: inner}
	if err := target.Close(testContext(t)); err != session.ErrSessionClosed {
		t.Fatalf("close err = %v", err)
	}
	if inner.closes != 1 {
		t.Fatalf("closes = %d", inner.closes)
	}
}
