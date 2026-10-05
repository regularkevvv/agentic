package realtime_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/testkit"
	"github.com/regularkevvv/agentic/realtime"
)

var errInjected = errors.New("injected")

// stubSession overrides selected methods of a real session.
type stubSession struct {
	sessionloop.Session
	subscribe func(context.Context, sessionloop.SubscribeOptions) (sessionloop.Stream, error)
	snapshot  func(context.Context) (sessionloop.Snapshot, error)
	dispatch  func(context.Context, sessionloop.Command) (sessionloop.Receipt, error)
}

func (s stubSession) Subscribe(ctx context.Context, options sessionloop.SubscribeOptions) (sessionloop.Stream, error) {
	if s.subscribe != nil {
		return s.subscribe(ctx, options)
	}
	return s.Session.Subscribe(ctx, options)
}

func (s stubSession) Snapshot(ctx context.Context) (sessionloop.Snapshot, error) {
	if s.snapshot != nil {
		return s.snapshot(ctx)
	}
	return s.Session.Snapshot(ctx)
}

func (s stubSession) Dispatch(ctx context.Context, command sessionloop.Command) (sessionloop.Receipt, error) {
	if s.dispatch != nil {
		return s.dispatch(ctx, command)
	}
	return s.Session.Dispatch(ctx, command)
}

type failingStream struct{ err error }

func (f failingStream) Next(context.Context) (sessionloop.Event, error) {
	return sessionloop.Event{}, f.err
}
func (failingStream) Close() error { return nil }

func sessionWith(t *testing.T, run testkit.RunFunc) sessionloop.Session {
	t.Helper()
	session, err := testkit.New(testkit.WithRunFunc(run)).NewSession(t.Context(), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close(context.Background()) })
	return session
}

// delegateOnce runs one delegate call against session and returns the tool
// output the voice model receives.
func delegateOnce(t *testing.T, session sessionloop.Session) string {
	t.Helper()
	conn, done := start(t, session)
	conn.next(t)
	conn.events <- delegateCall("call-1", "do it")
	result := conn.next(t)
	if result.Kind != realtime.ActionToolResult {
		t.Fatalf("got %+v, want tool result", result)
	}
	conn.next(t)
	end(t, conn, done)
	return result.Result.Output
}

func TestRunRejectsMissingConnOrSession(t *testing.T) {
	if err := realtime.Run(t.Context(), nil, newSession(t), realtime.Options{}); err == nil {
		t.Fatal("nil conn accepted")
	}
	if err := realtime.Run(t.Context(), newFakeConn(), nil, realtime.Options{}); err == nil {
		t.Fatal("nil session accepted")
	}
}

func TestRunReportsSetupFailures(t *testing.T) {
	seeded := newSession(t)
	if output := delegateOnce(t, seeded); output != "echo: do it" {
		t.Fatalf("seed run = %q", output)
	}
	cases := map[string]struct {
		session sessionloop.Session
		fail    realtime.ActionKind
	}{
		"subscribe": {session: stubSession{Session: newSession(t),
			subscribe: func(context.Context, sessionloop.SubscribeOptions) (sessionloop.Stream, error) {
				return nil, errInjected
			}}},
		"snapshot": {session: stubSession{Session: newSession(t),
			snapshot: func(context.Context) (sessionloop.Snapshot, error) { return sessionloop.Snapshot{}, errInjected }}},
		"configure": {session: newSession(t), fail: realtime.ActionConfigure},
		"history":   {session: seeded, fail: realtime.ActionHistory},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			conn := newFakeConn()
			conn.fail = map[realtime.ActionKind]error{tc.fail: errInjected}
			if err := realtime.Run(t.Context(), conn, tc.session, realtime.Options{}); !errors.Is(err, errInjected) {
				t.Fatalf("Run = %v, want injected failure", err)
			}
		})
	}
}

func TestLostSessionStreamEndsTheCall(t *testing.T) {
	session := stubSession{Session: newSession(t),
		subscribe: func(context.Context, sessionloop.SubscribeOptions) (sessionloop.Stream, error) {
			return failingStream{err: sessionloop.ErrLagged}, nil
		}}
	if err := realtime.Run(t.Context(), newFakeConn(), session, realtime.Options{}); !errors.Is(err, sessionloop.ErrLagged) {
		t.Fatalf("Run = %v, want ErrLagged", err)
	}
}

func TestFailedSendsEndTheCall(t *testing.T) {
	for _, kind := range []realtime.ActionKind{realtime.ActionToolResult, realtime.ActionRespond} {
		t.Run(string(kind), func(t *testing.T) {
			conn := newFakeConn()
			conn.fail = map[realtime.ActionKind]error{kind: errInjected}
			done := make(chan error, 1)
			go func() { done <- realtime.Run(t.Context(), conn, newSession(t), realtime.Options{}) }()
			conn.next(t)
			conn.events <- delegateCall("call-1", "hello")
			if err := <-done; !errors.Is(err, errInjected) {
				t.Fatalf("Run = %v, want injected failure", err)
			}
		})
	}
}

func TestRunOutcomesBecomeToolOutputs(t *testing.T) {
	cases := map[string]struct {
		run  testkit.RunFunc
		want string
	}{
		"silent completion": {run: func(*testkit.RunContext) error { return nil }, want: "Done."},
		"failure":           {run: func(*testkit.RunContext) error { return errInjected }, want: "The task failed."},
		"approval": {run: func(run *testkit.RunContext) error {
			_, err := run.Suspend(sessionloop.Suspension{ID: "s1", Kind: "approval", Description: "send the email"})
			return err
		}, want: "The task is paused awaiting approval: send the email"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := delegateOnce(t, sessionWith(t, tc.run)); got != tc.want {
				t.Fatalf("output = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestInterruptedRunIsReported(t *testing.T) {
	session := sessionWith(t, func(run *testkit.RunContext) error {
		<-run.Interrupted()
		return nil
	})
	conn, done := start(t, session)
	conn.next(t)
	conn.events <- delegateCall("call-1", "long task")
	snapshot := waitRunning(t, session)
	if _, err := session.Dispatch(t.Context(), sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: snapshot.ActiveRunID}); err != nil {
		t.Fatal(err)
	}
	if result := conn.next(t); result.Result.Output != "The task was interrupted before it finished." {
		t.Fatalf("output = %+v", result.Result)
	}
	conn.next(t)
	end(t, conn, done)
}

func TestRejectedDispatchesAreExplained(t *testing.T) {
	inner := newSession(t)
	busy := func(_ context.Context, command sessionloop.Command) (sessionloop.Receipt, error) {
		if command.Kind == sessionloop.CommandStart {
			return sessionloop.Receipt{}, sessionloop.ErrSessionBusy
		}
		return sessionloop.Receipt{}, errInjected
	}
	steerable := sessionloop.Snapshot{Capabilities: sessionloop.NewCapabilities(sessionloop.CapabilitySteer)}
	snapshots := func(later sessionloop.Snapshot, err error) func(context.Context) (sessionloop.Snapshot, error) {
		calls := 0
		return func(context.Context) (sessionloop.Snapshot, error) {
			calls++
			if calls == 1 {
				return steerable, nil
			}
			return later, err
		}
	}
	const busyText = "The backend is busy; ask again in a moment."
	cases := map[string]struct {
		session sessionloop.Session
		want    string
	}{
		"not steerable": {session: stubSession{Session: inner, dispatch: busy,
			snapshot: func(context.Context) (sessionloop.Snapshot, error) { return sessionloop.Snapshot{}, nil }},
			want: "The backend did not accept the request: " + sessionloop.ErrSessionBusy.Error()},
		"snapshot fails": {session: stubSession{Session: inner, dispatch: busy,
			snapshot: snapshots(sessionloop.Snapshot{}, errInjected)}, want: busyText},
		"no active run": {session: stubSession{Session: inner, dispatch: busy,
			snapshot: snapshots(sessionloop.Snapshot{}, nil)}, want: busyText},
		"steer rejected": {session: stubSession{Session: inner, dispatch: busy,
			snapshot: snapshots(sessionloop.Snapshot{ActiveRunID: "run-1"}, nil)}, want: busyText},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := delegateOnce(t, tc.session); got != tc.want {
				t.Fatalf("output = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestEndingTheCallLeavesTheRunToTheSession(t *testing.T) {
	release := make(chan struct{})
	session := sessionWith(t, func(run *testkit.RunContext) error {
		<-release
		run.EmitAssistant("finished after the call")
		return nil
	})
	ctx, cancel := context.WithCancel(t.Context())
	conn := newFakeConn()
	done := make(chan error, 1)
	go func() { done <- realtime.Run(ctx, conn, session, realtime.Options{}) }()
	conn.next(t)
	conn.events <- delegateCall("call-1", "slow")
	waitRunning(t, session)
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("Run = %v, want context.Canceled", err)
	}
	conn.quiet(t)
	close(release)
	if snapshot := waitIdle(t, session); len(snapshot.Entries) != 2 {
		t.Fatalf("run did not finish in the session: %+v", snapshot.Entries)
	}
}

func TestObserveSeesEveryEventAndIgnoresMalformedCalls(t *testing.T) {
	var seen []realtime.EventKind
	conn := newFakeConn()
	done := make(chan error, 1)
	go func() {
		done <- realtime.Run(t.Context(), conn, newSession(t), realtime.Options{
			Observe: func(event realtime.Event) { seen = append(seen, event.Kind) },
		})
	}()
	conn.next(t)
	conn.events <- realtime.Event{Kind: realtime.EventToolCall}
	conn.events <- realtime.Event{Kind: realtime.EventUserTranscript, Text: "hi"}
	conn.events <- realtime.Event{Kind: realtime.EventResponseDone}
	end(t, conn, done)
	if len(seen) != 3 || seen[0] != realtime.EventToolCall || seen[2] != realtime.EventResponseDone {
		t.Fatalf("observed %v", seen)
	}
}

func TestHistorySkipsToolTrafficAndNonTextBlocks(t *testing.T) {
	session := stubSession{Session: newSession(t), snapshot: func(context.Context) (sessionloop.Snapshot, error) {
		return sessionloop.Snapshot{Entries: []sessionloop.Entry{
			{Role: sessionloop.RoleUser, Blocks: []sessionloop.EntryBlock{
				{Kind: sessionloop.EntryBlockText, Text: "first"},
				{Kind: sessionloop.EntryBlockData, Data: json.RawMessage(`{}`)},
				{Kind: sessionloop.EntryBlockText, Text: "second"},
			}},
			{Role: sessionloop.RoleTool, Blocks: []sessionloop.EntryBlock{{Kind: sessionloop.EntryBlockText, Text: "tool"}}},
			{Role: sessionloop.RoleAssistant, Blocks: []sessionloop.EntryBlock{{Kind: sessionloop.EntryBlockText, Text: "  "}}},
		}}, nil
	}}
	conn, done := start(t, session)
	conn.next(t)
	history := conn.next(t)
	if len(history.History) != 1 || history.History[0] != (realtime.Turn{Role: realtime.RoleUser, Text: "first\nsecond"}) {
		t.Fatalf("history = %+v", history.History)
	}
	end(t, conn, done)
}
