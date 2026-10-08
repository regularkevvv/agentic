package realtime_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/testkit"
	"github.com/regularkevvv/agentic/realtime"
)

// fakeConn is a scripted call: the test pushes provider events and reads the
// actions the bridge sent.
type fakeConn struct {
	events  chan realtime.Event
	actions chan realtime.Action
	fail    map[realtime.ActionKind]error
}

func newFakeConn() *fakeConn {
	return &fakeConn{events: make(chan realtime.Event, 16), actions: make(chan realtime.Action, 16)}
}

func (c *fakeConn) Send(_ context.Context, action realtime.Action) error {
	if err := c.fail[action.Kind]; err != nil {
		return err
	}
	c.actions <- action
	return nil
}

func (c *fakeConn) Recv(ctx context.Context) (realtime.Event, error) {
	select {
	case event, ok := <-c.events:
		if !ok {
			return realtime.Event{}, io.EOF
		}
		return event, nil
	case <-ctx.Done():
		return realtime.Event{}, ctx.Err()
	}
}

func (c *fakeConn) Close() error { return nil }

func (c *fakeConn) next(t *testing.T) realtime.Action {
	t.Helper()
	select {
	case action := <-c.actions:
		return action
	case <-time.After(5 * time.Second):
		t.Fatal("no action sent")
		return realtime.Action{}
	}
}

func (c *fakeConn) quiet(t *testing.T) {
	t.Helper()
	select {
	case action := <-c.actions:
		t.Fatalf("unexpected action %+v", action)
	case <-time.After(50 * time.Millisecond):
	}
}

func delegateCall(id, request string) realtime.Event {
	args, _ := json.Marshal(map[string]string{"request": request})
	return realtime.Event{Kind: realtime.EventToolCall, Call: &realtime.ToolCall{
		ID: id, Name: realtime.DelegateTool, Arguments: args,
	}}
}

func start(t *testing.T, session sessionloop.Session) (*fakeConn, <-chan error) {
	t.Helper()
	conn := newFakeConn()
	done := make(chan error, 1)
	go func() { done <- realtime.Run(t.Context(), conn, session, realtime.Options{Voice: "test"}) }()
	return conn, done
}

func end(t *testing.T, conn *fakeConn, done <-chan error) {
	t.Helper()
	close(conn.events)
	if err := <-done; !errors.Is(err, io.EOF) {
		t.Fatalf("Run returned %v, want io.EOF", err)
	}
}

func newSession(t *testing.T) sessionloop.Session {
	t.Helper()
	session, err := testkit.New().NewSession(t.Context(), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close(context.Background()) })
	return session
}

func TestDelegateCallBecomesSessionRunAndSpokenResult(t *testing.T) {
	session := newSession(t)
	conn, done := start(t, session)

	configure := conn.next(t)
	if configure.Kind != realtime.ActionConfigure || configure.Config.Voice != "test" ||
		len(configure.Config.Tools) != 1 || configure.Config.Tools[0].Name != realtime.DelegateTool ||
		configure.Config.Instructions != realtime.DefaultInstructions {
		t.Fatalf("configure = %+v", configure)
	}

	conn.events <- delegateCall("call-1", "what is on my calendar")
	result := conn.next(t)
	if result.Kind != realtime.ActionToolResult || result.Result.CallID != "call-1" ||
		result.Result.Output != "echo: what is on my calendar" {
		t.Fatalf("result = %+v", result.Result)
	}
	if respond := conn.next(t); respond.Kind != realtime.ActionRespond {
		t.Fatalf("after result got %+v, want respond", respond)
	}
	end(t, conn, done)

	snapshot, err := session.Snapshot(t.Context())
	if err != nil || len(snapshot.Entries) != 2 || snapshot.Entries[0].Blocks[0].Text != "what is on my calendar" {
		t.Fatalf("session did not own the exchange: %+v %v", snapshot.Entries, err)
	}
}

func TestNextCallIsSeededFromCommittedConversation(t *testing.T) {
	session := newSession(t)
	conn, done := start(t, session)
	conn.next(t)
	conn.events <- delegateCall("call-1", "remember lima")
	conn.next(t)
	conn.next(t)
	end(t, conn, done)

	conn, done = start(t, session)
	conn.next(t)
	history := conn.next(t)
	want := []realtime.Turn{{Role: realtime.RoleUser, Text: "remember lima"}, {Role: realtime.RoleAssistant, Text: "echo: remember lima"}}
	if history.Kind != realtime.ActionHistory || len(history.History) != 2 ||
		history.History[0] != want[0] || history.History[1] != want[1] {
		t.Fatalf("history = %+v", history)
	}
	end(t, conn, done)
}

func TestRespondWaitsForTheResponseInProgress(t *testing.T) {
	conn, done := start(t, newSession(t))
	conn.next(t)
	conn.events <- realtime.Event{Kind: realtime.EventResponseStarted}
	conn.events <- delegateCall("call-1", "hello")
	if result := conn.next(t); result.Kind != realtime.ActionToolResult {
		t.Fatalf("got %+v, want tool result", result)
	}
	conn.quiet(t)
	conn.events <- realtime.Event{Kind: realtime.EventResponseDone, Text: "completed"}
	if respond := conn.next(t); respond.Kind != realtime.ActionRespond {
		t.Fatalf("got %+v, want deferred respond", respond)
	}
	end(t, conn, done)
}

func TestBusySessionIsSteered(t *testing.T) {
	steered := make(chan sessionloop.Input, 1)
	host := testkit.New(testkit.WithRunFunc(func(run *testkit.RunContext) error {
		input := <-run.Steered()
		steered <- input
		run.EmitAssistant("answered both")
		return nil
	}))
	session, err := host.NewSession(t.Context(), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	conn, done := start(t, session)
	conn.next(t)

	conn.events <- delegateCall("call-1", "book a table")
	waitRunning(t, session)
	conn.events <- delegateCall("call-2", "for four people")

	first, second := conn.next(t), conn.next(t)
	if second.Kind == realtime.ActionToolResult && second.Result.CallID == "call-2" {
		first, second = second, first
	}
	if first.Result == nil || first.Result.CallID != "call-2" ||
		first.Result.Output != "Added to the task already in progress. Its answer will follow; don't repeat this request." {
		t.Fatalf("steer result = %+v", first)
	}
	if input := <-steered; input.Blocks[0].Text != "for four people" {
		t.Fatalf("steered input = %+v", input)
	}
	// call-2's result and respond, then call-1's result. Its respond is
	// deferred: the fake never reports the first response as done.
	for _, action := range []realtime.Action{second, conn.next(t)} {
		if action.Kind == realtime.ActionToolResult && action.Result.CallID == "call-1" {
			if action.Result.Output != "answered both" {
				t.Fatalf("run result = %q", action.Result.Output)
			}
			end(t, conn, done)
			return
		}
	}
	t.Fatal("first call never received the run's answer")
}

func TestUnknownToolAndMissingRequestAreAnsweredWithoutDispatch(t *testing.T) {
	session := newSession(t)
	conn, done := start(t, session)
	conn.next(t)
	conn.events <- realtime.Event{Kind: realtime.EventToolCall, Call: &realtime.ToolCall{ID: "a", Name: "transfer_funds"}}
	if result := conn.next(t); result.Result.Output != `Unknown tool "transfer_funds". Only "delegate" is available.` {
		t.Fatalf("unknown tool result = %+v", result.Result)
	}
	conn.next(t)
	conn.events <- realtime.Event{Kind: realtime.EventResponseDone}
	conn.events <- realtime.Event{Kind: realtime.EventToolCall, Call: &realtime.ToolCall{
		ID: "b", Name: realtime.DelegateTool, Arguments: json.RawMessage(`{"request":"  "}`),
	}}
	if result := conn.next(t); result.Result.Output != "The request argument is required." {
		t.Fatalf("missing request result = %+v", result.Result)
	}
	conn.next(t)
	end(t, conn, done)
	if snapshot, _ := session.Snapshot(t.Context()); len(snapshot.Entries) != 0 {
		t.Fatalf("invalid calls reached the session: %+v", snapshot.Entries)
	}
}

func waitRunning(t *testing.T, session sessionloop.Session) sessionloop.Snapshot {
	t.Helper()
	return waitState(t, session, sessionloop.StateRunning)
}

func waitIdle(t *testing.T, session sessionloop.Session) sessionloop.Snapshot {
	t.Helper()
	return waitState(t, session, sessionloop.StateIdle)
}

func waitState(t *testing.T, session sessionloop.Session, state sessionloop.State) sessionloop.Snapshot {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if snapshot, err := session.Snapshot(t.Context()); err == nil && snapshot.State == state {
			return snapshot
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("session never reached %s", state)
	return sessionloop.Snapshot{}
}
