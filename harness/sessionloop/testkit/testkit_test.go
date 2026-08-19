package testkit_test

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/testkit"
)

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func openSession(t *testing.T, host *testkit.Host) sessionloop.Session {
	t.Helper()
	session, err := host.NewSession(testContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}
	t.Cleanup(func() { _ = session.Close(context.Background()) })
	return session
}

func textInput(text string) *sessionloop.Input {
	return &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: text}}}
}

func start(t *testing.T, session sessionloop.Session, text string) sessionloop.Receipt {
	t.Helper()
	receipt, err := session.Dispatch(testContext(t), sessionloop.Command{Kind: sessionloop.CommandStart, Input: textInput(text)})
	if err != nil {
		t.Fatalf("start dispatch failed: %v", err)
	}
	return receipt
}

func awaitSettled(t *testing.T, stream sessionloop.Stream, runID sessionloop.RunID) (sessionloop.Event, []sessionloop.Event) {
	t.Helper()
	var seen []sessionloop.Event
	for {
		event, err := stream.Next(testContext(t))
		if err != nil {
			t.Fatalf("Next while waiting for settlement of %q failed: %v", runID, err)
		}
		seen = append(seen, event)
		if event.Kind == sessionloop.EventRunSettled && event.RunID == runID {
			return event, seen
		}
	}
}

func firstInputText(input sessionloop.Input) string {
	for _, block := range input.Blocks {
		if block.Kind == sessionloop.InputBlockText {
			return block.Text
		}
	}
	return ""
}

func TestSteeredInputsAreCommittedAsEntriesBeforeDeliveryInDispatchOrder(t *testing.T) {
	t.Parallel()
	host := testkit.New(testkit.WithRunFunc(func(run *testkit.RunContext) error {
		first := <-run.Steered()
		second := <-run.Steered()
		run.EmitAssistant(firstInputText(first) + "|" + firstInputText(second))
		return nil
	}))
	session := openSession(t, host)
	stream, err := session.Subscribe(testContext(t), sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	release := host.HoldNextRun()
	receipt := start(t, session, "primary")
	if _, err := session.Dispatch(testContext(t), sessionloop.Command{Kind: sessionloop.CommandSteer, RunID: receipt.RunID, Input: textInput("one")}); err != nil {
		t.Fatalf("steer failed: %v", err)
	}
	if _, err := session.Dispatch(testContext(t), sessionloop.Command{Kind: sessionloop.CommandFollowUp, RunID: receipt.RunID, Input: textInput("two")}); err != nil {
		t.Fatalf("follow-up failed: %v", err)
	}
	release()

	_, seen := awaitSettled(t, stream, receipt.RunID)
	var origins []sessionloop.EntryOrigin
	var assistantText string
	for _, event := range seen {
		if event.Kind != sessionloop.EventEntryCommitted {
			continue
		}
		origins = append(origins, event.Entry.Origin)
		if event.Entry.Role == sessionloop.RoleAssistant {
			assistantText = event.Entry.Blocks[0].Text
		}
	}
	want := []sessionloop.EntryOrigin{
		sessionloop.OriginStart,
		sessionloop.OriginSteer,
		sessionloop.OriginFollowUp,
		sessionloop.OriginAssistant,
	}
	if len(origins) != len(want) {
		t.Fatalf("entry origins = %v, want %v", origins, want)
	}
	for index := range want {
		if origins[index] != want[index] {
			t.Fatalf("entry origins = %v, want %v", origins, want)
		}
	}
	if assistantText != "one|two" {
		t.Fatalf("steered inputs delivered out of order: assistant saw %q", assistantText)
	}
}

func TestSlowSubscriberFailsTerminallyWithErrLaggedOnAuthoritativeOverflow(t *testing.T) {
	t.Parallel()
	host := testkit.New()
	session := openSession(t, host)
	lagging, err := session.Subscribe(testContext(t), sessionloop.SubscribeOptions{Buffer: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = lagging.Close() }()
	healthy, err := session.Subscribe(testContext(t), sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = healthy.Close() }()

	receipt := start(t, session, "overflow the one-slot buffer")
	awaitSettled(t, healthy, receipt.RunID)

	if _, err := lagging.Next(testContext(t)); !errors.Is(err, sessionloop.ErrLagged) {
		t.Fatalf("Next on the lagging stream = %v, want ErrLagged", err)
	}
	if _, err := lagging.Next(testContext(t)); !errors.Is(err, sessionloop.ErrLagged) {
		t.Fatalf("lag must be terminal, second Next = %v, want ErrLagged", err)
	}
	if err := lagging.Close(); err != nil {
		t.Fatalf("closing a lagged stream failed: %v", err)
	}
}

func TestReopenRestoresDurableHistoryAndConcurrentOpenIsRefused(t *testing.T) {
	t.Parallel()
	host := testkit.New()
	session := openSession(t, host)
	id := session.ID()
	stream, err := session.Subscribe(testContext(t), sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	receipt := start(t, session, "durable fact")
	awaitSettled(t, stream, receipt.RunID)

	if _, err := host.OpenSession(testContext(t), id); !errors.Is(err, sessionloop.ErrSessionOpen) {
		t.Fatalf("second concurrent open = %v, want ErrSessionOpen", err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := stream.Next(testContext(t)); err != nil {
			if !errors.Is(err, io.EOF) {
				t.Fatalf("stream after close ended with %v, want io.EOF", err)
			}
			break
		}
	}

	reopened, err := host.OpenSession(testContext(t), id)
	if err != nil {
		t.Fatalf("reopen failed: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close(context.Background()) })
	snapshot, err := reopened.Snapshot(testContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != sessionloop.StateIdle || len(snapshot.Entries) != 2 {
		t.Fatalf("reopened snapshot = state %q entries %d, want idle with the 2 durable entries", snapshot.State, len(snapshot.Entries))
	}

	replay, err := reopened.Subscribe(testContext(t), sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = replay.Close() }()
	event, err := replay.Next(testContext(t))
	if err != nil || event.Position.Sequence != 1 {
		t.Fatalf("replay from zero must begin at sequence 1, got %#v, %v", event.Position, err)
	}

	if _, err := host.OpenSession(testContext(t), "session-does-not-exist"); err == nil {
		t.Fatal("opening an unknown session succeeded")
	}
}

func TestRunTargetedCommandsAreRejectedByStateAndRunIdentity(t *testing.T) {
	t.Parallel()
	host := testkit.New(testkit.WithRunFunc(testkit.ScenarioRunFunc()))
	session := openSession(t, host)
	ctx := testContext(t)

	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandSteer, RunID: "run-x", Input: textInput("early")}); !errors.Is(err, sessionloop.ErrNotRunning) {
		t.Fatalf("steer while idle = %v, want ErrNotRunning", err)
	}
	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "run-x"}); !errors.Is(err, sessionloop.ErrNotRunning) {
		t.Fatalf("interrupt while idle = %v, want ErrNotRunning", err)
	}
	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: "run-x", Resolution: &sessionloop.Resolution{SuspensionID: "susp-1"}}); !errors.Is(err, sessionloop.ErrNotRunning) {
		t.Fatalf("resolve while idle = %v, want ErrNotRunning", err)
	}

	release := host.HoldNextRun()
	held := start(t, session, "held run")
	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: held.RunID, Resolution: &sessionloop.Resolution{SuspensionID: "susp-1"}}); !errors.Is(err, sessionloop.ErrCommandConflict) {
		t.Fatalf("resolve while running unsuspended = %v, want ErrCommandConflict", err)
	}
	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandSteer, RunID: "run-stale", Input: textInput("wrong run")}); !errors.Is(err, sessionloop.ErrStaleRun) {
		t.Fatalf("steer with a foreign run ID = %v, want ErrStaleRun", err)
	}
	release()

	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	awaitSettled(t, stream, held.RunID)
}

func TestSuspensionRejectsMismatchesAndResolvesWithContinuationInput(t *testing.T) {
	t.Parallel()
	host := testkit.New(testkit.WithRunFunc(testkit.ScenarioRunFunc()))
	session := openSession(t, host)
	ctx := testContext(t)
	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	input := textInput("pause here")
	input.Meta = map[string]string{"conformance.scenario": "suspend"}
	receipt, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandStart, Input: input})
	if err != nil {
		t.Fatal(err)
	}
	for {
		event, nextErr := stream.Next(ctx)
		if nextErr != nil {
			t.Fatal(nextErr)
		}
		if event.Kind == sessionloop.EventRunSuspended {
			break
		}
	}
	snapshot, err := session.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	suspensionID := snapshot.Suspension.ID

	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandStart, Input: textInput("busy")}); !errors.Is(err, sessionloop.ErrSessionBusy) {
		t.Fatalf("start while suspended = %v, want ErrSessionBusy", err)
	}
	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandSteer, RunID: receipt.RunID, Input: textInput("nudge")}); !errors.Is(err, sessionloop.ErrSuspended) {
		t.Fatalf("steer while suspended = %v, want ErrSuspended", err)
	}
	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: "run-other", Resolution: &sessionloop.Resolution{SuspensionID: suspensionID}}); !errors.Is(err, sessionloop.ErrStaleRun) {
		t.Fatalf("resolve with a foreign run = %v, want ErrStaleRun", err)
	}
	if _, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: receipt.RunID, Resolution: &sessionloop.Resolution{SuspensionID: "susp-wrong"}}); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatalf("resolve with a wrong suspension ID = %v, want ErrInvalidCommand", err)
	}

	resolveInput := textInput("continue with this")
	if _, err := session.Dispatch(ctx, sessionloop.Command{
		Kind:       sessionloop.CommandResolve,
		RunID:      receipt.RunID,
		Input:      resolveInput,
		Resolution: &sessionloop.Resolution{SuspensionID: suspensionID, Decisions: []sessionloop.ResolutionDecision{{ID: "decision-1", Action: sessionloop.ResolutionApprove}}},
	}); err != nil {
		t.Fatalf("matching resolve failed: %v", err)
	}
	settled, seen := awaitSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome after resolve = %q, want completed", settled.Outcome.Kind)
	}
	continuation := false
	for _, event := range seen {
		if event.Kind == sessionloop.EventEntryCommitted && event.Entry.Origin == sessionloop.OriginFollowUp {
			continuation = true
		}
	}
	if !continuation {
		t.Fatal("the resolve continuation input was not committed as an entry")
	}
}

func TestInterruptWhileSuspendedSettlesTheRunAsInterrupted(t *testing.T) {
	t.Parallel()
	host := testkit.New(
		testkit.WithRunFunc(testkit.ScenarioRunFunc()),
		testkit.WithIdempotentDispatch(),
	)
	session := openSession(t, host)
	ctx := testContext(t)
	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	input := textInput("pause then interrupt")
	input.Meta = map[string]string{"conformance.scenario": "suspend"}
	receipt, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandStart, Input: input})
	if err != nil {
		t.Fatal(err)
	}
	for {
		event, nextErr := stream.Next(ctx)
		if nextErr != nil {
			t.Fatal(nextErr)
		}
		if event.Kind == sessionloop.EventRunSuspended {
			break
		}
	}
	interrupt := sessionloop.Command{
		Kind:           sessionloop.CommandInterrupt,
		RunID:          receipt.RunID,
		IdempotencyKey: "interrupt-suspended-run",
	}
	first, err := session.Dispatch(ctx, interrupt)
	if err != nil {
		t.Fatal(err)
	}
	second, err := session.Dispatch(ctx, interrupt)
	if err != nil {
		t.Fatalf("idempotent interrupt replay failed: %v", err)
	}
	if first != second {
		t.Fatalf("idempotent interrupt receipts diverge:\nfirst  %#v\nsecond %#v", first, second)
	}
	settled, _ := awaitSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunInterrupted {
		t.Fatalf("outcome = %q, want interrupted", settled.Outcome.Kind)
	}
}

func TestCloseKeepsASuspendedRunAndReopenRestoresTheSameSuspension(t *testing.T) {
	t.Parallel()
	host := testkit.New(testkit.WithRunFunc(testkit.ScenarioRunFunc()))
	session := openSession(t, host)
	id := session.ID()
	ctx := testContext(t)
	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}

	input := textInput("pause across close")
	input.Meta = map[string]string{"conformance.scenario": "suspend"}
	receipt, err := session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandStart, Input: input})
	if err != nil {
		t.Fatal(err)
	}
	for {
		event, nextErr := stream.Next(ctx)
		if nextErr != nil {
			t.Fatal(nextErr)
		}
		if event.Kind == sessionloop.EventRunSuspended {
			break
		}
	}
	before, err := session.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	suspensionID := before.Suspension.ID

	// Close must NOT settle the suspended run (law L11: a suspension is a
	// durable pause; closing releases the handle, never the session).
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	for {
		event, nextErr := stream.Next(ctx)
		if nextErr != nil {
			if !errors.Is(nextErr, io.EOF) {
				t.Fatalf("stream after close ended with %v, want io.EOF", nextErr)
			}
			break
		}
		if event.Kind == sessionloop.EventRunSettled {
			t.Fatalf("closing a suspended session settled its run: %#v", event)
		}
	}

	reopened, err := host.OpenSession(ctx, id)
	if err != nil {
		t.Fatalf("reopen of a suspended session failed: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close(context.Background()) })
	snapshot, err := reopened.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != sessionloop.StateSuspended || snapshot.ActiveRunID != receipt.RunID ||
		snapshot.Suspension == nil || snapshot.Suspension.ID != suspensionID {
		t.Fatalf("reopened snapshot = state %q run %q suspension %#v, want the surviving suspension %q",
			snapshot.State, snapshot.ActiveRunID, snapshot.Suspension, suspensionID)
	}

	replay, err := reopened.Subscribe(ctx, sessionloop.SubscribeOptions{Buffer: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = replay.Close() }()
	if _, err := reopened.Dispatch(ctx, sessionloop.Command{
		Kind:  sessionloop.CommandResolve,
		RunID: receipt.RunID,
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspensionID,
			Decisions:    []sessionloop.ResolutionDecision{{ID: "decision-1", Action: sessionloop.ResolutionApprove}},
		},
	}); err != nil {
		t.Fatalf("resolve after reopen failed: %v", err)
	}
	settled, _ := awaitSettled(t, replay, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome after reopen and resolve = %q, want completed", settled.Outcome.Kind)
	}
}

func TestIdempotencyKeysReturnTheOriginalReceiptAndSurviveReopen(t *testing.T) {
	t.Parallel()
	host := testkit.New(testkit.WithIdempotentDispatch())
	session := openSession(t, host)
	id := session.ID()
	ctx := testContext(t)
	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	command := sessionloop.Command{Kind: sessionloop.CommandStart, Input: textInput("exactly once"), IdempotencyKey: "key-1"}
	first, err := session.Dispatch(ctx, command)
	if err != nil {
		t.Fatal(err)
	}
	second, err := session.Dispatch(ctx, command)
	if err != nil {
		t.Fatalf("idempotent replay failed: %v", err)
	}
	if first != second {
		t.Fatalf("idempotent receipts diverge:\nfirst  %#v\nsecond %#v", first, second)
	}
	_, seen := awaitSettled(t, stream, first.RunID)
	starts := 0
	for _, event := range seen {
		if event.Kind == sessionloop.EventRunStarted {
			starts++
		}
	}
	if starts != 1 {
		t.Fatalf("idempotent start produced %d run.started events, want exactly one", starts)
	}

	// Keys share the durable log's lifetime: a reopened handle still replays.
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	reopened, err := host.OpenSession(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reopened.Close(context.Background()) })
	replayed, err := reopened.Dispatch(ctx, command)
	if err != nil {
		t.Fatalf("idempotent replay after reopen failed: %v", err)
	}
	if replayed != first {
		t.Fatalf("replay after reopen diverged:\nfirst    %#v\nreplayed %#v", first, replayed)
	}
	if snapshot, err := reopened.Snapshot(ctx); err != nil || snapshot.ActiveRunID != "" {
		t.Fatalf("replay after reopen started a duplicate run: %#v err=%v", snapshot, err)
	}
}

func TestRunFuncErrorSettlesTheRunAsFailed(t *testing.T) {
	t.Parallel()
	host := testkit.New(testkit.WithRunFunc(func(*testkit.RunContext) error {
		return errors.New("model exploded")
	}))
	session := openSession(t, host)
	stream, err := session.Subscribe(testContext(t), sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	receipt := start(t, session, "doomed")
	settled, _ := awaitSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunFailed || !strings.Contains(settled.Outcome.Failure, "model exploded") {
		t.Fatalf("outcome = %#v, want failed with the sanitized message", settled.Outcome)
	}
	snapshot, err := session.Snapshot(testContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != sessionloop.StateIdle {
		t.Fatalf("a failed run must settle back to idle, state = %q", snapshot.State)
	}
}

func TestCloseInterruptsTheActiveRunAndEndsStreamsAfterTheClosedState(t *testing.T) {
	t.Parallel()
	host := testkit.New()
	session := openSession(t, host)
	stream, err := session.Subscribe(testContext(t), sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	release := host.HoldNextRun()
	defer release()
	receipt := start(t, session, "held forever")
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("Close is not idempotent: %v", err)
	}

	var settled *sessionloop.Event
	var last sessionloop.Event
	for {
		event, nextErr := stream.Next(testContext(t))
		if nextErr != nil {
			if !errors.Is(nextErr, io.EOF) {
				t.Fatalf("stream ended with %v, want io.EOF", nextErr)
			}
			break
		}
		last = event
		if event.Kind == sessionloop.EventRunSettled {
			settledCopy := event
			settled = &settledCopy
		}
	}
	if settled == nil || settled.RunID != receipt.RunID || settled.Outcome.Kind != sessionloop.RunInterrupted {
		t.Fatalf("close must settle the held run as interrupted, got %#v", settled)
	}
	if last.Kind != sessionloop.EventSessionState || last.State != sessionloop.StateClosed {
		t.Fatalf("the final event must report the closed state, got %#v", last)
	}
	if _, err := session.Subscribe(testContext(t), sessionloop.SubscribeOptions{}); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatalf("Subscribe after Close = %v, want ErrSessionClosed", err)
	}
}

func TestSubscribeRejectsForeignPositionsAndDispatchHonorsAcceptanceContext(t *testing.T) {
	t.Parallel()
	host := testkit.New()
	session := openSession(t, host)

	if _, err := session.Subscribe(testContext(t), sessionloop.SubscribeOptions{After: sessionloop.Position{Sequence: 999, Token: "tk-999"}}); !errors.Is(err, sessionloop.ErrUnknownPosition) {
		t.Fatalf("subscribing beyond history = %v, want ErrUnknownPosition", err)
	}
	if _, err := session.Subscribe(testContext(t), sessionloop.SubscribeOptions{After: sessionloop.Position{Sequence: 1, Token: "forged"}}); !errors.Is(err, sessionloop.ErrUnknownPosition) {
		t.Fatalf("subscribing with a forged token = %v, want ErrUnknownPosition", err)
	}
	if _, err := session.Dispatch(testContext(t), sessionloop.Command{Kind: sessionloop.CommandStart, Input: textInput("x"), IdempotencyKey: "key"}); !errors.Is(err, sessionloop.ErrUnsupported) {
		t.Fatalf("idempotency key without the capability = %v, want ErrUnsupported", err)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := session.Dispatch(canceled, sessionloop.Command{Kind: sessionloop.CommandStart, Input: textInput("x")}); !errors.Is(err, context.Canceled) {
		t.Fatalf("dispatch with a canceled context = %v, want context.Canceled", err)
	}
	if _, err := host.NewSession(canceled, sessionloop.SessionOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("NewSession with a canceled context = %v, want context.Canceled", err)
	}
}
