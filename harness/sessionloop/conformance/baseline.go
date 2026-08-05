package conformance

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func caseStartReceiptRunIdentity(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	receipt := dispatch(t, session, startCommand(textInput("hello conformance")))
	if receipt.RunID == "" {
		t.Fatalf("start receipt carries no run identity: %#v", receipt)
	}
	if receipt.SessionID != session.ID() {
		t.Fatalf("start receipt session = %q, session ID = %q", receipt.SessionID, session.ID())
	}

	started, _ := awaitKind(t, stream, sessionloop.EventRunStarted)
	if started.RunID != receipt.RunID {
		t.Fatalf("run.started run = %q, receipt run = %q", started.RunID, receipt.RunID)
	}
	settled, seen := awaitSettled(t, stream, receipt.RunID)
	if settled.Outcome.RunID != receipt.RunID {
		t.Fatalf("settled outcome run = %q, receipt run = %q", settled.Outcome.RunID, receipt.RunID)
	}
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome = %q, want completed", settled.Outcome.Kind)
	}
	assistant := 0
	for _, entry := range committedEntries(seen, receipt.RunID) {
		if entry.Role == sessionloop.RoleAssistant {
			assistant++
		}
	}
	if assistant == 0 {
		t.Fatal("the factory contract requires at least one committed assistant entry per start")
	}
}

func caseAuthoritativeEntryOrder(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	receipt := dispatch(t, session, startCommand(textInput("order the transcript")))
	_, seen := awaitSettled(t, stream, receipt.RunID)

	entries := committedEntries(seen, receipt.RunID)
	if len(entries) < 2 {
		t.Fatalf("expected at least a user and an assistant entry, got %#v", entries)
	}
	if entries[0].Role != sessionloop.RoleUser || entries[0].Origin != sessionloop.OriginStart {
		t.Fatalf("first entry = role %q origin %q, want the start user entry", entries[0].Role, entries[0].Origin)
	}
	firstAssistant := -1
	for index, entry := range entries {
		if entry.Role == sessionloop.RoleAssistant {
			firstAssistant = index
			break
		}
	}
	if firstAssistant <= 0 {
		t.Fatalf("no assistant entry after the start user entry: %#v", entries)
	}
	if seen[len(seen)-1].Kind != sessionloop.EventRunSettled {
		t.Fatalf("settlement was not the last authoritative fact of the run: %#v", seen[len(seen)-1])
	}
}

func caseOneSettlementPerRun(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	first := dispatch(t, session, startCommand(textInput("first run")))
	_, seenFirst := awaitSettled(t, stream, first.RunID)
	second := dispatch(t, session, startCommand(textInput("second run")))
	_, seenSecond := awaitSettled(t, stream, second.RunID)

	settlements := map[sessionloop.RunID]int{}
	for _, event := range append(seenFirst, seenSecond...) {
		if event.Kind == sessionloop.EventRunSettled {
			settlements[event.RunID]++
		}
	}
	if settlements[first.RunID] != 1 || settlements[second.RunID] != 1 {
		t.Fatalf("settlement counts = %v, want exactly one per run", settlements)
	}
}

func caseIdleAfterSettlement(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	first := dispatch(t, session, startCommand(textInput("settle me")))
	awaitSettled(t, stream, first.RunID)

	snapshot := snapshotOf(t, session)
	if snapshot.State != sessionloop.StateIdle {
		t.Fatalf("state after settlement = %q, want idle", snapshot.State)
	}
	if snapshot.ActiveRunID != "" {
		t.Fatalf("active run after settlement = %q, want none", snapshot.ActiveRunID)
	}

	second := dispatch(t, session, startCommand(textInput("run again")))
	settled, _ := awaitSettled(t, stream, second.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("second run outcome = %q, want completed", settled.Outcome.Kind)
	}
}

func caseSnapshotCopyOwnership(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})
	receipt := dispatch(t, session, startCommand(textInput("own your copies")))
	awaitSettled(t, stream, receipt.RunID)

	snapshot := snapshotOf(t, session)
	original := snapshot.Clone()

	snapshot.SessionID = "mutated"
	snapshot.State = sessionloop.StateFaulted
	snapshot.Position.Token = "mutated"
	if len(snapshot.Entries) > 0 && len(snapshot.Entries[0].Content) > 0 {
		snapshot.Entries[0].Content[0].Text = "mutated"
	}
	if len(snapshot.Capabilities) > 0 {
		snapshot.Capabilities[0] = "mutated.capability"
	}
	snapshot.Pending = append(snapshot.Pending, sessionloop.QueuedInput{ID: "forged"})
	snapshot.Usage.TotalTokens += 1000

	fresh := snapshotOf(t, session)
	if !reflect.DeepEqual(fresh, original) {
		t.Fatalf("mutating a returned snapshot leaked into the host:\nbefore %#v\nafter  %#v", original, fresh)
	}
}

func caseStreamCloseAndCanceledNext(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	first := subscribe(t, session, sessionloop.SubscribeOptions{})
	if _, err := first.Next(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("Next with a canceled context = %v, want context.Canceled", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("closing a stream after a canceled Next failed: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("Stream.Close is not idempotent: %v", err)
	}
	if _, err := first.Next(watchdogContext(t)); !errors.Is(err, io.EOF) {
		t.Fatalf("Next after Close = %v, want io.EOF", err)
	}

	second := subscribe(t, session, sessionloop.SubscribeOptions{})
	result := make(chan error, 1)
	waiting := watchdogContext(t)
	go func() {
		for {
			if _, err := second.Next(waiting); err != nil {
				result <- err
				return
			}
		}
	}()
	if err := second.Close(); err != nil {
		t.Fatalf("closing a stream mid wait failed: %v", err)
	}
	if err := <-result; !errors.Is(err, io.EOF) {
		t.Fatalf("Next during Close = %v, want io.EOF", err)
	}
}

func caseInvalidCommandMatrix(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	resolution := &sessionloop.Resolution{SuspensionID: "susp-1"}
	input := textInput("structurally present")

	violations := []struct {
		name    string
		command sessionloop.Command
	}{
		{"start with a run target", sessionloop.Command{Kind: sessionloop.CommandStart, RunID: "run-1", Input: input}},
		{"start without input", sessionloop.Command{Kind: sessionloop.CommandStart}},
		{"start with a resolution", sessionloop.Command{Kind: sessionloop.CommandStart, Input: input, Resolution: resolution}},
		{"steer without a run target", sessionloop.Command{Kind: sessionloop.CommandSteer, Input: input}},
		{"steer without input", sessionloop.Command{Kind: sessionloop.CommandSteer, RunID: "run-1"}},
		{"steer with a resolution", sessionloop.Command{Kind: sessionloop.CommandSteer, RunID: "run-1", Input: input, Resolution: resolution}},
		{"follow-up without a run target", sessionloop.Command{Kind: sessionloop.CommandFollowUp, Input: input}},
		{"follow-up without input", sessionloop.Command{Kind: sessionloop.CommandFollowUp, RunID: "run-1"}},
		{"follow-up with a resolution", sessionloop.Command{Kind: sessionloop.CommandFollowUp, RunID: "run-1", Input: input, Resolution: resolution}},
		{"next-turn with a run target", sessionloop.Command{Kind: sessionloop.CommandNextTurn, RunID: "run-1", Input: input}},
		{"next-turn without input", sessionloop.Command{Kind: sessionloop.CommandNextTurn}},
		{"next-turn with a resolution", sessionloop.Command{Kind: sessionloop.CommandNextTurn, Input: input, Resolution: resolution}},
		{"resolve without a run target", sessionloop.Command{Kind: sessionloop.CommandResolve, Resolution: resolution}},
		{"resolve without a resolution", sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: "run-1"}},
		{"interrupt without a run target", sessionloop.Command{Kind: sessionloop.CommandInterrupt}},
		{"interrupt with input", sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "run-1", Input: input}},
		{"interrupt with a resolution", sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "run-1", Resolution: resolution}},
	}
	for _, violation := range violations {
		if _, err := session.Dispatch(watchdogContext(t), violation.command); !errors.Is(err, sessionloop.ErrInvalidCommand) {
			t.Errorf("%s: err = %v, want ErrInvalidCommand", violation.name, err)
		}
	}
}

func caseCloseAndReopenLifecycle(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session, err := env.Host.NewSession(watchdogContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}
	id := session.ID()
	if id == "" {
		t.Fatal("NewSession returned an empty session ID")
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("Close is not idempotent: %v", err)
	}
	if _, err := session.Dispatch(watchdogContext(t), startCommand(textInput("too late"))); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatalf("Dispatch after Close = %v, want ErrSessionClosed", err)
	}
	if _, err := session.Snapshot(watchdogContext(t)); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatalf("Snapshot after Close = %v, want ErrSessionClosed", err)
	}

	reopened, err := env.Host.OpenSession(watchdogContext(t), id)
	if err != nil {
		t.Fatalf("OpenSession after Close failed: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close(context.Background()) })
	if reopened.ID() != id {
		t.Fatalf("reopened session ID = %q, want %q", reopened.ID(), id)
	}
}

func caseStaleTargetedCommand(t *testing.T, env Env, capabilities sessionloop.Capabilities) {
	if !capabilities.Supports(sessionloop.CapabilitySteer) && !capabilities.Supports(sessionloop.CapabilityInterrupt) {
		t.Skip("host advertises neither input.steer nor run.interrupt; no targeted command to test")
	}
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	first := dispatch(t, session, startCommand(textInput("first run")))
	awaitSettled(t, stream, first.RunID)

	release := env.Gate.HoldNextRun()
	defer release()
	second := dispatch(t, session, startCommand(textInput("second run")))

	var err error
	if capabilities.Supports(sessionloop.CapabilitySteer) {
		_, err = session.Dispatch(watchdogContext(t), sessionloop.Command{
			Kind:  sessionloop.CommandSteer,
			RunID: first.RunID,
			Input: textInput("stale steer"),
		})
	} else {
		_, err = session.Dispatch(watchdogContext(t), sessionloop.Command{
			Kind:  sessionloop.CommandInterrupt,
			RunID: first.RunID,
		})
	}
	if !errors.Is(err, sessionloop.ErrStaleRun) {
		t.Fatalf("targeting the settled run %q = %v, want ErrStaleRun", first.RunID, err)
	}
	release()
	awaitSettled(t, stream, second.RunID)
}

func caseConcurrentStartSingleFlight(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	release := env.Gate.HoldNextRun()
	defer release()

	type outcome struct {
		receipt sessionloop.Receipt
		err     error
	}
	results := make(chan outcome, 2)
	barrier := make(chan struct{})
	contexts := [2]context.Context{watchdogContext(t), watchdogContext(t)}
	for index := range 2 {
		go func(ctx context.Context) {
			<-barrier
			receipt, err := session.Dispatch(ctx, startCommand(textInput("race to start")))
			results <- outcome{receipt: receipt, err: err}
		}(contexts[index])
	}
	close(barrier)

	var accepted []sessionloop.Receipt
	var rejected []error
	for range 2 {
		result := <-results
		if result.err == nil {
			accepted = append(accepted, result.receipt)
		} else {
			rejected = append(rejected, result.err)
		}
	}
	if len(accepted) != 1 {
		t.Fatalf("accepted %d concurrent starts, want exactly one (rejections: %v)", len(accepted), rejected)
	}
	if !errors.Is(rejected[0], sessionloop.ErrSessionBusy) {
		t.Fatalf("rejected start = %v, want ErrSessionBusy", rejected[0])
	}
	release()
	awaitSettled(t, stream, accepted[0].RunID)
}

func caseNoEventsAfterClose(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	receipt := dispatch(t, session, startCommand(textInput("final run")))
	awaitSettled(t, stream, receipt.RunID)
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	for {
		_, err := stream.Next(watchdogContext(t))
		if err == nil {
			continue
		}
		if !errors.Is(err, io.EOF) {
			t.Fatalf("stream after Close ended with %v, want io.EOF", err)
		}
		break
	}
	if _, err := session.Dispatch(watchdogContext(t), startCommand(textInput("after close"))); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatalf("Dispatch after Close = %v, want ErrSessionClosed", err)
	}
}
