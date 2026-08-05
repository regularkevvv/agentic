package session

// Focused race suite for the sessionloop view (plan 12.3). Every scenario
// synchronizes with barriers/channels only — no sleeps — and must stay green
// under `go test -race -count=100 -run 'TestLoopRace' ./session/`.

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/permission"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestLoopRaceTwoSimultaneousStarts(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "winner"), entered: entered, release: release},
	}}
	view, _ := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64})

	type outcome struct {
		receipt sessionloop.Receipt
		err     error
	}
	results := make(chan outcome, 2)
	barrier := make(chan struct{})
	for range 2 {
		go func() {
			<-barrier
			receipt, err := view.Dispatch(loopTestContext(t), sessionloopStartCommand("race"))
			results <- outcome{receipt: receipt, err: err}
		}()
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
	if len(accepted) != 1 || len(rejected) != 1 {
		t.Fatalf("accepted=%d rejected=%v, want exactly one of each", len(accepted), rejected)
	}
	if !errors.Is(rejected[0], sessionloop.ErrSessionBusy) {
		t.Fatalf("loser err = %v, want ErrSessionBusy", rejected[0])
	}
	close(release)
	settled, _ := awaitLoopSettled(t, stream, accepted[0].RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome = %#v", settled.Outcome)
	}
}

func TestLoopRaceStartVersusNextTurnAcceptance(t *testing.T) {
	view, session := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	barrier := make(chan struct{})
	startErr := make(chan error, 1)
	queueErr := make(chan error, 1)
	go func() {
		<-barrier
		_, err := view.Dispatch(loopTestContext(t), sessionloopStartCommand("start race"))
		startErr <- err
	}()
	go func() {
		<-barrier
		_, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
			Kind: sessionloop.CommandNextTurn, Input: sessionloopTextInput("queued race"),
		})
		queueErr <- err
	}()
	close(barrier)
	if err := <-startErr; err != nil {
		t.Fatalf("start dispatch = %v", err)
	}
	if err := <-queueErr; err != nil {
		t.Fatalf("next_turn dispatch = %v", err)
	}
	if err := session.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	occurrences := 0
	for _, pending := range snapshot.Pending {
		for _, block := range pending.Content {
			if block.Text == "queued race" {
				occurrences++
			}
		}
	}
	for _, entry := range snapshot.Entries {
		if entry.Origin == sessionloop.OriginNextTurn {
			for _, block := range entry.Content {
				if block.Text == "queued race" {
					occurrences++
				}
			}
		}
	}
	if occurrences != 1 {
		t.Fatalf("queued input appeared %d times across pending+entries", occurrences)
	}
}

func TestLoopRaceSteerAtClosingBoundary(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "final"), entered: entered, release: release},
	}}
	view, _ := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64})
	receipt := loopDispatch(t, view, sessionloopStartCommand("closing race"))
	awaitSignal(t, entered, "model entered")

	steerErr := make(chan error, 1)
	barrier := make(chan struct{})
	go func() {
		<-barrier
		_, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
			Kind: sessionloop.CommandSteer, RunID: receipt.RunID, Input: sessionloopTextInput("boundary steer"),
		})
		steerErr <- err
	}()
	go func() {
		<-barrier
		close(release)
	}()
	close(barrier)
	err := <-steerErr
	switch {
	case err == nil,
		errors.Is(err, sessionloop.ErrCommandConflict),
		errors.Is(err, sessionloop.ErrStaleRun),
		errors.Is(err, sessionloop.ErrNotRunning):
	default:
		t.Fatalf("boundary steer err = %v", err)
	}
	settled, _ := awaitLoopSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome = %#v", settled.Outcome)
	}
}

func TestLoopRaceInterruptVersusLateResult(t *testing.T) {
	model := &lateResultModel{
		entered: make(chan struct{}),
		release: make(chan struct{}),
		message: agentic.NewTextMessage(agentic.RoleAssistant, "late answer"),
	}
	view, session := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64})
	receipt := loopDispatch(t, view, sessionloopStartCommand("interrupt late"))
	awaitSignal(t, model.entered, "model entered")

	interruptReceipt := loopDispatch(t, view, sessionloop.Command{
		Kind: sessionloop.CommandInterrupt, RunID: receipt.RunID,
	})
	if interruptReceipt.Guarantee != sessionloop.AcceptanceAccepted {
		t.Fatalf("interrupt receipt = %#v", interruptReceipt)
	}
	awaitState(t, session, Interrupting)
	close(model.release)

	settled, seen := awaitLoopSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunInterrupted {
		t.Fatalf("outcome = %#v", settled.Outcome)
	}
	lateIndex := -1
	for index, event := range seen {
		if event.Kind == sessionloop.EventEntryCommitted && event.Entry.Role == sessionloop.RoleAssistant {
			lateIndex = index
		}
	}
	if lateIndex < 0 || lateIndex >= len(seen)-1 {
		t.Fatalf("late assistant result not delivered before settlement (index %d of %d)", lateIndex, len(seen))
	}
}

func TestLoopRaceResolveVersusClose(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "gate-r", Name: "danger", Input: map[string]any{"value": "z"}})},
		textStep("resumed"),
	}}
	agent := agentic.NewAgent("", model)
	agentic.AddTool(agent,
		func(context.Context, loopToolInput) (string, error) { return "ok", nil },
		agentic.AutoToolName("danger"),
		agentic.AutoToolDescription("gated"),
	)
	policy, err := permission.New(permission.DecisionDeny,
		permission.Rule{Pattern: "tool/danger/**", Decision: permission.DecisionAsk})
	if err != nil {
		t.Fatal(err)
	}
	permissionCapability, err := permission.NewCapability(policy)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(permissionCapability)
	if err != nil {
		t.Fatal(err)
	}
	view, session := newLoopViewForTest(t, agent, storememory.New(), func(config *Config[string], _ *LoopConfig[string]) {
		config.ToolGate = plan.ToolGate()
		config.Context = plan.ContextPolicy()
	})
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 128})
	receipt := loopDispatch(t, view, sessionloopStartCommand("suspend then race"))
	suspended, _ := awaitLoopKind(t, stream, sessionloop.EventRunSuspended)

	barrier := make(chan struct{})
	resolveErr := make(chan error, 1)
	closeErr := make(chan error, 1)
	go func() {
		<-barrier
		_, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
			Kind:  sessionloop.CommandResolve,
			RunID: receipt.RunID,
			Resolution: &sessionloop.Resolution{
				SuspensionID: suspended.Suspension.ID,
				Decisions:    []sessionloop.ResolutionDecision{{ID: "gate-r", Action: sessionloop.ResolutionApprove}},
			},
		})
		resolveErr <- err
	}()
	go func() {
		<-barrier
		closeErr <- view.Close(context.Background())
	}()
	close(barrier)

	if err := <-closeErr; err != nil {
		t.Fatalf("Close = %v", err)
	}
	err = <-resolveErr
	switch {
	case err == nil,
		errors.Is(err, sessionloop.ErrSessionClosed),
		errors.Is(err, sessionloop.ErrNotRunning),
		errors.Is(err, sessionloop.ErrStaleRun),
		errors.Is(err, sessionloop.ErrInvalidCommand),
		errors.Is(err, sessionloop.ErrSuspended):
	default:
		t.Fatalf("resolve during close err = %v", err)
	}
	if session.State() != Closed {
		t.Fatalf("state after close race = %s", session.State())
	}
}

func TestLoopRaceDispatchCancelAroundDurableAppend(t *testing.T) {
	t.Run("before append", func(t *testing.T) {
		driver := &countingDriver{}
		view, session := newLoopViewForTest(t, driver, storememory.New(), nil)
		before := loadJournalEntries(t, session)
		canceled, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := view.Dispatch(canceled, sessionloopStartCommand("never")); !errors.Is(err, context.Canceled) {
			t.Fatalf("dispatch err = %v", err)
		}
		if after := loadJournalEntries(t, session); len(after) != len(before) {
			t.Fatalf("journal grew %d -> %d", len(before), len(after))
		}
		if session.State() != Idle || driver.Count() != 0 {
			t.Fatalf("state=%s drives=%d", session.State(), driver.Count())
		}
	})
	t.Run("failing journal", func(t *testing.T) {
		driver := &countingDriver{}
		repository := newHookRepository()
		view, session := newLoopViewForTest(t, driver, repository, nil)
		boom := errors.New("append rejected")
		repository.journal().set(func(entries []store.PendingEntry) error {
			if batchHasKind(entries, kindRunOpened) {
				return boom
			}
			return nil
		}, nil)
		if _, err := view.Dispatch(loopTestContext(t), sessionloopStartCommand("rejected")); !errors.Is(err, boom) {
			t.Fatalf("dispatch err = %v", err)
		}
		repository.journal().set(nil, nil)
		if session.State() != Idle || driver.Count() != 0 {
			t.Fatalf("state=%s drives=%d", session.State(), driver.Count())
		}
	})
	t.Run("after append", func(t *testing.T) {
		driver := &countingDriver{}
		repository := newHookRepository()
		view, session := newLoopViewForTest(t, driver, repository, nil)
		dispatchCtx, cancel := context.WithCancel(context.Background())
		defer cancel()
		repository.journal().set(nil, func(entries []store.PendingEntry) {
			if batchHasKind(entries, kindRunOpened) {
				cancel()
			}
		})
		if _, err := view.Dispatch(dispatchCtx, sessionloopStartCommand("canceled late")); !errors.Is(err, context.Canceled) {
			t.Fatalf("dispatch err = %v", err)
		}
		if err := session.WaitForIdle(loopTestContext(t)); err != nil {
			t.Fatal(err)
		}
		entries := loadJournalEntries(t, session)
		if countEntries(entries, kindRunClosed) != 1 || driver.Count() != 0 {
			t.Fatalf("kinds=%v drives=%d", journalKinds(entries), driver.Count())
		}
	})
}

func TestLoopRaceStreamLagWhileFactsContinue(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 1})

	first := loopDispatch(t, view, sessionloopStartCommand("first"))
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	second := loopDispatch(t, view, sessionloopStartCommand("second"))
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	_, _ = first, second

	sawLag := false
	for {
		_, err := stream.Next(loopTestContext(t))
		if err == nil {
			continue
		}
		if errors.Is(err, sessionloop.ErrLagged) {
			sawLag = true
		} else if !errors.Is(err, io.EOF) {
			t.Fatalf("stream ended with %v", err)
		}
		break
	}
	if !sawLag {
		t.Fatal("subscriber never observed ErrLagged")
	}

	// Law L7 recovery: snapshot, then subscribe after its position.
	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := view.Subscribe(loopTestContext(t), sessionloop.SubscribeOptions{After: sessionloop.Position{Sequence: snapshot.Position.Sequence}})
	if err != nil {
		t.Fatalf("recovery subscribe = %v", err)
	}
	_ = recovered.Close()
}

func TestLoopRaceCloseWhileNextBlocked(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{})
	result := make(chan error, 1)
	waiting := loopTestContext(t)
	go func() {
		for {
			if _, err := stream.Next(waiting); err != nil {
				result <- err
				return
			}
		}
	}()
	if err := view.Close(context.Background()); err != nil {
		t.Fatalf("Close = %v", err)
	}
	if err := <-result; !errors.Is(err, io.EOF) {
		t.Fatalf("blocked Next ended with %v, want io.EOF", err)
	}
}

func TestLoopRaceCloseWhileDriveActive(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "held"), entered: entered, release: release},
	}}
	view, session := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64})
	receipt := loopDispatch(t, view, sessionloopStartCommand("close mid drive"))
	awaitSignal(t, entered, "model entered")

	if err := view.Close(context.Background()); err != nil {
		t.Fatalf("Close = %v", err)
	}
	if session.State() != Closed {
		t.Fatalf("state = %s", session.State())
	}
	settlements := 0
	for {
		event, err := stream.Next(loopTestContext(t))
		if err != nil {
			if !errors.Is(err, io.EOF) {
				t.Fatalf("stream ended with %v", err)
			}
			break
		}
		if event.Kind == sessionloop.EventRunSettled && event.RunID == receipt.RunID {
			settlements++
			if event.Outcome.Kind != sessionloop.RunInterrupted {
				t.Fatalf("outcome = %#v", event.Outcome)
			}
		}
	}
	if settlements != 1 {
		t.Fatalf("run settled %d times, want exactly once", settlements)
	}
	select {
	case <-release:
	default:
		close(release)
	}
}

func TestLoopRaceReopenAfterFault(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{
		textStep("will fault"),
		textStep("recovered"),
	}}
	repository := newHookRepository()
	config := sessionConfig(t, agentic.NewAgent("", model), repository, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	view, err := NewLoopView(session, LoopConfig[string]{CloseRoot: session.Close})
	if err != nil {
		t.Fatal(err)
	}
	boom := errors.New("mid-run append failure")
	repository.journal().set(func(entries []store.PendingEntry) error {
		if batchHasKind(entries, kindAssistantCommitted) {
			return boom
		}
		return nil
	}, nil)
	if _, err := view.Dispatch(loopTestContext(t), sessionloopStartCommand("fault me")); err != nil {
		t.Fatalf("dispatch = %v", err)
	}
	awaitState(t, session, Faulted)
	if err := view.Close(context.Background()); err != nil {
		t.Fatalf("Close of faulted session = %v", err)
	}
	if session.State() != Closed {
		t.Fatalf("state after close = %s", session.State())
	}

	// The durable session reopens: recovery closes the torn run and drives a
	// continuation with a fresh journal (the hook is not re-armed).
	recoveredConfig := sessionConfig(t, agentic.NewAgent("", model), repository, artifactmemory.New(), spill.Config{})
	recoveredConfig.ID = config.ID
	recovered, err := Recover(context.Background(), recoveredConfig)
	if err != nil {
		t.Fatalf("Recover after fault = %v", err)
	}
	if err := recovered.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatalf("recovered WaitForIdle = %v", err)
	}
	if err := recovered.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(boom.Error(), "append failure") {
		t.Fatal("unreachable")
	}
}
