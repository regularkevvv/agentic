package session

// Focused race suite for the sessionloop view (plan 12.3). Every scenario
// synchronizes with barriers/channels only — no sleeps — and must stay green
// under `go test -race -count=100 -run 'TestLoopRace' ./session/`.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"
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
		for _, block := range pending.Blocks {
			if block.Text == "queued race" {
				occurrences++
			}
		}
	}
	for _, entry := range snapshot.Entries {
		if entry.Origin == sessionloop.OriginNextTurn {
			for _, block := range entry.Blocks {
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

// gatedQueueIDs parks the first "queue" ID generation until released. It
// pins a steer dispatch inside acceptWithCursor exactly between the
// Dispatch-level stale pre-check (which already passed) and the locked
// durable append, so the run boundary can be crossed while the command is in
// flight. Every other prefix passes straight through.
type gatedQueueIDs struct {
	parked  chan struct{}
	release chan struct{}
	once    sync.Once
	counter atomic.Uint64
}

func (g *gatedQueueIDs) New(prefix string) (string, error) {
	if prefix == "queue" {
		g.once.Do(func() { close(g.parked) })
		<-g.release
	}
	return fmt.Sprintf("%s-%d", prefix, g.counter.Add(1)), nil
}

// TestLoopRaceStaleSteerRevalidatedInsideLockedAcceptance reproduces the L8
// check-then-act defect deterministically: a steer targeted at run A parks
// inside acceptWithCursor after the Dispatch-level pre-check, run A settles
// and run B starts while it is parked, and on release the locked recheck
// must reject the command with ErrStaleRun. Run B's journal and transcript
// must carry no trace of the stale steer.
func TestLoopRaceStaleSteerRevalidatedInsideLockedAcceptance(t *testing.T) {
	enteredA := make(chan struct{})
	releaseA := make(chan struct{})
	enteredB := make(chan struct{})
	releaseB := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "answer a"), entered: enteredA, release: releaseA},
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "answer b"), entered: enteredB, release: releaseB},
	}}
	ids := &gatedQueueIDs{parked: make(chan struct{}), release: make(chan struct{})}
	view, session := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(),
		func(config *Config[string], _ *LoopConfig[string]) {
			config.IDs = ids
		})
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 256})

	first := loopDispatch(t, view, sessionloopStartCommand("run a"))
	awaitSignal(t, enteredA, "run A model entered")

	steerErr := make(chan error, 1)
	go func() {
		_, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
			ID:    "cmd-stale-steer",
			Kind:  sessionloop.CommandSteer,
			RunID: first.RunID,
			Input: sessionloopTextInput("stale steer"),
		})
		steerErr <- err
	}()
	awaitSignal(t, ids.parked, "steer parked inside acceptWithCursor")

	// Cross the run boundary while the steer is parked: settle A, start B.
	close(releaseA)
	if err := session.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	second := loopDispatch(t, view, sessionloopStartCommand("run b"))
	awaitSignal(t, enteredB, "run B model entered")

	close(ids.release)
	if err := <-steerErr; !errors.Is(err, sessionloop.ErrStaleRun) {
		t.Fatalf("parked stale steer err = %v, want ErrStaleRun", err)
	}

	close(releaseB)
	settled, seen := awaitLoopSettled(t, stream, second.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("run B outcome = %#v", settled.Outcome)
	}
	for _, event := range seen {
		if event.Kind == sessionloop.EventQueueAccepted {
			t.Fatalf("the stale steer was durably accepted: %#v", event)
		}
		if event.Kind == sessionloop.EventEntryCommitted && event.Entry.Origin == sessionloop.OriginSteer {
			t.Fatalf("the stale steer leaked into run B's transcript: %#v", event.Entry)
		}
	}
	entries := loadJournalEntries(t, session)
	if countEntries(entries, kindQueueAccepted) != 0 {
		t.Fatalf("stale steer reached the journal: %v", journalKinds(entries))
	}
	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range snapshot.Entries {
		if entry.Origin == sessionloop.OriginSteer {
			t.Fatalf("stale steer visible in the snapshot: %#v", entry)
		}
	}
}

// TestLoopRaceStaleInterruptCannotCancelSuccessorRun is the interrupt
// variant: an interrupt aimed at run A is parked at a barrier after its
// dispatch-level pre-check would have passed, run A settles and run B
// starts, and the released requestInterrupt must fail stale without
// touching run B.
func TestLoopRaceStaleInterruptCannotCancelSuccessorRun(t *testing.T) {
	enteredA := make(chan struct{})
	releaseA := make(chan struct{})
	enteredB := make(chan struct{})
	releaseB := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "answer a"), entered: enteredA, release: releaseA},
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "answer b"), entered: enteredB, release: releaseB},
	}}
	view, session := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 256})

	first := loopDispatch(t, view, sessionloopStartCommand("run a"))
	awaitSignal(t, enteredA, "run A model entered")

	barrier := make(chan struct{})
	interruptErr := make(chan error, 1)
	go func() {
		<-barrier
		_, err := session.requestInterrupt(context.Background(), string(first.RunID))
		interruptErr <- err
	}()

	close(releaseA)
	if err := session.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	second := loopDispatch(t, view, sessionloopStartCommand("run b"))
	awaitSignal(t, enteredB, "run B model entered")

	close(barrier)
	err := <-interruptErr
	if !errors.Is(err, errStaleRunTarget) {
		t.Fatalf("stale requestInterrupt = %v, want errStaleRunTarget", err)
	}
	if mapped := mapLoopError(err); !errors.Is(mapped, sessionloop.ErrStaleRun) {
		t.Fatalf("mapLoopError(%v) = %v, want ErrStaleRun identity", err, mapped)
	}
	if state := session.State(); state != Running {
		t.Fatalf("run B was disturbed by the stale interrupt: state = %s", state)
	}

	close(releaseB)
	settled, _ := awaitLoopSettled(t, stream, second.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("run B outcome = %#v, want completed (not cancelled by the stale interrupt)", settled.Outcome)
	}
}

func TestLoopRaceCloseWhileNextBlocked(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{})
	result := make(chan error, 1)
	started := make(chan struct{})
	waiting := loopTestContext(t)
	go func() {
		// No dispatch ever happens on this view, so the first Next has
		// nothing to read and parks; the barrier orders its entry before
		// Close fires, proving Close races a genuinely waiting Next.
		close(started)
		for {
			if _, err := stream.Next(waiting); err != nil {
				result <- err
				return
			}
		}
	}()
	awaitSignal(t, started, "consumer entered Next before Close")
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
	// The recovered session settled back to idle: the torn run is durably
	// closed as interrupted and the recovery continuation completed.
	if state := recovered.State(); state != Idle {
		t.Fatalf("recovered state = %s, want idle", state)
	}
	entries := loadJournalEntries(t, recovered)
	if countEntries(entries, kindRunClosed) != 2 {
		t.Fatalf("recovered journal run.closed count = %d in %v",
			countEntries(entries, kindRunClosed), journalKinds(entries))
	}
	torn, err := decodePayload[runClosedPayload](recoveredConfig.Codec, entries[firstEntryIndex(entries, kindRunClosed)])
	if err != nil {
		t.Fatal(err)
	}
	if torn.Status != agentic.ExecutionInterrupted ||
		!strings.Contains(torn.Error, "process stopped before run termination") {
		t.Fatalf("torn run.closed payload = %#v", torn)
	}
	lastClosed := -1
	for index, entry := range entries {
		if entry.Kind == kindRunClosed {
			lastClosed = index
		}
	}
	continuation, err := decodePayload[runClosedPayload](recoveredConfig.Codec, entries[lastClosed])
	if err != nil {
		t.Fatal(err)
	}
	if continuation.Status != agentic.ExecutionCompleted {
		t.Fatalf("continuation run.closed payload = %#v", continuation)
	}
	if err := recovered.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}
