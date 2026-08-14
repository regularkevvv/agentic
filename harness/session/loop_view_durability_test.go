package session

// Durable idempotent dispatch across restarts: journaled acceptance markers
// must restore on open, replay receipts for retried keys, and reject corrupt
// or conflicting acceptance history instead of guessing.

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/permission"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

// flakyJournal wraps a real journal and fails the configured operations so
// tests can reach the dispatch and restore error paths deterministically.
type flakyJournal struct {
	store.Journal
	loadErr   error
	appendErr error
}

func (f flakyJournal) Load(ctx context.Context) (store.Snapshot, error) {
	if f.loadErr != nil {
		return store.Snapshot{}, f.loadErr
	}
	return f.Journal.Load(ctx)
}

func (f flakyJournal) Append(ctx context.Context, cursor store.Cursor, entries ...store.PendingEntry) (store.Commit, error) {
	if f.appendErr != nil {
		return store.Commit{}, f.appendErr
	}
	return f.Journal.Append(ctx, cursor, entries...)
}

func failJournal(session *Session[string], loadErr, appendErr error) (restore func()) {
	session.mu.Lock()
	previous := session.journal
	session.journal = flakyJournal{Journal: previous, loadErr: loadErr, appendErr: appendErr}
	session.mu.Unlock()
	return func() {
		session.mu.Lock()
		session.journal = previous
		session.mu.Unlock()
	}
}

func reopenLoopView(t *testing.T, config Config[string]) (*LoopView[string], error) {
	t.Helper()
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	view, err := NewLoopView(recovered, LoopConfig[string]{CloseRoot: recovered.Close})
	if err != nil {
		_ = recovered.Close(context.Background())
		return nil, err
	}
	t.Cleanup(func() { _ = view.Close(context.Background()) })
	return view, nil
}

func appendJournalEntries(t *testing.T, config Config[string], entries ...store.PendingEntry) {
	t.Helper()
	ctx := loopTestContext(t)
	journal, err := config.Repository.Open(ctx, config.ID)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = journal.Close(ctx) }()
	snapshot, err := journal.Load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := journal.Append(ctx, snapshot.Cursor, entries...); err != nil {
		t.Fatal(err)
	}
}

func acceptanceEntry(t *testing.T, config Config[string], payload loopCommandAcceptedPayload) store.PendingEntry {
	t.Helper()
	entry, err := pending(config.Codec, kindCommandAccepted, payload)
	if err != nil {
		t.Fatal(err)
	}
	return entry
}

func TestLoopViewDurableAcceptancesRestoreAcrossReopen(t *testing.T) {
	repository := storememory.New()
	model := &scriptedModel{steps: []modelStep{textStep("first"), textStep("second")}}
	config := sessionConfig(t, agentic.NewAgent("", model), repository, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	view, err := NewLoopView(session, LoopConfig[string]{CloseRoot: session.Close})
	if err != nil {
		t.Fatal(err)
	}
	start := sessionloopStartCommand("first run")
	start.IdempotencyKey = "durable-start"
	startReceipt := loopDispatch(t, view, start)
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	next := sessionloop.Command{
		Kind:           sessionloop.CommandNextTurn,
		Input:          sessionloopTextInput("queued turn"),
		IdempotencyKey: "durable-next",
	}
	nextReceipt := loopDispatch(t, view, next)
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	if err := view.Close(context.Background()); err != nil {
		t.Fatal(err)
	}

	// Acceptances this process never replayed in memory: a steer marker plus
	// a resolution and its resolve marker, so restore rebuilds the queue,
	// run, and resolution command indexes from the log alone.
	resolution, err := pending(config.Codec, kindResolutionAccepted, resolutionAcceptedPayload{SuspensionID: "susp-1"})
	if err != nil {
		t.Fatal(err)
	}
	appendJournalEntries(t, config,
		acceptanceEntry(t, config, loopCommandAcceptedPayload{
			CommandID: "cmd-steer", Kind: "steer", IdempotencyKey: "durable-steer",
			Digest: "sha256:steer", QueueID: "queue-steer",
		}),
		resolution,
		acceptanceEntry(t, config, loopCommandAcceptedPayload{
			CommandID: "cmd-resolve", Kind: "resolve", IdempotencyKey: "durable-resolve",
			Digest: "sha256:resolve", RunID: "run-resolved",
		}),
	)

	reopened, err := reopenLoopView(t, config)
	if err != nil {
		t.Fatalf("reopen failed: %v", err)
	}
	replayedStart, err := reopened.Dispatch(loopTestContext(t), start)
	if err != nil {
		t.Fatalf("replayed start failed: %v", err)
	}
	if replayedStart.CommandID != startReceipt.CommandID ||
		replayedStart.RunID != startReceipt.RunID ||
		replayedStart.Guarantee != sessionloop.AcceptanceDurable {
		t.Fatalf("restored start receipt = %#v, want %#v", replayedStart, startReceipt)
	}
	replayedNext, err := reopened.Dispatch(loopTestContext(t), next)
	if err != nil {
		t.Fatalf("replayed next_turn failed: %v", err)
	}
	if replayedNext.CommandID != nextReceipt.CommandID ||
		replayedNext.QueueID != nextReceipt.QueueID ||
		replayedNext.Guarantee != sessionloop.AcceptanceDurable {
		t.Fatalf("restored next_turn receipt = %#v, want %#v", replayedNext, nextReceipt)
	}
	if got := reopened.commandForQueue("queue-steer"); got != "cmd-steer" {
		t.Fatalf("restored steer queue command = %q", got)
	}
	if got := reopened.commandForRun("run-resolved"); got != "cmd-resolve" {
		t.Fatalf("restored resolve run command = %q", got)
	}
	conflict := sessionloopStartCommand("other content")
	conflict.IdempotencyKey = "durable-start"
	if _, err := reopened.Dispatch(loopTestContext(t), conflict); !errors.Is(err, sessionloop.ErrCommandConflict) {
		t.Fatalf("conflicting replay err = %v, want ErrCommandConflict", err)
	}
}

func TestLoopViewRestoreRejectsCorruptAcceptances(t *testing.T) {
	cases := []struct {
		name    string
		entries []loopCommandAcceptedPayload
		want    string
	}{
		{
			name: "missing digest",
			entries: []loopCommandAcceptedPayload{
				{CommandID: "cmd-1", Kind: "start", IdempotencyKey: "key-1"},
			},
			want: "invalid durable command acceptance",
		},
		{
			name: "conflicting duplicate key",
			entries: []loopCommandAcceptedPayload{
				{CommandID: "cmd-1", Kind: "start", IdempotencyKey: "key-1", Digest: "sha256:one"},
				{CommandID: "cmd-2", Kind: "start", IdempotencyKey: "key-1", Digest: "sha256:two"},
			},
			want: "conflicting durable idempotency key",
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			repository := storememory.New()
			config := sessionConfig(t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
			session, err := New(context.Background(), config)
			if err != nil {
				t.Fatal(err)
			}
			if err := session.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
			pendings := make([]store.PendingEntry, 0, len(testCase.entries))
			for _, payload := range testCase.entries {
				pendings = append(pendings, acceptanceEntry(t, config, payload))
			}
			appendJournalEntries(t, config, pendings...)
			if _, err := reopenLoopView(t, config); err == nil || !strings.Contains(err.Error(), testCase.want) {
				t.Fatalf("reopen err = %v, want %q", err, testCase.want)
			}
		})
	}
}

func TestClaimIdempotencyKeyContentionAndReplay(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	_, owned, claim, err := view.claimIdempotencyKey(context.Background(), "contended", "digest-1")
	if err != nil || !owned || claim == nil {
		t.Fatalf("first claim = owned %t, claim %v, err %v", owned, claim, err)
	}
	// A rival dispatch must wait for the owner, and its context governs the
	// wait: with the claim still held a canceled context surfaces instead of
	// blocking forever.
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, err := view.claimIdempotencyKey(canceled, "contended", "digest-1"); !errors.Is(err, context.Canceled) {
		t.Fatalf("waiting claim err = %v, want context.Canceled", err)
	}
	// The owner records the acceptance and releases; a late retry replays the
	// recorded receipt without taking ownership.
	receipt := sessionloop.Receipt{CommandID: "cmd-1", SessionID: view.ID(), Guarantee: sessionloop.AcceptanceDurable}
	view.rememberCommandAcceptance(&loopCommandAcceptedPayload{
		CommandID: "cmd-1", Kind: "start", IdempotencyKey: "contended", Digest: "digest-1",
	}, receipt)
	view.releaseIdempotencyClaim("contended", claim)
	replayed, owned, _, err := view.claimIdempotencyKey(context.Background(), "contended", "digest-1")
	if err != nil || owned || replayed != receipt {
		t.Fatalf("replayed claim = %#v, owned %t, err %v", replayed, owned, err)
	}
}

func TestLoopViewKeyedInterruptReplaysReceipt(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "blocked"), entered: entered, release: release},
	}}
	view, session := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	receipt := loopDispatch(t, view, sessionloopStartCommand("interrupt me"))
	awaitSignal(t, entered, "model entered")

	followUp := sessionloop.Command{
		Kind:           sessionloop.CommandFollowUp,
		RunID:          receipt.RunID,
		Input:          sessionloopTextInput("queued follow-up"),
		IdempotencyKey: "durable-follow-up",
	}
	if followUpReceipt := loopDispatch(t, view, followUp); followUpReceipt.QueueID == "" {
		t.Fatalf("keyed follow-up receipt = %#v", followUpReceipt)
	}

	// While the run is live, a broken journal must fail the durable interrupt
	// acceptance itself instead of interrupting without the marker.
	restore := failJournal(session, nil, errors.New("append failed"))
	broken := sessionloop.Command{
		Kind:           sessionloop.CommandInterrupt,
		RunID:          receipt.RunID,
		IdempotencyKey: "durable-interrupt-broken",
	}
	if _, err := view.Dispatch(loopTestContext(t), broken); err == nil || !strings.Contains(err.Error(), "append failed") {
		t.Fatalf("keyed interrupt with broken journal err = %v", err)
	}
	restore()

	interrupt := sessionloop.Command{
		Kind:           sessionloop.CommandInterrupt,
		RunID:          receipt.RunID,
		IdempotencyKey: "durable-interrupt",
	}
	first := loopDispatch(t, view, interrupt)
	second := loopDispatch(t, view, interrupt)
	if first != second {
		t.Fatalf("keyed interrupt receipts differ: first=%#v second=%#v", first, second)
	}
	close(release)
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
}

// newSuspendedGateView builds a view whose first run suspends on the gated
// "danger" tool, the fixture behind every durable-resolve scenario here.
func newSuspendedGateView(t *testing.T, mutate func(*Config[string], *LoopConfig[string])) (*LoopView[string], *Session[string]) {
	t.Helper()
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "gate-1", Name: "danger", Input: map[string]any{"value": "x"}})},
		textStep("finished"),
	}}
	agent := agentic.NewAgent("base", model)
	agentic.AddTool(agent,
		func(context.Context, loopToolInput) (string, error) { return "gated ok", nil },
		agentic.AutoToolName("danger"),
		agentic.AutoToolDescription("Perform a gated action"),
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
	return newLoopViewForTest(t, agent, storememory.New(), func(config *Config[string], loopConfig *LoopConfig[string]) {
		config.ToolGate = plan.ToolGate()
		config.Context = plan.ContextPolicy()
		if mutate != nil {
			mutate(config, loopConfig)
		}
	})
}

func TestLoopViewKeyedResolveJournalsAndReplays(t *testing.T) {
	view, session := newSuspendedGateView(t, nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 256})

	receipt := loopDispatch(t, view, sessionloopStartCommand("begin"))
	suspended, _ := awaitLoopKind(t, stream, sessionloop.EventRunSuspended)
	if suspended.Suspension == nil || suspended.Suspension.ID == "" {
		t.Fatalf("run.suspended = %#v", suspended)
	}
	// A resolution prompt is still v1 text-only input.
	unsupported := sessionloop.Command{
		Kind:  sessionloop.CommandResolve,
		RunID: receipt.RunID,
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspended.Suspension.ID,
			Decisions:    []sessionloop.ResolutionDecision{{ID: "gate-1", Action: sessionloop.ResolutionApprove}},
		},
		Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{
			{Kind: sessionloop.InputBlockData, Data: json.RawMessage(`{"a":1}`)},
		}},
	}
	if _, err := view.Dispatch(loopTestContext(t), unsupported); err == nil {
		t.Fatal("data-block resolution prompt was accepted")
	}
	// A broken journal must fail the durable resolve acceptance itself.
	restoreJournal := failJournal(session, nil, errors.New("append failed"))
	brokenResolve := sessionloop.Command{
		Kind:           sessionloop.CommandResolve,
		RunID:          receipt.RunID,
		IdempotencyKey: "durable-resolve-broken",
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspended.Suspension.ID,
			Decisions:    []sessionloop.ResolutionDecision{{ID: "gate-1", Action: sessionloop.ResolutionApprove}},
		},
	}
	if _, err := view.Dispatch(loopTestContext(t), brokenResolve); err == nil ||
		!strings.Contains(err.Error(), "append failed") {
		t.Fatalf("keyed resolve with broken journal err = %v", err)
	}
	restoreJournal()
	invalid := sessionloop.Command{
		Kind:  sessionloop.CommandResolve,
		RunID: receipt.RunID,
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspended.Suspension.ID,
			Decisions: []sessionloop.ResolutionDecision{
				{ID: "gate-1", Action: sessionloop.ResolutionExternalResult, Data: json.RawMessage("{")},
			},
		},
	}
	if _, err := view.Dispatch(loopTestContext(t), invalid); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatalf("undecodable resolution data err = %v, want ErrInvalidCommand", err)
	}

	resolve := sessionloop.Command{
		Kind:           sessionloop.CommandResolve,
		RunID:          receipt.RunID,
		IdempotencyKey: "durable-resolve",
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspended.Suspension.ID,
			Decisions:    []sessionloop.ResolutionDecision{{ID: "gate-1", Action: sessionloop.ResolutionApprove}},
		},
	}
	first := loopDispatch(t, view, resolve)
	second := loopDispatch(t, view, resolve)
	if first != second {
		t.Fatalf("keyed resolve receipts differ: first=%#v second=%#v", first, second)
	}
	settled, _ := awaitLoopSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome = %#v", settled.Outcome)
	}
}

func TestRecoverSettlesPendingInterruptFromDurableAcceptance(t *testing.T) {
	repository := storememory.New()
	config := sessionConfig[string](t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	// Crash without a run closure: release the journal lease directly so the
	// log ends with an open run and a durable interrupt acceptance behind it.
	if err := session.journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	runOpened, err := pending(config.Codec, kindRunOpened, runOpenedPayload{ID: "run-crashed", Mode: "start"})
	if err != nil {
		t.Fatal(err)
	}
	appendJournalEntries(t, config,
		runOpened,
		acceptanceEntry(t, config, loopCommandAcceptedPayload{
			CommandID: "cmd-interrupt", Kind: "interrupt", IdempotencyKey: "durable-interrupt",
			Digest: "sha256:interrupt", RunID: "run-crashed",
		}),
	)
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatalf("recover with pending interrupt failed: %v", err)
	}
	t.Cleanup(func() { _ = recovered.Close(context.Background()) })
	if state := recovered.State(); state != Idle {
		t.Fatalf("recovered state = %s, want Idle", state)
	}
}

func TestLoopViewSurfacesJournalLoadFailures(t *testing.T) {
	repository := storememory.New()
	config := sessionConfig[string](t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	restore := failJournal(session, errors.New("load failed"), nil)
	if _, err := NewLoopView(session, LoopConfig[string]{CloseRoot: session.Close}); err == nil ||
		!strings.Contains(err.Error(), "load failed") {
		t.Fatalf("NewLoopView with failing journal err = %v", err)
	}
	restore()
	view, err := NewLoopView(session, LoopConfig[string]{CloseRoot: session.Close})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = view.Close(context.Background()) })
	restore = failJournal(session, errors.New("load failed"), nil)
	if _, err := view.Subscribe(loopTestContext(t), sessionloop.SubscribeOptions{
		After: sessionloop.Position{Sequence: 1},
	}); err == nil || !strings.Contains(err.Error(), "load failed") {
		t.Fatalf("Subscribe replay with failing journal err = %v", err)
	}
	restore()
}

func TestRecoverRejectsUndecodableAcceptancePayload(t *testing.T) {
	repository := storememory.New()
	config := sessionConfig[string](t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	appendJournalEntries(t, config, store.PendingEntry{
		Kind: kindCommandAccepted, Payload: []byte("{"), Durability: store.DurabilitySync,
	})
	if _, err := Recover(context.Background(), config); err == nil {
		t.Fatal("recover accepted an undecodable durable acceptance payload")
	}
}

func TestRecoverRejectsInvalidPersistedScope(t *testing.T) {
	repository := storememory.New()
	config := sessionConfig[string](t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	created, err := pending(config.Codec, kindSessionCreated, sessionCreatedPayload{
		Scope: &harnessruntime.Scope{SessionID: config.ID, Depth: -1},
	})
	if err != nil {
		t.Fatal(err)
	}
	appendJournalEntries(t, config, created)
	if _, err := Recover(context.Background(), config); err == nil ||
		!strings.Contains(err.Error(), "invalid persisted session scope") {
		t.Fatalf("recover with foreign scope err = %v", err)
	}
}

func awaitSuspended(t *testing.T, view *LoopView[string]) {
	t.Helper()
	deadline := time.Now().Add(loopTestWait)
	for view.inner.State() != Suspended {
		if time.Now().After(deadline) {
			t.Fatalf("session never suspended, state = %s", view.inner.State())
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestLoopViewSnapshotSurfacesSuspensionProjectorFailure(t *testing.T) {
	view, _ := newSuspendedGateView(t, func(_ *Config[string], loopConfig *LoopConfig[string]) {
		loopConfig.SuspensionProjector = func(agentic.Suspension) (sessionloop.Suspension, error) {
			return sessionloop.Suspension{}, errors.New("projector failed")
		}
	})
	loopDispatch(t, view, sessionloopStartCommand("begin"))
	awaitSuspended(t, view)
	if _, err := view.Snapshot(loopTestContext(t)); err == nil || !strings.Contains(err.Error(), "projector failed") {
		t.Fatalf("suspended snapshot err = %v", err)
	}
}

func TestLoopViewSnapshotSurfacesLateSuspensionProjectorFailure(t *testing.T) {
	// The event projection succeeds, so the failure comes from the snapshot's
	// own suspension branch rather than the entry replay.
	var projectorCalls atomic.Int32
	view, _ := newSuspendedGateView(t, func(_ *Config[string], loopConfig *LoopConfig[string]) {
		loopConfig.SuspensionProjector = func(value agentic.Suspension) (sessionloop.Suspension, error) {
			if projectorCalls.Add(1) > 1 {
				return sessionloop.Suspension{}, errors.New("projector failed late")
			}
			return defaultSuspensionProjector(value)
		}
	})
	loopDispatch(t, view, sessionloopStartCommand("begin"))
	awaitSuspended(t, view)
	if _, err := view.Snapshot(loopTestContext(t)); err == nil || !strings.Contains(err.Error(), "projector failed late") {
		t.Fatalf("suspended snapshot err = %v", err)
	}
}

func TestLoopViewSnapshotHonorsCanceledContext(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := view.Snapshot(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled snapshot err = %v", err)
	}
}
