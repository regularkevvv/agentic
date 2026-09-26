package session

// Fault injection covers both possible outcomes of an unsuccessful Append
// response. No sleep or automatic retry can stand in for journal reconstruction.

import (
	"context"
	"errors"
	"sync"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type acceptanceOutcomeJournal struct {
	store.Journal
	committed   bool
	cause       error
	afterCommit func()
}

type acceptanceReadKey struct{}

type acceptanceReadBarrier struct {
	store.Journal
	entered chan struct{}
	proceed <-chan struct{}
	loaded  chan struct{}
}

func (j acceptanceReadBarrier) Load(ctx context.Context) (store.Snapshot, error) {
	if ctx.Value(acceptanceReadKey{}) == nil {
		return j.Journal.Load(ctx)
	}
	close(j.entered)
	select {
	case <-j.proceed:
	case <-ctx.Done():
		return store.Snapshot{}, ctx.Err()
	}
	loaded, err := j.Journal.Load(ctx)
	close(j.loaded)
	return loaded, err
}

func TestLoopRaceJournalReaderDuringUnknownAppend(t *testing.T) {
	view, _, command := acceptanceFailureFixture(t, "start")
	readEntered, readProceed, readLoaded := make(chan struct{}), make(chan struct{}), make(chan struct{})
	committed, appendReturn := make(chan struct{}), make(chan struct{})
	var loadOnce, appendOnce sync.Once
	unblockLoad := func() { loadOnce.Do(func() { close(readProceed) }) }
	unblockAppend := func() { appendOnce.Do(func() { close(appendReturn) }) }
	defer unblockLoad()
	defer unblockAppend()
	view.inner.mu.Lock()
	view.inner.journal = acceptanceReadBarrier{
		Journal: acceptanceOutcomeJournal{Journal: view.inner.journal, committed: true, cause: errors.New("lost reply"),
			afterCommit: func() { close(committed); <-appendReturn }},
		entered: readEntered, proceed: readProceed, loaded: readLoaded,
	}
	view.inner.mu.Unlock()
	readErr := make(chan error, 1)
	ctx := context.WithValue(loopTestContext(t), acceptanceReadKey{}, true)
	go func() { _, err := view.Snapshot(ctx); readErr <- err }()
	awaitSignal(t, readEntered, "reader passed availability check")
	dispatched := dispatchAsync(loopTestContext(t), view, command)
	awaitSignal(t, committed, "append committed without returning")
	unblockLoad()
	awaitSignal(t, readLoaded, "reader loaded committed bytes")
	unblockAppend()
	requireAcceptanceFault(t, <-dispatched)
	requireAcceptanceFault(t, <-readErr)
}

func TestLoopRaceWaitingReaderWakesOnAcceptanceError(t *testing.T) {
	view, _, command := acceptanceFailureFixture(t, "start")
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64}).(*loopStream)
	checked := make(chan struct{})
	var once sync.Once
	availability := stream.projector.availability
	stream.projector.availability = func() (<-chan struct{}, error) {
		changed, err := availability()
		once.Do(func() { close(checked) })
		return changed, err
	}
	readErr := make(chan error, 1)
	go func() { _, err := stream.Next(loopTestContext(t)); readErr <- err }()
	awaitSignal(t, checked, "reader captured state-change signal")
	view.inner.mu.Lock()
	view.inner.journal = acceptanceOutcomeJournal{Journal: view.inner.journal, cause: errors.New("append failed")}
	view.inner.mu.Unlock()
	_, err := view.Dispatch(loopTestContext(t), command)
	requireAcceptanceFault(t, err)
	requireAcceptanceFault(t, <-readErr)
}

func (j acceptanceOutcomeJournal) Append(ctx context.Context, cursor store.Cursor, entries ...store.PendingEntry) (store.Commit, error) {
	if !batchHasKind(entries, kindCommandAccepted) {
		return j.Journal.Append(ctx, cursor, entries...)
	}
	if j.committed {
		if _, err := j.Journal.Append(ctx, cursor, entries...); err != nil {
			return store.Commit{}, err
		}
		if j.afterCommit != nil {
			j.afterCommit()
		}
	}
	return store.Commit{}, j.cause
}

func acceptanceFailureFixture(t *testing.T, path string) (*LoopView[string], Config[string], sessionloop.Command) {
	t.Helper()
	var config Config[string]
	capture := func(c *Config[string], _ *LoopConfig[string]) { config = *c }
	command := sessionloopStartCommand("ambiguous input")
	var view *LoopView[string]
	switch path {
	case "resolve":
		view, _ = newSuspendedGateView(t, capture)
		stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 128})
		start := loopDispatch(t, view, sessionloopStartCommand("begin"))
		suspended, _ := awaitLoopKind(t, stream, sessionloop.EventRunSuspended)
		command = sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: start.RunID,
			Resolution: &sessionloop.Resolution{SuspensionID: suspended.Suspension.ID,
				Decisions: []sessionloop.ResolutionDecision{{ID: "gate-1", Action: sessionloop.ResolutionApprove}}}}
	case "recovery_resolve":
		call := agentic.ToolUse{ID: "effect-1", Name: "effect"}
		config, _, _ = crashedConfig(t, []agentic.ToolUse{call}, "started")
		var err error
		view, err = reopenLoopView(t, config)
		if err != nil {
			t.Fatal(err)
		}
		snapshot, err := view.Snapshot(loopTestContext(t))
		if err != nil {
			t.Fatal(err)
		}
		command = sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: snapshot.ActiveRunID,
			Input: sessionloopTextInput("continue"),
			Resolution: &sessionloop.Resolution{SuspensionID: snapshot.Suspension.ID,
				Decisions: []sessionloop.ResolutionDecision{{ID: call.ID, Action: sessionloop.ResolutionDeny}}}}
	case "steer", "follow_up", "interrupt":
		entered, release := make(chan struct{}), make(chan struct{})
		t.Cleanup(func() { close(release) })
		model := &scriptedModel{steps: []modelStep{{message: agentic.NewTextMessage(agentic.RoleAssistant, "done"), entered: entered, release: release}}}
		view, _ = newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), capture)
		start := loopDispatch(t, view, sessionloopStartCommand("begin"))
		awaitSignal(t, entered, "model entered")
		command.Kind = sessionloop.CommandKind(path)
		command.RunID = start.RunID
		if path == "interrupt" {
			command.Input = nil
		}
	default:
		view, _ = newLoopViewForTest(t, &countingDriver{}, storememory.New(), capture)
		if path == "next_turn" {
			command.Kind = sessionloop.CommandNextTurn
		}
	}
	command.ID = "cmd-uncertain"
	command.IdempotencyKey = "key-uncertain"
	return view, config, command
}

func requireAcceptanceFault(t *testing.T, err error) {
	t.Helper()
	if !errors.Is(err, sessionloop.ErrSessionFaulted) {
		t.Fatalf("error = %v, want ErrSessionFaulted", err)
	}
}

func acceptanceEntries(t *testing.T, s *Session[string]) int {
	t.Helper()
	count := 0
	for _, entry := range loadJournalEntries(t, s) {
		if entry.Kind == kindCommandAccepted {
			count++
		}
	}
	return count
}

func TestLoopRaceAcceptanceErrorRequiresReconstruction(t *testing.T) {
	for _, path := range []string{"start", "next_turn", "steer", "follow_up", "resolve", "recovery_resolve", "interrupt", "reject"} {
		for _, committed := range []bool{false, true} {
			outcome := "not_committed"
			if committed {
				outcome = "committed_reply_lost"
			}
			t.Run(path+"/"+outcome, func(t *testing.T) {
				view, config, command := acceptanceFailureFixture(t, path)
				stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 256})
				cause := errors.New("injected append error")
				view.inner.mu.Lock()
				view.inner.journal = acceptanceOutcomeJournal{Journal: view.inner.journal, committed: committed, cause: cause}
				view.inner.mu.Unlock()
				var err error
				if path == "reject" {
					_, err = view.Reject(loopTestContext(t), command, sessionloop.RejectionUnsupported)
				} else {
					_, err = view.Dispatch(loopTestContext(t), command)
				}
				requireAcceptanceFault(t, err)
				if !errors.Is(err, cause) {
					t.Fatalf("lost original cause: %v", err)
				}
				if _, err = view.Snapshot(loopTestContext(t)); err == nil {
					t.Fatal("snapshot exposed an ambiguous view")
				}
				_, err = view.Replay(loopTestContext(t), sessionloop.Position{}, sessionloop.Position{})
				requireAcceptanceFault(t, err)
				_, err = view.Subscribe(loopTestContext(t), sessionloop.SubscribeOptions{})
				requireAcceptanceFault(t, err)
				_, err = stream.Next(loopTestContext(t))
				requireAcceptanceFault(t, err)
				_, _, err = view.Acceptance(loopTestContext(t), command)
				requireAcceptanceFault(t, err)
				_, err = view.Dispatch(loopTestContext(t), command)
				requireAcceptanceFault(t, err)
				_, err = view.Reject(loopTestContext(t), command, sessionloop.RejectionUnsupported)
				requireAcceptanceFault(t, err)
				_, err = NewLoopView(view.inner, LoopConfig[string]{CloseRoot: view.inner.Close})
				requireAcceptanceFault(t, err)
				for _, lookup := range []func() (sessionloop.CommandID, error){
					func() (sessionloop.CommandID, error) { return view.commandForRun("run") },
					func() (sessionloop.CommandID, error) { return view.commandForQueue("queue") },
					func() (sessionloop.CommandID, error) { return view.commandForResolution(1) },
				} {
					_, err = lookup()
					requireAcceptanceFault(t, err)
				}
				if err = view.Close(loopTestContext(t)); err != nil {
					t.Fatal(err)
				}
				reopened, err := reopenLoopView(t, config)
				if err != nil {
					t.Fatal(err)
				}
				receipt, found, err := reopened.Acceptance(loopTestContext(t), command)
				if err != nil || found != committed {
					t.Fatalf("reconstructed acceptance=%v, committed=%v, err=%v", found, committed, err)
				}
				if committed {
					before := acceptanceEntries(t, reopened.inner)
					again, err := reopened.Dispatch(loopTestContext(t), command)
					if err != nil || again != receipt {
						t.Fatalf("retry=%#v %v, want %#v", again, err, receipt)
					}
					if after := acceptanceEntries(t, reopened.inner); before != 1 || after != before {
						t.Fatal("retry duplicated journal acceptance")
					}
				} else if path == "start" || path == "next_turn" || path == "reject" {
					if path == "reject" {
						_, err = reopened.Reject(loopTestContext(t), command, sessionloop.RejectionUnsupported)
					} else {
						_, err = reopened.Dispatch(loopTestContext(t), command)
					}
					if err != nil {
						t.Fatalf("healthy retry after reconstruction: %v", err)
					}
				}
				if _, err := reopened.Snapshot(loopTestContext(t)); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}
