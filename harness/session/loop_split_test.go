package session

// Tests for the private acceptance/execution split (plan S3): single-use
// accepted values, cancellation before durable acceptance, deterministic
// interrupted settlement after acceptance-time cancellation, and the
// cannot-outlive-Close guarantee.

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

// hookJournal wraps a journal with test barriers around Append.
type hookJournal struct {
	store.Journal
	mu          sync.Mutex
	failAppend  func(entries []store.PendingEntry) error
	afterAppend func(entries []store.PendingEntry)
}

func (j *hookJournal) hooks() (func([]store.PendingEntry) error, func([]store.PendingEntry)) {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.failAppend, j.afterAppend
}

func (j *hookJournal) set(fail func([]store.PendingEntry) error, after func([]store.PendingEntry)) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.failAppend, j.afterAppend = fail, after
}

func (j *hookJournal) Append(ctx context.Context, cursor store.Cursor, entries ...store.PendingEntry) (store.Commit, error) {
	fail, after := j.hooks()
	if fail != nil {
		if err := fail(entries); err != nil {
			return store.Commit{}, err
		}
	}
	commit, err := j.Journal.Append(ctx, cursor, entries...)
	if err == nil && after != nil {
		after(entries)
	}
	return commit, err
}

// hookRepository wraps every journal it opens in a hookJournal and remembers
// the latest one for test coordination.
type hookRepository struct {
	store.Repository
	mu     sync.Mutex
	latest *hookJournal
}

func newHookRepository() *hookRepository {
	return &hookRepository{Repository: storememory.New()}
}

func (r *hookRepository) journal() *hookJournal {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.latest
}

func (r *hookRepository) wrap(journal store.Journal) store.Journal {
	wrapped := &hookJournal{Journal: journal}
	r.mu.Lock()
	r.latest = wrapped
	r.mu.Unlock()
	return wrapped
}

func (r *hookRepository) Create(ctx context.Context, id string, entries ...store.PendingEntry) (store.Journal, store.Commit, error) {
	journal, commit, err := r.Repository.Create(ctx, id, entries...)
	if err != nil {
		return nil, store.Commit{}, err
	}
	return r.wrap(journal), commit, nil
}

func (r *hookRepository) Open(ctx context.Context, id string) (store.Journal, error) {
	journal, err := r.Repository.Open(ctx, id)
	if err != nil {
		return nil, err
	}
	return r.wrap(journal), nil
}

func batchHasKind(entries []store.PendingEntry, kind string) bool {
	for _, entry := range entries {
		if entry.Kind == kind {
			return true
		}
	}
	return false
}

func TestAcceptedStartIsSingleUse(t *testing.T) {
	driver := &countingDriver{}
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	accepted, err := session.prepareStart(context.Background(),
		agentic.NewTextMessage(agentic.RoleUser, "single use"), context.Background())
	if err != nil {
		t.Fatal(err)
	}
	execution, err := session.driveAccepted(accepted)
	if err != nil || execution.Status != agentic.ExecutionCompleted {
		t.Fatalf("first drive execution=%#v err=%v", execution, err)
	}
	if _, err := session.driveAccepted(accepted); err == nil || !strings.Contains(err.Error(), "already driven") {
		t.Fatalf("second drive err = %v, want already-driven", err)
	}
	if driver.Count() != 1 {
		t.Fatalf("driver ran %d times, want exactly once", driver.Count())
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestAcceptedResumeIsSingleUse(t *testing.T) {
	resume := &acceptedResume[string]{}
	if err := resume.consume(); err != nil {
		t.Fatalf("first consume = %v", err)
	}
	driver := &countingDriver{}
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(context.Background()) }()
	if _, err := session.driveResumed(resume); err == nil || !strings.Contains(err.Error(), "already driven") {
		t.Fatalf("consumed resume drive err = %v, want already-driven", err)
	}
	indeterminate := &acceptedResume[string]{indeterminate: true}
	if err := indeterminate.consume(); err != nil {
		t.Fatalf("first consume = %v", err)
	}
	if _, err := session.driveResumedIndeterminate(indeterminate); err == nil ||
		!strings.Contains(err.Error(), "already driven") {
		t.Fatalf("consumed indeterminate drive err = %v, want already-driven", err)
	}
	if driver.Count() != 0 {
		t.Fatalf("driver ran %d times for consumed resumes", driver.Count())
	}
}

// TestPrepareStartCanceledBeforeAcceptance proves a dispatch canceled before
// the durable append leaves no run and no journal growth.
func TestPrepareStartCanceledBeforeAcceptance(t *testing.T) {
	driver := &countingDriver{}
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	before := loadJournalEntries(t, session)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := session.prepareStart(canceled,
		agentic.NewTextMessage(agentic.RoleUser, "never accepted"), canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("prepareStart err = %v, want context.Canceled", err)
	}
	after := loadJournalEntries(t, session)
	if len(after) != len(before) {
		t.Fatalf("journal grew from %d to %d entries on canceled acceptance", len(before), len(after))
	}
	if state := session.State(); state != Idle {
		t.Fatalf("state = %s, want idle", state)
	}
	if session.currentRunID() != "" {
		t.Fatal("canceled acceptance left an active run")
	}
	if driver.Count() != 0 {
		t.Fatalf("driver ran %d times", driver.Count())
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// TestDispatchCancelAfterAcceptanceSettlesInterrupted uses a journal barrier
// to cancel the dispatch context exactly after the durable acceptance append
// succeeds: the accepted run must settle interrupted deterministically, the
// driver must never run, and no unowned run may survive (plan 8.4).
func TestDispatchCancelAfterAcceptanceSettlesInterrupted(t *testing.T) {
	driver := &countingDriver{}
	repository := newHookRepository()
	config := sessionConfig(t, driver, repository, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	view, err := NewLoopView(session, LoopConfig[string]{CloseRoot: session.Close})
	if err != nil {
		t.Fatal(err)
	}
	dispatchCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	repository.journal().set(nil, func(entries []store.PendingEntry) {
		if batchHasKind(entries, kindRunOpened) {
			cancel()
		}
	})
	_, err = view.Dispatch(dispatchCtx, sessionloopStartCommand("canceled after acceptance"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("dispatch err = %v, want context.Canceled", err)
	}
	if err := session.WaitForIdle(context.Background()); err != nil {
		t.Fatalf("WaitForIdle after handshake = %v", err)
	}
	entries := loadJournalEntries(t, session)
	if countEntries(entries, kindRunOpened) != 1 || countEntries(entries, kindRunClosed) != 1 {
		t.Fatalf("journal kinds after canceled dispatch = %v", journalKinds(entries))
	}
	closed, err := decodePayload[runClosedPayload](config.Codec, entries[firstEntryIndex(entries, kindRunClosed)])
	if err != nil {
		t.Fatal(err)
	}
	if closed.Status != agentic.ExecutionInterrupted {
		t.Fatalf("run settled %v, want interrupted", closed.Status)
	}
	if driver.Count() != 0 {
		t.Fatalf("driver ran %d times for a canceled dispatch", driver.Count())
	}
	if err := view.Close(context.Background()); err != nil {
		t.Fatalf("Close = %v", err)
	}
}

// TestDriveAcceptedAfterCloseSettlesThroughFinishPath documents why the
// drive halves carry no Closed pre-check: through the view the scenario is
// unreachable (Close joins every drive goroutine before releasing the root),
// so a post-Close drive exists only by fabrication. Like the legacy fused
// Prompt, the fabricated drive settles through finishExecution, which
// reports the absent run instead of a state pre-check.
func TestDriveAcceptedAfterCloseSettlesThroughFinishPath(t *testing.T) {
	driver := &countingDriver{}
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	accepted, err := session.prepareStart(context.Background(),
		agentic.NewTextMessage(agentic.RoleUser, "outlive close"), context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.requestInterrupt(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	if err := session.finishInterrupt(&agentic.Execution[string]{Status: agentic.ExecutionInterrupted}); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := session.driveAccepted(accepted); err == nil ||
		!strings.Contains(err.Error(), "without an active session run") {
		t.Fatalf("fabricated post-close drive err = %v, want the finish path's absent-run report", err)
	}
	if _, err := session.driveAccepted(accepted); err == nil || !strings.Contains(err.Error(), "already driven") {
		t.Fatalf("second fabricated drive err = %v, want single-use consumption", err)
	}
	if driver.Count() != 1 {
		t.Fatalf("driver ran %d times, want exactly the one fabricated drive", driver.Count())
	}
}

// TestRequestInterruptStateMatrix pins the request half's returns against
// the legacy Interrupt state matrix.
func TestRequestInterruptStateMatrix(t *testing.T) {
	driver := &countingDriver{}
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.requestInterrupt(context.Background(), ""); !errors.Is(err, ErrNotRunning) {
		t.Fatalf("idle requestInterrupt = %v, want ErrNotRunning", err)
	}
	accepted, err := session.prepareStart(context.Background(),
		agentic.NewTextMessage(agentic.RoleUser, "interrupt states"), context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// A stale expected run never transitions the session (law L8).
	if _, err := session.requestInterrupt(context.Background(), "run-other"); !errors.Is(err, errStaleRunTarget) {
		t.Fatalf("stale requestInterrupt = %v, want errStaleRunTarget", err)
	}
	if state := session.State(); state != Running {
		t.Fatalf("state after stale requestInterrupt = %s, want running", state)
	}
	// The matching expected run interrupts exactly like the legacy no-check call.
	runID, err := session.requestInterrupt(context.Background(), accepted.runID)
	if err != nil || runID != accepted.runID {
		t.Fatalf("running requestInterrupt = (%q, %v), want (%q, nil)", runID, err, accepted.runID)
	}
	// Interrupting joins without error.
	if joined, err := session.requestInterrupt(context.Background(), ""); err != nil || joined != accepted.runID {
		t.Fatalf("interrupting requestInterrupt = (%q, %v)", joined, err)
	}
	if err := session.finishInterrupt(&agentic.Execution[string]{Status: agentic.ExecutionInterrupted}); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := session.requestInterrupt(context.Background(), ""); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("closed requestInterrupt = %v, want ErrSessionClosed", err)
	}
}
