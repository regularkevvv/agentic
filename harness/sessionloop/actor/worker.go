// The shared Worker implements the handoff order in spec/CONTRACT.md. Waiting
// for sessions is adapter-owned; active sessions keep checking eligible inputs.

package actor

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// Config belongs to service bootstrap, never to the submission use case.
type Config struct {
	Owner          string
	Adapter        Adapter
	SessionOpener  SessionOpener
	EventSink      EventSink
	OnError        func(ActorID, error)
	LeaseTTL       time.Duration
	PollInterval   time.Duration // active-session input checks, not recovery cron
	RetryInterval  time.Duration
	CleanupTimeout time.Duration
	BatchSize      int
	EventBuffer    int // independent of mailbox page size
	MaxActors      int
}

type worker struct {
	cfg     Config
	running atomic.Bool
}

// NewWorker does not start execution. Bootstrap independently calls Run.
func NewWorker(cfg Config) (Worker, error) {
	if cfg.Owner == "" || cfg.Adapter == nil || cfg.SessionOpener == nil {
		return nil, errors.New("session actor: owner, adapter and session opener are required")
	}
	if g := cfg.Adapter.Guarantee(); g != sessionloop.AcceptanceAccepted && g != sessionloop.AcceptanceDurable {
		return nil, errors.New("session actor: adapter must declare its acceptance guarantee")
	}
	if cfg.LeaseTTL <= 0 {
		cfg.LeaseTTL = 30 * time.Second
	}
	if cfg.LeaseTTL < 3*time.Nanosecond {
		return nil, errors.New("session actor: lease TTL is too short")
	}
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 20 * time.Millisecond
	}
	if cfg.RetryInterval <= 0 {
		cfg.RetryInterval = time.Second
	}
	if cfg.CleanupTimeout <= 0 {
		cfg.CleanupTimeout = 5 * time.Second
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.EventBuffer <= 0 {
		cfg.EventBuffer = 256
	}
	if cfg.MaxActors <= 0 {
		cfg.MaxActors = 8
	}
	return &worker{cfg: cfg}, nil
}

func (w *worker) Run(ctx context.Context) error {
	if !w.running.CompareAndSwap(false, true) {
		return ErrWorkerRunning
	}
	defer w.running.Store(false)
	var workers sync.WaitGroup
	for range w.cfg.MaxActors {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for ctx.Err() == nil {
				id, err := w.cfg.Adapter.Receive(ctx)
				if err == nil {
					err = w.runActor(ctx, id)
				}
				if err == nil || errors.Is(err, ErrLeaseHeld) || errors.Is(err, ErrNoWork) {
					continue
				}
				if ctx.Err() != nil {
					return
				}
				if w.cfg.OnError != nil {
					w.cfg.OnError(id, err)
				}
				if !wait(ctx, w.cfg.RetryInterval) {
					return
				}
			}
		}()
	}
	workers.Wait()
	return ctx.Err()
}

func wait(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

func (w *worker) runActor(ctx context.Context, id ActorID) (result error) {
	lease, err := w.cfg.Adapter.Acquire(ctx, id, w.cfg.Owner, w.cfg.LeaseTTL)
	if err != nil {
		return err
	}
	runCtx, cancel := context.WithCancelCause(ctx)
	renewDone := make(chan struct{})
	go func() { defer close(renewDone); w.renew(runCtx, cancel, lease) }()
	var session Session
	defer func() {
		cleanup, finish := context.WithTimeout(context.Background(), w.cfg.CleanupTimeout)
		defer finish()
		if session != nil {
			result = errors.Join(result, session.Close(cleanup))
		}
		// Keep renewal alive through normal Close; then join it before release.
		cause := context.Cause(runCtx)
		cancel(context.Canceled)
		<-renewDone
		if cause != nil {
			result = errors.Join(result, cause)
		}
		if result == nil {
			result = w.cfg.Adapter.Release(cleanup, lease)
		}
		// Any failure leaves unfinished execution discoverable after expiry.
	}()
	session, err = w.cfg.SessionOpener.Open(runCtx, lease)
	if err != nil {
		return err
	}
	if session == nil {
		return errors.New("session actor: opener returned nil session")
	}
	caps := session.Capabilities()
	if !caps.Supports(sessionloop.CapabilityIdempotentDispatch) ||
		(w.cfg.Adapter.Guarantee() == sessionloop.AcceptanceDurable && !caps.Supports(sessionloop.CapabilityDurableAcceptance)) {
		return fmt.Errorf("actor requires idempotent dispatch and matching durability: %w", sessionloop.ErrUnsupported)
	}
	return w.drive(runCtx, lease, session)
}

func (w *worker) renew(ctx context.Context, cancel context.CancelCauseFunc, lease Lease) {
	ticker := time.NewTicker(w.cfg.LeaseTTL / 3)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if _, err := w.cfg.Adapter.Renew(ctx, lease, w.cfg.LeaseTTL); err != nil {
				cancel(fmt.Errorf("%w: %w", ErrLeaseLost, err))
				return
			}
		}
	}
}

type streamResult struct {
	event sessionloop.Event
	err   error
}

func (w *worker) drive(ctx context.Context, lease Lease, session Session) error {
	snapshot, err := session.Snapshot(ctx)
	if err != nil {
		return err
	}
	waiting := make(map[sessionloop.RunID]struct{})
	if snapshot.ActiveRunID != "" && !quiescent(snapshot) {
		waiting[snapshot.ActiveRunID] = struct{}{}
	}
	if sink, ok := w.cfg.EventSink.(SnapshotSink); ok {
		if err := sink.ObserveSnapshot(ctx, lease, snapshot); err != nil {
			return err
		}
	}
	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{After: snapshot.Position, Preview: true, Buffer: w.cfg.EventBuffer})
	if err != nil {
		return err
	}
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	events := make(chan streamResult, 1)
	pumpDone := make(chan struct{})
	go func() {
		defer close(pumpDone)
		for {
			event, err := stream.Next(streamCtx)
			select {
			case events <- streamResult{event, err}:
			case <-streamCtx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()
	defer func() { cancel(); _ = stream.Close(); <-pumpDone }()
	ticker := time.NewTicker(w.cfg.PollInterval)
	defer ticker.Stop()
	var after uint64
	var sweepHasPending bool
	for {
		if err := ctx.Err(); err != nil {
			return context.Cause(ctx)
		}
		snapshot, err = session.Snapshot(ctx)
		if err != nil {
			return err
		}
		if snapshot.State == sessionloop.StateFaulted || snapshot.State == sessionloop.StateClosed {
			return fmt.Errorf("session actor: session cannot progress in state %s", snapshot.State)
		}
		pending, err := w.cfg.Adapter.Pending(ctx, lease, after, w.cfg.BatchSize)
		if err != nil {
			return err
		}
		sweepHasPending = sweepHasPending || len(pending) != 0
		for _, input := range pending {
			if input.ActorID != lease.ActorID || input.Sequence <= after {
				return errors.New("session actor: invalid mailbox order or actor binding")
			}
			after = input.Sequence
			run, err := w.deliver(ctx, lease, session, snapshot, input)
			if err != nil {
				return err
			}
			if run != "" {
				waiting[run] = struct{}{}
			}
			snapshot, err = session.Snapshot(ctx)
			if err != nil {
				return err
			}
		}
		if len(pending) < w.cfg.BatchSize {
			if !sweepHasPending && quiescent(snapshot) && len(waiting) == 0 {
				// Projection repair must not depend on the event pump winning a
				// scheduling race against the final idle snapshot.
				if sink, ok := w.cfg.EventSink.(SnapshotSink); ok {
					if err := sink.ObserveSnapshot(ctx, lease, snapshot); err != nil {
						return err
					}
				}
				return nil
			}
			after, sweepHasPending = 0, false
		} else {
			// Finish the page sweep without delaying controls behind busy Starts.
			select {
			case item := <-events:
				if err := w.observe(ctx, lease, item, waiting); err != nil {
					return err
				}
			default:
			}
			continue
		}
		select {
		case <-ctx.Done():
			return context.Cause(ctx)
		case <-ticker.C:
		case item := <-events:
			if err := w.observe(ctx, lease, item, waiting); err != nil {
				return err
			}
		}
	}
}

// Quiescence is a harness fact, not an inferred run outcome. Next-turn input
// and a persisted suspension are intentionally parked until external input.
func quiescent(s sessionloop.Snapshot) bool {
	return s.State == sessionloop.StateIdle || (s.State == sessionloop.StateSuspended && s.Suspension != nil)
}

func (w *worker) deliver(ctx context.Context, lease Lease, session Session, snapshot sessionloop.Snapshot, input Command) (sessionloop.RunID, error) {
	canonical, err := input.Normalize()
	if err != nil {
		return "", err
	}
	command := canonical.Command
	receipt, found, err := session.Acceptance(ctx, command)
	if err != nil {
		return "", err
	}
	if !found {
		if command.Kind == sessionloop.CommandStart && snapshot.State != sessionloop.StateIdle {
			return "", nil
		}
		receipt, err = session.Dispatch(ctx, command)
		if errors.Is(err, sessionloop.ErrSessionBusy) || errors.Is(err, sessionloop.ErrSuspended) {
			return "", nil
		}
		if reason := terminalRejection(err); reason != "" {
			receipt, err = session.Reject(ctx, command, reason)
		}
		if err != nil {
			return "", err
		} // ambiguous errors retain input for exact retry
		// A returned receipt alone is not permission to delete delivery state.
		persisted, accepted, lookupErr := session.Acceptance(ctx, command)
		if lookupErr != nil {
			return "", lookupErr
		}
		if !accepted || persisted != receipt {
			return "", ErrInvalidReceipt
		}
	}
	if receipt.CommandID != command.ID || receipt.SessionID == "" || receipt.SessionID != session.ID() ||
		(receipt.Rejection != "" && (!receipt.Rejection.Valid() || receipt.RunID != "" || receipt.QueueID != "")) ||
		(receipt.Guarantee != sessionloop.AcceptanceAccepted && receipt.Guarantee != sessionloop.AcceptanceDurable) ||
		(w.cfg.Adapter.Guarantee() == sessionloop.AcceptanceDurable &&
			(receipt.Guarantee != sessionloop.AcceptanceDurable || receipt.Position.IsZero())) {
		return "", ErrInvalidReceipt
	}
	if err := w.cfg.Adapter.Acknowledge(ctx, lease, input.ID, receipt); err != nil {
		return "", err
	}
	// Observe terminal events before retiring, but continue delivering controls
	// meanwhile. A historical receipt for an already quiet run must not wait
	// for an event preceding our subscription's snapshot.
	if (command.Kind == sessionloop.CommandStart || command.Kind == sessionloop.CommandResolve) &&
		(!found || (snapshot.ActiveRunID == receipt.RunID && !quiescent(snapshot))) {
		return receipt.RunID, nil
	}
	return "", nil
}

func terminalRejection(err error) sessionloop.Rejection {
	switch {
	case errors.Is(err, sessionloop.ErrStaleRun):
		return sessionloop.RejectionStaleRun
	case errors.Is(err, sessionloop.ErrNotRunning):
		return sessionloop.RejectionNotRunning
	case errors.Is(err, sessionloop.ErrUnsupported):
		return sessionloop.RejectionUnsupported
	case errors.Is(err, sessionloop.ErrInvalidCommand):
		return sessionloop.RejectionInvalidCommand
	default:
		return "" // conflicts, storage and unknown errors remain recoverable
	}
}

func (w *worker) observe(ctx context.Context, lease Lease, item streamResult, waiting map[sessionloop.RunID]struct{}) error {
	if item.err != nil {
		return item.err
	}
	if w.cfg.EventSink != nil {
		if err := w.cfg.EventSink.Observe(ctx, lease, item.event); err != nil {
			return err
		}
	}
	if item.event.Nature == sessionloop.EventAuthoritative &&
		(item.event.Kind == sessionloop.EventRunSettled || item.event.Kind == sessionloop.EventRunSuspended ||
			(item.event.Kind == sessionloop.EventSessionState && item.event.State == sessionloop.StateSuspended)) {
		delete(waiting, item.event.RunID)
	}
	return nil
}
