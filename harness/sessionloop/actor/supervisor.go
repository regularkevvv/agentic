package actor

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

type Config struct {
	Owner     string
	Commands  CommandStore
	Leases    LeaseStore
	Notifier  Notifier
	Activator Activator
	Observer  Observer
	// OnError observes an activation failure after durable work has already
	// been accepted. Lease contention and normal cancellation are omitted.
	OnError      func(ActorID, error)
	LeaseTTL     time.Duration
	ScanInterval time.Duration
	// NotificationRetry controls resubscription after the lossy notifier fails.
	// Durable mailbox scans continue throughout an outage.
	NotificationRetry time.Duration
	BatchSize         int
	MaxActors         int
}

type Supervisor struct {
	cfg Config

	mu     sync.Mutex
	active map[ActorID]struct{}
	sem    chan struct{}
	wg     sync.WaitGroup
}

func New(config Config) (*Supervisor, error) {
	if config.Owner == "" || config.Commands == nil || config.Leases == nil ||
		config.Notifier == nil || config.Activator == nil {
		return nil, errors.New("session actor: owner, commands, leases, notifier, and activator are required")
	}
	if config.LeaseTTL <= 0 {
		config.LeaseTTL = 30 * time.Second
	}
	if config.ScanInterval <= 0 {
		config.ScanInterval = time.Second
	}
	if config.NotificationRetry <= 0 {
		config.NotificationRetry = time.Second
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 64
	}
	if config.MaxActors <= 0 {
		config.MaxActors = 128
	}
	return &Supervisor{cfg: config, active: make(map[ActorID]struct{}), sem: make(chan struct{}, config.MaxActors)}, nil
}

// Submit durably enqueues before publishing the best-effort wake-up. A
// notification error is returned only as advisory context: the successful
// Submission remains valid and the periodic scan will recover it.
func (s *Supervisor) Submit(ctx context.Context, command Command) (Submission, error) {
	submission, err := s.cfg.Commands.Enqueue(ctx, command)
	if err != nil {
		return Submission{}, err
	}
	if err := s.cfg.Notifier.Publish(ctx, submission.ActorID); err != nil {
		return submission, fmt.Errorf("session actor: command is durable but wake publish failed: %w", err)
	}
	return submission, nil
}

func (s *Supervisor) Run(ctx context.Context) error {
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	wakes := make(chan ActorID, s.cfg.BatchSize)
	go s.listen(runCtx, wakes)
	ticker := time.NewTicker(s.cfg.ScanInterval)
	defer ticker.Stop()
	for {
		select {
		case <-runCtx.Done():
			s.wg.Wait()
			return runCtx.Err()
		case id := <-wakes:
			s.activate(runCtx, id)
		case <-ticker.C:
			ids, scanErr := s.cfg.Commands.ReadyActors(runCtx, s.cfg.BatchSize)
			if scanErr != nil {
				return scanErr
			}
			for _, id := range ids {
				s.activate(runCtx, id)
			}
		}
	}
}

func (s *Supervisor) listen(ctx context.Context, wakes chan<- ActorID) {
	for ctx.Err() == nil {
		subscription, err := s.cfg.Notifier.Subscribe(ctx)
		if err != nil {
			s.notificationError(err)
			if !waitRetry(ctx, s.cfg.NotificationRetry) {
				return
			}
			continue
		}
		for ctx.Err() == nil {
			id, nextErr := subscription.Next(ctx)
			if nextErr != nil {
				_ = subscription.Close()
				if ctx.Err() != nil || errors.Is(nextErr, context.Canceled) {
					return
				}
				s.notificationError(nextErr)
				break
			}
			select {
			case wakes <- id:
			case <-ctx.Done():
				_ = subscription.Close()
				return
			}
		}
		if !waitRetry(ctx, s.cfg.NotificationRetry) {
			return
		}
	}
}

func (s *Supervisor) notificationError(err error) {
	if err != nil && s.cfg.OnError != nil {
		s.cfg.OnError("", fmt.Errorf("session actor: lossy notifier unavailable; durable scanning continues: %w", err))
	}
}

func waitRetry(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

func (s *Supervisor) activate(ctx context.Context, id ActorID) {
	if id == "" {
		return
	}
	s.mu.Lock()
	if _, exists := s.active[id]; exists {
		s.mu.Unlock()
		return
	}
	s.active[id] = struct{}{}
	s.mu.Unlock()
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		select {
		case s.sem <- struct{}{}:
		case <-ctx.Done():
			s.forget(id)
			return
		}
		defer func() {
			<-s.sem
			s.forget(id)
		}()
		if err := s.RunActor(ctx, id); err != nil && !errors.Is(err, ErrLeaseHeld) &&
			!errors.Is(err, context.Canceled) && s.cfg.OnError != nil {
			s.cfg.OnError(id, err)
		}
	}()
}

func (s *Supervisor) forget(id ActorID) {
	s.mu.Lock()
	delete(s.active, id)
	s.mu.Unlock()
}

// RunActor owns one actor until its active run settles and its durable inbox
// is empty. ErrLeaseHeld is a normal competing-replica outcome.
func (s *Supervisor) RunActor(ctx context.Context, id ActorID) error {
	lease, err := s.cfg.Leases.Acquire(ctx, id, s.cfg.Owner, s.cfg.LeaseTTL)
	if err != nil {
		return err
	}
	actorCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	defer func() { _ = s.cfg.Leases.Release(context.Background(), lease) }()
	leaseUpdates := make(chan Lease, 1)
	leaseErr := make(chan error, 1)
	go s.renew(actorCtx, lease, leaseUpdates, leaseErr)

	session, err := s.cfg.Activator.Open(actorCtx, id)
	if err != nil {
		return err
	}
	defer func() { _ = session.Close(context.Background()) }()
	snapshot, err := session.Snapshot(actorCtx)
	if err != nil {
		return err
	}
	if observer, ok := s.cfg.Observer.(SnapshotObserver); ok {
		if err := observer.ObserveSnapshot(actorCtx, lease, snapshot); err != nil {
			return err
		}
	}
	stream, err := session.Subscribe(actorCtx, sessionloop.SubscribeOptions{After: snapshot.Position, Preview: true, Buffer: s.cfg.BatchSize})
	if err != nil {
		return err
	}
	defer func() { _ = stream.Close() }()

	current := lease
	for {
		select {
		case updated := <-leaseUpdates:
			current = updated
		default:
		}
		pending, pendingErr := s.cfg.Commands.Pending(actorCtx, current, s.cfg.BatchSize)
		if pendingErr != nil {
			return pendingErr
		}
		dispatched, waitRun, dispatchErr := s.dispatchEligible(actorCtx, current, session, snapshot, pending)
		if dispatchErr != nil {
			return dispatchErr
		}
		if snapshot.State == sessionloop.StateIdle && len(pending) == 0 {
			return nil
		}
		if dispatched {
			for waitRun != "" {
				event, nextErr := nextActorEvent(actorCtx, stream, leaseUpdates, leaseErr, &current)
				if nextErr != nil {
					return nextErr
				}
				if err := s.observeEvent(actorCtx, current, event); err != nil {
					return err
				}
				if event.Kind == sessionloop.EventRunSettled && event.RunID == waitRun {
					waitRun = ""
				}
			}
			snapshot, err = session.Snapshot(actorCtx)
			if err != nil {
				return err
			}
			continue
		}
		event, nextErr := nextActorEvent(actorCtx, stream, leaseUpdates, leaseErr, &current)
		if nextErr != nil {
			return nextErr
		}
		if err := s.observeEvent(actorCtx, current, event); err != nil {
			return err
		}
		if event.Nature == sessionloop.EventAuthoritative {
			snapshot, err = session.Snapshot(actorCtx)
			if err != nil {
				return err
			}
		}
	}
}

func (s *Supervisor) dispatchEligible(
	ctx context.Context,
	lease Lease,
	session sessionloop.Session,
	snapshot sessionloop.Snapshot,
	pending []Command,
) (bool, sessionloop.RunID, error) {
	for _, envelope := range pending {
		if envelope.State == CommandDispatched {
			if envelope.Receipt != nil && envelope.Receipt.RunID != "" &&
				snapshot.ActiveRunID == envelope.Receipt.RunID {
				continue
			}
			if err := s.cfg.Commands.MarkSettled(ctx, lease, envelope.ID, nil); err != nil {
				return false, "", err
			}
			return true, "", nil
		}
		command := envelope.Command.Clone()
		if command.Kind == sessionloop.CommandStart && snapshot.State != sessionloop.StateIdle {
			continue
		}
		if command.ID == "" {
			command.ID = sessionloop.CommandID(envelope.ID)
		}
		if session.Capabilities().Supports(sessionloop.CapabilityIdempotentDispatch) {
			command.IdempotencyKey = string(envelope.ID)
		}
		receipt, err := session.Dispatch(ctx, command)
		if err != nil {
			if errors.Is(err, sessionloop.ErrSessionBusy) {
				continue
			}
			_ = s.cfg.Commands.MarkFailed(ctx, lease, envelope.ID, err)
			return false, "", err
		}
		if err := s.cfg.Commands.MarkDispatched(ctx, lease, envelope.ID, receipt); err != nil {
			return false, "", err
		}
		if command.Kind != sessionloop.CommandStart && command.Kind != sessionloop.CommandResolve {
			if err := s.cfg.Commands.MarkSettled(ctx, lease, envelope.ID, nil); err != nil {
				return false, "", err
			}
		}
		waitRun := sessionloop.RunID("")
		if command.Kind == sessionloop.CommandStart || command.Kind == sessionloop.CommandResolve {
			waitRun = receipt.RunID
		}
		return true, waitRun, nil
	}
	return false, "", nil
}

func (s *Supervisor) observeEvent(ctx context.Context, lease Lease, event sessionloop.Event) error {
	if s.cfg.Observer != nil {
		if err := s.cfg.Observer.Observe(ctx, lease, event); err != nil {
			return err
		}
	}
	if event.Kind == sessionloop.EventRunSettled && event.CommandID != "" {
		if err := s.cfg.Commands.MarkSettled(ctx, lease, CommandID(event.CommandID), event.Outcome); err != nil {
			return err
		}
	}
	return nil
}

func (s *Supervisor) renew(ctx context.Context, initial Lease, updates chan<- Lease, result chan<- error) {
	interval := s.cfg.LeaseTTL / 3
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	lease := initial
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			updated, err := s.cfg.Leases.Renew(ctx, lease, s.cfg.LeaseTTL)
			if err != nil {
				result <- err
				return
			}
			lease = updated
			select {
			case updates <- updated:
			default:
			}
		}
	}
}

func nextActorEvent(
	ctx context.Context,
	stream sessionloop.Stream,
	updates <-chan Lease,
	leaseErr <-chan error,
	current *Lease,
) (sessionloop.Event, error) {
	type result struct {
		event sessionloop.Event
		err   error
	}
	next := make(chan result, 1)
	go func() {
		event, err := stream.Next(ctx)
		next <- result{event: event, err: err}
	}()
	for {
		select {
		case <-ctx.Done():
			return sessionloop.Event{}, ctx.Err()
		case err := <-leaseErr:
			return sessionloop.Event{}, fmt.Errorf("%w: %v", ErrLeaseLost, err)
		case updated := <-updates:
			*current = updated
		case result := <-next:
			return result.event, result.err
		}
	}
}
