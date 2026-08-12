package actor

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func TestSupervisorDefaultsSubmitAndActivationEdges(t *testing.T) {
	commands := &commandStoreStub{}
	leases := &leaseStoreStub{}
	notifier := &notifierStub{}
	activator := ActivatorFunc(func(context.Context, ActorID) (sessionloop.Session, error) {
		return &sessionStub{}, nil
	})
	supervisor, err := New(Config{Owner: "pod", Commands: commands, Leases: leases, Notifier: notifier, Activator: activator})
	if err != nil {
		t.Fatal(err)
	}
	if supervisor.cfg.LeaseTTL != 30*time.Second || supervisor.cfg.ScanInterval != time.Second ||
		supervisor.cfg.NotificationRetry != time.Second || supervisor.cfg.BatchSize != 64 || supervisor.cfg.MaxActors != 128 {
		t.Fatalf("defaults = %+v", supervisor.cfg)
	}
	want := errors.New("enqueue")
	commands.enqueueErr = want
	if _, err := supervisor.Submit(context.Background(), Command{}); !errors.Is(err, want) {
		t.Fatalf("enqueue error = %v", err)
	}
	commands.enqueueErr = nil
	commands.submission = Submission{ID: "command", ActorID: "actor"}
	if submission, err := supervisor.Submit(context.Background(), Command{}); err != nil || submission.ID != "command" || notifier.published != "actor" {
		t.Fatalf("submit = %+v, %v published=%q", submission, err, notifier.published)
	}

	supervisor.activate(context.Background(), "")
	supervisor.active["already"] = struct{}{}
	supervisor.activate(context.Background(), "already")
	if len(supervisor.active) != 1 {
		t.Fatalf("duplicate activation changed active set: %+v", supervisor.active)
	}
	delete(supervisor.active, "already")
	supervisor.sem <- struct{}{}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	supervisor.activate(canceled, "blocked")
	<-supervisor.sem
	supervisor.wg.Wait()
	if len(supervisor.active) != 0 {
		t.Fatalf("canceled semaphore activation remained active: %+v", supervisor.active)
	}
}

func TestDispatchEligibleTransitionsAndFailures(t *testing.T) {
	want := errors.New("injected")
	start := func(id CommandID) Command {
		return Command{ID: id, State: CommandPending, Command: sessionloop.Command{
			Kind:  sessionloop.CommandStart,
			Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "hello"}}},
		}}
	}
	t.Run("dispatched active is retained", func(t *testing.T) {
		commands := &commandStoreStub{}
		supervisor := &Supervisor{cfg: Config{Commands: commands}}
		value := start("active")
		value.State = CommandDispatched
		value.Receipt = &sessionloop.Receipt{RunID: "run"}
		dispatched, run, err := supervisor.dispatchEligible(context.Background(), Lease{}, &sessionStub{}, sessionloop.Snapshot{ActiveRunID: "run"}, []Command{value})
		if err != nil || dispatched || run != "" || commands.settled != 0 {
			t.Fatalf("result = %v %q %v settled=%d", dispatched, run, err, commands.settled)
		}
	})
	t.Run("recovered dispatched is settled", func(t *testing.T) {
		commands := &commandStoreStub{}
		supervisor := &Supervisor{cfg: Config{Commands: commands}}
		value := start("recovered")
		value.State = CommandDispatched
		dispatched, _, err := supervisor.dispatchEligible(context.Background(), Lease{}, &sessionStub{}, sessionloop.Snapshot{}, []Command{value})
		if err != nil || !dispatched || commands.settled != 1 {
			t.Fatalf("result = %v %v settled=%d", dispatched, err, commands.settled)
		}
		commands.markSettledErr = want
		if _, _, err := supervisor.dispatchEligible(context.Background(), Lease{}, &sessionStub{}, sessionloop.Snapshot{}, []Command{value}); !errors.Is(err, want) {
			t.Fatalf("settle error = %v", err)
		}
	})
	t.Run("busy start is skipped", func(t *testing.T) {
		supervisor := &Supervisor{cfg: Config{Commands: &commandStoreStub{}}}
		dispatched, _, err := supervisor.dispatchEligible(context.Background(), Lease{}, &sessionStub{}, sessionloop.Snapshot{State: sessionloop.StateRunning}, []Command{start("busy")})
		if err != nil || dispatched {
			t.Fatalf("busy = %v, %v", dispatched, err)
		}
	})
	t.Run("host busy is skipped and host failure marks failed", func(t *testing.T) {
		commands := &commandStoreStub{}
		supervisor := &Supervisor{cfg: Config{Commands: commands}}
		busy := &sessionStub{dispatchErr: sessionloop.ErrSessionBusy}
		if dispatched, _, err := supervisor.dispatchEligible(context.Background(), Lease{}, busy, sessionloop.Snapshot{State: sessionloop.StateIdle}, []Command{start("busy")}); err != nil || dispatched {
			t.Fatalf("host busy = %v, %v", dispatched, err)
		}
		failed := &sessionStub{dispatchErr: want}
		if _, _, err := supervisor.dispatchEligible(context.Background(), Lease{}, failed, sessionloop.Snapshot{State: sessionloop.StateIdle}, []Command{start("failed")}); !errors.Is(err, want) || commands.failed != 1 {
			t.Fatalf("host failure = %v failed=%d", err, commands.failed)
		}
	})
	t.Run("idempotent start dispatches and waits", func(t *testing.T) {
		commands := &commandStoreStub{}
		supervisor := &Supervisor{cfg: Config{Commands: commands}}
		session := &sessionStub{caps: sessionloop.NewCapabilities(sessionloop.CapabilityIdempotentDispatch), receipt: sessionloop.Receipt{RunID: "run"}}
		value := start("envelope")
		value.Command.ID = ""
		dispatched, run, err := supervisor.dispatchEligible(context.Background(), Lease{}, session, sessionloop.Snapshot{State: sessionloop.StateIdle}, []Command{value})
		if err != nil || !dispatched || run != "run" || session.command.ID != "envelope" || session.command.IdempotencyKey != "envelope" || commands.dispatched != 1 {
			t.Fatalf("result = %v %q %v command=%+v marked=%d", dispatched, run, err, session.command, commands.dispatched)
		}
		commands.markDispatchedErr = want
		if _, _, err := supervisor.dispatchEligible(context.Background(), Lease{}, session, sessionloop.Snapshot{State: sessionloop.StateIdle}, []Command{start("mark")}); !errors.Is(err, want) {
			t.Fatalf("mark dispatch error = %v", err)
		}
	})
	t.Run("non-run command settles immediately", func(t *testing.T) {
		commands := &commandStoreStub{}
		supervisor := &Supervisor{cfg: Config{Commands: commands}}
		value := Command{ID: "interrupt", State: CommandPending, Command: sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "run"}}
		dispatched, run, err := supervisor.dispatchEligible(context.Background(), Lease{}, &sessionStub{}, sessionloop.Snapshot{}, []Command{value})
		if err != nil || !dispatched || run != "" || commands.settled != 1 {
			t.Fatalf("result = %v %q %v settled=%d", dispatched, run, err, commands.settled)
		}
		commands.markSettledErr = want
		if _, _, err := supervisor.dispatchEligible(context.Background(), Lease{}, &sessionStub{}, sessionloop.Snapshot{}, []Command{value}); !errors.Is(err, want) {
			t.Fatalf("settle error = %v", err)
		}
	})
	t.Run("resolve waits for resumed run", func(t *testing.T) {
		commands := &commandStoreStub{}
		supervisor := &Supervisor{cfg: Config{Commands: commands}}
		value := Command{ID: "resolve", State: CommandPending, Command: sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: "old", Resolution: &sessionloop.Resolution{}}}
		dispatched, run, err := supervisor.dispatchEligible(context.Background(), Lease{}, &sessionStub{receipt: sessionloop.Receipt{RunID: "resumed"}}, sessionloop.Snapshot{}, []Command{value})
		if err != nil || !dispatched || run != "resumed" {
			t.Fatalf("result = %v %q %v", dispatched, run, err)
		}
	})
}

func TestObserveRenewAndNextEventBranches(t *testing.T) {
	want := errors.New("injected")
	commands := &commandStoreStub{}
	supervisor := &Supervisor{cfg: Config{Commands: commands}}
	observer := &observerStub{err: want}
	supervisor.cfg.Observer = observer
	if err := supervisor.observeEvent(context.Background(), Lease{}, sessionloop.Event{}); !errors.Is(err, want) {
		t.Fatalf("observer error = %v", err)
	}
	observer.err = nil
	commands.markSettledErr = want
	settled := sessionloop.Event{Kind: sessionloop.EventRunSettled, CommandID: "command"}
	if err := supervisor.observeEvent(context.Background(), Lease{}, settled); !errors.Is(err, want) {
		t.Fatalf("mark settled error = %v", err)
	}
	commands.markSettledErr = nil
	if err := supervisor.observeEvent(context.Background(), Lease{}, settled); err != nil || commands.settled == 0 {
		t.Fatalf("settle observation = %v count=%d", err, commands.settled)
	}
	supervisor.cfg.Observer = nil
	if err := supervisor.observeEvent(context.Background(), Lease{}, sessionloop.Event{}); err != nil {
		t.Fatal(err)
	}

	updates := make(chan Lease, 1)
	leaseErrors := make(chan error, 1)
	current := Lease{}
	stream := &streamStub{events: make(chan streamResult, 1)}
	updates <- Lease{Fence: 2}
	go func() {
		time.Sleep(time.Millisecond)
		stream.events <- streamResult{event: sessionloop.Event{RunID: "run"}}
	}()
	event, err := nextActorEvent(context.Background(), stream, updates, leaseErrors, &current)
	if err != nil || event.RunID != "run" || current.Fence != 2 {
		t.Fatalf("next with update = %+v, %v current=%+v", event, err, current)
	}
	leaseErrors <- want
	if _, err := nextActorEvent(context.Background(), &streamStub{events: make(chan streamResult)}, make(chan Lease), leaseErrors, &current); !errors.Is(err, ErrLeaseLost) {
		t.Fatalf("lease error = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := nextActorEvent(canceled, &streamStub{events: make(chan streamResult)}, make(chan Lease), make(chan error), &current); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled next = %v", err)
	}
	stream = &streamStub{events: make(chan streamResult, 1)}
	stream.events <- streamResult{err: io.EOF}
	if _, err := nextActorEvent(context.Background(), stream, make(chan Lease), make(chan error), &current); !errors.Is(err, io.EOF) {
		t.Fatalf("stream error = %v", err)
	}

	leases := &leaseStoreStub{renewErr: want}
	supervisor.cfg.Leases = leases
	supervisor.cfg.LeaseTTL = 3 * time.Millisecond
	renewErr := make(chan error, 1)
	go supervisor.renew(context.Background(), Lease{}, make(chan Lease, 1), renewErr)
	select {
	case err := <-renewErr:
		if !errors.Is(err, want) {
			t.Fatalf("renew error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("renew did not report failure")
	}
	cancelCtx, cancelRenew := context.WithCancel(context.Background())
	cancelRenew()
	supervisor.cfg.LeaseTTL = -1
	supervisor.renew(cancelCtx, Lease{}, make(chan Lease), make(chan error, 1))

	renewCtx, stopRenew := context.WithCancel(context.Background())
	defer stopRenew()
	wantLease := Lease{ActorID: "actor", Owner: "pod", Fence: 2}
	supervisor.cfg.Leases = &leaseStoreStub{renewed: wantLease}
	supervisor.cfg.LeaseTTL = 3 * time.Millisecond
	renewed := make(chan Lease, 1)
	go supervisor.renew(renewCtx, Lease{ActorID: "actor", Owner: "pod", Fence: 1}, renewed, make(chan error, 1))
	select {
	case lease := <-renewed:
		if lease.Fence != wantLease.Fence {
			t.Fatalf("renewed lease = %+v", lease)
		}
		stopRenew()
	case <-time.After(time.Second):
		t.Fatal("renew did not publish the updated lease")
	}
}

func TestListenerRecoversAndStopsCleanly(t *testing.T) {
	want := errors.New("injected")
	t.Run("subscribe failure", func(t *testing.T) {
		reported := make(chan error, 1)
		supervisor := &Supervisor{cfg: Config{
			Notifier:          &notifierStub{subscribeErr: want},
			NotificationRetry: time.Hour,
			OnError: func(_ ActorID, err error) {
				reported <- err
			},
		}}
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		go func() {
			supervisor.listen(ctx, make(chan ActorID, 1))
			close(done)
		}()
		select {
		case err := <-reported:
			if !errors.Is(err, want) {
				t.Fatalf("reported error = %v", err)
			}
			cancel()
		case <-time.After(time.Second):
			t.Fatal("subscribe failure was not reported")
		}
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("listener did not stop during retry")
		}
	})
	t.Run("next failure", func(t *testing.T) {
		reported := make(chan error, 1)
		supervisor := &Supervisor{cfg: Config{
			Notifier:          &notifierStub{sub: &subscriptionStub{nextErr: want}},
			NotificationRetry: time.Hour,
			OnError: func(_ ActorID, err error) {
				reported <- err
			},
		}}
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		go func() {
			supervisor.listen(ctx, make(chan ActorID, 1))
			close(done)
		}()
		select {
		case err := <-reported:
			if !errors.Is(err, want) {
				t.Fatalf("reported error = %v", err)
			}
			cancel()
		case <-time.After(time.Second):
			t.Fatal("subscription failure was not reported")
		}
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("listener did not stop after subscription failure")
		}
	})
	t.Run("delivery", func(t *testing.T) {
		supervisor := &Supervisor{cfg: Config{
			Notifier:          &notifierStub{sub: &subscriptionStub{nextIDs: []ActorID{"actor"}}},
			NotificationRetry: time.Hour,
		}}
		ctx, cancel := context.WithCancel(context.Background())
		wakes := make(chan ActorID, 1)
		done := make(chan struct{})
		go func() {
			supervisor.listen(ctx, wakes)
			close(done)
		}()
		select {
		case id := <-wakes:
			if id != "actor" {
				t.Fatalf("wake = %q", id)
			}
			cancel()
		case <-time.After(time.Second):
			t.Fatal("wake was not delivered")
		}
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("listener did not stop")
		}
	})
	t.Run("canceled blocked delivery", func(t *testing.T) {
		supervisor := &Supervisor{cfg: Config{
			Notifier:          &notifierStub{sub: &subscriptionStub{nextIDs: []ActorID{"actor"}}},
			NotificationRetry: time.Hour,
		}}
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		go func() {
			supervisor.listen(ctx, make(chan ActorID))
			close(done)
		}()
		time.Sleep(10 * time.Millisecond)
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("blocked delivery did not observe cancellation")
		}
	})

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if waitRetry(canceled, time.Hour) {
		t.Fatal("canceled retry unexpectedly continued")
	}
}

func TestRunConsumesNotificationWake(t *testing.T) {
	want := errors.New("activation")
	reported := make(chan ActorID, 1)
	supervisor, err := New(Config{
		Owner:        "pod",
		Commands:     &commandStoreStub{},
		Leases:       &leaseStoreStub{},
		Notifier:     &notifierStub{sub: &subscriptionStub{nextIDs: []ActorID{"actor"}}},
		ScanInterval: time.Hour,
		Activator: ActivatorFunc(func(context.Context, ActorID) (sessionloop.Session, error) {
			return nil, want
		}),
		OnError: func(id ActorID, err error) {
			if errors.Is(err, want) {
				reported <- id
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- supervisor.Run(ctx) }()
	select {
	case id := <-reported:
		if id != "actor" {
			t.Fatalf("activation id = %q", id)
		}
		cancel()
	case <-time.After(time.Second):
		t.Fatal("notification wake was not activated")
	}
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("Run = %v", err)
	}
}

func TestRunAndRunActorFailureSurfaces(t *testing.T) {
	want := errors.New("injected")
	t.Run("scan failure", func(t *testing.T) {
		commands := &commandStoreStub{readyErr: want}
		supervisor, err := New(Config{
			Owner: "pod", Commands: commands, Leases: &leaseStoreStub{}, Notifier: &notifierStub{sub: &subscriptionStub{}},
			Activator:    ActivatorFunc(func(context.Context, ActorID) (sessionloop.Session, error) { return &sessionStub{}, nil }),
			ScanInterval: time.Millisecond,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := supervisor.Run(context.Background()); !errors.Is(err, want) {
			t.Fatalf("scan error = %v", err)
		}
	})
	tests := []struct {
		name      string
		commands  *commandStoreStub
		leases    *leaseStoreStub
		activator Activator
		observer  Observer
		want      error
	}{
		{name: "snapshot", activator: ActivatorFunc(func(context.Context, ActorID) (sessionloop.Session, error) {
			return &sessionStub{snapshotErr: want}, nil
		}), want: want},
		{name: "snapshot observer", activator: ActivatorFunc(func(context.Context, ActorID) (sessionloop.Session, error) { return &sessionStub{}, nil }), observer: &snapshotObserverStub{err: want}, want: want},
		{name: "subscribe", activator: ActivatorFunc(func(context.Context, ActorID) (sessionloop.Session, error) {
			return &sessionStub{subscribeErr: want}, nil
		}), want: want},
		{name: "pending", commands: &commandStoreStub{pendingErr: want}, activator: ActivatorFunc(func(context.Context, ActorID) (sessionloop.Session, error) { return &sessionStub{}, nil }), want: want},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			commands := test.commands
			if commands == nil {
				commands = &commandStoreStub{}
			}
			leases := test.leases
			if leases == nil {
				leases = &leaseStoreStub{}
			}
			supervisor, err := New(Config{Owner: "pod", Commands: commands, Leases: leases, Notifier: &notifierStub{}, Activator: test.activator, Observer: test.observer})
			if err != nil {
				t.Fatal(err)
			}
			if err := supervisor.RunActor(context.Background(), "actor"); !errors.Is(err, test.want) {
				t.Fatalf("RunActor = %v", err)
			}
		})
	}
}

func TestRunActorLoopFailureSurfaces(t *testing.T) {
	want := errors.New("injected")
	start := Command{ID: "command", State: CommandPending, Command: sessionloop.Command{
		Kind:  sessionloop.CommandStart,
		Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "hello"}}},
	}}
	tests := []struct {
		name     string
		commands *commandStoreStub
		session  *sessionStub
		observer Observer
	}{
		{
			name:     "dispatch",
			commands: &commandStoreStub{pending: []Command{start}},
			session:  &sessionStub{snapshot: sessionloop.Snapshot{State: sessionloop.StateIdle}, dispatchErr: want},
		},
		{
			name:     "dispatched stream",
			commands: &commandStoreStub{pending: []Command{start}},
			session: &sessionStub{snapshot: sessionloop.Snapshot{State: sessionloop.StateIdle}, receipt: sessionloop.Receipt{RunID: "run"},
				stream: streamWith(streamResult{err: want})},
		},
		{
			name:     "dispatched observer",
			commands: &commandStoreStub{pending: []Command{start}},
			session: &sessionStub{snapshot: sessionloop.Snapshot{State: sessionloop.StateIdle}, receipt: sessionloop.Receipt{RunID: "run"},
				stream: streamWith(streamResult{event: sessionloop.Event{Kind: sessionloop.EventUsage, RunID: "run"}})},
			observer: &observerStub{err: want},
		},
		{
			name:     "post-settlement snapshot",
			commands: &commandStoreStub{pending: []Command{start}},
			session: &sessionStub{snapshot: sessionloop.Snapshot{State: sessionloop.StateIdle}, snapshotErr: want, snapshotErrAfter: 2,
				receipt: sessionloop.Receipt{RunID: "run"}, stream: streamWith(streamResult{event: sessionloop.Event{Kind: sessionloop.EventRunSettled, RunID: "run"}})},
		},
		{
			name:    "idle stream",
			session: &sessionStub{snapshot: sessionloop.Snapshot{State: sessionloop.StateRunning}, stream: streamWith(streamResult{err: want})},
		},
		{
			name:     "idle observer",
			session:  &sessionStub{snapshot: sessionloop.Snapshot{State: sessionloop.StateRunning}, stream: streamWith(streamResult{event: sessionloop.Event{Kind: sessionloop.EventUsage}})},
			observer: &observerStub{err: want},
		},
		{
			name: "authoritative snapshot",
			session: &sessionStub{snapshot: sessionloop.Snapshot{State: sessionloop.StateRunning}, snapshotErr: want, snapshotErrAfter: 2,
				stream: streamWith(streamResult{event: sessionloop.Event{Kind: sessionloop.EventRunStarted, Nature: sessionloop.EventAuthoritative}})},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			commands := test.commands
			if commands == nil {
				commands = &commandStoreStub{}
			}
			supervisor, err := New(Config{
				Owner: "pod", Commands: commands, Leases: &leaseStoreStub{}, Notifier: &notifierStub{},
				Activator: ActivatorFunc(func(context.Context, ActorID) (sessionloop.Session, error) { return test.session, nil }),
				Observer:  test.observer,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := supervisor.RunActor(context.Background(), "actor"); !errors.Is(err, want) {
				t.Fatalf("RunActor = %v", err)
			}
		})
	}
}

func TestRunActorConsumesRenewedFenceBeforePolling(t *testing.T) {
	commands := &commandStoreStub{}
	leases := &leaseStoreStub{
		lease:   Lease{ActorID: "actor", Owner: "pod", Fence: 1},
		renewed: Lease{ActorID: "actor", Owner: "pod", Fence: 2},
	}
	supervisor, err := New(Config{
		Owner: "pod", Commands: commands, Leases: leases, Notifier: &notifierStub{}, LeaseTTL: 3 * time.Millisecond,
		Activator: ActivatorFunc(func(context.Context, ActorID) (sessionloop.Session, error) {
			return &sessionStub{snapshot: sessionloop.Snapshot{State: sessionloop.StateIdle}, snapshotDelay: 10 * time.Millisecond}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := supervisor.RunActor(context.Background(), "actor"); err != nil {
		t.Fatal(err)
	}
	if commands.pendingLease.Fence != 2 {
		t.Fatalf("pending fence = %d, want renewed fence 2", commands.pendingLease.Fence)
	}
}

func streamWith(results ...streamResult) sessionloop.Stream {
	events := make(chan streamResult, len(results))
	for _, result := range results {
		events <- result
	}
	return &streamStub{events: events}
}

type commandStoreStub struct {
	submission                        Submission
	enqueueErr, pendingErr, readyErr  error
	markDispatchedErr, markSettledErr error
	settled, dispatched, failed       int
	pending                           []Command
	pendingLease                      Lease
}

func (s *commandStoreStub) Enqueue(context.Context, Command) (Submission, error) {
	return s.submission, s.enqueueErr
}

func (s *commandStoreStub) Pending(_ context.Context, lease Lease, _ int) ([]Command, error) {
	s.pendingLease = lease
	return s.pending, s.pendingErr
}
func (s *commandStoreStub) MarkDispatched(context.Context, Lease, CommandID, sessionloop.Receipt) error {
	s.dispatched++
	return s.markDispatchedErr
}
func (s *commandStoreStub) MarkSettled(context.Context, Lease, CommandID, *sessionloop.RunOutcome) error {
	s.settled++
	return s.markSettledErr
}
func (s *commandStoreStub) MarkFailed(context.Context, Lease, CommandID, error) error {
	s.failed++
	return nil
}
func (s *commandStoreStub) ReadyActors(context.Context, int) ([]ActorID, error) {
	return nil, s.readyErr
}

type leaseStoreStub struct {
	lease, renewed       Lease
	acquireErr, renewErr error
}

func (s *leaseStoreStub) Acquire(context.Context, ActorID, string, time.Duration) (Lease, error) {
	if s.lease.ActorID == "" {
		s.lease = Lease{ActorID: "actor", Owner: "pod", Fence: 1, Expires: time.Now().Add(time.Minute)}
	}
	return s.lease, s.acquireErr
}
func (s *leaseStoreStub) Renew(context.Context, Lease, time.Duration) (Lease, error) {
	if s.renewed.ActorID != "" {
		return s.renewed, s.renewErr
	}
	return s.lease, s.renewErr
}
func (*leaseStoreStub) Release(context.Context, Lease) error { return nil }

type notifierStub struct {
	published                ActorID
	publishErr, subscribeErr error
	sub                      Subscription
}

func (s *notifierStub) Publish(_ context.Context, id ActorID) error {
	s.published = id
	return s.publishErr
}
func (s *notifierStub) Subscribe(context.Context) (Subscription, error) {
	if s.sub == nil {
		s.sub = &subscriptionStub{}
	}
	return s.sub, s.subscribeErr
}

type subscriptionStub struct {
	nextErr error
	nextIDs []ActorID
	next    int
}

func (s *subscriptionStub) Next(ctx context.Context) (ActorID, error) {
	if s.nextErr != nil {
		return "", s.nextErr
	}
	if s.next < len(s.nextIDs) {
		id := s.nextIDs[s.next]
		s.next++
		return id, nil
	}
	<-ctx.Done()
	return "", ctx.Err()
}
func (*subscriptionStub) Close() error { return nil }

type sessionStub struct {
	caps                                   sessionloop.Capabilities
	snapshot                               sessionloop.Snapshot
	snapshotErr, subscribeErr, dispatchErr error
	receipt                                sessionloop.Receipt
	command                                sessionloop.Command
	stream                                 sessionloop.Stream
	snapshotCalls                          int
	snapshotErrAfter                       int
	snapshotDelay                          time.Duration
}

func (s *sessionStub) ID() sessionloop.SessionID              { return "session" }
func (s *sessionStub) Capabilities() sessionloop.Capabilities { return s.caps }
func (s *sessionStub) Dispatch(_ context.Context, command sessionloop.Command) (sessionloop.Receipt, error) {
	s.command = command.Clone()
	return s.receipt, s.dispatchErr
}
func (s *sessionStub) Snapshot(context.Context) (sessionloop.Snapshot, error) {
	if s.snapshotDelay > 0 {
		time.Sleep(s.snapshotDelay)
	}
	s.snapshotCalls++
	if s.snapshotErr != nil && (s.snapshotErrAfter == 0 || s.snapshotCalls >= s.snapshotErrAfter) {
		return sessionloop.Snapshot{}, s.snapshotErr
	}
	return s.snapshot.Clone(), nil
}
func (s *sessionStub) Subscribe(context.Context, sessionloop.SubscribeOptions) (sessionloop.Stream, error) {
	if s.stream == nil {
		s.stream = &streamStub{events: make(chan streamResult)}
	}
	return s.stream, s.subscribeErr
}
func (*sessionStub) Close(context.Context) error { return nil }

type streamResult struct {
	event sessionloop.Event
	err   error
}

type streamStub struct{ events chan streamResult }

func (s *streamStub) Next(ctx context.Context) (sessionloop.Event, error) {
	select {
	case <-ctx.Done():
		return sessionloop.Event{}, ctx.Err()
	case result := <-s.events:
		return result.event, result.err
	}
}
func (*streamStub) Close() error { return nil }

type observerStub struct{ err error }

func (s *observerStub) Observe(context.Context, Lease, sessionloop.Event) error { return s.err }

type snapshotObserverStub struct{ err error }

func (s *snapshotObserverStub) Observe(context.Context, Lease, sessionloop.Event) error { return nil }
func (s *snapshotObserverStub) ObserveSnapshot(context.Context, Lease, sessionloop.Snapshot) error {
	return s.err
}
