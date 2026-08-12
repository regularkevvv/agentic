package actor_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	actormemory "github.com/regularkevvv/agentic/harness/sessionloop/actor/memory"
	"github.com/regularkevvv/agentic/harness/sessionloop/testkit"
)

func actorCommand(id, actorID, text string) actor.Command {
	return actor.Command{
		ID: actor.CommandID(id), ActorID: actor.ActorID(actorID),
		Command: sessionloop.Command{Kind: sessionloop.CommandStart, Input: &sessionloop.Input{
			Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: text}},
		}},
	}
}

func newSupervisor(t *testing.T, store *actormemory.Store, notifier actor.Notifier, host sessionloop.Host, observer actor.Observer) *actor.Supervisor {
	t.Helper()
	supervisor, err := actor.New(actor.Config{
		Owner: "pod-a", Commands: store, Leases: store, Notifier: notifier,
		Activator: actor.ActivatorFunc(func(ctx context.Context, _ actor.ActorID) (sessionloop.Session, error) {
			return host.NewSession(ctx, sessionloop.SessionOptions{})
		}),
		Observer: observer, LeaseTTL: time.Second, ScanInterval: 5 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	return supervisor
}

func TestRunActorDrainsMessagesInOrderAndObservesEvents(t *testing.T) {
	store := actormemory.NewStore()
	notifier := actormemory.NewNotifier()
	host := testkit.New(testkit.WithIdempotentDispatch())
	var mu sync.Mutex
	var settled []sessionloop.RunID
	supervisor := newSupervisor(t, store, notifier, host, actor.ObserverFunc(func(_ context.Context, _ actor.Lease, event sessionloop.Event) error {
		if event.Kind == sessionloop.EventRunSettled {
			mu.Lock()
			settled = append(settled, event.RunID)
			mu.Unlock()
		}
		return nil
	}))
	for _, command := range []actor.Command{actorCommand("one", "conversation", "first"), actorCommand("two", "conversation", "second")} {
		if _, err := store.Enqueue(context.Background(), command); err != nil {
			t.Fatal(err)
		}
	}
	if err := supervisor.RunActor(context.Background(), "conversation"); err != nil {
		t.Fatal(err)
	}
	for _, id := range []actor.CommandID{"one", "two"} {
		command, err := store.Command(id)
		if err != nil {
			t.Fatal(err)
		}
		if command.State != actor.CommandSettled || command.Receipt == nil || command.Receipt.RunID == "" {
			t.Fatalf("command %s = %#v", id, command)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(settled) != 2 || settled[0] == settled[1] {
		t.Fatalf("settled runs = %v", settled)
	}
}

func TestSupervisorScanRecoversDroppedNotification(t *testing.T) {
	store := actormemory.NewStore()
	notifier := actormemory.NewNotifier()
	host := testkit.New(testkit.WithIdempotentDispatch())
	supervisor := newSupervisor(t, store, notifier, host, nil)
	if _, err := store.Enqueue(context.Background(), actorCommand("lost-wake", "conversation", "hello")); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- supervisor.Run(ctx) }()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		command, err := store.Command("lost-wake")
		if err == nil && command.State == actor.CommandSettled {
			cancel()
			if err := <-done; !errors.Is(err, context.Canceled) {
				t.Fatalf("Run = %v", err)
			}
			return
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	<-done
	t.Fatal("periodic ReadyActors scan did not recover the dropped notification")
}

func TestSupervisorSubmitReportsDurableWakeFailure(t *testing.T) {
	store := actormemory.NewStore()
	notifier := actormemory.NewNotifier()
	notifier.Close()
	supervisor := newSupervisor(t, store, notifier, testkit.New(), nil)
	submission, err := supervisor.Submit(context.Background(), actorCommand("durable", "actor", "message"))
	if err == nil || submission.ID != "durable" {
		t.Fatalf("Submit = %#v, %v", submission, err)
	}
	command, commandErr := store.Command("durable")
	if commandErr != nil || command.State != actor.CommandPending {
		t.Fatalf("durable command = %#v, %v", command, commandErr)
	}
}

func TestSupervisorValidationAndCompetingLease(t *testing.T) {
	if _, err := actor.New(actor.Config{}); err == nil {
		t.Fatal("empty config succeeded")
	}
	store := actormemory.NewStore()
	notifier := actormemory.NewNotifier()
	host := testkit.New()
	supervisor := newSupervisor(t, store, notifier, host, nil)
	lease, err := store.Acquire(context.Background(), "busy", "other", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Release(context.Background(), lease) }()
	if err := supervisor.RunActor(context.Background(), "busy"); !errors.Is(err, actor.ErrLeaseHeld) {
		t.Fatalf("RunActor = %v", err)
	}
}

func TestSupervisorReportsActivationFailure(t *testing.T) {
	store := actormemory.NewStore()
	notifier := actormemory.NewNotifier()
	boom := errors.New("activate")
	failures := make(chan error, 1)
	supervisor, err := actor.New(actor.Config{
		Owner: "pod-a", Commands: store, Leases: store, Notifier: notifier,
		Activator:    actor.ActivatorFunc(func(context.Context, actor.ActorID) (sessionloop.Session, error) { return nil, boom }),
		OnError:      func(_ actor.ActorID, err error) { failures <- err },
		ScanInterval: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Enqueue(context.Background(), actorCommand("failure", "actor", "message")); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- supervisor.Run(ctx) }()
	select {
	case err := <-failures:
		if !errors.Is(err, boom) {
			t.Fatalf("failure = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("activation failure was not reported")
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("Run = %v", err)
	}
}

func TestSupervisorKeepsScanningAndResubscribesDuringNotifierOutage(t *testing.T) {
	store := actormemory.NewStore()
	notifier := &flakyNotifier{delegate: actormemory.NewNotifier(), subscribeFailures: 1}
	failures := make(chan error, 2)
	host := testkit.New(testkit.WithIdempotentDispatch())
	supervisor, err := actor.New(actor.Config{
		Owner: "pod-a", Commands: store, Leases: store, Notifier: notifier,
		Activator: actor.ActivatorFunc(func(ctx context.Context, _ actor.ActorID) (sessionloop.Session, error) {
			return host.NewSession(ctx, sessionloop.SessionOptions{})
		}),
		OnError:  func(_ actor.ActorID, err error) { failures <- err },
		LeaseTTL: time.Second, ScanInterval: time.Millisecond, NotificationRetry: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Enqueue(context.Background(), actorCommand("during-outage", "conversation", "hello")); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- supervisor.Run(ctx) }()
	select {
	case err := <-failures:
		if !strings.Contains(err.Error(), "durable scanning continues") {
			t.Fatalf("notifier error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("notifier outage was not reported")
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		command, commandErr := store.Command("during-outage")
		if commandErr == nil && command.State == actor.CommandSettled {
			cancel()
			if err := <-done; !errors.Is(err, context.Canceled) {
				t.Fatalf("Run = %v", err)
			}
			return
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	<-done
	t.Fatal("durable scan stopped during notifier outage")
}

func TestRunActorPublishesRecoveredSnapshotBeforeDraining(t *testing.T) {
	store := actormemory.NewStore()
	notifier := actormemory.NewNotifier()
	host := testkit.New(testkit.WithIdempotentDispatch())
	observer := &snapshotObserver{snapshots: make(chan sessionloop.Snapshot, 1)}
	supervisor := newSupervisor(t, store, notifier, host, observer)
	if _, err := store.Enqueue(context.Background(), actorCommand("snapshot", "conversation", "hello")); err != nil {
		t.Fatal(err)
	}
	if err := supervisor.RunActor(context.Background(), "conversation"); err != nil {
		t.Fatal(err)
	}
	select {
	case snapshot := <-observer.snapshots:
		if snapshot.State != sessionloop.StateIdle {
			t.Fatalf("activation snapshot = %+v", snapshot)
		}
	default:
		t.Fatal("activation snapshot was not observed")
	}
}

type snapshotObserver struct{ snapshots chan sessionloop.Snapshot }

func (o *snapshotObserver) Observe(context.Context, actor.Lease, sessionloop.Event) error { return nil }

func (o *snapshotObserver) ObserveSnapshot(_ context.Context, _ actor.Lease, snapshot sessionloop.Snapshot) error {
	o.snapshots <- snapshot.Clone()
	return nil
}

type flakyNotifier struct {
	mu                sync.Mutex
	delegate          actor.Notifier
	subscribeFailures int
}

func (n *flakyNotifier) Publish(ctx context.Context, id actor.ActorID) error {
	return n.delegate.Publish(ctx, id)
}

func (n *flakyNotifier) Subscribe(ctx context.Context) (actor.Subscription, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.subscribeFailures > 0 {
		n.subscribeFailures--
		return nil, errors.New("notification transport down")
	}
	return n.delegate.Subscribe(ctx)
}
