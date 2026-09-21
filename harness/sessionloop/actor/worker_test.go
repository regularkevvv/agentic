// Integration tests interrupt the actual Worker at each handoff boundary and
// verify retained work against a restartable reference harness.
package actor_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor/memory"
	"github.com/regularkevvv/agentic/harness/sessionloop/testkit"
)

func input(id string) actor.Command {
	return actor.Command{ActorID: "a", ID: actor.CommandID(id), Command: sessionloop.Command{Kind: sessionloop.CommandStart,
		Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: id}}}}}
}

func eventually(t *testing.T, check func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if check() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("condition did not become true")
}

type boundOpener struct {
	host     *testkit.Host
	mu       sync.Mutex
	bindings map[actor.ActorID]sessionloop.SessionID
	opened   chan actor.Session
	wrap     func(actor.Session) actor.Session
	count    atomic.Int32
}

func newOpener(host *testkit.Host) *boundOpener {
	return &boundOpener{host: host, bindings: make(map[actor.ActorID]sessionloop.SessionID), opened: make(chan actor.Session, 128)}
}

func (o *boundOpener) Open(ctx context.Context, l actor.Lease) (actor.Session, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	var session sessionloop.Session
	var err error
	if id, exists := o.bindings[l.ActorID]; exists {
		session, err = o.host.OpenSession(ctx, id)
	} else {
		session, err = o.host.NewSession(ctx, sessionloop.SessionOptions{})
		if err == nil {
			o.bindings[l.ActorID] = session.ID()
		}
	}
	if err != nil {
		return nil, err
	}
	o.count.Add(1)
	result := session.(actor.Session)
	o.opened <- result
	if o.wrap != nil {
		result = o.wrap(result)
	}
	return result, nil
}

type faultAdapter struct {
	actor.Adapter
	failAckBefore atomic.Bool
	failAckAfter  atomic.Bool
	released      chan actor.Lease
	acks          atomic.Int32
	beforeRelease func()
}

func (a *faultAdapter) Acknowledge(ctx context.Context, l actor.Lease, id actor.CommandID, r sessionloop.Receipt) error {
	if a.failAckBefore.CompareAndSwap(true, false) {
		return errors.New("before mailbox deletion")
	}
	if err := a.Adapter.Acknowledge(ctx, l, id, r); err != nil {
		return err
	}
	a.acks.Add(1)
	if a.failAckAfter.CompareAndSwap(true, false) {
		return errors.New("after mailbox deletion")
	}
	return nil
}

func (a *faultAdapter) Release(ctx context.Context, l actor.Lease) error {
	if a.beforeRelease != nil {
		a.beforeRelease()
	}
	if err := a.Adapter.Release(ctx, l); err != nil {
		return err
	}
	a.released <- l
	return nil
}

type lostReplySession struct {
	actor.Session
	lose *atomic.Bool
}

func (s *lostReplySession) Dispatch(ctx context.Context, c sessionloop.Command) (sessionloop.Receipt, error) {
	r, err := s.Session.Dispatch(ctx, c)
	if err == nil && s.lose.CompareAndSwap(true, false) {
		return sessionloop.Receipt{}, errors.New("journal committed; reply lost")
	}
	return r, err
}

type failedCloseSession struct {
	actor.Session
	fail *atomic.Bool
}

func (s *failedCloseSession) Close(ctx context.Context) error {
	err := s.Session.Close(ctx)
	if err == nil && s.fail.CompareAndSwap(true, false) {
		return errors.New("close reply lost")
	}
	return err
}

func startWorker(t *testing.T, a actor.Adapter, o actor.SessionOpener, edit func(*actor.Config)) (actor.Worker, func(), <-chan error) {
	t.Helper()
	cfg := actor.Config{Owner: "worker", Adapter: a, SessionOpener: o, LeaseTTL: 45 * time.Millisecond,
		PollInterval: time.Millisecond, RetryInterval: time.Millisecond, MaxActors: 1, BatchSize: 4}
	if edit != nil {
		edit(&cfg)
	}
	w, err := actor.NewWorker(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()
	t.Cleanup(cancel)
	return w, cancel, done
}

func stop(t *testing.T, cancel func(), done <-chan error) {
	t.Helper()
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("worker did not stop")
	}
}

func awaitRelease(t *testing.T, a *faultAdapter) {
	t.Helper()
	select {
	case <-a.released:
	case <-time.After(3 * time.Second):
		t.Fatal("session never retired")
	}
}

func TestSubmitDoesNotExecuteAndWorkerDrainsInOrder(t *testing.T) {
	store := memory.New()
	o := newOpener(testkit.New(testkit.WithIdempotentDispatch()))
	for _, id := range []string{"one", "two"} {
		if _, err := store.Submit(t.Context(), input(id)); err != nil {
			t.Fatal(err)
		}
	}
	if o.count.Load() != 0 {
		t.Fatal("submission opened a harness")
	}
	a := &faultAdapter{Adapter: store, released: make(chan actor.Lease, 8)}
	var settlements atomic.Int32
	_, cancel, done := startWorker(t, a, o, func(c *actor.Config) {
		c.EventSink = actor.EventSinkFunc(func(_ context.Context, _ actor.Lease, e sessionloop.Event) error {
			if e.Kind == sessionloop.EventRunSettled {
				settlements.Add(1)
			}
			return nil
		})
	})
	awaitRelease(t, a)
	stop(t, cancel, done)
	if settlements.Load() != 2 {
		t.Fatalf("closed before observing terminal events: %d", settlements.Load())
	}
	s, err := o.Open(t.Context(), actor.Lease{ActorID: "a"})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(context.Background())
	snapshot, err := s.Snapshot(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var users []string
	for _, entry := range snapshot.Entries {
		if entry.Role == sessionloop.RoleUser {
			users = append(users, entry.Blocks[0].Text)
		}
	}
	if fmt.Sprint(users) != "[one two]" || a.acks.Load() != 2 {
		t.Fatalf("users=%v acknowledgments=%d", users, a.acks.Load())
	}
}

func TestActiveRunReceivesSteeringBehindMultipleBusyStartPages(t *testing.T) {
	host := testkit.New(testkit.WithIdempotentDispatch())
	release := host.HoldNextRun()
	defer release()
	store := memory.New()
	o := newOpener(host)
	a := &faultAdapter{Adapter: store, released: make(chan actor.Lease, 8)}
	if _, err := store.Submit(t.Context(), input("start")); err != nil {
		t.Fatal(err)
	}
	_, cancel, done := startWorker(t, a, o, func(c *actor.Config) { c.BatchSize = 1 })
	var session actor.Session
	select {
	case session = <-o.opened:
	case <-time.After(time.Second):
		t.Fatal("not opened")
	}
	var runID sessionloop.RunID
	eventually(t, func() bool {
		s, e := session.Snapshot(t.Context())
		runID = s.ActiveRunID
		return e == nil && runID != "" && a.acks.Load() == 1
	})
	for _, id := range []string{"next1", "next2", "next3"} {
		if _, err := store.Submit(t.Context(), input(id)); err != nil {
			t.Fatal(err)
		}
	}
	steer := input("steer")
	steer.Command.Kind = sessionloop.CommandSteer
	steer.Command.RunID = runID
	if _, err := store.Submit(t.Context(), steer); err != nil {
		t.Fatal(err)
	}
	canonical, _ := steer.Normalize()
	eventually(t, func() bool {
		_, found, err := session.Acceptance(t.Context(), canonical.Command)
		return err == nil && found
	})
	if a.acks.Load() < 1 {
		t.Fatal("start not removed while run was held")
	}
	snapshot, err := session.Snapshot(t.Context())
	if err != nil || snapshot.ActiveRunID != runID {
		t.Fatalf("steering missed active run=%+v %v", snapshot, err)
	}
	release()
	awaitRelease(t, a)
	stop(t, cancel, done)
}

func TestWorkerRecoversEachAmbiguousHandoffWithoutReexecutingInput(t *testing.T) {
	for _, point := range []string{"accept reply", "before deletion", "after deletion", "close reply"} {
		t.Run(point, func(t *testing.T) {
			store := memory.New()
			o := newOpener(testkit.New(testkit.WithIdempotentDispatch()))
			a := &faultAdapter{Adapter: store, released: make(chan actor.Lease, 8)}
			var once atomic.Bool
			once.Store(true)
			switch point {
			case "accept reply":
				o.wrap = func(s actor.Session) actor.Session { return &lostReplySession{s, &once} }
			case "before deletion":
				a.failAckBefore.Store(true)
			case "after deletion":
				a.failAckAfter.Store(true)
			case "close reply":
				o.wrap = func(s actor.Session) actor.Session { return &failedCloseSession{s, &once} }
			}
			failures := make(chan error, 16)
			if _, err := store.Submit(t.Context(), input("one")); err != nil {
				t.Fatal(err)
			}
			_, cancel, done := startWorker(t, a, o, func(c *actor.Config) { c.OnError = func(_ actor.ActorID, e error) { failures <- e } })
			awaitRelease(t, a)
			stop(t, cancel, done)
			if len(failures) == 0 || o.count.Load() < 2 {
				t.Fatalf("failure was not recovered: errors=%d opens=%d", len(failures), o.count.Load())
			}
			s, err := o.Open(t.Context(), actor.Lease{ActorID: "a"})
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close(context.Background())
			snapshot, err := s.Snapshot(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			count := 0
			for _, entry := range snapshot.Entries {
				if entry.Role == sessionloop.RoleUser {
					count++
				}
			}
			if count != 1 {
				t.Fatalf("input executed %d times", count)
			}
			retry, err := store.Submit(t.Context(), input("one"))
			if err != nil || !retry.Duplicate {
				t.Fatalf("retry after cleanup=%+v %v", retry, err)
			}
		})
	}
}

func TestSubmissionInCloseReleaseWindowRunsWithoutAnotherWake(t *testing.T) {
	store := memory.New()
	o := newOpener(testkit.New(testkit.WithIdempotentDispatch()))
	a := &faultAdapter{Adapter: store, released: make(chan actor.Lease, 8)}
	var once sync.Once
	a.beforeRelease = func() {
		once.Do(func() {
			if _, err := store.Submit(context.Background(), input("racing")); err != nil {
				panic(err)
			}
		})
	}
	if _, err := store.Submit(t.Context(), input("one")); err != nil {
		t.Fatal(err)
	}
	_, cancel, done := startWorker(t, a, o, nil)
	awaitRelease(t, a)
	awaitRelease(t, a)
	stop(t, cancel, done)
	if a.acks.Load() != 2 {
		t.Fatalf("racing command left unexecuted: %d", a.acks.Load())
	}
}

func TestFailedRestorationNeverReplacesSessionOrDropsInput(t *testing.T) {
	store := memory.New()
	if _, err := store.Submit(t.Context(), input("one")); err != nil {
		t.Fatal(err)
	}
	failure := errors.New("missing bound journal")
	failures := make(chan error, 4)
	opener := actor.SessionOpenerFunc(func(context.Context, actor.Lease) (actor.Session, error) { return nil, failure })
	_, cancel, done := startWorker(t, store, opener, func(c *actor.Config) { c.OnError = func(_ actor.ActorID, e error) { failures <- e } })
	select {
	case err := <-failures:
		if !errors.Is(err, failure) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("open failure not reported")
	}
	stop(t, cancel, done)
	ctx, end := context.WithTimeout(t.Context(), time.Second)
	defer end()
	id, err := store.Receive(ctx)
	if err != nil || id != "a" {
		t.Fatalf("not recoverable: %s %v", id, err)
	}
	l, err := store.Acquire(ctx, "a", "test", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	pending, err := store.Pending(ctx, l, 0, 10)
	if err != nil || len(pending) != 1 {
		t.Fatalf("lost input=%v %v", pending, err)
	}
}

func TestStaleCommandIsJournaledAsRejectedAndDoesNotBlockNextInput(t *testing.T) {
	store := memory.New()
	o := newOpener(testkit.New(testkit.WithIdempotentDispatch()))
	a := &faultAdapter{Adapter: store, released: make(chan actor.Lease, 8)}
	stale := input("stale")
	stale.Command = sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "finished"}
	if _, err := store.Submit(t.Context(), stale); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Submit(t.Context(), input("valid")); err != nil {
		t.Fatal(err)
	}
	_, cancel, done := startWorker(t, a, o, nil)
	awaitRelease(t, a)
	stop(t, cancel, done)
	s, err := o.Open(t.Context(), actor.Lease{ActorID: "a"})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(t.Context())
	canonical, _ := stale.Normalize()
	r, found, err := s.Acceptance(t.Context(), canonical.Command)
	if err != nil || !found || !r.Rejection.Valid() {
		t.Fatalf("terminal failure not recorded: %+v %v %v", r, found, err)
	}
	if a.acks.Load() != 2 {
		t.Fatalf("next input blocked: ack=%d", a.acks.Load())
	}
}
