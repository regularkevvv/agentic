// Tests exercise the local flavor's linearization points, wake/expiry behavior
// and authority boundary, in addition to the reusable adapter contract suite.
package localchannel

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor/conformance"
)

func command(id, session string) actor.Command {
	return actor.Command{ID: actor.CommandID(id), ActorID: actor.ActorID(session), Command: sessionloop.Command{Kind: sessionloop.CommandStart,
		Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: id}}}}}
}

func seed(t *testing.T, s *Store) actor.Lease {
	t.Helper()
	if _, err := s.Submit(t.Context(), command("one", "a")); err != nil {
		t.Fatal(err)
	}
	l, err := s.Acquire(t.Context(), "a", "owner", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func TestContract(t *testing.T) {
	conformance.Run(t, func(t *testing.T) conformance.Fixture {
		s := New()
		return conformance.Fixture{Adapter: s, Expire: func(l actor.Lease) {
			s.mu.Lock()
			s.sessions[l.ActorID].lease.Expires = time.Time{}
			s.signal()
			s.mu.Unlock()
		}}
	})
}

func TestRecoverWithoutNewInput(t *testing.T) {
	s := New()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := s.Recover(ctx, "session"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := s.Recover(t.Context(), ""); err == nil {
		t.Fatal("empty session accepted")
	}
	if err := s.Recover(t.Context(), "session"); err != nil {
		t.Fatal(err)
	}
	if err := s.Recover(t.Context(), "session"); err != nil {
		t.Fatal(err)
	}
	if id, err := s.Receive(t.Context()); err != nil || id != "session" {
		t.Fatal(id, err)
	}
	l, err := s.Acquire(t.Context(), "session", "worker", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Release(t.Context(), l); err != nil {
		t.Fatal(err)
	}
	if err := s.Recover(t.Context(), "session"); err != nil {
		t.Fatal(err)
	}
	if id, err := s.Receive(t.Context()); err != nil || id != "session" {
		t.Fatal(id, err)
	}
}

func TestReceiveWaitsForSubmissionAndExpiry(t *testing.T) {
	s := New()
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	done := make(chan actor.ActorID, 1)
	go func() { id, _ := s.Receive(ctx); done <- id }()
	if _, err := s.Submit(ctx, command("one", "a")); err != nil {
		t.Fatal(err)
	}
	select {
	case id := <-done:
		if id != "a" {
			t.Fatal(id)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	l, err := s.Acquire(ctx, "a", "owner", 20*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Renew(ctx, l, 20*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	id, err := s.Receive(ctx)
	if err != nil || id != "a" {
		t.Fatalf("expiry wake=%s %v", id, err)
	}
}

func TestReceiveFairnessAndNoLostWakeAtRelease(t *testing.T) {
	for range 30 {
		s := New()
		l := seed(t, s)
		if _, err := s.Submit(t.Context(), command("one", "b")); err != nil {
			t.Fatal(err)
		}
		if id, err := s.Receive(t.Context()); err != nil || id != "b" {
			t.Fatalf("skip owned=%s %v", id, err)
		}
		var wg sync.WaitGroup
		wg.Add(2)
		go func() { defer wg.Done(); _, _ = s.Submit(t.Context(), command("two", "a")) }()
		go func() { defer wg.Done(); _ = s.Release(t.Context(), l) }()
		wg.Wait()
		id, err := s.Receive(t.Context())
		if err != nil || id != "a" {
			t.Fatalf("shutdown race=%s %v", id, err)
		}
		id, err = s.Receive(t.Context())
		if err != nil || id != "b" {
			t.Fatalf("round robin=%s %v", id, err)
		}
	}
}

func TestAuthoritySerializesTakeoverWithActualMutation(t *testing.T) {
	s := New()
	l := seed(t, s)
	entered, finish := make(chan struct{}), make(chan struct{})
	committed := make(chan error, 1)
	go func() {
		committed <- s.Authority(l).Commit(t.Context(), func(context.Context) error { close(entered); <-finish; return nil })
	}()
	<-entered
	takeover := make(chan actor.Lease, 1)
	go func() {
		s.mu.Lock()
		s.sessions["a"].lease.Expires = time.Time{}
		s.mu.Unlock()
		fresh, _ := s.Acquire(t.Context(), "a", "new", time.Minute)
		takeover <- fresh
	}()
	select {
	case <-takeover:
		t.Fatal("takeover crossed guarded mutation")
	default:
	}
	close(finish)
	if err := <-committed; err != nil {
		t.Fatal(err)
	}
	fresh := <-takeover
	if fresh.Fence <= l.Fence {
		t.Fatalf("fresh=%+v", fresh)
	}
	called := false
	if err := s.Authority(l).Commit(t.Context(), func(context.Context) error { called = true; return nil }); !errors.Is(err, actor.ErrLeaseLost) || called {
		t.Fatalf("stale commit=%v called=%v", err, called)
	}
}

func TestValidationCancellationOverflowAndIsolation(t *testing.T) {
	s := New()
	ctx := t.Context()
	l := seed(t, s)
	if s.Guarantee() != sessionloop.AcceptanceAccepted {
		t.Fatal("false crash durability")
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := s.Submit(canceled, command("x", "a")); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.Acquire(canceled, "a", "x", time.Second); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.Renew(canceled, l, time.Second); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := s.Release(canceled, l); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.Pending(canceled, l, 0, 1); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := s.Acknowledge(canceled, l, "one", sessionloop.Receipt{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := s.Authority(l).Commit(canceled, nil); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.Submit(ctx, actor.Command{}); err == nil {
		t.Fatal("invalid submission")
	}
	if _, err := s.Acquire(ctx, "", "", 0); err == nil {
		t.Fatal("invalid acquire")
	}
	if _, err := s.Acquire(ctx, "missing", "owner", time.Second); !errors.Is(err, actor.ErrNoWork) {
		t.Fatal(err)
	}
	if _, err := s.Renew(ctx, l, 0); err == nil {
		t.Fatal("invalid renewal")
	}
	if _, err := s.Pending(ctx, actor.Lease{}, 0, 1); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatal(err)
	}
	if err := s.Acknowledge(ctx, l, "missing", sessionloop.Receipt{}); !errors.Is(err, actor.ErrCommandNotFound) {
		t.Fatal(err)
	}
	if err := s.Acknowledge(ctx, l, "one", sessionloop.Receipt{}); !errors.Is(err, actor.ErrInvalidReceipt) {
		t.Fatal(err)
	}
	if _, err := s.Submit(ctx, command("one", "b")); err != nil {
		t.Fatal("IDs are session scoped", err)
	}
	s.mu.Lock()
	s.sessions["a"].sequence = math.MaxUint64
	s.mu.Unlock()
	if _, err := s.Submit(ctx, command("two", "a")); !errors.Is(err, actor.ErrGenerationExhausted) {
		t.Fatal(err)
	}
	s.mu.Lock()
	s.sessions["a"].generation = actor.Fence(math.MaxUint64)
	s.sessions["a"].lease.Expires = time.Time{}
	s.mu.Unlock()
	if _, err := s.Acquire(ctx, "a", "new", time.Second); !errors.Is(err, actor.ErrGenerationExhausted) {
		t.Fatal(err)
	}
	other, err := s.Acquire(ctx, "b", "owner", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	r := sessionloop.Receipt{CommandID: "one", SessionID: "b", Guarantee: sessionloop.AcceptanceAccepted}
	if err := s.Acknowledge(ctx, other, "one", r); err != nil {
		t.Fatal(err)
	}
	if err := s.Release(ctx, other); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Acquire(ctx, "b", "owner", time.Minute); !errors.Is(err, actor.ErrNoWork) {
		t.Fatal(err)
	}
}
