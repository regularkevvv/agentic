package memory

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
)

func command(id, actorID, text string) actor.Command {
	return actor.Command{ID: actor.CommandID(id), ActorID: actor.ActorID(actorID), Command: sessionloop.Command{
		Kind:  sessionloop.CommandStart,
		Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: text}}},
	}}
}

func TestStoreCommandLifecycleAndIdempotency(t *testing.T) {
	store := NewStore()
	first, err := store.Enqueue(context.Background(), command("c1", "a1", "hello"))
	if err != nil || first.Sequence != 1 || first.Duplicate {
		t.Fatalf("first = %#v, %v", first, err)
	}
	duplicate, err := store.Enqueue(context.Background(), command("c1", "a1", "hello"))
	if err != nil || !duplicate.Duplicate || duplicate.Sequence != first.Sequence {
		t.Fatalf("duplicate = %#v, %v", duplicate, err)
	}
	if _, err := store.Enqueue(context.Background(), command("c1", "a1", "different")); !errors.Is(err, actor.ErrCommandConflict) {
		t.Fatalf("conflict = %v", err)
	}
	lease, err := store.Acquire(context.Background(), "a1", "pod", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	pending, err := store.Pending(context.Background(), lease, 10)
	if err != nil || len(pending) != 1 || pending[0].State != actor.CommandPending {
		t.Fatalf("pending = %#v, %v", pending, err)
	}
	receipt := sessionloop.Receipt{CommandID: "c1", SessionID: "s", RunID: "r", Guarantee: sessionloop.AcceptanceDurable}
	if err := store.MarkDispatched(context.Background(), lease, "c1", receipt); err != nil {
		t.Fatal(err)
	}
	if err := store.MarkDispatched(context.Background(), lease, "c1", receipt); err != nil {
		t.Fatal(err)
	}
	outcome := sessionloop.RunOutcome{RunID: "r", Kind: sessionloop.RunCompleted}
	if err := store.MarkSettled(context.Background(), lease, "c1", &outcome); err != nil {
		t.Fatal(err)
	}
	if err := store.MarkSettled(context.Background(), lease, "c1", &outcome); err != nil {
		t.Fatal(err)
	}
	stored, err := store.Command("c1")
	if err != nil || stored.State != actor.CommandSettled || stored.Outcome == nil || stored.Outcome.RunID != "r" {
		t.Fatalf("stored = %#v, %v", stored, err)
	}
	ready, err := store.ReadyActors(context.Background(), 10)
	if err != nil || len(ready) != 0 {
		t.Fatalf("ready = %v, %v", ready, err)
	}
}

func TestStoreFencingExpiryAndFailures(t *testing.T) {
	store := NewStore()
	now := time.Unix(100, 0).UTC()
	store.now = func() time.Time { return now }
	if _, err := store.Enqueue(context.Background(), command("c", "actor", "hello")); err != nil {
		t.Fatal(err)
	}
	lease, err := store.Acquire(context.Background(), "actor", "one", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Acquire(context.Background(), "actor", "two", time.Second); !errors.Is(err, actor.ErrLeaseHeld) {
		t.Fatalf("competing Acquire = %v", err)
	}
	now = now.Add(2 * time.Second)
	replacement, err := store.Acquire(context.Background(), "actor", "two", time.Second)
	if err != nil || replacement.Fence <= lease.Fence {
		t.Fatalf("replacement = %#v, %v", replacement, err)
	}
	if err := store.MarkFailed(context.Background(), lease, "c", errors.New("stale")); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("stale mutation = %v", err)
	}
	renewed, err := store.Renew(context.Background(), replacement, time.Minute)
	if err != nil || !renewed.Expires.After(replacement.Expires) {
		t.Fatalf("renew = %#v, %v", renewed, err)
	}
	if err := store.MarkFailed(context.Background(), renewed, "c", errors.New("boom")); err != nil {
		t.Fatal(err)
	}
	stored, _ := store.Command("c")
	if stored.State != actor.CommandFailed || stored.Error != "boom" {
		t.Fatalf("failed command = %#v", stored)
	}
	if err := store.Release(context.Background(), renewed); err != nil {
		t.Fatal(err)
	}
	if err := store.Release(context.Background(), renewed); err != nil {
		t.Fatal(err)
	}
}

func TestDoorbellFanoutCloseAndCancellation(t *testing.T) {
	notifier := NewDoorbell()
	first, err := notifier.Subscribe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	second, _ := notifier.Subscribe(context.Background())
	if err := notifier.Ring(context.Background(), "actor"); err != nil {
		t.Fatal(err)
	}
	for _, sub := range []actor.Subscription{first, second} {
		id, err := sub.Next(context.Background())
		if err != nil || id != "actor" {
			t.Fatalf("Next = %q, %v", id, err)
		}
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := first.Next(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("closed Next = %v", err)
	}
	notifier.Close()
	if err := notifier.Ring(context.Background(), "actor"); !errors.Is(err, io.EOF) {
		t.Fatalf("closed Ring = %v", err)
	}
	if _, err := notifier.Subscribe(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("closed Subscribe = %v", err)
	}
	if _, err := second.Next(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("notifier-close Next = %v", err)
	}
}

func TestStoreValidationCancellationLimitsAndStateConflicts(t *testing.T) {
	store := NewStore()
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Enqueue(canceled, command("c", "a", "text")); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled enqueue = %v", err)
	}
	for _, invalid := range []actor.Command{
		{ActorID: "actor", Command: command("x", "actor", "text").Command},
		{ID: "command", Command: command("x", "actor", "text").Command},
		{ID: "command", ActorID: "actor", Command: sessionloop.Command{Kind: sessionloop.CommandStart}},
	} {
		if _, err := store.Enqueue(context.Background(), invalid); err == nil {
			t.Fatalf("invalid enqueue succeeded: %+v", invalid)
		}
	}
	created := command("one", "beta", "one")
	created.Created = time.Unix(1, 0)
	if _, err := store.Enqueue(context.Background(), created); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Enqueue(context.Background(), command("one", "other", "one")); !errors.Is(err, actor.ErrCommandConflict) {
		t.Fatalf("cross-actor replay = %v", err)
	}
	if _, err := store.Enqueue(context.Background(), command("two", "alpha", "two")); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Enqueue(context.Background(), command("three", "beta", "three")); err != nil {
		t.Fatal(err)
	}
	ready, err := store.ReadyActors(context.Background(), 1)
	if err != nil || len(ready) != 1 || ready[0] != "alpha" {
		t.Fatalf("limited ready = %v, %v", ready, err)
	}
	if _, err := store.ReadyActors(canceled, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled ready = %v", err)
	}
	if _, err := store.Acquire(canceled, "beta", "pod", time.Second); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled acquire = %v", err)
	}
	for _, input := range []struct {
		id    actor.ActorID
		owner string
		ttl   time.Duration
	}{{owner: "pod", ttl: time.Second}, {id: "beta", ttl: time.Second}, {id: "beta", owner: "pod"}} {
		if _, err := store.Acquire(context.Background(), input.id, input.owner, input.ttl); err == nil {
			t.Fatalf("invalid acquire succeeded: %+v", input)
		}
	}
	lease, err := store.Acquire(context.Background(), "beta", "pod", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Pending(canceled, lease, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled pending = %v", err)
	}
	pending, err := store.Pending(context.Background(), lease, 0)
	if err != nil || len(pending) != 2 {
		t.Fatalf("default-limit pending = %+v, %v", pending, err)
	}
	stale := lease
	stale.Fence++
	if _, err := store.Pending(context.Background(), stale, 1); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("stale pending = %v", err)
	}
	if err := store.MarkDispatched(canceled, lease, "one", sessionloop.Receipt{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled mutate = %v", err)
	}
	if err := store.MarkDispatched(context.Background(), lease, "missing", sessionloop.Receipt{}); !errors.Is(err, actor.ErrCommandNotFound) {
		t.Fatalf("missing mutate = %v", err)
	}
	if err := store.MarkDispatched(context.Background(), lease, "one", sessionloop.Receipt{RunID: "run"}); err != nil {
		t.Fatal(err)
	}
	if err := store.MarkDispatched(context.Background(), lease, "one", sessionloop.Receipt{RunID: "different"}); !errors.Is(err, actor.ErrCommandConflict) {
		t.Fatalf("changed dispatch = %v", err)
	}
	if err := store.MarkSettled(context.Background(), lease, "three", nil); !errors.Is(err, actor.ErrCommandConflict) {
		t.Fatalf("settle pending = %v", err)
	}
	if err := store.MarkSettled(context.Background(), lease, "one", nil); err != nil {
		t.Fatal(err)
	}
	if err := store.MarkDispatched(context.Background(), lease, "one", sessionloop.Receipt{}); !errors.Is(err, actor.ErrCommandConflict) {
		t.Fatalf("dispatch settled = %v", err)
	}
	if err := store.MarkFailed(context.Background(), lease, "three", nil); err != nil {
		t.Fatal(err)
	}
	remaining, err := store.Pending(context.Background(), lease, 0)
	if err != nil || len(remaining) != 0 {
		t.Fatalf("pending skips terminal = %+v, %v", remaining, err)
	}
	if _, err := store.Renew(canceled, lease, time.Second); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled renew = %v", err)
	}
	if _, err := store.Renew(context.Background(), stale, time.Second); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("stale renew = %v", err)
	}
	if err := store.Release(canceled, lease); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled release = %v", err)
	}
	if err := store.Release(context.Background(), stale); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("stale release = %v", err)
	}
	if _, err := store.Command("missing"); !errors.Is(err, actor.ErrCommandNotFound) {
		t.Fatalf("missing command = %v", err)
	}
}

func TestDoorbellCancellationAndLossyBuffer(t *testing.T) {
	notifier := NewDoorbell()
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := notifier.Ring(canceled, "actor"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled publish = %v", err)
	}
	if _, err := notifier.Subscribe(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled subscribe = %v", err)
	}
	cancelSubscription, err := notifier.Subscribe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cancelSubscription.Next(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled next = %v", err)
	}
	if err := cancelSubscription.Close(); err != nil {
		t.Fatal(err)
	}
	subscription, err := notifier.Subscribe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 70; index++ {
		if err := notifier.Ring(context.Background(), actor.ActorID("actor")); err != nil {
			t.Fatal(err)
		}
	}
	if err := subscription.Close(); err != nil {
		t.Fatal(err)
	}
	if err := subscription.Close(); err != nil {
		t.Fatalf("idempotent subscription close = %v", err)
	}
}
