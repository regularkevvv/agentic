// Package conformance exercises actual adapters against the actor delivery laws.
// These are shared implementation tests, not a substitute for a Lean code proof.
package conformance

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
)

// Fixture supplies isolated storage and a deterministic ownership-expiry hook.
// Expire must revoke the grant without deleting mailbox or journal history.
type Fixture struct {
	Adapter actor.Adapter
	Expire  func(actor.Lease)
}

func input(id string) actor.Command {
	return actor.Command{ActorID: "session", ID: actor.CommandID(id), Command: sessionloop.Command{
		Kind: sessionloop.CommandStart, Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: id}}}}}
}

func receipt(id string) sessionloop.Receipt {
	return sessionloop.Receipt{CommandID: sessionloop.CommandID(id), SessionID: "journal", Guarantee: sessionloop.AcceptanceDurable, Position: sessionloop.Position{Sequence: 1, Token: id}}
}

// Run tests conservation, immutable identity, fencing, pagination and both
// shutdown-race orders. Receipt fixtures stand for previously journaled facts;
// Worker integration tests separately verify the real acceptance-before-delete.
func Run(t *testing.T, factory func(*testing.T) Fixture) {
	t.Helper()
	t.Run("identity survives mailbox deletion", func(t *testing.T) {
		f := factory(t)
		ctx := t.Context()
		first, err := f.Adapter.Submit(ctx, input("one"))
		if err != nil {
			t.Fatal(err)
		}
		lease, err := f.Adapter.Acquire(ctx, "session", "worker", time.Minute)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Adapter.Acknowledge(ctx, lease, "one", receipt("one")); err != nil {
			t.Fatal(err)
		}
		if err := f.Adapter.Acknowledge(ctx, lease, "one", receipt("one")); err != nil {
			t.Fatal(err)
		}
		pending, err := f.Adapter.Pending(ctx, lease, 0, 10)
		if err != nil || len(pending) != 0 {
			t.Fatalf("mailbox=%v %v", pending, err)
		}
		again, err := f.Adapter.Submit(ctx, input("one"))
		if err != nil || !again.Duplicate || again.Sequence != first.Sequence || again.Guarantee != first.Guarantee {
			t.Fatalf("retry=%+v %v", again, err)
		}
		changed := input("one")
		changed.Command.Input.Blocks[0].Text = "different"
		if _, err := f.Adapter.Submit(ctx, changed); !errors.Is(err, actor.ErrCommandConflict) {
			t.Fatalf("conflict=%v", err)
		}
		// Simulated crash AFTER deletion: empty mailbox must still be discovered.
		f.Expire(lease)
		id, err := f.Adapter.Receive(ctx)
		if err != nil || id != "session" {
			t.Fatalf("recovery=%s %v", id, err)
		}
	})
	for _, before := range []bool{true, false} {
		name := "submission after retirement"
		if before {
			name = "submission before retirement"
		}
		t.Run(name, func(t *testing.T) {
			f := factory(t)
			ctx := t.Context()
			if _, err := f.Adapter.Submit(ctx, input("one")); err != nil {
				t.Fatal(err)
			}
			lease, err := f.Adapter.Acquire(ctx, "session", "worker", time.Minute)
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Adapter.Acknowledge(ctx, lease, "one", receipt("one")); err != nil {
				t.Fatal(err)
			}
			// The fixture's journal is quiescent and closed at this boundary.
			if before {
				if _, err := f.Adapter.Submit(ctx, input("two")); err != nil {
					t.Fatal(err)
				}
			}
			if err := f.Adapter.Release(ctx, lease); err != nil {
				t.Fatal(err)
			}
			if !before {
				if _, err := f.Adapter.Submit(ctx, input("two")); err != nil {
					t.Fatal(err)
				}
			}
			id, err := f.Adapter.Receive(ctx)
			if err != nil || id != "session" {
				t.Fatalf("ready=%s %v", id, err)
			}
		})
	}
	t.Run("expired owner cannot mutate successor", func(t *testing.T) {
		f := factory(t)
		ctx := t.Context()
		if _, err := f.Adapter.Submit(ctx, input("one")); err != nil {
			t.Fatal(err)
		}
		old, err := f.Adapter.Acquire(ctx, "session", "first", time.Minute)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Adapter.Acquire(ctx, "session", "second", time.Minute); !errors.Is(err, actor.ErrLeaseHeld) {
			t.Fatalf("double owner=%v", err)
		}
		f.Expire(old)
		fresh, err := f.Adapter.Acquire(ctx, "session", "second", time.Minute)
		if err != nil || fresh.Fence <= old.Fence {
			t.Fatalf("takeover=%+v %v", fresh, err)
		}
		if err := f.Adapter.Acknowledge(ctx, old, "one", receipt("one")); !errors.Is(err, actor.ErrLeaseLost) {
			t.Fatalf("stale ack=%v", err)
		}
		if _, err := f.Adapter.Renew(ctx, old, time.Minute); !errors.Is(err, actor.ErrLeaseLost) {
			t.Fatalf("stale renew=%v", err)
		}
		if err := f.Adapter.Release(ctx, old); !errors.Is(err, actor.ErrLeaseLost) {
			t.Fatalf("stale release=%v", err)
		}
		pending, err := f.Adapter.Pending(ctx, fresh, 0, 10)
		if err != nil || len(pending) != 1 {
			t.Fatalf("successor mailbox=%v %v", pending, err)
		}
	})
	t.Run("ordered copy owned pagination", func(t *testing.T) {
		f := factory(t)
		ctx := t.Context()
		for _, id := range []string{"one", "two", "three"} {
			if _, err := f.Adapter.Submit(ctx, input(id)); err != nil {
				t.Fatal(err)
			}
		}
		lease, err := f.Adapter.Acquire(ctx, "session", "worker", time.Minute)
		if err != nil {
			t.Fatal(err)
		}
		first, err := f.Adapter.Pending(ctx, lease, 0, 1)
		if err != nil || len(first) != 1 || first[0].ID != "one" {
			t.Fatalf("first=%v %v", first, err)
		}
		first[0].Command.Input.Blocks[0].Text = "corrupted copy"
		rest, err := f.Adapter.Pending(ctx, lease, first[0].Sequence, 1)
		if err != nil || len(rest) != 1 || rest[0].ID != "two" {
			t.Fatalf("rest=%v %v", rest, err)
		}
		again, err := f.Adapter.Pending(ctx, lease, 0, 1)
		if err != nil || again[0].Command.Input.Blocks[0].Text != "one" {
			t.Fatalf("alias=%v %v", again, err)
		}
	})
	t.Run("receive cancellation", func(t *testing.T) {
		f := factory(t)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := f.Adapter.Receive(ctx); !errors.Is(err, context.Canceled) {
			t.Fatalf("receive=%v", err)
		}
	})
}
