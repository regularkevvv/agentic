// Authority tests use the actual memory journal, not a mocked append result.
package store_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	actormemory "github.com/regularkevvv/agentic/harness/sessionloop/actor/memory"
	"github.com/regularkevvv/agentic/harness/store"
	"github.com/regularkevvv/agentic/harness/store/memory"
)

func TestJournalAuthorityFencesEveryMutationAndAllowsCleanup(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	mailbox := actormemory.New()
	_, err := mailbox.Submit(ctx, actor.Command{ActorID: "a", ID: "one", Command: sessionloop.Command{Kind: sessionloop.CommandStart, Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "hello"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := mailbox.Acquire(ctx, "a", "old", 30*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	base := memory.New()
	bound := store.WithAuthority(base, mailbox.Authority(lease))
	j, commit, err := bound.Create(ctx, "journal", store.PendingEntry{Kind: "created"})
	if err != nil {
		t.Fatal(err)
	}
	if j.SessionID() != "journal" {
		t.Fatal(j.SessionID())
	}
	if _, err := j.Load(ctx); err != nil {
		t.Fatal(err)
	}
	commit, err = j.Append(ctx, commit.Cursor, store.PendingEntry{Kind: "accepted"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := mailbox.Receive(ctx); err != nil {
		t.Fatal(err)
	} // timer, no extra message
	fresh, err := mailbox.Acquire(ctx, "a", "new", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := j.Append(ctx, commit.Cursor, store.PendingEntry{Kind: "stale"}); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("old append=%v", err)
	}
	if _, err := j.Load(ctx); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("old load=%v", err)
	}
	if _, _, err := bound.Create(ctx, "another"); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("old create=%v", err)
	}
	if _, err := bound.Open(ctx, "journal"); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("old open=%v", err)
	}
	if err := j.Close(ctx); err != nil {
		t.Fatal(err)
	}
	next, err := store.WithAuthority(base, mailbox.Authority(fresh)).Open(ctx, "journal")
	if err != nil {
		t.Fatal(err)
	}
	defer next.Close(ctx)
	if _, err := next.Append(ctx, commit.Cursor, store.PendingEntry{Kind: "successor"}); err != nil {
		t.Fatal(err)
	}
	if err := j.Close(ctx); err != nil {
		t.Fatal(err)
	} // old cleanup cannot close successor
	state, err := next.Load(ctx)
	if err != nil || len(state.Entries) != 3 {
		t.Fatalf("history=%+v %v", state, err)
	}
}

type authorityFunc func(context.Context, func(context.Context) error) error

func (f authorityFunc) Commit(ctx context.Context, fn func(context.Context) error) error {
	return f(ctx, fn)
}

func TestAuthorityPropagatesTransactionAndAmbiguousFailures(t *testing.T) {
	type txKey struct{}
	boom := errors.New("commit reply lost")
	base := memory.New()
	var lose bool
	a := authorityFunc(func(ctx context.Context, fn func(context.Context) error) error {
		err := fn(context.WithValue(ctx, txKey{}, true))
		if err == nil && lose {
			return boom
		}
		return err
	})
	r := store.WithAuthority(base, a)
	lose = true
	if _, _, err := r.Create(t.Context(), "one", store.PendingEntry{Kind: "created"}); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	// The underlying create committed; failed wrapper construction closed its
	// handle rather than leaking it. The caller must recover, not create again.
	lose = false
	j, err := r.Open(t.Context(), "one")
	if err != nil {
		t.Fatal(err)
	}
	state, err := j.Load(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	lose = true
	if _, err := j.Append(t.Context(), state.Cursor, store.PendingEntry{Kind: "accepted"}); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	if _, err := j.Load(t.Context()); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	if err := j.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Open(t.Context(), "one"); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	lose = false
	j, err = r.Open(t.Context(), "one")
	if err != nil {
		t.Fatal(err)
	}
	defer j.Close(t.Context())
	state, err = j.Load(t.Context())
	if err != nil || len(state.Entries) != 2 {
		t.Fatalf("durable ambiguous commit=%+v %v", state, err)
	}
	if _, _, err := r.Create(t.Context(), "one"); !errors.Is(err, store.ErrSessionExists) {
		t.Fatal(err)
	}
	if _, err := r.Open(t.Context(), "missing"); !errors.Is(err, store.ErrSessionNotFound) {
		t.Fatal(err)
	}
	for _, f := range []func(){func() { store.WithAuthority(nil, a) }, func() { store.WithAuthority(base, nil) }} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("nil dependency accepted")
				}
			}()
			f()
		}()
	}
}
