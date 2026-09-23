// Adapter conformance checks the physical SQL boundaries; native worker tests
// separately establish the semantic acceptance-before-mailbox-cleanup mapping.
package postgres

import (
	"bytes"
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor/conformance"
	"github.com/regularkevvv/agentic/harness/store"
)

func seed(t *testing.T, s *Store, id string) {
	t.Helper()
	j, _, err := s.Bootstrap().Create(t.Context(), id, store.PendingEntry{Kind: "fixture", Payload: []byte("original"), Durability: store.DurabilitySync})
	if err != nil {
		t.Fatal(err)
	}
	if err := j.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func expire(t *testing.T, s *Store, lease actor.Lease) {
	t.Helper()
	err := s.transaction(t.Context(), func(tx pgx.Tx) error {
		if _, err := lock(t.Context(), tx, lease.ActorID); err != nil {
			return err
		}
		_, err := tx.Exec(t.Context(), "UPDATE sessions SET expires_at=clock_timestamp()-interval '1 second' WHERE id=$1 AND fence=$2", string(lease.ActorID), int64(lease.Fence))
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
}

// The shared suite's receipts are synthetic witnesses. Replace just those
// witnesses with actual SQL journal references, without changing adapter logic.
type conformanceAdapter struct{ *Store }

func (a conformanceAdapter) Acknowledge(ctx context.Context, lease actor.Lease, id actor.CommandID, r sessionloop.Receipt) error {
	var pos sessionloop.Position
	err := a.transaction(ctx, func(tx pgx.Tx) error {
		row, err := owned(ctx, tx, lease)
		if err != nil {
			return err
		}
		var seq *int64
		if err := tx.QueryRow(ctx, "SELECT accepted_at_seq FROM commands WHERE session_id=$1 AND id=$2", string(lease.ActorID), string(id)).Scan(&seq); err != nil {
			return err
		}
		if seq != nil {
			pos.Sequence = uint64(*seq)
			return tx.QueryRow(ctx, "SELECT entry_id FROM journal WHERE session_id=$1 AND sequence=$2", string(lease.ActorID), *seq).Scan(&pos.Token)
		}
		commit, err := appendEntries(ctx, tx, string(lease.ActorID), row.cursor, []store.PendingEntry{{Kind: "conformance-witness", Payload: []byte(id), Durability: store.DurabilitySync}})
		pos = sessionloop.Position{Sequence: commit.Cursor.Seq, Token: commit.Cursor.EntryID}
		return err
	})
	if err != nil {
		return err
	}
	r.SessionID, r.Position = sessionloop.SessionID(lease.ActorID), pos
	return a.Store.Acknowledge(ctx, lease, id, r)
}

func TestActorConformance(t *testing.T) {
	conformance.Run(t, func(t *testing.T) conformance.Fixture {
		s := database(t)
		seed(t, s, "session")
		return conformance.Fixture{Adapter: conformanceAdapter{s}, Expire: func(l actor.Lease) { expire(t, s, l) }}
	})
}

func TestJournalFencingAtomicityAndBytes(t *testing.T) {
	s := database(t)
	seed(t, s, "session")
	submit(t, s, command("session", "one", "one"))
	lease, err := s.Acquire(t.Context(), "session", "first", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	repo := s.Repository(lease)
	j, err := repo.Open(t.Context(), "session")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := repo.Open(t.Context(), "session"); !errors.Is(err, store.ErrSessionOpen) {
		t.Fatalf("double open: %v", err)
	}
	before, err := j.Load(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	raw := []byte{0, 255, ' ', '{', '\n', '}', 0}
	entry := store.PendingEntry{Kind: "opaque", Payload: raw, Durability: store.DurabilitySync}
	// A bad second entry is discovered after the first INSERT; the whole batch
	// and journal leaf must still roll back.
	if _, err := j.Append(t.Context(), before.Cursor, entry, store.PendingEntry{}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("partial append: %v", err)
	}
	after, err := j.Load(t.Context())
	if err != nil || !after.Cursor.Equal(before.Cursor) {
		t.Fatalf("rollback: %+v %v", after, err)
	}
	commit, err := j.Append(t.Context(), before.Cursor, entry)
	if err != nil {
		t.Fatal(err)
	}
	raw[0] = 9
	loaded, err := j.Load(t.Context())
	if err != nil || !bytes.Equal(loaded.Entries[1].Payload, []byte{0, 255, ' ', '{', '\n', '}', 0}) {
		t.Fatalf("opaque bytes: %+v %v", loaded, err)
	}
	if _, err := j.Append(t.Context(), before.Cursor, entry); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("stale leaf: %v", err)
	}
	bad := sessionloop.Receipt{SessionID: "session", CommandID: "one", Guarantee: sessionloop.AcceptanceDurable, Position: sessionloop.Position{Sequence: commit.Cursor.Seq, Token: "wrong"}}
	if err := s.Acknowledge(t.Context(), lease, "one", bad); !errors.Is(err, actor.ErrInvalidReceipt) {
		t.Fatalf("invented receipt: %v", err)
	}
	expire(t, s, lease)
	if _, err := s.Renew(t.Context(), lease, time.Minute); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("expired renew: %v", err)
	}
	next, err := reconnect(t, s).Acquire(t.Context(), "session", "successor", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	fresh, err := s.Repository(next).Open(t.Context(), "session")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = fresh.Close(context.Background()) })
	if _, err := j.Append(t.Context(), commit.Cursor, entry); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("stale write: %v", err)
	}
	if err := j.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := fresh.Load(t.Context()); err != nil {
		t.Fatalf("old close revoked successor: %v", err)
	}
	if _, err := fresh.Append(t.Context(), commit.Cursor, entry); err != nil {
		t.Fatal(err)
	}
}

func TestConcurrentSubmissionAndOwnership(t *testing.T) {
	s := database(t)
	seed(t, s, "session")
	const clients = 16
	var wg sync.WaitGroup
	results := make(chan actor.Submission, clients)
	for range clients {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := s.Submit(t.Context(), command("session", "same", "same"))
			if err != nil {
				t.Error(err)
			}
			results <- r
		}()
	}
	wg.Wait()
	close(results)
	originals := 0
	for r := range results {
		if !r.Duplicate {
			originals++
		}
		if r.Sequence != 1 {
			t.Errorf("sequence=%d", r.Sequence)
		}
	}
	if originals != 1 {
		t.Fatalf("original submissions=%d", originals)
	}
	winners := make(chan actor.Lease, clients)
	for i := range clients {
		wg.Add(1)
		go func() {
			defer wg.Done()
			l, err := s.Acquire(t.Context(), "session", string(rune('a'+i)), time.Minute)
			if err == nil {
				winners <- l
			} else if !errors.Is(err, actor.ErrLeaseHeld) {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	close(winners)
	if len(winners) != 1 {
		t.Fatalf("owners=%d", len(winners))
	}
}
