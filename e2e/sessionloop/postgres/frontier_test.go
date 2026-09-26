// Real committed journal boundaries, lease revocation and autonomous takeover.
// The test never opens a session to rescue it or resubmits an accepted command.
package postgres

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
)

type frontierRepository struct {
	store.Repository
	after func([]store.Entry) error
}

type failRenewal struct {
	actor.Adapter
	fail atomic.Bool
	err  error
}

func (a *failRenewal) Renew(ctx context.Context, l actor.Lease, ttl time.Duration) (actor.Lease, error) {
	if a.fail.CompareAndSwap(true, false) {
		return actor.Lease{}, a.err
	}
	return a.Adapter.Renew(ctx, l, ttl)
}

func TestRenewalIOFailureDoesNotCancelAcceptedSteering(t *testing.T) {
	f := newFixture(t)
	gate := make(chan struct{})
	f.model.gate = gate
	failed := make(chan struct{}, 1)
	boom := errors.New("injected renewal transport failure while grant is still valid")
	f.onWorkerError = func(err error) {
		if !errors.Is(err, boom) {
			t.Errorf("unexpected error: %v", err)
			return
		}
		failed <- struct{}{}
	}
	s := reconnect(t, f.store)
	broken := &failRenewal{Adapter: s, err: boom}
	a := observe(broken)
	submit(t, f.store, command(f.id, "start", "first"))
	stop := f.worker(t, s, a)
	start := await(t, a.acks)
	await(t, f.model.entered)
	steer := command(f.id, "steer", "steering")
	steer.Command.Kind, steer.Command.RunID = sessionloop.CommandSteer, start.RunID
	submit(t, f.store, steer)
	r := await(t, a.acks)
	if r.CommandID != "steer" || r.QueueID == "" {
		t.Fatalf("not accepted: %+v", r)
	}
	broken.fail.Store(true)
	await(t, failed) // the failed owner's cleanup has finished before model release
	close(gate)
	await(t, a.releases)
	stop()
	snap, err := f.inspect(t).Snapshot(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var users []string
	for _, e := range snap.Entries {
		if e.Role == sessionloop.RoleUser {
			users = append(users, e.Blocks[0].Text)
		}
	}
	if !reflect.DeepEqual(users, []string{"first", "steering"}) {
		t.Fatalf("transport error deleted accepted steering: %v", users)
	}
}

func (r frontierRepository) Open(ctx context.Context, id string) (store.Journal, error) {
	j, err := r.Repository.Open(ctx, id)
	if err != nil {
		return nil, err
	}
	return frontierJournal{j, r.after}, nil
}

type frontierJournal struct {
	store.Journal
	after func([]store.Entry) error
}

func (j frontierJournal) Append(ctx context.Context, cursor store.Cursor, entries ...store.PendingEntry) (store.Commit, error) {
	commit, err := j.Journal.Append(ctx, cursor, entries...)
	if err == nil {
		err = j.after(commit.Entries)
	}
	if err != nil {
		return store.Commit{}, err
	}
	return commit, nil
}

func TestAcceptedSteeringSurvivesEveryLeaseLossBoundary(t *testing.T) {
	for _, boundary := range []string{"agentic.assistant_committed", "agentic.output_validated", "agentic.turn_ended", "queue.drained", "agentic.messages_injected", "agentic.run_completed", "run.closed"} {
		t.Run(boundary, func(t *testing.T) {
			f := newFixture(t)
			var injected atomic.Bool
			f.repository = func(lease actor.Lease, repo store.Repository) store.Repository {
				return frontierRepository{repo, func(entries []store.Entry) error {
					for _, entry := range entries {
						if entry.Kind != boundary || !injected.CompareAndSwap(false, true) {
							continue
						}
						err := f.store.transaction(t.Context(), func(tx pgx.Tx) error {
							_, err := tx.Exec(t.Context(), `UPDATE sessions SET expires_at=clock_timestamp()-interval '1 second' WHERE id=$1 AND owner=$2 AND fence=$3`, string(lease.ActorID), lease.Owner, int64(lease.Fence))
							return err
						})
						if err != nil {
							return err
						}
						return actor.ErrLeaseLost // committed bytes; no successful reply
					}
					return nil
				}}
			}
			f.onWorkerError = func(err error) {
				expectedFault := injected.Load() && errors.Is(err, sessionloop.ErrSessionFaulted)
				if !errors.Is(err, actor.ErrLeaseLost) && !expectedFault {
					t.Errorf("unexpected worker error: %v", err)
				}
			}
			gate := make(chan struct{})
			f.model.gate = gate
			submit(t, f.store, command(f.id, "start", "first"))
			a := observe(reconnect(t, f.store))
			stop := f.worker(t, a.Adapter.(*Store), a)
			start := await(t, a.acks)
			await(t, f.model.entered)
			steer := command(f.id, "steer", "steering")
			steer.Command.Kind, steer.Command.RunID = sessionloop.CommandSteer, start.RunID
			submit(t, f.store, steer)
			r := await(t, a.acks)
			if r.CommandID != "steer" || r.QueueID == "" || r.Rejection != "" {
				t.Fatalf("steering not accepted: %+v", r)
			}
			close(gate)
			await(t, a.releases)
			stop()
			if !injected.Load() {
				t.Fatal("crash boundary was not reached")
			}
			// Retirement above, not inspection, proves autonomous recovery.
			s := f.inspect(t)
			snap, err := s.Snapshot(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			var users []string
			for _, e := range snap.Entries {
				if e.Role == sessionloop.RoleUser {
					users = append(users, e.Blocks[0].Text)
				}
			}
			if !reflect.DeepEqual(users, []string{"first", "steering"}) || snap.State != sessionloop.StateIdle {
				t.Fatalf("users=%v state=%s", users, snap.State)
			}
			history, err := s.(interface {
				Replay(context.Context, sessionloop.Position, sessionloop.Position) ([]sessionloop.Event, error)
			}).Replay(t.Context(), sessionloop.Position{}, snap.Position)
			if err != nil {
				t.Fatal(err)
			}
			settled := 0
			for _, e := range history {
				if e.Kind != sessionloop.EventRunSettled {
					continue
				}
				settled++
				if e.RunID != start.RunID || e.Outcome == nil || e.Outcome.Kind != sessionloop.RunCompleted {
					t.Errorf("wrong recovered settlement: %+v", e)
				}
			}
			if settled != 1 {
				t.Errorf("logical run settlements=%d", settled)
			}
			for _, original := range []struct {
				command actor.Command
				receipt sessionloop.Receipt
			}{{command(f.id, "start", "first"), start}, {steer, r}} {
				normalized, err := original.command.Normalize()
				if err != nil {
					t.Fatal(err)
				}
				replayed, found, err := s.Acceptance(t.Context(), normalized.Command)
				if err != nil || !found || replayed != original.receipt {
					t.Fatalf("acceptance identity changed: %+v, %v", replayed, err)
				}
			}
			f.model.mu.Lock()
			defer f.model.mu.Unlock()
			if len(f.model.requests) != 2 {
				t.Fatalf("model requests=%d, expected exactly first reply and steered reply", len(f.model.requests))
			}
			for i, prefix := range f.model.requests[0].messages {
				if !bytes.Equal(prefix, f.model.requests[1].messages[i]) {
					t.Fatalf("recovery changed provider prefix at %d", i)
				}
			}
		})
	}
}
