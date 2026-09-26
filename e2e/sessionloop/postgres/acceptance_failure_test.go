// Append-error recovery crosses the real SQL journal, the shared Worker and
// independent session reopen. The same suite runs through transaction pooling.
package postgres

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
)

type uncertainRepository struct {
	store.Repository
	committed bool
	injected  *atomic.Bool
	cause     error
}

func (r uncertainRepository) Open(ctx context.Context, id string) (store.Journal, error) {
	j, err := r.Repository.Open(ctx, id)
	if err != nil {
		return nil, err
	}
	return &uncertainJournal{Journal: j, fault: r}, nil
}

type uncertainJournal struct {
	store.Journal
	fault uncertainRepository
}

func (j *uncertainJournal) Append(ctx context.Context, cursor store.Cursor, entries ...store.PendingEntry) (store.Commit, error) {
	for _, entry := range entries {
		if entry.Kind != "sessionloop.command.accepted" || !j.fault.injected.CompareAndSwap(false, true) {
			continue
		}
		if j.fault.committed {
			if _, err := j.Journal.Append(ctx, cursor, entries...); err != nil {
				return store.Commit{}, err
			}
		}
		return store.Commit{}, j.fault.cause
	}
	return j.Journal.Append(ctx, cursor, entries...)
}

func TestWorkerRecoversUnknownAcceptanceResult(t *testing.T) {
	for _, committed := range []bool{false, true} {
		name := "before_commit"
		if committed {
			name = "after_commit"
		}
		t.Run(name, func(t *testing.T) {
			f := newFixture(t)
			connection := reconnect(t, f.store)
			observed := observe(connection)
			boom := errors.New("acceptance reply lost")
			var injected atomic.Bool
			errorsSeen := make(chan error, 8)
			opener := actor.SessionOpenerFunc(func(ctx context.Context, lease actor.Lease) (actor.Session, error) {
				repo := uncertainRepository{Repository: connection.Repository(lease), committed: committed, injected: &injected, cause: boom}
				h, err := host(f.config, f.model, repo)
				if err != nil {
					return nil, err
				}
				s, err := h.OpenSession(ctx, sessionloop.SessionID(lease.ActorID))
				if err != nil {
					return nil, err
				}
				return s.(actor.Session), nil
			})
			w, err := actor.NewWorker(actor.Config{Owner: "recovery-test", Adapter: observed, SessionOpener: opener,
				LeaseTTL: time.Second, PollInterval: 5 * time.Millisecond, RetryInterval: 10 * time.Millisecond,
				MaxActors: 1, BatchSize: 1, OnError: func(_ actor.ActorID, err error) { errorsSeen <- err }})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			done := make(chan error, 1)
			go func() { done <- w.Run(ctx) }()
			var once sync.Once
			stop := func() {
				once.Do(func() {
					cancel()
					if err := await(t, done); !errors.Is(err, context.Canceled) {
						t.Errorf("worker stop: %v", err)
					}
				})
			}
			t.Cleanup(stop)
			input := command(f.id, "uncertain", "survive the lost reply")
			submit(t, f.store, input)
			if err := await(t, errorsSeen); !errors.Is(err, boom) || !errors.Is(err, sessionloop.ErrSessionFaulted) {
				t.Fatalf("expected fail-closed acceptance: %v", err)
			}
			// No manual opener/resubmission: the worker must discover and recover.
			receipt := await(t, observed.acks)
			if receipt.CommandID != "uncertain" || receipt.Rejection != "" {
				t.Fatalf("recovered receipt=%+v", receipt)
			}
			await(t, observed.releases)
			stop()
			select {
			case err := <-errorsSeen:
				t.Fatalf("unexpected recovery failure: %v", err)
			default:
			}
			err = f.store.transaction(t.Context(), func(tx pgx.Tx) error {
				var accepted, pending int
				if err := tx.QueryRow(t.Context(), "SELECT count(*) FROM journal WHERE session_id=$1 AND kind='sessionloop.command.accepted'", string(f.id)).Scan(&accepted); err != nil {
					return err
				}
				if err := tx.QueryRow(t.Context(), "SELECT count(*) FROM commands WHERE session_id=$1 AND payload IS NOT NULL", string(f.id)).Scan(&pending); err != nil {
					return err
				}
				if accepted != 1 || pending != 0 {
					t.Errorf("acceptances=%d pending=%d", accepted, pending)
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			// Inspection happens only AFTER autonomous recovery and retirement.
			s := f.inspect(t)
			snapshot, err := s.Snapshot(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			users := 0
			for _, entry := range snapshot.Entries {
				if entry.Role == sessionloop.RoleUser {
					users++
					if entry.CommandID != receipt.CommandID {
						t.Errorf("lost attribution: %+v", entry)
					}
				}
			}
			if users != 1 || snapshot.State != sessionloop.StateIdle {
				t.Fatalf("users=%d state=%s", users, snapshot.State)
			}
			normalized, _ := input.Normalize()
			persisted, found, err := s.Acceptance(t.Context(), normalized.Command)
			if err != nil || !found || persisted != receipt {
				t.Fatalf("receipt changed after reopen: %+v found=%v err=%v", persisted, found, err)
			}
		})
	}
}
