// Transaction-boundary regressions cover database-clock expiry after a real
// lock wait and corruption refusing restoration instead of creating new history.
package postgres

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
)

func TestExpiryCheckedAfterLockWait(t *testing.T) {
	s := database(t)
	seed(t, s, "session")
	submit(t, s, command("session", "one", "one"))
	lease, err := s.Acquire(t.Context(), "session", "owner", 250*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	locked := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	held := make(chan error, 1)
	go func() {
		held <- s.transaction(t.Context(), func(tx pgx.Tx) error {
			if _, err := lock(t.Context(), tx, "session"); err != nil {
				return err
			}
			close(locked)
			// Hold the row until the DB itself confirms expiry, not a local clock.
			if _, err := tx.Exec(t.Context(), "SELECT pg_sleep(greatest(0,extract(epoch from ($1::timestamptz-clock_timestamp())))+0.02)", lease.Expires); err != nil {
				return err
			}
			<-release
			return nil
		})
	}()
	await(t, locked)
	renewed := make(chan error, 1)
	go func() { _, err := s.Renew(t.Context(), lease, time.Minute); renewed <- err }()
	// The only other backend waits on the session row, while the holder's
	// pg_sleep expires the lease. Waiting longer than the TTL is intentional.
	timer := time.NewTimer(350 * time.Millisecond)
	defer timer.Stop()
	select {
	case err := <-renewed:
		t.Fatalf("renew did not wait on the row: %v", err)
	case <-timer.C:
	}
	release <- struct{}{}
	if err := await(t, held); err != nil {
		t.Fatal(err)
	}
	if err := await(t, renewed); !errors.Is(err, actor.ErrLeaseLost) {
		t.Fatalf("expired owner renewed after lock wait: %v", err)
	}
}

func TestCorruptJournalCannotBecomeFreshSession(t *testing.T) {
	s := database(t)
	seed(t, s, "session")
	submit(t, s, command("session", "one", "one"))
	lease, err := s.Acquire(t.Context(), "session", "owner", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	j, err := s.Repository(lease).Open(t.Context(), "session")
	if err != nil {
		t.Fatal(err)
	}
	defer j.Close(context.Background())
	err = s.transaction(t.Context(), func(tx pgx.Tx) error {
		if _, err := lock(t.Context(), tx, "session"); err != nil {
			return err
		}
		_, err := tx.Exec(t.Context(), "DELETE FROM journal WHERE session_id='session'")
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := j.Load(t.Context()); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("missing bound history=%v", err)
	}
	if _, err := s.Submit(t.Context(), command("missing", "one", "one")); !errors.Is(err, store.ErrSessionNotFound) {
		t.Fatalf("invented journal binding=%v", err)
	}
}
