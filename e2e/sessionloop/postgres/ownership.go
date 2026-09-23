// Ownership uses the session row as the sole serialization point. Receive is
// ordinary worker polling over retained state, including unfinished journals.

package postgres

import (
	"context"
	"errors"
	"math"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
)

func (s *Store) Receive(ctx context.Context) (actor.ActorID, error) {
	for {
		var id actor.ActorID
		err := s.transaction(ctx, func(tx pgx.Tx) error {
			return tx.QueryRow(ctx, `SELECT id FROM sessions s
				WHERE (owner IS NULL OR expires_at <= clock_timestamp())
				AND (needs_execution OR EXISTS (SELECT 1 FROM commands c
					WHERE c.session_id=s.id AND c.accepted_at_seq IS NULL))
				ORDER BY last_claimed_at NULLS FIRST, id LIMIT 1`).Scan(&id)
		})
		if !errors.Is(err, pgx.ErrNoRows) {
			return id, err
		}
		timer := time.NewTimer(10 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return "", ctx.Err()
		case <-timer.C:
		}
	}
}

func (s *Store) Acquire(ctx context.Context, id actor.ActorID, owner string, ttl time.Duration) (actor.Lease, error) {
	if id == "" || owner == "" {
		return actor.Lease{}, errors.New("postgres flavor: actor and owner required")
	}
	if err := validTTL(ttl); err != nil {
		return actor.Lease{}, err
	}
	var lease actor.Lease
	err := s.transaction(ctx, func(tx pgx.Tx) error {
		r, err := lock(ctx, tx, id)
		if errors.Is(err, store.ErrSessionNotFound) {
			return actor.ErrNoWork
		}
		if err != nil {
			return err
		}
		now, err := databaseTime(ctx, tx)
		if err != nil {
			return err
		}
		if r.owner != "" && r.expires.After(now) {
			return actor.ErrLeaseHeld
		}
		var pending bool
		if err := tx.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM commands WHERE session_id=$1 AND accepted_at_seq IS NULL)", string(id)).Scan(&pending); err != nil {
			return err
		}
		if !r.needs && !pending {
			return actor.ErrNoWork
		}
		if r.fence == math.MaxInt64 {
			return actor.ErrGenerationExhausted
		}
		lease = actor.Lease{ActorID: id, Owner: owner, Fence: actor.Fence(r.fence + 1), Expires: now.Add(ttl)}
		_, err = tx.Exec(ctx, `UPDATE sessions SET owner=$2, fence=$3, expires_at=$4,
			needs_execution=true, journal_handle=NULL, last_claimed_at=$5 WHERE id=$1`, string(id), owner, int64(lease.Fence), lease.Expires, now)
		return err
	})
	if err != nil {
		return actor.Lease{}, err
	}
	return lease, nil
}

func (s *Store) Renew(ctx context.Context, lease actor.Lease, ttl time.Duration) (actor.Lease, error) {
	if err := validTTL(ttl); err != nil {
		return actor.Lease{}, err
	}
	err := s.transaction(ctx, func(tx pgx.Tx) error {
		if _, err := owned(ctx, tx, lease); err != nil {
			return err
		}
		now, err := databaseTime(ctx, tx)
		if err != nil {
			return err
		}
		lease.Expires = now.Add(ttl)
		_, err = tx.Exec(ctx, "UPDATE sessions SET expires_at=$2 WHERE id=$1", string(lease.ActorID), lease.Expires)
		return err
	})
	if err != nil {
		return actor.Lease{}, err
	}
	return lease, nil
}

func (s *Store) Release(ctx context.Context, lease actor.Lease) error {
	return s.transaction(ctx, func(tx pgx.Tx) error {
		r, err := owned(ctx, tx, lease)
		if err != nil {
			return err
		}
		if r.handle != "" {
			return store.ErrSessionOpen
		}
		// The shared worker establishes quiescence and closes its journal first.
		// Input arriving before OR after this transaction remains discoverable.
		_, err = tx.Exec(ctx, "UPDATE sessions SET owner=NULL, expires_at=NULL, needs_execution=false WHERE id=$1", string(lease.ActorID))
		return err
	})
}
