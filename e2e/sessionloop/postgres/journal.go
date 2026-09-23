// Journal stores codec-owned bytes and an expected leaf. The worker's lease and
// the journal handle are checked inside the SAME transaction as every append.

package postgres

import (
	"context"
	"errors"
	"math"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
)

type repository struct {
	store     *Store
	lease     actor.Lease
	bootstrap bool
}

// Repository binds restoration and all subsequent journal calls to this grant.
// It performs fencing internally; an unrelated WithAuthority wrapper is neither
// needed nor sufficient. Only the independently scheduled worker renews/releases.
func (s *Store) Repository(lease actor.Lease) store.Repository {
	return &repository{store: s, lease: lease}
}

// Bootstrap creates the initial native journal before publishing its identity.
// Its returned handle permits Load/Close, but not execution/Append or reopening.
// Close retires the known-idle session. A creator crash leaves needs_execution
// set, so a worker can restore the committed initial journal after lease expiry.
func (s *Store) Bootstrap() store.Repository { return &repository{store: s, bootstrap: true} }

func (r *repository) Create(ctx context.Context, id string, entries ...store.PendingEntry) (store.Journal, store.Commit, error) {
	if err := store.ValidateSessionID(id); err != nil {
		return nil, store.Commit{}, err
	}
	if !r.bootstrap {
		return nil, store.Commit{}, errors.New("postgres flavor: use Bootstrap to create a session")
	}
	j := &journal{store: r.store, handle: uuid.NewString(), bootstrap: true}
	var commit store.Commit
	err := r.store.transaction(ctx, func(tx pgx.Tx) error {
		now, err := databaseTime(ctx, tx)
		if err != nil {
			return err
		}
		j.lease = actor.Lease{ActorID: actor.ActorID(id), Owner: uuid.NewString(), Fence: 1, Expires: now.Add(30 * time.Second)}
		if _, err := tx.Exec(ctx, `INSERT INTO sessions(id,owner,fence,expires_at,needs_execution,journal_handle)
			VALUES($1,$2,1,$3,true,$4)`, id, j.lease.Owner, j.lease.Expires, j.handle); err != nil {
			var pgerr *pgconn.PgError
			if errors.As(err, &pgerr) && pgerr.Code == "23505" {
				return store.ErrSessionExists
			}
			return err
		}
		commit, err = appendEntries(ctx, tx, id, store.Cursor{}, entries)
		return err
	})
	if err != nil {
		return nil, store.Commit{}, err
	}
	return j, commit, nil
}

func (r *repository) Open(ctx context.Context, id string) (store.Journal, error) {
	if err := store.ValidateSessionID(id); err != nil {
		return nil, err
	}
	if r.bootstrap || string(r.lease.ActorID) != id {
		return nil, actor.ErrLeaseLost
	}
	j := &journal{store: r.store, lease: r.lease, handle: uuid.NewString()}
	err := r.store.transaction(ctx, func(tx pgx.Tx) error {
		row, err := owned(ctx, tx, r.lease)
		if err != nil {
			return err
		}
		if row.handle != "" {
			return store.ErrSessionOpen
		}
		_, err = tx.Exec(ctx, "UPDATE sessions SET journal_handle=$2 WHERE id=$1", id, j.handle)
		return err
	})
	if err != nil {
		return nil, err
	}
	return j, nil
}

type journal struct {
	store     *Store
	lease     actor.Lease
	handle    string
	bootstrap bool
	mu        sync.Mutex
	closed    bool
}

func (j *journal) SessionID() string { return string(j.lease.ActorID) }

func (j *journal) owned(ctx context.Context, tx pgx.Tx) (sessionRow, error) {
	if j.closed {
		return sessionRow{}, store.ErrJournalClosed
	}
	r, err := owned(ctx, tx, j.lease)
	if err != nil {
		return r, err
	}
	if r.handle != j.handle {
		return r, store.ErrJournalClosed
	}
	return r, nil
}

func (j *journal) Load(ctx context.Context) (store.Snapshot, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	var result store.Snapshot
	err := j.store.transaction(ctx, func(tx pgx.Tx) error {
		r, err := j.owned(ctx, tx)
		if err != nil {
			return err
		}
		rows, err := tx.Query(ctx, `SELECT schema_version,sequence,entry_id,parent_id,kind,payload,durability
			FROM journal WHERE session_id=$1 ORDER BY sequence`, j.SessionID())
		if err != nil {
			return err
		}
		defer rows.Close()
		var cursor store.Cursor
		for rows.Next() {
			var entry store.Entry
			var seq int64
			var schema, durability int16
			if err := rows.Scan(&schema, &seq, &entry.ID, &entry.ParentID, &entry.Kind, &entry.Payload, &durability); err != nil {
				return err
			}
			if schema != int16(store.CurrentSchema) || seq != int64(cursor.Seq)+1 || entry.ParentID != cursor.EntryID ||
				entry.ID == "" || entry.Kind == "" || durability < 0 || durability > int16(store.DurabilitySync) {
				return store.ErrCorruptLog
			}
			entry.Schema, entry.Seq, entry.Durability = uint16(schema), uint64(seq), store.Durability(durability)
			cursor = entry.Cursor()
			result.Entries = append(result.Entries, entry)
		}
		if err := rows.Err(); err != nil {
			return err
		}
		if !cursor.Equal(r.cursor) {
			return store.ErrCorruptLog
		}
		result.Cursor = cursor
		return nil
	})
	if err != nil {
		return store.Snapshot{}, err
	}
	return result, nil
}

func (j *journal) Append(ctx context.Context, expected store.Cursor, pending ...store.PendingEntry) (store.Commit, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	var result store.Commit
	err := j.store.transaction(ctx, func(tx pgx.Tx) error {
		r, err := j.owned(ctx, tx)
		if err != nil {
			return err
		}
		if j.bootstrap {
			return errors.New("postgres flavor: bootstrap handle cannot execute; submit to the mailbox")
		}
		if !r.cursor.Equal(expected) {
			return &store.ConflictError{Expected: expected, Actual: r.cursor}
		}
		result, err = appendEntries(ctx, tx, j.SessionID(), r.cursor, pending)
		return err
	})
	if err != nil {
		return store.Commit{}, err
	}
	return result, nil
}

func appendEntries(ctx context.Context, tx pgx.Tx, id string, cursor store.Cursor, pending []store.PendingEntry) (store.Commit, error) {
	initial := cursor
	entries := make([]store.Entry, 0, len(pending))
	for _, item := range pending {
		if item.Kind == "" || item.Durability > store.DurabilitySync {
			return store.Commit{}, store.ErrCorruptLog
		}
		if cursor.Seq >= math.MaxInt64 {
			return store.Commit{}, actor.ErrGenerationExhausted
		}
		entry := store.Entry{Schema: store.CurrentSchema, Seq: cursor.Seq + 1, ID: uuid.NewString(),
			ParentID: cursor.EntryID, Kind: item.Kind, Payload: append([]byte{}, item.Payload...), Durability: item.Durability}
		if _, err := tx.Exec(ctx, `INSERT INTO journal(session_id,schema_version,sequence,entry_id,parent_id,kind,payload,durability)
			VALUES($1,$2,$3,$4,$5,$6,$7,$8)`, id, int16(entry.Schema), int64(entry.Seq), entry.ID, entry.ParentID, entry.Kind, entry.Payload, int16(entry.Durability)); err != nil {
			return store.Commit{}, err
		}
		entries = append(entries, entry)
		cursor = entry.Cursor()
	}
	if _, err := tx.Exec(ctx, "UPDATE sessions SET journal_seq=$2,journal_entry_id=$3 WHERE id=$1", id, int64(cursor.Seq), cursor.EntryID); err != nil {
		return store.Commit{}, err
	}
	return store.NewCommit(entries, initial), nil
}

// Close is handle cleanup, not worker retirement. Matching fence AND handle
// prevents an old Close from releasing a successor, even after expiry/takeover.
func (j *journal) Close(ctx context.Context) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed {
		return nil
	}
	err := j.store.transaction(ctx, func(tx pgx.Tx) error {
		r, err := lock(ctx, tx, j.lease.ActorID)
		if err != nil {
			return err
		}
		if r.owner != j.lease.Owner || actor.Fence(r.fence) != j.lease.Fence || r.handle != j.handle {
			return nil
		}
		query := "UPDATE sessions SET journal_handle=NULL WHERE id=$1"
		if j.bootstrap {
			// No execution writes are possible through this creation-only handle.
			query = "UPDATE sessions SET journal_handle=NULL,owner=NULL,expires_at=NULL,needs_execution=false WHERE id=$1"
		}
		_, err = tx.Exec(ctx, query, j.SessionID())
		return err
	})
	if err == nil {
		j.closed = true
	}
	return err
}

var _ store.Repository = (*repository)(nil)
var _ store.Journal = (*journal)(nil)
