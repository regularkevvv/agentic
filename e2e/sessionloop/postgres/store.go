// Package postgres is an executable e2e reference adapter, not a shipped
// Harness dependency. Mailbox, ownership and journal share short row-locked
// transactions. No connection survives a storage call or enters model execution.
package postgres

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
)

//go:embed schema.sql
var schemaSQL string

// Store owns only its pool; the caller owns schema provisioning and retention.
// Sharing the same DSN/schema across processes shares all execution state.
type Store struct {
	pool   *pgxpool.Pool
	schema string
}

// Connect uses uncached extended queries, including through transaction pools.
// DSN credentials must stay outside source code and logs.
func Connect(ctx context.Context, dsn, schema string) (*Store, error) {
	if schema == "" {
		return nil, errors.New("postgres flavor: schema required")
	}
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, err
	}
	cfg.ConnConfig.DefaultQueryExecMode = pgx.QueryExecModeExec
	cfg.MaxConns = 16
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return &Store{pool: pool, schema: schema}, nil
}

func (s *Store) Close() { s.pool.Close() }

// Migrate requires a fresh, caller-owned schema. It never alters existing data.
func (s *Store) Migrate(ctx context.Context) error {
	return s.transaction(ctx, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, "CREATE SCHEMA "+pgx.Identifier{s.schema}.Sanitize()); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, schemaSQL, pgx.QueryExecModeSimpleProtocol)
		return err
	})
}

func (*Store) Guarantee() sessionloop.AcceptanceGuarantee { return sessionloop.AcceptanceDurable }

// transaction is the only connection-owning boundary. SET LOCAL cannot leak to
// the next PgBouncer client. Failed/ambiguous COMMIT is returned, never retried.
func (s *Store) transaction(ctx context.Context, fn func(pgx.Tx) error) error {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.ReadCommitted})
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	if _, err := tx.Exec(ctx, "SET LOCAL search_path TO "+pgx.Identifier{s.schema}.Sanitize()+", pg_catalog"); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, "SET LOCAL synchronous_commit TO on"); err != nil {
		return err
	}
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

type sessionRow struct {
	owner      string
	fence      int64
	expires    *time.Time
	needs      bool
	commandSeq int64
	cursor     store.Cursor
	handle     string
}

func lock(ctx context.Context, tx pgx.Tx, id actor.ActorID) (sessionRow, error) {
	var r sessionRow
	var seq int64
	err := tx.QueryRow(ctx, `SELECT coalesce(owner,''), fence, expires_at, needs_execution,
		next_command_seq, journal_seq, journal_entry_id, coalesce(journal_handle,'')
		FROM sessions WHERE id=$1 FOR UPDATE`, string(id)).Scan(
		&r.owner, &r.fence, &r.expires, &r.needs, &r.commandSeq, &seq, &r.cursor.EntryID, &r.handle)
	if errors.Is(err, pgx.ErrNoRows) {
		return r, store.ErrSessionNotFound
	}
	r.cursor.Seq = uint64(seq)
	return r, err
}

// Database time is sampled AFTER the row lock, never before a lock wait.
func databaseTime(ctx context.Context, tx pgx.Tx) (time.Time, error) {
	var now time.Time
	err := tx.QueryRow(ctx, "SELECT clock_timestamp()").Scan(&now)
	return now, err
}

func owned(ctx context.Context, tx pgx.Tx, lease actor.Lease) (sessionRow, error) {
	r, err := lock(ctx, tx, lease.ActorID)
	if errors.Is(err, store.ErrSessionNotFound) {
		return r, actor.ErrLeaseLost
	}
	if err != nil {
		return r, err
	}
	now, err := databaseTime(ctx, tx)
	if err != nil {
		return r, err
	}
	if lease.Owner == "" || r.owner != lease.Owner || actor.Fence(r.fence) != lease.Fence ||
		r.expires == nil || !r.expires.After(now) {
		return r, actor.ErrLeaseLost
	}
	return r, nil
}

func validTTL(ttl time.Duration) error {
	if ttl < time.Microsecond {
		return fmt.Errorf("postgres flavor: TTL must be at least one microsecond")
	}
	return nil
}

var _ actor.Adapter = (*Store)(nil)
