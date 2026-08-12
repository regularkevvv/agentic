// Package postgres implements the Agentic Harness exclusive journal on
// PostgreSQL. One acquired pgx connection holds a session-scoped advisory lock
// for the lifetime of each open Journal, providing cross-process ownership.
package postgres

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/regularkevvv/agentic/harness/store"
)

const defaultSchema = "agentic"

var schemaPattern = regexp.MustCompile(`^[a-z_][a-z0-9_]{0,62}$`)

type Option func(*Repository) error

func WithSchema(schema string) Option {
	return func(repository *Repository) error {
		if !schemaPattern.MatchString(schema) {
			return fmt.Errorf("agentic postgres: invalid schema %q", schema)
		}
		repository.schema = schema
		return nil
	}
}

type Repository struct {
	pool   *pgxpool.Pool
	schema string
}

func Open(ctx context.Context, dsn string, options ...Option) (*Repository, error) {
	if dsn == "" {
		return nil, errors.New("agentic postgres: DSN is required")
	}
	repository := &Repository{schema: defaultSchema}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("agentic postgres: option must not be nil")
		}
		if err := option(repository); err != nil {
			return nil, err
		}
	}
	config, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("agentic postgres: parse DSN: %w", err)
	}
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("agentic postgres: open pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("agentic postgres: ping: %w", err)
	}
	repository.pool = pool
	return repository, nil
}

func (r *Repository) Close() {
	if r != nil && r.pool != nil {
		r.pool.Close()
	}
}

func (r *Repository) Migrate(ctx context.Context) error {
	ddl := fmt.Sprintf(`
CREATE SCHEMA IF NOT EXISTS %s;
CREATE TABLE IF NOT EXISTS %s.sessions (
    session_id text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS %s.entries (
    session_id text NOT NULL REFERENCES %s.sessions(session_id) ON DELETE CASCADE,
    seq bigint NOT NULL CHECK (seq > 0),
    id text NOT NULL,
    parent_id text NOT NULL,
    kind text NOT NULL CHECK (kind <> ''),
    payload bytea NOT NULL,
    durability smallint NOT NULL CHECK (durability BETWEEN 0 AND 1),
    schema_version smallint NOT NULL,
    PRIMARY KEY (session_id, seq),
    UNIQUE (session_id, id)
);`, r.schema, r.schema, r.schema, r.schema)
	if _, err := r.pool.Exec(ctx, ddl); err != nil {
		return fmt.Errorf("agentic postgres: migrate: %w", err)
	}
	return nil
}

func (r *Repository) Create(ctx context.Context, sessionID string, pending ...store.PendingEntry) (store.Journal, store.Commit, error) {
	if err := ctx.Err(); err != nil {
		return nil, store.Commit{}, err
	}
	if err := store.ValidateSessionID(sessionID); err != nil {
		return nil, store.Commit{}, err
	}
	entries, err := sequence(store.Cursor{}, pending)
	if err != nil {
		return nil, store.Commit{}, err
	}
	connection, err := r.acquire(ctx, sessionID)
	if err != nil {
		return nil, store.Commit{}, err
	}
	journal := &Journal{repository: r, connection: connection, sessionID: sessionID}
	tx, err := connection.Begin(ctx)
	if err != nil {
		_ = journal.Close(context.Background())
		return nil, store.Commit{}, err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	if _, err := tx.Exec(ctx, fmt.Sprintf(`INSERT INTO %s.sessions (session_id) VALUES ($1)`, r.schema), sessionID); err != nil {
		_ = tx.Rollback(context.Background())
		_ = journal.Close(context.Background())
		if isUniqueViolation(err) {
			return nil, store.Commit{}, fmt.Errorf("%w: %s", store.ErrSessionExists, sessionID)
		}
		return nil, store.Commit{}, fmt.Errorf("agentic postgres: create session: %w", err)
	}
	if err := insertEntries(ctx, tx, r.schema, sessionID, entries); err != nil {
		_ = tx.Rollback(context.Background())
		_ = journal.Close(context.Background())
		return nil, store.Commit{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		_ = tx.Rollback(context.Background())
		_ = journal.Close(context.Background())
		return nil, store.Commit{}, fmt.Errorf("agentic postgres: commit create: %w", err)
	}
	return journal, store.NewCommit(entries, store.Cursor{}), nil
}

func (r *Repository) Open(ctx context.Context, sessionID string) (store.Journal, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := store.ValidateSessionID(sessionID); err != nil {
		return nil, err
	}
	connection, err := r.acquire(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	journal := &Journal{repository: r, connection: connection, sessionID: sessionID}
	var exists bool
	if err := connection.QueryRow(ctx, fmt.Sprintf(`SELECT EXISTS(SELECT 1 FROM %s.sessions WHERE session_id=$1)`, r.schema), sessionID).Scan(&exists); err != nil {
		_ = journal.Close(context.Background())
		return nil, fmt.Errorf("agentic postgres: inspect session: %w", err)
	}
	if !exists {
		_ = journal.Close(context.Background())
		return nil, fmt.Errorf("%w: %s", store.ErrSessionNotFound, sessionID)
	}
	if _, err := journal.Load(ctx); err != nil {
		_ = journal.Close(context.Background())
		return nil, err
	}
	return journal, nil
}

func (r *Repository) acquire(ctx context.Context, sessionID string) (*pgxpool.Conn, error) {
	connection, err := r.pool.Acquire(ctx)
	if err != nil {
		return nil, fmt.Errorf("agentic postgres: acquire connection: %w", err)
	}
	var locked bool
	query := `SELECT pg_try_advisory_lock(hashtextextended($1, 0))`
	if err := connection.QueryRow(ctx, query, r.schema+":"+sessionID).Scan(&locked); err != nil {
		connection.Release()
		return nil, fmt.Errorf("agentic postgres: acquire session lock: %w", err)
	}
	if !locked {
		connection.Release()
		return nil, fmt.Errorf("%w: %s", store.ErrSessionOpen, sessionID)
	}
	return connection, nil
}

type Journal struct {
	repository *Repository
	connection *pgxpool.Conn
	sessionID  string
	mu         sync.Mutex
	closed     bool
}

func (j *Journal) SessionID() string { return j.sessionID }

func (j *Journal) Load(ctx context.Context) (store.Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return store.Snapshot{}, err
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed {
		return store.Snapshot{}, fmt.Errorf("%w: %s", store.ErrJournalClosed, j.sessionID)
	}
	entries, err := loadEntries(ctx, j.connection, j.repository.schema, j.sessionID)
	if err != nil {
		return store.Snapshot{}, err
	}
	return store.Snapshot{Entries: entries, Cursor: tail(entries)}, nil
}

func (j *Journal) Append(ctx context.Context, expected store.Cursor, pending ...store.PendingEntry) (store.Commit, error) {
	if err := ctx.Err(); err != nil {
		return store.Commit{}, err
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed {
		return store.Commit{}, fmt.Errorf("%w: %s", store.ErrJournalClosed, j.sessionID)
	}
	tx, err := j.connection.Begin(ctx)
	if err != nil {
		return store.Commit{}, err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	actual, err := loadTail(ctx, tx, j.repository.schema, j.sessionID)
	if err != nil {
		return store.Commit{}, err
	}
	if !expected.Equal(actual) {
		return store.Commit{}, &store.ConflictError{Expected: expected, Actual: actual}
	}
	entries, err := sequence(actual, pending)
	if err != nil {
		return store.Commit{}, err
	}
	if err := insertEntries(ctx, tx, j.repository.schema, j.sessionID, entries); err != nil {
		return store.Commit{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return store.Commit{}, fmt.Errorf("agentic postgres: commit append: %w", err)
	}
	return store.NewCommit(entries, actual), nil
}

func (j *Journal) Close(context.Context) error {
	j.mu.Lock()
	if j.closed {
		j.mu.Unlock()
		return nil
	}
	j.closed = true
	connection := j.connection
	j.connection = nil
	j.mu.Unlock()
	if connection == nil {
		return nil
	}
	cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var unlocked bool
	err := connection.QueryRow(cleanupCtx, `SELECT pg_advisory_unlock(hashtextextended($1, 0))`, j.repository.schema+":"+j.sessionID).Scan(&unlocked)
	if err != nil {
		underlying := connection.Hijack()
		_ = underlying.Close(cleanupCtx)
	} else {
		connection.Release()
	}
	if err != nil {
		return fmt.Errorf("agentic postgres: release session lock: %w", err)
	}
	if !unlocked {
		return fmt.Errorf("agentic postgres: session lock was not held: %w", store.ErrJournalClosed)
	}
	return nil
}

type queryer interface {
	Query(context.Context, string, ...any) (pgx.Rows, error)
}

func loadEntries(ctx context.Context, queryer queryer, schema, sessionID string) ([]store.Entry, error) {
	rows, err := queryer.Query(ctx, fmt.Sprintf(`SELECT schema_version, seq, id, parent_id, kind, payload, durability FROM %s.entries WHERE session_id=$1 ORDER BY seq`, schema), sessionID)
	if err != nil {
		return nil, fmt.Errorf("agentic postgres: load entries: %w", err)
	}
	defer rows.Close()
	var entries []store.Entry
	parent := ""
	for rows.Next() {
		var entry store.Entry
		var durability int16
		if err := rows.Scan(&entry.Schema, &entry.Seq, &entry.ID, &entry.ParentID, &entry.Kind, &entry.Payload, &durability); err != nil {
			return nil, fmt.Errorf("agentic postgres: scan entry: %w", err)
		}
		entry.Durability = store.Durability(durability)
		if entry.Schema != store.CurrentSchema || entry.Seq != uint64(len(entries)+1) || entry.ID == "" ||
			entry.Kind == "" || entry.ParentID != parent || entry.Durability > store.DurabilitySync {
			return nil, fmt.Errorf("%w: invalid chain at sequence %d", store.ErrCorruptLog, entry.Seq)
		}
		entries = append(entries, entry)
		parent = entry.ID
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("agentic postgres: iterate entries: %w", err)
	}
	return store.CloneEntries(entries), nil
}

type rowQueryer interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}

func loadTail(ctx context.Context, queryer rowQueryer, schema, sessionID string) (store.Cursor, error) {
	var cursor store.Cursor
	err := queryer.QueryRow(ctx, fmt.Sprintf(`SELECT seq, id FROM %s.entries WHERE session_id=$1 ORDER BY seq DESC LIMIT 1 FOR UPDATE`, schema), sessionID).Scan(&cursor.Seq, &cursor.EntryID)
	if errors.Is(err, pgx.ErrNoRows) {
		return store.Cursor{}, nil
	}
	if err != nil {
		return store.Cursor{}, fmt.Errorf("agentic postgres: load tail: %w", err)
	}
	return cursor, nil
}

type execer interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
}

func insertEntries(ctx context.Context, execer execer, schema, sessionID string, entries []store.Entry) error {
	query := fmt.Sprintf(`INSERT INTO %s.entries (session_id, seq, id, parent_id, kind, payload, durability, schema_version) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`, schema)
	for _, entry := range entries {
		payload := entry.Payload
		if payload == nil {
			payload = []byte{}
		}
		if _, err := execer.Exec(ctx, query, sessionID, entry.Seq, entry.ID, entry.ParentID, entry.Kind, payload, entry.Durability, entry.Schema); err != nil {
			return fmt.Errorf("agentic postgres: insert sequence %d: %w", entry.Seq, err)
		}
	}
	return nil
}

func sequence(parent store.Cursor, pending []store.PendingEntry) ([]store.Entry, error) {
	entries := make([]store.Entry, len(pending))
	baseSequence := parent.Seq
	for index, value := range pending {
		if value.Kind == "" || value.Durability > store.DurabilitySync {
			return nil, fmt.Errorf("%w: invalid pending entry %d", store.ErrCorruptLog, index)
		}
		id := newID()
		entries[index] = store.Entry{Schema: store.CurrentSchema, Seq: baseSequence + uint64(index) + 1,
			ID: id, ParentID: parent.EntryID, Kind: value.Kind, Payload: append([]byte(nil), value.Payload...), Durability: value.Durability}
		parent = entries[index].Cursor()
	}
	return entries, nil
}

func newID() string {
	value := make([]byte, 16)
	_, _ = rand.Read(value) // crypto/rand.Read is fail-stop on modern Go.
	return hex.EncodeToString(value)
}

func tail(entries []store.Entry) store.Cursor {
	if len(entries) == 0 {
		return store.Cursor{}
	}
	return entries[len(entries)-1].Cursor()
}

func isUniqueViolation(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "23505"
}

var _ store.Repository = (*Repository)(nil)
var _ store.Journal = (*Journal)(nil)
