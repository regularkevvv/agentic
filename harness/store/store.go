// Package store defines the append-only journal port used by the harness.
//
// It deliberately knows nothing about JSON, files, databases, or in-memory
// maps. Payloads are opaque bytes produced by the configured codec.
package store

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

const CurrentSchema uint16 = 1

// Durability is an acknowledgement requirement, not a storage technology.
type Durability uint8

const (
	// DurabilityDefault allows an adapter to use its ordinary acknowledgement
	// semantics. It is suitable only for facts the application can recreate.
	DurabilityDefault Durability = iota
	// DurabilitySync requires the adapter to make the append crash-durable
	// before returning success.
	DurabilitySync
)

// PendingEntry is an unsequenced domain fact ready to append.
type PendingEntry struct {
	Kind       string
	Payload    []byte
	Durability Durability
}

// Entry is one committed append-only fact. Repositories assign Schema, Seq,
// ID, and ParentID.
type Entry struct {
	Schema     uint16
	Seq        uint64
	ID         string
	ParentID   string
	Kind       string
	Payload    []byte
	Durability Durability
}

// Cursor identifies the exact active journal leaf. Seq alone is useful to
// public subscribers; EntryID prevents a stale writer from appending to a
// different branch or overwritten test double.
type Cursor struct {
	Seq     uint64
	EntryID string
}

func (c Cursor) IsZero() bool { return c.Seq == 0 && c.EntryID == "" }

func (c Cursor) Equal(other Cursor) bool {
	return c.Seq == other.Seq && c.EntryID == other.EntryID
}

func (e Entry) Cursor() Cursor {
	return Cursor{Seq: e.Seq, EntryID: e.ID}
}

type Commit struct {
	Entries []Entry
	Cursor  Cursor
}

type Snapshot struct {
	Entries []Entry
	Cursor  Cursor
}

// Repository creates or opens a single-writer session journal.
//
// At most one owner may commit. An adapter may hold an exclusive handle until
// Close (a competing Open returns ErrSessionOpen), or bind a revocable external
// authority and atomically fence every mutation. Revocation must reject old
// writes even if the old process still holds a Journal. Neither strategy
// requires a long-lived database connection in this interface.
type Repository interface {
	Create(context.Context, string, ...PendingEntry) (Journal, Commit, error)
	Open(context.Context, string) (Journal, error)
}

// Journal is a handle with mutation authority. Append atomically validates that
// authority AND the expected leaf; checking the cursor alone does not fence an
// old owner that happens to know the current cursor.
type Journal interface {
	SessionID() string
	Load(context.Context) (Snapshot, error)
	// Append commits the whole batch or none of it. A returned error does not
	// certify rollback: the commit may be durable even if its reply was lost.
	// Callers must reconstruct durable truth before retrying an uncertain write.
	Append(context.Context, Cursor, ...PendingEntry) (Commit, error)
	// Close releases this handle, never a successor's authority. It is
	// idempotent so cleanup can be retried after cancellation or I/O failure.
	Close(context.Context) error
}

var (
	ErrInvalidSessionID = errors.New("invalid session ID")
	ErrSessionExists    = errors.New("session already exists")
	ErrSessionNotFound  = errors.New("session not found")
	ErrSessionOpen      = errors.New("session already has an open journal")
	ErrJournalClosed    = errors.New("session journal is closed")
	ErrConflict         = errors.New("session journal cursor conflict")
	ErrCorruptLog       = errors.New("corrupt session log")
)

type ConflictError struct {
	Expected Cursor
	Actual   Cursor
}

func (e *ConflictError) Error() string {
	return fmt.Sprintf("%s: expected (%d,%q), actual (%d,%q)",
		ErrConflict, e.Expected.Seq, e.Expected.EntryID, e.Actual.Seq, e.Actual.EntryID)
}

func (e *ConflictError) Is(target error) bool { return target == ErrConflict }

func ValidateSessionID(sessionID string) error {
	if sessionID == "" || len(sessionID) > 128 {
		return fmt.Errorf("%w: %q", ErrInvalidSessionID, sessionID)
	}
	for _, r := range sessionID {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || strings.ContainsRune("_-", r) {
			continue
		}
		return fmt.Errorf("%w: %q", ErrInvalidSessionID, sessionID)
	}
	return nil
}

func ClonePending(entries []PendingEntry) []PendingEntry {
	if len(entries) == 0 {
		return nil
	}
	cloned := make([]PendingEntry, len(entries))
	for i, entry := range entries {
		cloned[i] = entry
		cloned[i].Payload = append([]byte(nil), entry.Payload...)
	}
	return cloned
}

func CloneEntries(entries []Entry) []Entry {
	if len(entries) == 0 {
		return nil
	}
	cloned := make([]Entry, len(entries))
	for i, entry := range entries {
		cloned[i] = entry
		cloned[i].Payload = append([]byte(nil), entry.Payload...)
	}
	return cloned
}

func NewCommit(entries []Entry, fallback Cursor) Commit {
	cloned := CloneEntries(entries)
	cursor := fallback
	if len(cloned) > 0 {
		cursor = cloned[len(cloned)-1].Cursor()
	}
	return Commit{Entries: cloned, Cursor: cursor}
}
