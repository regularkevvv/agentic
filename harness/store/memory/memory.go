// Package memory provides an in-memory session-journal adapter.
package memory

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	"github.com/regularkevvv/agentic/harness/store"
)

type Repository struct {
	mu       sync.RWMutex
	sessions map[string][]store.Entry
	open     map[string]*Journal
}

func New() *Repository {
	return &Repository{
		sessions: make(map[string][]store.Entry),
		open:     make(map[string]*Journal),
	}
}

func (r *Repository) Create(ctx context.Context, sessionID string, pending ...store.PendingEntry) (store.Journal, store.Commit, error) {
	if err := ctx.Err(); err != nil {
		return nil, store.Commit{}, err
	}
	if err := store.ValidateSessionID(sessionID); err != nil {
		return nil, store.Commit{}, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.sessions[sessionID]; exists {
		return nil, store.Commit{}, fmt.Errorf("%w: %s", store.ErrSessionExists, sessionID)
	}
	entries, err := sequence(store.Cursor{}, pending)
	if err != nil {
		return nil, store.Commit{}, err
	}
	journal := &Journal{repository: r, sessionID: sessionID}
	r.sessions[sessionID] = store.CloneEntries(entries)
	r.open[sessionID] = journal
	return journal, store.NewCommit(entries, store.Cursor{}), nil
}

func (r *Repository) Open(ctx context.Context, sessionID string) (store.Journal, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := store.ValidateSessionID(sessionID); err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.sessions[sessionID]; !exists {
		return nil, fmt.Errorf("%w: %s", store.ErrSessionNotFound, sessionID)
	}
	if r.open[sessionID] != nil {
		return nil, fmt.Errorf("%w: %s", store.ErrSessionOpen, sessionID)
	}
	journal := &Journal{repository: r, sessionID: sessionID}
	r.open[sessionID] = journal
	return journal, nil
}

type Journal struct {
	repository *Repository
	sessionID  string
	closed     bool
}

func (j *Journal) SessionID() string { return j.sessionID }

func (j *Journal) Load(ctx context.Context) (store.Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return store.Snapshot{}, err
	}
	j.repository.mu.RLock()
	defer j.repository.mu.RUnlock()
	if err := j.validateOpenLocked(); err != nil {
		return store.Snapshot{}, err
	}
	entries := store.CloneEntries(j.repository.sessions[j.sessionID])
	return store.Snapshot{Entries: entries, Cursor: tail(entries)}, nil
}

func (j *Journal) Append(ctx context.Context, expected store.Cursor, pending ...store.PendingEntry) (store.Commit, error) {
	if err := ctx.Err(); err != nil {
		return store.Commit{}, err
	}
	j.repository.mu.Lock()
	defer j.repository.mu.Unlock()
	if err := j.validateOpenLocked(); err != nil {
		return store.Commit{}, err
	}
	existing := j.repository.sessions[j.sessionID]
	actual := tail(existing)
	if !expected.Equal(actual) {
		return store.Commit{}, &store.ConflictError{Expected: expected, Actual: actual}
	}
	entries, err := sequence(actual, pending)
	if err != nil {
		return store.Commit{}, err
	}
	j.repository.sessions[j.sessionID] = append(existing, store.CloneEntries(entries)...)
	return store.NewCommit(entries, actual), nil
}

func (j *Journal) Close(context.Context) error {
	j.repository.mu.Lock()
	defer j.repository.mu.Unlock()
	if j.closed {
		return nil
	}
	j.closed = true
	if j.repository.open[j.sessionID] == j {
		delete(j.repository.open, j.sessionID)
	}
	return nil
}

func (j *Journal) validateOpenLocked() error {
	if j.closed || j.repository.open[j.sessionID] != j {
		return fmt.Errorf("%w: %s", store.ErrJournalClosed, j.sessionID)
	}
	return nil
}

func tail(entries []store.Entry) store.Cursor {
	if len(entries) == 0 {
		return store.Cursor{}
	}
	return entries[len(entries)-1].Cursor()
}

func sequence(cursor store.Cursor, pending []store.PendingEntry) ([]store.Entry, error) {
	entries := make([]store.Entry, len(pending))
	for i, item := range store.ClonePending(pending) {
		if item.Kind == "" {
			return nil, fmt.Errorf("%w: entry kind is empty", store.ErrCorruptLog)
		}
		if item.Durability > store.DurabilitySync {
			return nil, fmt.Errorf("%w: invalid durability %d", store.ErrCorruptLog, item.Durability)
		}
		id, err := newEntryID()
		if err != nil {
			return nil, err
		}
		cursor.Seq++
		entries[i] = store.Entry{
			Schema:     store.CurrentSchema,
			Seq:        cursor.Seq,
			ID:         id,
			ParentID:   cursor.EntryID,
			Kind:       item.Kind,
			Payload:    append([]byte(nil), item.Payload...),
			Durability: item.Durability,
		}
		cursor.EntryID = id
	}
	return entries, nil
}

// newEntryID returns a UUIDv7-shaped identifier without a module dependency.
func newEntryID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", fmt.Errorf("generate entry ID: %w", err)
	}
	millis := uint64(time.Now().UnixMilli())
	var timestamp [8]byte
	binary.BigEndian.PutUint64(timestamp[:], millis)
	copy(value[0:6], timestamp[2:])
	value[6] = (value[6] & 0x0f) | 0x70
	value[8] = (value[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		value[0:4], value[4:6], value[6:8], value[8:10], value[10:16]), nil
}

var _ store.Repository = (*Repository)(nil)
var _ store.Journal = (*Journal)(nil)
