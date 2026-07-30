// Package jsonl provides a filesystem-backed JSON-lines journal adapter.
//
// JSON is an envelope choice of this adapter. Domain payloads remain opaque
// bytes and are base64-encoded by encoding/json.
package jsonl

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/regularkevvv/agentic/harness/store"
)

type Repository struct {
	root string
	mu   sync.Mutex
	open map[string]*Journal
}

func New(root string) (*Repository, error) {
	if root == "" {
		return nil, errors.New("session journal root is required")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("canonicalize session journal root: %w", err)
	}
	if err := os.MkdirAll(abs, 0o700); err != nil {
		return nil, fmt.Errorf("create session journal root: %w", err)
	}
	return &Repository{root: filepath.Clean(abs), open: make(map[string]*Journal)}, nil
}

func (r *Repository) Root() string { return r.root }

func (r *Repository) Create(ctx context.Context, sessionID string, pending ...store.PendingEntry) (store.Journal, store.Commit, error) {
	if err := ctx.Err(); err != nil {
		return nil, store.Commit{}, err
	}
	if err := store.ValidateSessionID(sessionID); err != nil {
		return nil, store.Commit{}, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.open[sessionID] != nil {
		return nil, store.Commit{}, fmt.Errorf("%w: %s", store.ErrSessionOpen, sessionID)
	}
	lease, err := acquireFileLock(r.lockPath(sessionID))
	if err != nil {
		if errors.Is(err, errFileLocked) {
			return nil, store.Commit{}, fmt.Errorf("%w: %s", store.ErrSessionOpen, sessionID)
		}
		return nil, store.Commit{}, fmt.Errorf("lock session journal: %w", err)
	}
	release := true
	defer func() {
		if release {
			_ = lease.Close()
		}
	}()

	file, err := os.OpenFile(r.path(sessionID), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		if errors.Is(err, fs.ErrExist) {
			return nil, store.Commit{}, fmt.Errorf("%w: %s", store.ErrSessionExists, sessionID)
		}
		return nil, store.Commit{}, fmt.Errorf("create session journal: %w", err)
	}
	entries, err := sequence(store.Cursor{}, pending)
	if err == nil {
		err = writeEntries(file, entries)
	}
	if err == nil && anySync(entries) {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return nil, store.Commit{}, fmt.Errorf("initialize session journal: %w", err)
	}
	if closeErr != nil {
		return nil, store.Commit{}, fmt.Errorf("close session journal: %w", closeErr)
	}
	if err := syncDirectory(r.root); err != nil {
		return nil, store.Commit{}, err
	}
	journal := &Journal{repository: r, sessionID: sessionID, lease: lease}
	r.open[sessionID] = journal
	release = false
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
	if r.open[sessionID] != nil {
		return nil, fmt.Errorf("%w: %s", store.ErrSessionOpen, sessionID)
	}
	lease, err := acquireFileLock(r.lockPath(sessionID))
	if err != nil {
		if errors.Is(err, errFileLocked) {
			return nil, fmt.Errorf("%w: %s", store.ErrSessionOpen, sessionID)
		}
		return nil, fmt.Errorf("lock session journal: %w", err)
	}
	journal := &Journal{repository: r, sessionID: sessionID, lease: lease}
	if _, err := journal.loadLocked(); err != nil {
		_ = lease.Close()
		return nil, err
	}
	r.open[sessionID] = journal
	return journal, nil
}

type Journal struct {
	repository *Repository
	sessionID  string
	lease      *fileLock
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
	entries, err := j.loadLocked()
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
	existing, err := j.loadLocked()
	if err != nil {
		return store.Commit{}, err
	}
	actual := tail(existing)
	if !expected.Equal(actual) {
		return store.Commit{}, &store.ConflictError{Expected: expected, Actual: actual}
	}
	entries, err := sequence(actual, pending)
	if err != nil {
		return store.Commit{}, err
	}
	file, err := os.OpenFile(j.repository.path(j.sessionID), os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return store.Commit{}, fmt.Errorf("open session journal for append: %w", err)
	}
	if err = writeEntries(file, entries); err == nil && anySync(entries) {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return store.Commit{}, fmt.Errorf("append session journal: %w", err)
	}
	if closeErr != nil {
		return store.Commit{}, fmt.Errorf("close session journal: %w", closeErr)
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
	lease := j.lease
	j.lease = nil
	j.mu.Unlock()

	j.repository.mu.Lock()
	if j.repository.open[j.sessionID] == j {
		delete(j.repository.open, j.sessionID)
	}
	j.repository.mu.Unlock()
	if lease == nil {
		return nil
	}
	return lease.Close()
}

func (j *Journal) loadLocked() ([]store.Entry, error) {
	path := j.repository.path(j.sessionID)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s", store.ErrSessionNotFound, j.sessionID)
		}
		return nil, fmt.Errorf("read session journal: %w", err)
	}
	data, err = j.recoverPartial(path, data)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	lines := bytes.Split(data, []byte{'\n'})
	entries := make([]store.Entry, 0, len(lines)-1)
	var parent string
	for i, line := range lines {
		if len(line) == 0 {
			continue
		}
		var entry diskEntry
		if err := json.Unmarshal(line, &entry); err != nil {
			return nil, fmt.Errorf("%w: line %d: %v", store.ErrCorruptLog, i+1, err)
		}
		if entry.Schema != store.CurrentSchema || entry.Seq != uint64(len(entries)+1) ||
			entry.ID == "" || entry.Kind == "" || entry.ParentID != parent ||
			entry.Durability > store.DurabilitySync {
			return nil, fmt.Errorf("%w: invalid chain at line %d", store.ErrCorruptLog, i+1)
		}
		entries = append(entries, entry.toDomain())
		parent = entry.ID
	}
	return store.CloneEntries(entries), nil
}

func (j *Journal) recoverPartial(path string, data []byte) ([]byte, error) {
	if len(data) == 0 || data[len(data)-1] == '\n' {
		return data, nil
	}
	cut := bytes.LastIndexByte(data, '\n') + 1
	partial := append([]byte(nil), data[cut:]...)
	sidecar := fmt.Sprintf("%s.partial-%d", path, time.Now().UnixNano())
	if err := os.WriteFile(sidecar, partial, 0o600); err != nil {
		return nil, fmt.Errorf("preserve partial session-journal tail: %w", err)
	}
	sidecarFile, err := os.OpenFile(sidecar, os.O_WRONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("open partial-tail sidecar: %w", err)
	}
	if err := sidecarFile.Sync(); err != nil {
		_ = sidecarFile.Close()
		return nil, fmt.Errorf("sync partial-tail sidecar: %w", err)
	}
	if err := sidecarFile.Close(); err != nil {
		return nil, fmt.Errorf("close partial-tail sidecar: %w", err)
	}
	if err := syncDirectory(j.repository.root); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("open session journal for tail recovery: %w", err)
	}
	if err = file.Truncate(int64(cut)); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return nil, fmt.Errorf("truncate partial session-journal tail: %w", err)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("close recovered session journal: %w", closeErr)
	}
	return data[:cut], nil
}

func (r *Repository) path(sessionID string) string {
	return filepath.Join(r.root, sessionID+".jsonl")
}

func (r *Repository) lockPath(sessionID string) string {
	return filepath.Join(r.root, sessionID+".lock")
}

type diskEntry struct {
	Schema     uint16           `json:"schema"`
	Seq        uint64           `json:"seq"`
	ID         string           `json:"id"`
	ParentID   string           `json:"parent_id,omitempty"`
	Kind       string           `json:"kind"`
	Payload    []byte           `json:"payload,omitempty"`
	Durability store.Durability `json:"durability,omitempty"`
}

func (e diskEntry) toDomain() store.Entry {
	return store.Entry{
		Schema: e.Schema, Seq: e.Seq, ID: e.ID, ParentID: e.ParentID,
		Kind: e.Kind, Payload: append([]byte(nil), e.Payload...), Durability: e.Durability,
	}
}

func writeEntries(file *os.File, entries []store.Entry) error {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	for _, entry := range entries {
		disk := diskEntry{
			Schema: entry.Schema, Seq: entry.Seq, ID: entry.ID, ParentID: entry.ParentID,
			Kind: entry.Kind, Payload: append([]byte(nil), entry.Payload...), Durability: entry.Durability,
		}
		if err := encoder.Encode(disk); err != nil {
			return err
		}
	}
	if buffer.Len() == 0 {
		return nil
	}
	written, err := file.Write(buffer.Bytes())
	if err != nil {
		return err
	}
	if written != buffer.Len() {
		return errors.New("short session-journal write")
	}
	return nil
}

func anySync(entries []store.Entry) bool {
	for _, entry := range entries {
		if entry.Durability == store.DurabilitySync {
			return true
		}
	}
	return false
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

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open session journal directory: %w", err)
	}
	if err := directory.Sync(); err != nil {
		_ = directory.Close()
		return fmt.Errorf("sync session journal directory: %w", err)
	}
	if err := directory.Close(); err != nil {
		return fmt.Errorf("close session journal directory: %w", err)
	}
	return nil
}

var _ store.Repository = (*Repository)(nil)
var _ store.Journal = (*Journal)(nil)
