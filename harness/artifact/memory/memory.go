// Package memory keeps artifacts in process memory, for tests and for runs
// that should leave nothing behind.
//
// It is an adapter for [artifact.Store]; see the file adapter beside it for
// what that means for imports.
package memory

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/regularkevvv/agentic/harness/artifact"
)

type memoryItem struct {
	handle artifact.Handle
	digest [sha256.Size]byte
	data   []byte
}

type Store struct {
	mu       sync.RWMutex
	sessions map[string]map[string]memoryItem
	byHandle map[string]map[artifact.Handle][]byte
}

func New() *Store {
	return &Store{
		sessions: make(map[string]map[string]memoryItem),
		byHandle: make(map[string]map[artifact.Handle][]byte),
	}
}

func (m *Store) Put(ctx context.Context, sessionID, key string, data []byte) (artifact.Handle, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := artifact.ValidateSessionID(sessionID); err != nil {
		return "", err
	}
	if key == "" {
		return "", fmt.Errorf("artifact key is required")
	}
	digest := sha256.Sum256(data)
	m.mu.Lock()
	defer m.mu.Unlock()
	items := m.sessions[sessionID]
	if items == nil {
		items = make(map[string]memoryItem)
		m.sessions[sessionID] = items
	}
	if item, ok := items[key]; ok {
		if item.digest != digest || !bytes.Equal(item.data, data) {
			return "", fmt.Errorf("%w: %s", artifact.ErrArtifactConflict, key)
		}
		return item.handle, nil
	}
	handle, err := newHandle()
	if err != nil {
		return "", err
	}
	copy := append([]byte(nil), data...)
	items[key] = memoryItem{handle: handle, digest: digest, data: copy}
	byHandle := m.byHandle[sessionID]
	if byHandle == nil {
		byHandle = make(map[artifact.Handle][]byte)
		m.byHandle[sessionID] = byHandle
	}
	byHandle[handle] = copy
	return handle, nil
}

func (m *Store) Get(ctx context.Context, sessionID string, handle artifact.Handle) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := artifact.ValidateSessionID(sessionID); err != nil {
		return nil, err
	}
	if err := artifact.ValidateHandle(handle); err != nil {
		return nil, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	data, ok := m.byHandle[sessionID][handle]
	if !ok {
		return nil, fmt.Errorf("%w: %s", artifact.ErrArtifactNotFound, handle)
	}
	return append([]byte(nil), data...), nil
}

func (m *Store) Count(sessionID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.byHandle[sessionID])
}

func newHandle() (artifact.Handle, error) {
	value := make([]byte, 32)
	if _, err := rand.Read(value); err != nil {
		return "", fmt.Errorf("generate artifact handle: %w", err)
	}
	return artifact.Handle("art_" + hex.EncodeToString(value)), nil
}

var _ artifact.Store = (*Store)(nil)
