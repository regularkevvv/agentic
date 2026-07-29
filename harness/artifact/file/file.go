package file

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"

	"github.com/regularkevvv/agentic/harness/artifact"
)

type Store struct {
	root  string
	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

type fileMetadata struct {
	Handle artifact.Handle `json:"handle"`
	Digest string          `json:"digest"`
	Size   int             `json:"size"`
}

func New(root string) (*Store, error) {
	if root == "" {
		return nil, errors.New("artifact root is required")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("canonicalize artifact root: %w", err)
	}
	if err := os.MkdirAll(abs, 0o700); err != nil {
		return nil, fmt.Errorf("create artifact root: %w", err)
	}
	return &Store{root: filepath.Clean(abs), locks: make(map[string]*sync.Mutex)}, nil
}

func (s *Store) Root() string { return s.root }

func (s *Store) Put(ctx context.Context, sessionID, key string, data []byte) (artifact.Handle, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := artifact.ValidateSessionID(sessionID); err != nil {
		return "", err
	}
	if key == "" {
		return "", errors.New("artifact key is required")
	}
	lock := s.sessionLock(sessionID)
	lock.Lock()
	defer lock.Unlock()
	directory := filepath.Join(s.root, sessionID)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return "", fmt.Errorf("create artifact session directory: %w", err)
	}
	digest := sha256.Sum256(data)
	keyDigest := hashKey(key)
	metadataPath := filepath.Join(directory, hex.EncodeToString(keyDigest[:])+".json")
	if encoded, err := os.ReadFile(metadataPath); err == nil {
		var metadata fileMetadata
		if err := json.Unmarshal(encoded, &metadata); err != nil {
			return "", fmt.Errorf("decode artifact metadata: %w", err)
		}
		if err := artifact.ValidateHandle(metadata.Handle); err != nil {
			return "", fmt.Errorf("decode artifact metadata: %w", err)
		}
		if metadata.Digest != hex.EncodeToString(digest[:]) || metadata.Size != len(data) {
			return "", fmt.Errorf("%w: %s", artifact.ErrArtifactConflict, key)
		}
		existing, err := os.ReadFile(filepath.Join(directory, metadata.Handle.String()+".blob"))
		if err != nil || !bytes.Equal(existing, data) {
			return "", fmt.Errorf("artifact metadata/data mismatch for %s", metadata.Handle)
		}
		return metadata.Handle, nil
	} else if !errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("read artifact metadata: %w", err)
	}

	handle, err := newHandle()
	if err != nil {
		return "", err
	}
	blobPath := filepath.Join(directory, handle.String()+".blob")
	if err := writeAtomic(directory, blobPath, data); err != nil {
		return "", err
	}
	metadata := fileMetadata{Handle: handle, Digest: hex.EncodeToString(digest[:]), Size: len(data)}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return "", err
	}
	if err := writeAtomic(directory, metadataPath, encoded); err != nil {
		return "", err
	}
	if err := syncDir(directory); err != nil {
		return "", err
	}
	return handle, nil
}

func (s *Store) Get(ctx context.Context, sessionID string, handle artifact.Handle) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := artifact.ValidateSessionID(sessionID); err != nil {
		return nil, err
	}
	if err := artifact.ValidateHandle(handle); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(s.root, sessionID, handle.String()+".blob"))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s", artifact.ErrArtifactNotFound, handle)
		}
		return nil, fmt.Errorf("read artifact: %w", err)
	}
	return data, nil
}

func (s *Store) sessionLock(sessionID string) *sync.Mutex {
	s.mu.Lock()
	defer s.mu.Unlock()
	lock := s.locks[sessionID]
	if lock == nil {
		lock = &sync.Mutex{}
		s.locks[sessionID] = lock
	}
	return lock
}

func newHandle() (artifact.Handle, error) {
	value := make([]byte, 32)
	if _, err := rand.Read(value); err != nil {
		return "", fmt.Errorf("generate artifact handle: %w", err)
	}
	return artifact.Handle("art_" + hex.EncodeToString(value)), nil
}

var _ artifact.Store = (*Store)(nil)

func hashKey(key string) [sha256.Size]byte { return sha256.Sum256([]byte(key)) }

func writeAtomic(directory, target string, data []byte) error {
	temporary, err := os.CreateTemp(directory, ".artifact-*")
	if err != nil {
		return fmt.Errorf("create artifact temporary file: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err = temporary.Chmod(0o600); err == nil {
		_, err = temporary.Write(data)
	}
	if err == nil {
		err = temporary.Sync()
	}
	closeErr := temporary.Close()
	if err != nil {
		return fmt.Errorf("write artifact: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("close artifact: %w", closeErr)
	}
	if err := os.Rename(temporaryPath, target); err != nil {
		return fmt.Errorf("commit artifact: %w", err)
	}
	return nil
}

func syncDir(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open artifact directory: %w", err)
	}
	if err := directory.Sync(); err != nil {
		_ = directory.Close()
		return fmt.Errorf("sync artifact directory: %w", err)
	}
	if err := directory.Close(); err != nil {
		return fmt.Errorf("close artifact directory: %w", err)
	}
	return nil
}
