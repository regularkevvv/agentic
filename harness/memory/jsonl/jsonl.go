// Package jsonl provides a durable JSON-lines implementation of the generic
// memory ports. JSONL is an adapter choice, not part of the core contract.
package jsonl

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	memorycore "github.com/regularkevvv/agentic/harness/memory"
)

var ErrCorruptLog = errors.New("corrupt memory JSONL log")

type Store struct {
	root string
	mu   sync.Mutex
}

func New(root string) (*Store, error) {
	if root == "" {
		return nil, errors.New("memory JSONL root is required")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("canonicalize memory JSONL root: %w", err)
	}
	if err := os.MkdirAll(abs, 0o700); err != nil {
		return nil, fmt.Errorf("create memory JSONL root: %w", err)
	}
	return &Store{root: filepath.Clean(abs)}, nil
}

func (s *Store) Root() string { return s.root }

type operation struct {
	fingerprint string
	result      memorycore.MutationResult
}

type state struct {
	files map[string]memorycore.File
	ops   map[string]operation
	next  uint64
}

type record struct {
	Version  int                       `json:"version"`
	Scope    memorycore.Scope          `json:"scope"`
	Sequence uint64                    `json:"sequence"`
	Mutation memorycore.Mutation       `json:"mutation"`
	Result   memorycore.MutationResult `json:"result"`
	File     *memorycore.File          `json:"file,omitempty"`
}

func (s *Store) Read(ctx context.Context, scope memorycore.Scope, path string, options memorycore.ReadOptions) (memorycore.File, error) {
	if err := memorycore.ValidateScope(scope); err != nil {
		return memorycore.File{}, err
	}
	if err := memorycore.ValidateRead(path, options); err != nil {
		return memorycore.File{}, err
	}
	loaded, err := s.load(ctx, scope)
	if err != nil {
		return memorycore.File{}, err
	}
	file, ok := loaded.files[path]
	if !ok {
		return memorycore.File{}, fmt.Errorf("%w: %s", memorycore.ErrNotFound, path)
	}
	if len(file.Content) > options.MaxBytes {
		return memorycore.File{}, fmt.Errorf("%w: %s has %d bytes", memorycore.ErrLimitExceeded, path, len(file.Content))
	}
	return memorycore.CloneFile(file), nil
}

func (s *Store) List(ctx context.Context, scope memorycore.Scope, options memorycore.ListOptions) ([]string, error) {
	if err := memorycore.ValidateScope(scope); err != nil {
		return nil, err
	}
	if err := memorycore.ValidateList(options); err != nil {
		return nil, err
	}
	loaded, err := s.load(ctx, scope)
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(loaded.files))
	for path := range loaded.files {
		if options.Prefix == "" || path == options.Prefix || strings.HasPrefix(path, options.Prefix+"/") {
			paths = append(paths, path)
		}
	}
	sort.Strings(paths)
	if len(paths) > options.Limit {
		paths = paths[:options.Limit]
	}
	return paths, nil
}

func (s *Store) Search(ctx context.Context, scope memorycore.Scope, options memorycore.SearchOptions) (memorycore.SearchResult, error) {
	if err := memorycore.ValidateScope(scope); err != nil {
		return memorycore.SearchResult{}, err
	}
	if err := memorycore.ValidateSearch(options); err != nil {
		return memorycore.SearchResult{}, err
	}
	loaded, err := s.load(ctx, scope)
	if err != nil {
		return memorycore.SearchResult{}, err
	}
	paths := make([]string, 0, len(loaded.files))
	for path := range loaded.files {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	remaining := options.MaxBytes
	result := memorycore.SearchResult{}
	for _, path := range paths {
		if len(result.Matches) == options.Limit || remaining == 0 {
			break
		}
		content := string(loaded.files[path].Content)
		if !strings.Contains(path, options.Query) && !strings.Contains(content, options.Query) {
			continue
		}
		value := []byte(content)
		if len(value) > remaining {
			value = value[:remaining]
		}
		result.Matches = append(result.Matches, memorycore.Match{Path: path, Content: string(value)})
		remaining -= len(value)
	}
	return result, nil
}

func (s *Store) Mutate(ctx context.Context, scope memorycore.Scope, mutation memorycore.Mutation) (memorycore.MutationResult, error) {
	if err := memorycore.ValidateScope(scope); err != nil {
		return memorycore.MutationResult{}, err
	}
	if err := memorycore.ValidateMutation(mutation); err != nil {
		return memorycore.MutationResult{}, err
	}
	mutation.Content = append([]byte(nil), mutation.Content...)
	s.mu.Lock()
	defer s.mu.Unlock()
	lock, err := s.acquire(ctx, scope)
	if err != nil {
		return memorycore.MutationResult{}, err
	}
	defer func() { _ = lock.Close() }()
	loaded, err := s.loadUnlocked(scope)
	if err != nil {
		return memorycore.MutationResult{}, err
	}
	if prior, ok := loaded.ops[mutation.IdempotencyKey]; ok {
		if prior.fingerprint != mutation.Fingerprint {
			return memorycore.MutationResult{}, fmt.Errorf("%w: %s", memorycore.ErrIdempotencyConflict, mutation.IdempotencyKey)
		}
		return prior.result, nil
	}
	current, exists := loaded.files[mutation.Path]
	if mutation.ExpectedVersion == "" {
		if exists {
			return memorycore.MutationResult{}, fmt.Errorf("%w: %s already exists", memorycore.ErrConflict, mutation.Path)
		}
	} else if !exists || current.Version != mutation.ExpectedVersion {
		return memorycore.MutationResult{}, fmt.Errorf("%w: %s version differs", memorycore.ErrConflict, mutation.Path)
	}
	if mutation.Kind == memorycore.MutationDelete && !exists {
		return memorycore.MutationResult{}, fmt.Errorf("%w: %s does not exist", memorycore.ErrConflict, mutation.Path)
	}
	sequence := loaded.next + 1
	version := makeVersion(sequence, mutation.Path, mutation.Content)
	result := memorycore.MutationResult{Path: mutation.Path, Version: version}
	var file *memorycore.File
	switch mutation.Kind {
	case memorycore.MutationAppend:
		content := append([]byte(nil), current.Content...)
		content = append(content, mutation.Content...)
		value := memorycore.File{Path: mutation.Path, Content: content, Version: version}
		file = &value
		result.Bytes = len(content)
	case memorycore.MutationReplace:
		value := memorycore.File{Path: mutation.Path, Content: append([]byte(nil), mutation.Content...), Version: version}
		file = &value
		result.Bytes = len(value.Content)
	case memorycore.MutationDelete:
		result.Deleted = true
	}
	entry := record{Version: 1, Scope: scope, Sequence: sequence, Mutation: mutation, Result: result, File: file}
	if err := s.append(scope, entry); err != nil {
		return memorycore.MutationResult{}, err
	}
	return result, nil
}

func (s *Store) load(ctx context.Context, scope memorycore.Scope) (state, error) {
	if err := ctx.Err(); err != nil {
		return state{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	lock, err := s.acquire(ctx, scope)
	if err != nil {
		return state{}, err
	}
	defer func() { _ = lock.Close() }()
	return s.loadUnlocked(scope)
}

func (s *Store) loadUnlocked(scope memorycore.Scope) (state, error) {
	result := state{files: make(map[string]memorycore.File), ops: make(map[string]operation)}
	path := s.path(scope)
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return result, nil
	}
	if err != nil {
		return state{}, fmt.Errorf("read memory JSONL: %w", err)
	}
	data, err = s.recoverPartial(path, data)
	if err != nil {
		return state{}, err
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 64<<10), 64<<20)
	line := 0
	for scanner.Scan() {
		line++
		var entry record
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			return state{}, fmt.Errorf("%w: line %d: %v", ErrCorruptLog, line, err)
		}
		if entry.Version != 1 || entry.Scope != scope || entry.Sequence != result.next+1 ||
			memorycore.ValidateMutation(entry.Mutation) != nil || entry.Result.Path != entry.Mutation.Path ||
			entry.Result.Version == "" {
			return state{}, fmt.Errorf("%w: invalid record at line %d", ErrCorruptLog, line)
		}
		if _, exists := result.ops[entry.Mutation.IdempotencyKey]; exists {
			return state{}, fmt.Errorf("%w: repeated operation at line %d", ErrCorruptLog, line)
		}
		current, exists := result.files[entry.Mutation.Path]
		if entry.Mutation.ExpectedVersion == "" {
			if exists {
				return state{}, fmt.Errorf("%w: invalid create frontier at line %d", ErrCorruptLog, line)
			}
		} else if !exists || current.Version != entry.Mutation.ExpectedVersion {
			return state{}, fmt.Errorf("%w: invalid CAS frontier at line %d", ErrCorruptLog, line)
		}
		expectedVersion := makeVersion(entry.Sequence, entry.Mutation.Path, entry.Mutation.Content)
		if entry.Result.Version != expectedVersion {
			return state{}, fmt.Errorf("%w: invalid result version at line %d", ErrCorruptLog, line)
		}
		switch entry.Mutation.Kind {
		case memorycore.MutationAppend, memorycore.MutationReplace:
			if entry.File == nil || entry.Result.Deleted || entry.File.Path != entry.Mutation.Path ||
				entry.File.Version != expectedVersion || entry.Result.Bytes != len(entry.File.Content) {
				return state{}, fmt.Errorf("%w: invalid file at line %d", ErrCorruptLog, line)
			}
			expectedContent := append([]byte(nil), entry.Mutation.Content...)
			if entry.Mutation.Kind == memorycore.MutationAppend {
				expectedContent = append(append([]byte(nil), current.Content...), entry.Mutation.Content...)
			}
			if !bytes.Equal(entry.File.Content, expectedContent) {
				return state{}, fmt.Errorf("%w: invalid file content at line %d", ErrCorruptLog, line)
			}
			result.files[entry.File.Path] = memorycore.CloneFile(*entry.File)
		case memorycore.MutationDelete:
			if !exists || entry.File != nil || !entry.Result.Deleted || entry.Result.Bytes != 0 {
				return state{}, fmt.Errorf("%w: invalid delete at line %d", ErrCorruptLog, line)
			}
			delete(result.files, entry.Mutation.Path)
		}
		result.ops[entry.Mutation.IdempotencyKey] = operation{fingerprint: entry.Mutation.Fingerprint, result: entry.Result}
		result.next = entry.Sequence
	}
	if err := scanner.Err(); err != nil {
		return state{}, fmt.Errorf("scan memory JSONL: %w", err)
	}
	return result, nil
}

func (s *Store) append(scope memorycore.Scope, entry record) error {
	path := s.path(scope)
	_, statErr := os.Stat(path)
	created := errors.Is(statErr, fs.ErrNotExist)
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("open memory JSONL: %w", err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetEscapeHTML(false)
	if err = encoder.Encode(entry); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return fmt.Errorf("append memory JSONL: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("close memory JSONL: %w", closeErr)
	}
	if created {
		return syncDirectory(s.root)
	}
	return nil
}

func (s *Store) recoverPartial(path string, data []byte) ([]byte, error) {
	if len(data) == 0 || data[len(data)-1] == '\n' {
		return data, nil
	}
	cut := bytes.LastIndexByte(data, '\n') + 1
	partial := append([]byte(nil), data[cut:]...)
	sidecar := fmt.Sprintf("%s.partial-%d", path, time.Now().UnixNano())
	if err := os.WriteFile(sidecar, partial, 0o600); err != nil {
		return nil, fmt.Errorf("preserve partial memory tail: %w", err)
	}
	sidecarFile, err := os.OpenFile(sidecar, os.O_WRONLY, 0)
	if err != nil {
		return nil, err
	}
	if err = sidecarFile.Sync(); err == nil {
		err = sidecarFile.Close()
	} else {
		_ = sidecarFile.Close()
	}
	if err != nil {
		return nil, fmt.Errorf("sync partial memory sidecar: %w", err)
	}
	if err := syncDirectory(s.root); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		return nil, err
	}
	if err = file.Truncate(int64(cut)); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return nil, fmt.Errorf("truncate partial memory tail: %w", err)
	}
	if closeErr != nil {
		return nil, closeErr
	}
	return data[:cut], nil
}

func (s *Store) acquire(ctx context.Context, scope memorycore.Scope) (*fileLock, error) {
	for {
		lock, err := acquireFileLock(s.lockPath(scope))
		if err == nil {
			return lock, nil
		}
		if !errors.Is(err, errFileLocked) {
			return nil, err
		}
		timer := time.NewTimer(5 * time.Millisecond)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

func (s *Store) path(scope memorycore.Scope) string {
	return filepath.Join(s.root, scopeName(scope)+".jsonl")
}

func (s *Store) lockPath(scope memorycore.Scope) string {
	return filepath.Join(s.root, scopeName(scope)+".lock")
}

func scopeName(scope memorycore.Scope) string {
	hash := sha256.Sum256([]byte(scope))
	return hex.EncodeToString(hash[:])
}

func makeVersion(sequence uint64, path string, content []byte) string {
	hash := sha256.New()
	_, _ = fmt.Fprintf(hash, "%d\x00%s\x00", sequence, path)
	_, _ = hash.Write(content)
	return "v1:" + hex.EncodeToString(hash.Sum(nil))
}

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open memory directory: %w", err)
	}
	err = directory.Sync()
	closeErr := directory.Close()
	if err != nil {
		return fmt.Errorf("sync memory directory: %w", err)
	}
	return closeErr
}

var _ memorycore.Store = (*Store)(nil)
var _ memorycore.Searcher = (*Store)(nil)
