// Package memory provides a concurrent in-process implementation of the
// generic memory.Store and memory.Searcher ports.
package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"

	memorycore "github.com/regularkevvv/agentic/harness/memory"
)

type operation struct {
	fingerprint string
	result      memorycore.MutationResult
}

type scopeState struct {
	files map[string]memorycore.File
	ops   map[string]operation
	next  uint64
}

type Store struct {
	mu     sync.RWMutex
	scopes map[memorycore.Scope]*scopeState
}

func New() *Store { return &Store{scopes: make(map[memorycore.Scope]*scopeState)} }

func (s *Store) Read(
	ctx context.Context,
	scope memorycore.Scope,
	path string,
	options memorycore.ReadOptions,
) (memorycore.File, error) {
	if err := ctx.Err(); err != nil {
		return memorycore.File{}, err
	}
	if err := memorycore.ValidateScope(scope); err != nil {
		return memorycore.File{}, err
	}
	if err := memorycore.ValidateRead(path, options); err != nil {
		return memorycore.File{}, err
	}
	s.mu.RLock()
	state := s.scopes[scope]
	file, ok := memorycore.File{}, false
	if state != nil {
		file, ok = state.files[path]
	}
	s.mu.RUnlock()
	if !ok {
		return memorycore.File{}, fmt.Errorf("%w: %s", memorycore.ErrNotFound, path)
	}
	if len(file.Content) > options.MaxBytes {
		return memorycore.File{}, fmt.Errorf("%w: %s has %d bytes", memorycore.ErrLimitExceeded, path, len(file.Content))
	}
	return memorycore.CloneFile(file), nil
}

func (s *Store) List(
	ctx context.Context,
	scope memorycore.Scope,
	options memorycore.ListOptions,
) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := memorycore.ValidateScope(scope); err != nil {
		return nil, err
	}
	if err := memorycore.ValidateList(options); err != nil {
		return nil, err
	}
	s.mu.RLock()
	paths := make([]string, 0)
	if state := s.scopes[scope]; state != nil {
		for path := range state.files {
			if options.Prefix == "" || path == options.Prefix || strings.HasPrefix(path, options.Prefix+"/") {
				paths = append(paths, path)
			}
		}
	}
	s.mu.RUnlock()
	sort.Strings(paths)
	if len(paths) > options.Limit {
		paths = paths[:options.Limit]
	}
	return paths, nil
}

func (s *Store) Mutate(
	ctx context.Context,
	scope memorycore.Scope,
	mutation memorycore.Mutation,
) (memorycore.MutationResult, error) {
	if err := ctx.Err(); err != nil {
		return memorycore.MutationResult{}, err
	}
	if err := memorycore.ValidateScope(scope); err != nil {
		return memorycore.MutationResult{}, err
	}
	if err := memorycore.ValidateMutation(mutation); err != nil {
		return memorycore.MutationResult{}, err
	}
	mutation.Content = append([]byte(nil), mutation.Content...)
	s.mu.Lock()
	defer s.mu.Unlock()
	state := s.scopes[scope]
	if state == nil {
		state = &scopeState{files: make(map[string]memorycore.File), ops: make(map[string]operation)}
		s.scopes[scope] = state
	}
	if prior, ok := state.ops[mutation.IdempotencyKey]; ok {
		if prior.fingerprint != mutation.Fingerprint {
			return memorycore.MutationResult{}, fmt.Errorf("%w: %s", memorycore.ErrIdempotencyConflict, mutation.IdempotencyKey)
		}
		return prior.result, nil
	}
	current, exists := state.files[mutation.Path]
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
	state.next++
	version := version(state.next, mutation.Path, mutation.Content)
	result := memorycore.MutationResult{Path: mutation.Path, Version: version}
	switch mutation.Kind {
	case memorycore.MutationAppend:
		content := append([]byte(nil), current.Content...)
		content = append(content, mutation.Content...)
		state.files[mutation.Path] = memorycore.File{Path: mutation.Path, Content: content, Version: version}
		result.Bytes = len(content)
	case memorycore.MutationReplace:
		content := append([]byte(nil), mutation.Content...)
		state.files[mutation.Path] = memorycore.File{Path: mutation.Path, Content: content, Version: version}
		result.Bytes = len(content)
	case memorycore.MutationDelete:
		delete(state.files, mutation.Path)
		result.Deleted = true
	}
	state.ops[mutation.IdempotencyKey] = operation{fingerprint: mutation.Fingerprint, result: result}
	return result, nil
}

func (s *Store) Search(
	ctx context.Context,
	scope memorycore.Scope,
	options memorycore.SearchOptions,
) (memorycore.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return memorycore.SearchResult{}, err
	}
	if err := memorycore.ValidateScope(scope); err != nil {
		return memorycore.SearchResult{}, err
	}
	if err := memorycore.ValidateSearch(options); err != nil {
		return memorycore.SearchResult{}, err
	}
	s.mu.RLock()
	files := make([]memorycore.File, 0)
	if state := s.scopes[scope]; state != nil {
		for _, file := range state.files {
			files = append(files, memorycore.CloneFile(file))
		}
	}
	s.mu.RUnlock()
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	remaining := options.MaxBytes
	result := memorycore.SearchResult{}
	for _, file := range files {
		if len(result.Matches) == options.Limit || remaining == 0 {
			break
		}
		content := string(file.Content)
		if !strings.Contains(file.Path, options.Query) && !strings.Contains(content, options.Query) {
			continue
		}
		bytes := []byte(content)
		if len(bytes) > remaining {
			bytes = bytes[:remaining]
		}
		result.Matches = append(result.Matches, memorycore.Match{Path: file.Path, Content: string(bytes)})
		remaining -= len(bytes)
	}
	return result, nil
}

func version(sequence uint64, path string, content []byte) string {
	hash := sha256.New()
	_, _ = fmt.Fprintf(hash, "%d\x00%s\x00", sequence, path)
	_, _ = hash.Write(content)
	return "v1:" + hex.EncodeToString(hash.Sum(nil))
}

var _ memorycore.Store = (*Store)(nil)
var _ memorycore.Searcher = (*Store)(nil)
