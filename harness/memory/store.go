// Package memory defines bounded, namespaced cross-session memory ports and an
// opt-in harness capability. It is separate from durable session storage.
package memory

import (
	"context"
	"errors"
	"fmt"
	"path"
	"strings"
)

var (
	ErrNotFound            = errors.New("memory file not found")
	ErrConflict            = errors.New("memory version conflict")
	ErrIdempotencyConflict = errors.New("memory idempotency conflict")
	ErrInvalidPath         = errors.New("invalid memory path")
	ErrLimitExceeded       = errors.New("memory bound exceeded")
)

type Scope string

type File struct {
	Path    string `json:"path"`
	Content []byte `json:"content"`
	Version string `json:"version"`
}

type ReadOptions struct {
	MaxBytes int
}

type ListOptions struct {
	Prefix string
	Limit  int
}

type MutationKind string

const (
	MutationAppend  MutationKind = "append"
	MutationReplace MutationKind = "replace"
	MutationDelete  MutationKind = "delete"
)

type Mutation struct {
	Path            string
	Kind            MutationKind
	Content         []byte
	ExpectedVersion string
	IdempotencyKey  string
	Fingerprint     string
}

type MutationResult struct {
	Path    string `json:"path"`
	Version string `json:"version"`
	Deleted bool   `json:"deleted,omitempty"`
	Bytes   int    `json:"bytes"`
}

type SearchOptions struct {
	Query    string
	Limit    int
	MaxBytes int
}

type Match struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type SearchResult struct {
	Matches []Match `json:"matches"`
}

type Store interface {
	Read(context.Context, Scope, string, ReadOptions) (File, error)
	List(context.Context, Scope, ListOptions) ([]string, error)
	Mutate(context.Context, Scope, Mutation) (MutationResult, error)
}

type Searcher interface {
	Search(context.Context, Scope, SearchOptions) (SearchResult, error)
}

func ValidateScope(scope Scope) error {
	value := string(scope)
	if value == "" || len(value) > 256 || strings.TrimSpace(value) != value || strings.ContainsAny(value, "\x00\r\n") {
		return errors.New("invalid memory scope")
	}
	return nil
}

func ValidatePath(value string) error {
	if value == "" || len(value) > 1024 || strings.HasPrefix(value, "/") || strings.HasSuffix(value, "/") ||
		strings.Contains(value, "\\") || path.Clean(value) != value {
		return fmt.Errorf("%w: %q", ErrInvalidPath, value)
	}
	for _, segment := range strings.Split(value, "/") {
		if segment == "" || segment == "." || segment == ".." || segment == ".agentic-memory.jsonl" {
			return fmt.Errorf("%w: %q", ErrInvalidPath, value)
		}
	}
	return nil
}

func ValidateRead(path string, options ReadOptions) error {
	if err := ValidatePath(path); err != nil {
		return err
	}
	if options.MaxBytes <= 0 {
		return errors.New("memory read bound must be positive")
	}
	return nil
}

func ValidateList(options ListOptions) error {
	if options.Limit <= 0 {
		return errors.New("memory list bound must be positive")
	}
	if options.Prefix != "" {
		if err := ValidatePath(options.Prefix); err != nil {
			return err
		}
	}
	return nil
}

func ValidateSearch(options SearchOptions) error {
	if options.Query == "" || options.Limit <= 0 || options.MaxBytes <= 0 {
		return errors.New("memory search query and positive bounds are required")
	}
	return nil
}

func ValidateMutation(mutation Mutation) error {
	if err := ValidatePath(mutation.Path); err != nil {
		return err
	}
	switch mutation.Kind {
	case MutationAppend, MutationReplace:
	case MutationDelete:
		if len(mutation.Content) != 0 {
			return errors.New("memory delete must not contain content")
		}
	default:
		return errors.New("invalid memory mutation kind")
	}
	if mutation.IdempotencyKey == "" || mutation.Fingerprint == "" {
		return errors.New("memory mutation idempotency key and fingerprint are required")
	}
	return nil
}

func CloneFile(file File) File {
	file.Content = append([]byte(nil), file.Content...)
	return file
}
