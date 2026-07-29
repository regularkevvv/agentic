// Package env defines execution-substrate ports. It does not define
// governance policy, and no implementation is implicitly an OS sandbox.
package env

import (
	"context"
	"errors"
	"io/fs"
	"time"
)

// CanonicalResource is a backend-qualified resource identity. ID is opaque to
// generic harness policy; callers may compare complete values but must not
// infer local-path semantics from an arbitrary backend.
type CanonicalResource struct {
	Scheme  string
	ID      string
	Display string
}

func (r CanonicalResource) Valid() bool { return r.Scheme != "" && r.ID != "" }

func (r CanonicalResource) String() string {
	if r.Display != "" {
		return r.Display
	}
	return r.Scheme + ":" + r.ID
}

type FileInfo struct {
	Name    string
	Size    int64
	Mode    fs.FileMode
	ModTime time.Time
	IsDir   bool
}

type DirEntry struct {
	Name  string
	Mode  fs.FileMode
	IsDir bool
}

type FileSystem interface {
	CanonicalPath(context.Context, string) (CanonicalResource, error)
	ReadFile(context.Context, string) ([]byte, error)
	WriteFile(context.Context, string, []byte, fs.FileMode) error
	MkdirAll(context.Context, string, fs.FileMode) error
	ReadDir(context.Context, string) ([]DirEntry, error)
	Stat(context.Context, string) (FileInfo, error)
	Remove(context.Context, string) error
}

type Command struct {
	Name  string
	Args  []string
	Dir   string
	Env   []string
	Stdin []byte
}

type CommandResult struct {
	Stdout   []byte
	Stderr   []byte
	ExitCode int
}

type Shell interface {
	Exec(context.Context, Command) (CommandResult, error)
}

// Environment exposes independently discoverable substrate facets.
type Environment interface {
	Files() FileSystem
	Shell() (Shell, bool)
}

// Lease owns one session-scoped environment instance. Close must be
// idempotent so a caller can retry cleanup after a canceled context or a
// transient backend error.
type Lease interface {
	Environment
	Close(context.Context) error
}

// Factory provisions one isolated environment lease per session. A factory may
// point leases at the same workspace, but ownership and cleanup are never
// shared implicitly by Harness.
type Factory interface {
	Open(context.Context, string) (Lease, error)
}

type FactoryFunc func(context.Context, string) (Lease, error)

func (f FactoryFunc) Open(ctx context.Context, sessionID string) (Lease, error) {
	return f(ctx, sessionID)
}

var ErrFactoryClosed = errors.New("environment factory is closed")
