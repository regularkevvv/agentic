package memory

import (
	"context"
	"errors"
	"io/fs"
	"path"
	"strings"
	"sync"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

// narrowLease is a virtual-root view over one memory environment. It owns only
// the view; closing it never closes the parent.
type narrowLease struct {
	parent *Environment
	root   string
	shell  bool

	mu     sync.RWMutex
	closed bool
}

func (l *narrowLease) CanonicalPath(ctx context.Context, name string) (harnessenv.CanonicalResource, error) {
	resolved, err := l.resolve(name)
	if err != nil {
		return harnessenv.CanonicalResource{}, err
	}
	return l.parent.CanonicalPath(ctx, resolved)
}

func (l *narrowLease) ReadFile(ctx context.Context, name string) ([]byte, error) {
	resolved, err := l.resolve(name)
	if err != nil {
		return nil, err
	}
	return l.parent.ReadFile(ctx, resolved)
}

func (l *narrowLease) WriteFile(ctx context.Context, name string, data []byte, mode fs.FileMode) error {
	resolved, err := l.resolve(name)
	if err != nil {
		return err
	}
	return l.parent.WriteFile(ctx, resolved, data, mode)
}

func (l *narrowLease) MkdirAll(ctx context.Context, name string, mode fs.FileMode) error {
	resolved, err := l.resolve(name)
	if err != nil {
		return err
	}
	return l.parent.MkdirAll(ctx, resolved, mode)
}

func (l *narrowLease) ReadDir(ctx context.Context, name string) ([]harnessenv.DirEntry, error) {
	resolved, err := l.resolve(name)
	if err != nil {
		return nil, err
	}
	return l.parent.ReadDir(ctx, resolved)
}

func (l *narrowLease) Stat(ctx context.Context, name string) (harnessenv.FileInfo, error) {
	resolved, err := l.resolve(name)
	if err != nil {
		return harnessenv.FileInfo{}, err
	}
	return l.parent.Stat(ctx, resolved)
}

func (l *narrowLease) Remove(ctx context.Context, name string) error {
	resolved, err := l.resolve(name)
	if err != nil {
		return err
	}
	return l.parent.Remove(ctx, resolved)
}

func (l *narrowLease) Exec(ctx context.Context, command harnessenv.Command) (harnessenv.CommandResult, error) {
	if !l.shell {
		return harnessenv.CommandResult{}, &harnessenv.Error{
			Code: harnessenv.CodeUnsupported,
			Op:   "exec",
			Err:  errors.New("shell is not enabled in the narrowed environment"),
		}
	}
	resolved, err := l.resolve(command.Dir)
	if err != nil {
		return harnessenv.CommandResult{}, err
	}
	command.Dir = resolved
	return l.parent.Exec(ctx, command)
}

func (l *narrowLease) Close(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.closed = true
	return nil
}

func (l *narrowLease) Files() harnessenv.FileSystem { return l }

func (l *narrowLease) Shell() (harnessenv.Shell, bool) {
	if !l.shell {
		return nil, false
	}
	return l, true
}

func (l *narrowLease) Narrow(ctx context.Context, request harnessenv.NarrowRequest) (harnessenv.Lease, error) {
	if request.Root == "" {
		return nil, &harnessenv.Error{
			Code: harnessenv.CodeInvalid,
			Op:   "narrow",
			Err:  errors.New("narrow root is required"),
		}
	}
	resolved, err := l.resolve(request.Root)
	if err != nil {
		return nil, err
	}
	request.Root = resolved
	request.Shell = request.Shell && l.shell
	return l.parent.Narrow(ctx, request)
}

func (l *narrowLease) resolve(name string) (string, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.closed {
		return "", &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "resolve", Path: name, Err: errors.New("narrowed environment closed")}
	}
	if name == "" {
		name = "."
	}
	virtual := path.Clean("/" + strings.TrimPrefix(name, "/"))
	return path.Join(l.root, strings.TrimPrefix(virtual, "/")), nil
}

var _ harnessenv.Lease = (*narrowLease)(nil)
var _ harnessenv.Narrower = (*narrowLease)(nil)
