// Package memory implements an environment backed by an in-process filesystem
// and a caller-supplied shell function.
//
// Tests are the point. A run against this adapter touches no disk and executes
// no command, so the shell is whatever [ShellFunc] you hand it, and a test can
// assert on the commands an agent tried to run rather than on their effects.
package memory

import (
	"context"
	"errors"
	"io/fs"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

type ShellFunc func(context.Context, harnessenv.Command) (harnessenv.CommandResult, error)

type memoryNode struct {
	data    []byte
	mode    fs.FileMode
	modTime time.Time
	dir     bool
}

type Environment struct {
	mu     sync.RWMutex
	cwd    string
	nodes  map[string]memoryNode
	shell  ShellFunc
	closed bool
}

func New(cwd string, shell ShellFunc) (*Environment, error) {
	if cwd == "" {
		cwd = "/"
	}
	canonical, err := memoryPath("/", cwd)
	if err != nil {
		return nil, err
	}
	memory := &Environment{cwd: canonical, nodes: make(map[string]memoryNode), shell: shell}
	memory.nodes["/"] = memoryNode{dir: true, mode: fs.ModeDir | 0o755, modTime: time.Now().UTC()}
	if err := memory.mkdirAllLocked(canonical, 0o755); err != nil {
		return nil, err
	}
	return memory, nil
}

func (m *Environment) CanonicalPath(ctx context.Context, name string) (harnessenv.CanonicalResource, error) {
	if err := ctx.Err(); err != nil {
		return harnessenv.CanonicalResource{}, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return harnessenv.CanonicalResource{}, &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "canonicalize", Path: name, Err: errors.New("environment closed")}
	}
	canonical, err := memoryPath(m.cwd, name)
	if err != nil {
		return harnessenv.CanonicalResource{}, err
	}
	return harnessenv.CanonicalResource{Scheme: "memory", ID: canonical, Display: canonical}, nil
}

func (m *Environment) ReadFile(ctx context.Context, name string) ([]byte, error) {
	canonical, err := m.CanonicalPath(ctx, name)
	if err != nil {
		return nil, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	node, ok := m.nodes[canonical.ID]
	if !ok {
		return nil, &harnessenv.Error{Code: harnessenv.CodeNotFound, Op: "read", Path: canonical.ID, Err: fs.ErrNotExist}
	}
	if node.dir {
		return nil, &harnessenv.Error{Code: harnessenv.CodeNotDirectory, Op: "read", Path: canonical.ID, Err: errors.New("path is a directory")}
	}
	return append([]byte(nil), node.data...), nil
}

func (m *Environment) WriteFile(ctx context.Context, name string, data []byte, mode fs.FileMode) error {
	canonical, err := m.CanonicalPath(ctx, name)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	parent := path.Dir(canonical.ID)
	if node, ok := m.nodes[parent]; !ok || !node.dir {
		return &harnessenv.Error{Code: harnessenv.CodeNotFound, Op: "write", Path: canonical.ID, Err: fs.ErrNotExist}
	}
	if node, ok := m.nodes[canonical.ID]; ok && node.dir {
		return &harnessenv.Error{Code: harnessenv.CodeNotDirectory, Op: "write", Path: canonical.ID, Err: errors.New("path is a directory")}
	}
	m.nodes[canonical.ID] = memoryNode{data: append([]byte(nil), data...), mode: mode, modTime: time.Now().UTC()}
	return nil
}

func (m *Environment) MkdirAll(ctx context.Context, name string, mode fs.FileMode) error {
	canonical, err := m.CanonicalPath(ctx, name)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.mkdirAllLocked(canonical.ID, mode)
}

func (m *Environment) mkdirAllLocked(canonical string, mode fs.FileMode) error {
	current := "/"
	for _, part := range strings.Split(strings.TrimPrefix(canonical, "/"), "/") {
		if part == "" {
			continue
		}
		current = path.Join(current, part)
		if node, ok := m.nodes[current]; ok {
			if !node.dir {
				return &harnessenv.Error{Code: harnessenv.CodeNotDirectory, Op: "mkdir", Path: current, Err: errors.New("path component is a file")}
			}
			continue
		}
		m.nodes[current] = memoryNode{dir: true, mode: fs.ModeDir | mode.Perm(), modTime: time.Now().UTC()}
	}
	return nil
}

func (m *Environment) ReadDir(ctx context.Context, name string) ([]harnessenv.DirEntry, error) {
	canonical, err := m.CanonicalPath(ctx, name)
	if err != nil {
		return nil, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if node, ok := m.nodes[canonical.ID]; !ok {
		return nil, &harnessenv.Error{Code: harnessenv.CodeNotFound, Op: "readdir", Path: canonical.ID, Err: fs.ErrNotExist}
	} else if !node.dir {
		return nil, &harnessenv.Error{Code: harnessenv.CodeNotDirectory, Op: "readdir", Path: canonical.ID, Err: errors.New("path is not a directory")}
	}
	var entries []harnessenv.DirEntry
	for name, node := range m.nodes {
		if name == canonical.ID || path.Dir(name) != canonical.ID {
			continue
		}
		entries = append(entries, harnessenv.DirEntry{Name: path.Base(name), Mode: node.mode, IsDir: node.dir})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	return entries, nil
}

func (m *Environment) Stat(ctx context.Context, name string) (harnessenv.FileInfo, error) {
	canonical, err := m.CanonicalPath(ctx, name)
	if err != nil {
		return harnessenv.FileInfo{}, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	node, ok := m.nodes[canonical.ID]
	if !ok {
		return harnessenv.FileInfo{}, &harnessenv.Error{Code: harnessenv.CodeNotFound, Op: "stat", Path: canonical.ID, Err: fs.ErrNotExist}
	}
	return harnessenv.FileInfo{Name: path.Base(canonical.ID), Size: int64(len(node.data)), Mode: node.mode, ModTime: node.modTime, IsDir: node.dir}, nil
}

func (m *Environment) Remove(ctx context.Context, name string) error {
	canonical, err := m.CanonicalPath(ctx, name)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if canonical.ID == "/" {
		return &harnessenv.Error{Code: harnessenv.CodePermission, Op: "remove", Path: canonical.ID, Err: errors.New("cannot remove root")}
	}
	node, ok := m.nodes[canonical.ID]
	if !ok {
		return &harnessenv.Error{Code: harnessenv.CodeNotFound, Op: "remove", Path: canonical.ID, Err: fs.ErrNotExist}
	}
	if node.dir {
		for name := range m.nodes {
			if path.Dir(name) == canonical.ID {
				return &harnessenv.Error{Code: harnessenv.CodeIO, Op: "remove", Path: canonical.ID, Err: errors.New("directory not empty")}
			}
		}
	}
	delete(m.nodes, canonical.ID)
	return nil
}

func (m *Environment) Exec(ctx context.Context, command harnessenv.Command) (harnessenv.CommandResult, error) {
	m.mu.RLock()
	if m.closed {
		m.mu.RUnlock()
		return harnessenv.CommandResult{}, &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "exec", Err: errors.New("environment closed")}
	}
	shell := m.shell
	m.mu.RUnlock()
	if shell == nil {
		return harnessenv.CommandResult{}, &harnessenv.Error{Code: harnessenv.CodeUnsupported, Op: "exec", Err: errors.New("memory shell is not configured")}
	}
	return shell(ctx, command)
}

func (m *Environment) Close(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

func (m *Environment) Files() harnessenv.FileSystem { return m }

func (m *Environment) Shell() (harnessenv.Shell, bool) {
	return m, m.shell != nil
}

func (m *Environment) Narrow(ctx context.Context, request harnessenv.NarrowRequest) (harnessenv.Lease, error) {
	if request.Root == "" {
		return nil, &harnessenv.Error{Code: harnessenv.CodeInvalid, Op: "narrow", Err: errors.New("narrow root is required")}
	}
	resource, err := m.CanonicalPath(ctx, request.Root)
	if err != nil {
		return nil, err
	}
	info, err := m.Stat(ctx, resource.ID)
	if err != nil {
		return nil, err
	}
	if !info.IsDir {
		return nil, &harnessenv.Error{
			Code: harnessenv.CodeNotDirectory,
			Op:   "narrow",
			Path: resource.ID,
			Err:  errors.New("narrow root is not a directory"),
		}
	}
	return &narrowLease{
		parent: m,
		root:   resource.ID,
		shell:  request.Shell && m.shell != nil,
	}, nil
}

func memoryPath(cwd, name string) (string, error) {
	if name == "" {
		name = "."
	}
	if !path.IsAbs(name) {
		name = path.Join(cwd, name)
	}
	canonical := path.Clean(name)
	if !path.IsAbs(canonical) {
		return "", &harnessenv.Error{Code: harnessenv.CodeInvalid, Op: "canonicalize", Path: name, Err: errors.New("invalid memory path")}
	}
	return canonical, nil
}

type Config struct {
	Cwd   string
	Shell ShellFunc
}

type Factory struct {
	config Config
}

func NewFactory(config Config) (*Factory, error) {
	if _, err := memoryPath("/", config.Cwd); err != nil {
		return nil, err
	}
	return &Factory{config: config}, nil
}

func (f *Factory) Open(ctx context.Context, _ string) (harnessenv.Lease, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return New(f.config.Cwd, f.config.Shell)
}

var _ harnessenv.Factory = (*Factory)(nil)
var _ harnessenv.Lease = (*Environment)(nil)
var _ harnessenv.Narrower = (*Environment)(nil)
