package local

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

type SymlinkPolicy uint8

const (
	SymlinkWithinRoot SymlinkPolicy = iota
	SymlinkDeny
)

type Config struct {
	Root     string
	Cwd      string
	Symlinks SymlinkPolicy
}

// Environment uses os.Root for traversal-resistant filesystem operations. Shell
// commands run as the current OS user and can affect resources outside Root;
// this adapter must never be represented as a sandbox.
type Environment struct {
	rootPath string
	cwdRel   string
	symlinks SymlinkPolicy
	root     *os.Root
	mu       sync.RWMutex
	closed   bool
}

func New(config Config) (*Environment, error) {
	if config.Root == "" || !filepath.IsAbs(config.Root) {
		return nil, &harnessenv.Error{Code: harnessenv.CodeInvalid, Op: "open", Path: config.Root, Err: errors.New("absolute root is required")}
	}
	rootPath, err := filepath.EvalSymlinks(filepath.Clean(config.Root))
	if err != nil {
		return nil, harnessenv.Wrap("open", config.Root, err)
	}
	info, err := os.Stat(rootPath)
	if err != nil {
		return nil, harnessenv.Wrap("open", rootPath, err)
	}
	if !info.IsDir() {
		return nil, &harnessenv.Error{Code: harnessenv.CodeNotDirectory, Op: "open", Path: rootPath, Err: errors.New("root is not a directory")}
	}
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		return nil, harnessenv.Wrap("open", rootPath, err)
	}
	local := &Environment{rootPath: rootPath, root: root, symlinks: config.Symlinks}
	cwd := config.Cwd
	if cwd == "" {
		cwd = "."
	}
	rel, err := local.relative(cwd)
	if err != nil {
		_ = root.Close()
		return nil, err
	}
	stat, err := root.Stat(rel)
	if err != nil || !stat.IsDir() {
		_ = root.Close()
		if err == nil {
			err = errors.New("cwd is not a directory")
		}
		return nil, harnessenv.Wrap("cwd", cwd, err)
	}
	local.cwdRel = rel
	return local, nil
}

func (l *Environment) Root() string { return l.rootPath }

func (l *Environment) CanonicalPath(ctx context.Context, name string) (harnessenv.CanonicalResource, error) {
	if err := ctx.Err(); err != nil {
		return harnessenv.CanonicalResource{}, err
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.closed {
		return harnessenv.CanonicalResource{}, &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "canonicalize", Path: name, Err: os.ErrClosed}
	}
	rel, err := l.relative(name)
	if err != nil {
		return harnessenv.CanonicalResource{}, err
	}
	if l.symlinks == SymlinkDeny {
		if err := l.rejectSymlinks(rel); err != nil {
			return harnessenv.CanonicalResource{}, err
		}
	}
	canonical, err := canonicalExistingPrefix(l.rootPath, rel)
	if err != nil {
		return harnessenv.CanonicalResource{}, harnessenv.Wrap("canonicalize", name, err)
	}
	if !within(l.rootPath, canonical) {
		return harnessenv.CanonicalResource{}, &harnessenv.Error{Code: harnessenv.CodeEscaped, Op: "canonicalize", Path: name, Err: errors.New("path escapes root")}
	}
	return harnessenv.CanonicalResource{Scheme: "file", ID: canonical, Display: canonical}, nil
}

func (l *Environment) ReadFile(ctx context.Context, name string) ([]byte, error) {
	if _, err := l.CanonicalPath(ctx, name); err != nil {
		return nil, err
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.closed {
		return nil, &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "read", Path: name, Err: os.ErrClosed}
	}
	rel, err := l.relative(name)
	if err != nil {
		return nil, err
	}
	data, err := l.root.ReadFile(rel)
	return data, harnessenv.Wrap("read", name, err)
}

func (l *Environment) WriteFile(ctx context.Context, name string, data []byte, mode fs.FileMode) error {
	if _, err := l.CanonicalPath(ctx, name); err != nil {
		return err
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.closed {
		return &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "write", Path: name, Err: os.ErrClosed}
	}
	rel, err := l.relative(name)
	if err != nil {
		return err
	}
	return harnessenv.Wrap("write", name, l.root.WriteFile(rel, data, mode))
}

func (l *Environment) MkdirAll(ctx context.Context, name string, mode fs.FileMode) error {
	if _, err := l.CanonicalPath(ctx, name); err != nil {
		return err
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.closed {
		return &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "mkdir", Path: name, Err: os.ErrClosed}
	}
	rel, err := l.relative(name)
	if err != nil {
		return err
	}
	return harnessenv.Wrap("mkdir", name, l.root.MkdirAll(rel, mode))
}

func (l *Environment) ReadDir(ctx context.Context, name string) ([]harnessenv.DirEntry, error) {
	if _, err := l.CanonicalPath(ctx, name); err != nil {
		return nil, err
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.closed {
		return nil, &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "readdir", Path: name, Err: os.ErrClosed}
	}
	rel, err := l.relative(name)
	if err != nil {
		return nil, err
	}
	entries, err := fs.ReadDir(l.root.FS(), rel)
	if err != nil {
		return nil, harnessenv.Wrap("readdir", name, err)
	}
	result := make([]harnessenv.DirEntry, len(entries))
	for i, entry := range entries {
		result[i] = harnessenv.DirEntry{Name: entry.Name(), Mode: entry.Type(), IsDir: entry.IsDir()}
	}
	return result, nil
}

func (l *Environment) Stat(ctx context.Context, name string) (harnessenv.FileInfo, error) {
	if _, err := l.CanonicalPath(ctx, name); err != nil {
		return harnessenv.FileInfo{}, err
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.closed {
		return harnessenv.FileInfo{}, &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "stat", Path: name, Err: os.ErrClosed}
	}
	rel, err := l.relative(name)
	if err != nil {
		return harnessenv.FileInfo{}, err
	}
	info, err := l.root.Stat(rel)
	if err != nil {
		return harnessenv.FileInfo{}, harnessenv.Wrap("stat", name, err)
	}
	return harnessenv.FileInfo{Name: info.Name(), Size: info.Size(), Mode: info.Mode(), ModTime: info.ModTime(), IsDir: info.IsDir()}, nil
}

func (l *Environment) Remove(ctx context.Context, name string) error {
	if _, err := l.CanonicalPath(ctx, name); err != nil {
		return err
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.closed {
		return &harnessenv.Error{Code: harnessenv.CodeClosed, Op: "remove", Path: name, Err: os.ErrClosed}
	}
	rel, err := l.relative(name)
	if err != nil {
		return err
	}
	return harnessenv.Wrap("remove", name, l.root.Remove(rel))
}

func (l *Environment) Exec(ctx context.Context, command harnessenv.Command) (harnessenv.CommandResult, error) {
	if command.Name == "" {
		return harnessenv.CommandResult{}, &harnessenv.Error{Code: harnessenv.CodeInvalid, Op: "exec", Err: errors.New("command name is required")}
	}
	dir := command.Dir
	if dir == "" {
		dir = filepath.Join(l.rootPath, l.cwdRel)
	}
	canonical, err := l.CanonicalPath(ctx, dir)
	if err != nil {
		return harnessenv.CommandResult{}, err
	}
	cmd := exec.CommandContext(ctx, command.Name, command.Args...)
	cmd.Dir = canonical.ID
	if command.Env != nil {
		cmd.Env = append(os.Environ(), command.Env...)
	}
	cmd.Stdin = bytes.NewReader(command.Stdin)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err = cmd.Run()
	result := harnessenv.CommandResult{Stdout: stdout.Bytes(), Stderr: stderr.Bytes()}
	if err == nil {
		return result, nil
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		result.ExitCode = exit.ExitCode()
		return result, nil
	}
	return harnessenv.CommandResult{}, harnessenv.Wrap("exec", command.Name, err)
}

func (l *Environment) Close(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	return harnessenv.Wrap("cleanup", l.rootPath, l.root.Close())
}

func (l *Environment) Files() harnessenv.FileSystem { return l }

func (l *Environment) Shell() (harnessenv.Shell, bool) { return l, true }

// Narrow opens an independently owned os.Root below this environment. Allowing
// shell preserves ordinary host execution semantics and is not containment.
func (l *Environment) Narrow(ctx context.Context, request harnessenv.NarrowRequest) (harnessenv.Lease, error) {
	if request.Root == "" {
		return nil, &harnessenv.Error{
			Code: harnessenv.CodeInvalid,
			Op:   "narrow",
			Err:  errors.New("narrow root is required"),
		}
	}
	resource, err := l.CanonicalPath(ctx, request.Root)
	if err != nil {
		return nil, err
	}
	child, err := New(Config{
		Root:     resource.ID,
		Cwd:      ".",
		Symlinks: l.symlinks,
	})
	if err != nil {
		return nil, err
	}
	return &narrowLease{Lease: child, shell: request.Shell}, nil
}

type narrowLease struct {
	harnessenv.Lease
	shell bool
}

func (l *narrowLease) Shell() (harnessenv.Shell, bool) {
	if !l.shell {
		return nil, false
	}
	return l.Lease.Shell()
}

func (l *narrowLease) Narrow(ctx context.Context, request harnessenv.NarrowRequest) (harnessenv.Lease, error) {
	narrower, ok := l.Lease.(harnessenv.Narrower)
	if !ok {
		return nil, &harnessenv.Error{
			Code: harnessenv.CodeUnsupported,
			Op:   "narrow",
			Err:  errors.New("nested narrowing is unsupported"),
		}
	}
	request.Shell = request.Shell && l.shell
	return narrower.Narrow(ctx, request)
}

func (l *Environment) relative(name string) (string, error) {
	if name == "" {
		name = "."
	}
	var rel string
	if filepath.IsAbs(name) {
		canonical, err := canonicalAbsolute(filepath.Clean(name))
		if err != nil {
			return "", harnessenv.Wrap("canonicalize", name, err)
		}
		rel, err = filepath.Rel(l.rootPath, canonical)
		if err != nil {
			return "", harnessenv.Wrap("canonicalize", name, err)
		}
	} else {
		rel = filepath.Clean(filepath.Join(l.cwdRel, name))
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", &harnessenv.Error{Code: harnessenv.CodeEscaped, Op: "canonicalize", Path: name, Err: errors.New("path escapes root")}
	}
	if rel == "" {
		rel = "."
	}
	return rel, nil
}

func (l *Environment) rejectSymlinks(rel string) error {
	current := "."
	for _, part := range strings.Split(filepath.Clean(rel), string(filepath.Separator)) {
		if part == "." || part == "" {
			continue
		}
		current = filepath.Join(current, part)
		info, err := l.root.Lstat(current)
		if errors.Is(err, fs.ErrNotExist) {
			break
		}
		if err != nil {
			return harnessenv.Wrap("lstat", rel, err)
		}
		if info.Mode()&fs.ModeSymlink != 0 {
			return &harnessenv.Error{Code: harnessenv.CodeEscaped, Op: "canonicalize", Path: rel, Err: errors.New("symlinks are disabled")}
		}
	}
	return nil
}

func canonicalExistingPrefix(root, rel string) (string, error) {
	return canonicalAbsolute(filepath.Join(root, rel))
}

func canonicalAbsolute(candidate string) (string, error) {
	remaining := []string{}
	current := candidate
	for {
		resolved, err := filepath.EvalSymlinks(current)
		if err == nil {
			parts := append([]string{resolved}, remaining...)
			return filepath.Clean(filepath.Join(parts...)), nil
		}
		if !errors.Is(err, fs.ErrNotExist) {
			return "", err
		}
		parent := filepath.Dir(current)
		if parent == current {
			return "", err
		}
		remaining = append([]string{filepath.Base(current)}, remaining...)
		current = parent
	}
}

func within(root, candidate string) bool {
	rel, err := filepath.Rel(root, candidate)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

type Factory struct {
	config Config
}

func NewFactory(config Config) (*Factory, error) {
	if config.Root == "" || !filepath.IsAbs(config.Root) {
		return nil, &harnessenv.Error{Code: harnessenv.CodeInvalid, Op: "factory", Path: config.Root, Err: errors.New("absolute root is required")}
	}
	return &Factory{config: config}, nil
}

func (f *Factory) Open(ctx context.Context, _ string) (harnessenv.Lease, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return New(f.config)
}

var _ harnessenv.Factory = (*Factory)(nil)
var _ harnessenv.Lease = (*Environment)(nil)
var _ harnessenv.Narrower = (*Environment)(nil)
var _ harnessenv.Narrower = (*narrowLease)(nil)

func (l *Environment) String() string {
	return fmt.Sprintf("Local(root=%q, cwd=%q; not an OS sandbox)", l.rootPath, l.cwdRel)
}
