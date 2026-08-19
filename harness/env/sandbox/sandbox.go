// Package sandbox provides a fail-closed local environment whose filesystem
// API remains rooted with os.Root and whose shell processes enter an operating-
// system sandbox before executing untrusted commands.
package sandbox

import (
	"context"
	"errors"
	"io/fs"
	"path/filepath"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/env/local"
)

// Config describes one workspace sandbox. Network access is denied by
// default; setting Network permits it while retaining filesystem confinement.
type Config struct {
	Root     string
	Cwd      string
	Symlinks local.SymlinkPolicy
	Network  bool
}

type Factory struct {
	config  Config
	backend string
}

// NewFactory validates that the current platform has a usable strict backend.
// Unsupported platforms fail closed instead of silently falling back to the
// ordinary local environment.
func NewFactory(config Config) (*Factory, error) {
	return newFactory(config, probeBackend)
}

func newFactory(config Config, probe func() (string, error)) (*Factory, error) {
	if config.Root == "" || !filepath.IsAbs(config.Root) {
		return nil, &harnessenv.Error{Code: harnessenv.CodeInvalid, Op: "sandbox", Path: config.Root, Err: errors.New("absolute root is required")}
	}
	backend, err := probe()
	if err != nil {
		return nil, &harnessenv.Error{Code: harnessenv.CodeUnsupported, Op: "sandbox", Path: config.Root, Err: err}
	}
	return &Factory{config: config, backend: backend}, nil
}

func (f *Factory) Backend() string {
	if f == nil {
		return ""
	}
	return f.backend
}

func (f *Factory) Open(ctx context.Context, _ string) (harnessenv.Lease, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	localEnvironment, err := local.New(local.Config{
		Root: f.config.Root, Cwd: f.config.Cwd, Symlinks: f.config.Symlinks,
	})
	if err != nil {
		return nil, err
	}
	return &Environment{
		local: localEnvironment, root: localEnvironment.Root(), shell: true,
		network: f.config.Network, symlinks: f.config.Symlinks,
	}, nil
}

type Environment struct {
	local    *local.Environment
	root     string
	shell    bool
	network  bool
	symlinks local.SymlinkPolicy
}

func (e *Environment) Files() harnessenv.FileSystem { return e }

func (e *Environment) Shell() (harnessenv.Shell, bool) {
	if !e.shell {
		return nil, false
	}
	return e, true
}

func (e *Environment) CanonicalPath(ctx context.Context, path string) (harnessenv.CanonicalResource, error) {
	return e.local.CanonicalPath(ctx, path)
}

func (e *Environment) ReadFile(ctx context.Context, path string) ([]byte, error) {
	return e.local.ReadFile(ctx, path)
}

func (e *Environment) WriteFile(ctx context.Context, path string, data []byte, mode fs.FileMode) error {
	return e.local.WriteFile(ctx, path, data, mode)
}

func (e *Environment) MkdirAll(ctx context.Context, path string, mode fs.FileMode) error {
	return e.local.MkdirAll(ctx, path, mode)
}

func (e *Environment) ReadDir(ctx context.Context, path string) ([]harnessenv.DirEntry, error) {
	return e.local.ReadDir(ctx, path)
}

func (e *Environment) Stat(ctx context.Context, path string) (harnessenv.FileInfo, error) {
	return e.local.Stat(ctx, path)
}

func (e *Environment) Remove(ctx context.Context, path string) error {
	return e.local.Remove(ctx, path)
}

func (e *Environment) Exec(ctx context.Context, command harnessenv.Command) (harnessenv.CommandResult, error) {
	if !e.shell {
		return harnessenv.CommandResult{}, &harnessenv.Error{Code: harnessenv.CodeUnsupported, Op: "exec", Err: errors.New("sandbox shell is disabled")}
	}
	if command.Name == "" {
		return harnessenv.CommandResult{}, &harnessenv.Error{Code: harnessenv.CodeInvalid, Op: "exec", Err: errors.New("command name is required")}
	}
	directory := command.Dir
	if directory == "" {
		directory = "."
	}
	canonical, err := e.local.CanonicalPath(ctx, directory)
	if err != nil {
		return harnessenv.CommandResult{}, err
	}
	result, err := execute(ctx, execution{
		root: e.root, cwd: canonical.ID, network: e.network, command: command,
	})
	if err != nil {
		return harnessenv.CommandResult{}, harnessenv.Wrap("exec", command.Name, err)
	}
	return result, nil
}

func (e *Environment) Narrow(ctx context.Context, request harnessenv.NarrowRequest) (harnessenv.Lease, error) {
	if request.Root == "" {
		return nil, &harnessenv.Error{Code: harnessenv.CodeInvalid, Op: "narrow", Err: errors.New("narrow root is required")}
	}
	resource, err := e.local.CanonicalPath(ctx, request.Root)
	if err != nil {
		return nil, err
	}
	child, err := local.New(local.Config{Root: resource.ID, Cwd: ".", Symlinks: e.symlinks})
	if err != nil {
		return nil, err
	}
	return &Environment{
		local: child, root: resource.ID, shell: e.shell && request.Shell,
		network: e.network, symlinks: e.symlinks,
	}, nil
}

func (e *Environment) Close(ctx context.Context) error { return e.local.Close(ctx) }

var _ harnessenv.Factory = (*Factory)(nil)
var _ harnessenv.Lease = (*Environment)(nil)
var _ harnessenv.Narrower = (*Environment)(nil)
