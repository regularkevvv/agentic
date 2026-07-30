package memory

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/env/envtest"
)

func TestEnvironmentConformance(t *testing.T) {
	t.Parallel()
	envtest.Run(t, func(t *testing.T) env.Lease {
		environment, err := New("/", nil)
		if err != nil {
			t.Fatal(err)
		}
		return environment
	})
}

func TestFilesystemAndShell(t *testing.T) {
	t.Parallel()
	memory, err := New("/workspace", func(_ context.Context, command env.Command) (env.CommandResult, error) {
		return env.CommandResult{Stdout: []byte(command.Name)}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = memory.Close(context.Background()) }()
	ctx := context.Background()
	if err := memory.MkdirAll(ctx, "dir/sub", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := memory.WriteFile(ctx, "dir/sub/file", []byte("value"), 0o640); err != nil {
		t.Fatal(err)
	}
	resource, err := memory.CanonicalPath(ctx, "/workspace/dir/sub/file")
	if err != nil || resource.Scheme != "memory" || resource.ID != "/workspace/dir/sub/file" {
		t.Fatalf("resource = %#v, %v", resource, err)
	}
	shell, ok := memory.Shell()
	if !ok {
		t.Fatal("configured shell unavailable")
	}
	result, err := shell.Exec(ctx, env.Command{Name: "mock"})
	if err != nil || string(result.Stdout) != "mock" {
		t.Fatalf("Exec = %#v, %v", result, err)
	}
}

func TestConcurrentAccessAndUnsupportedShell(t *testing.T) {
	t.Parallel()
	memory, err := New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = memory.Close(context.Background()) }()
	if err := memory.MkdirAll(context.Background(), "/data", 0o755); err != nil {
		t.Fatal(err)
	}
	const workers = 32
	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		go func() {
			defer wg.Done()
			_ = memory.WriteFile(context.Background(), "/data/shared", []byte("x"), 0o600)
			_, _ = memory.ReadFile(context.Background(), "/data/shared")
			_, _ = memory.ReadDir(context.Background(), "/data")
		}()
	}
	wg.Wait()
	if _, ok := memory.Shell(); ok {
		t.Fatal("unconfigured shell reported available")
	}
	if _, err := memory.Exec(context.Background(), env.Command{Name: "none"}); !env.HasCode(err, env.CodeUnsupported) {
		t.Fatalf("unsupported shell error = %v", err)
	}
}

func TestFactoryCreatesIndependentState(t *testing.T) {
	t.Parallel()
	factory, err := NewFactory(Config{Cwd: "/"})
	if err != nil {
		t.Fatal(err)
	}
	first, err := factory.Open(context.Background(), "one")
	if err != nil {
		t.Fatal(err)
	}
	second, err := factory.Open(context.Background(), "two")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = first.Close(context.Background()) }()
	defer func() { _ = second.Close(context.Background()) }()
	if err := first.Files().WriteFile(context.Background(), "shared", []byte("one"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := second.Files().ReadFile(context.Background(), "shared"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("second lease observed first state: %v", err)
	}
}

func TestMemoryFilesystemFailuresAndCancellation(t *testing.T) {
	t.Parallel()
	memory, err := New("", func(context.Context, env.Command) (env.CommandResult, error) {
		return env.CommandResult{}, errors.New("shell failed")
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := memory.CanonicalPath(ctx, "/"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled canonicalization = %v", err)
	}
	if err := memory.Close(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled close = %v", err)
	}
	if _, err := memory.ReadFile(context.Background(), "/"); !env.HasCode(err, env.CodeNotDirectory) {
		t.Fatalf("read directory = %v", err)
	}
	if err := memory.WriteFile(context.Background(), "/missing/file", nil, 0o600); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("write missing parent = %v", err)
	}
	if err := memory.WriteFile(context.Background(), "/", nil, 0o600); !env.HasCode(err, env.CodeNotDirectory) {
		t.Fatalf("write directory = %v", err)
	}
	if err := memory.WriteFile(context.Background(), "/file", []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := memory.MkdirAll(context.Background(), "/file/child", 0o755); !env.HasCode(err, env.CodeNotDirectory) {
		t.Fatalf("mkdir through file = %v", err)
	}
	if _, err := memory.ReadDir(context.Background(), "/missing"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("readdir missing = %v", err)
	}
	if _, err := memory.ReadDir(context.Background(), "/file"); !env.HasCode(err, env.CodeNotDirectory) {
		t.Fatalf("readdir file = %v", err)
	}
	if _, err := memory.Stat(context.Background(), "/missing"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("stat missing = %v", err)
	}
	if err := memory.Remove(context.Background(), "/"); !env.HasCode(err, env.CodePermission) {
		t.Fatalf("remove root = %v", err)
	}
	if err := memory.Remove(context.Background(), "/missing"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("remove missing = %v", err)
	}
	if err := memory.MkdirAll(context.Background(), "/dir", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := memory.WriteFile(context.Background(), "/dir/child", nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := memory.Remove(context.Background(), "/dir"); !env.HasCode(err, env.CodeIO) {
		t.Fatalf("remove nonempty directory = %v", err)
	}
	if _, err := memory.Exec(context.Background(), env.Command{Name: "fail"}); err == nil {
		t.Fatal("shell error was hidden")
	}
	if err := memory.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := memory.CanonicalPath(context.Background(), "/"); !env.HasCode(err, env.CodeClosed) {
		t.Fatalf("closed canonicalization = %v", err)
	}
	if _, err := memory.Exec(context.Background(), env.Command{Name: "fail"}); !env.HasCode(err, env.CodeClosed) {
		t.Fatalf("closed shell = %v", err)
	}
}

func TestMemoryFactoryCancellationAndPathHelpers(t *testing.T) {
	t.Parallel()
	factory, err := NewFactory(Config{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := factory.Open(ctx, "session"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled factory = %v", err)
	}
	if path, err := memoryPath("/workspace", ""); err != nil || path != "/workspace" {
		t.Fatalf("empty memory path = %q, %v", path, err)
	}
}

func TestNarrowedEnvironmentConfinesPathsAndDoesNotOwnParent(t *testing.T) {
	t.Parallel()
	var commandDir string
	parent, err := New("/", func(_ context.Context, command env.Command) (env.CommandResult, error) {
		commandDir = command.Dir
		return env.CommandResult{Stdout: []byte("ok")}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = parent.Close(context.Background()) }()
	ctx := context.Background()
	if err := parent.MkdirAll(ctx, "/children/one", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := parent.MkdirAll(ctx, "/outside", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := parent.WriteFile(ctx, "/children/one/input", []byte("inside"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := parent.WriteFile(ctx, "/outside/secret", []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	narrow, err := parent.Narrow(ctx, env.NarrowRequest{Root: "/children/one"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := narrow.Shell(); ok {
		t.Fatal("narrowed shell was enabled implicitly")
	}
	if _, err := narrow.(*narrowLease).Exec(ctx, env.Command{Name: "disabled"}); !env.HasCode(err, env.CodeUnsupported) {
		t.Fatalf("disabled narrow shell = %v", err)
	}
	if data, err := narrow.Files().ReadFile(ctx, "/input"); err != nil || string(data) != "inside" {
		t.Fatalf("narrow read = %q, %v", data, err)
	}
	if err := narrow.Files().MkdirAll(ctx, "dir", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := narrow.Files().WriteFile(ctx, "dir/output", []byte("value"), 0o600); err != nil {
		t.Fatal(err)
	}
	if info, err := narrow.Files().Stat(ctx, "dir/output"); err != nil || info.Size != 5 {
		t.Fatalf("narrow stat = %#v, %v", info, err)
	}
	if entries, err := narrow.Files().ReadDir(ctx, "dir"); err != nil || len(entries) != 1 {
		t.Fatalf("narrow readdir = %#v, %v", entries, err)
	}
	resource, err := narrow.Files().CanonicalPath(ctx, "../outside/secret")
	if err != nil || resource.ID != "/children/one/outside/secret" {
		t.Fatalf("narrow canonical path = %#v, %v", resource, err)
	}
	if _, err := narrow.Files().ReadFile(ctx, "../outside/secret"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("narrow traversal reached parent outside path: %v", err)
	}
	if err := narrow.Files().Remove(ctx, "dir/output"); err != nil {
		t.Fatal(err)
	}
	nestedNarrower, ok := narrow.(env.Narrower)
	if !ok {
		t.Fatal("narrowed memory environment lost nested narrowing")
	}
	nested, err := nestedNarrower.Narrow(ctx, env.NarrowRequest{Root: ".", Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := nested.Shell(); ok {
		t.Fatal("nested memory narrowing broadened disabled shell access")
	}
	if data, err := nested.Files().ReadFile(ctx, "input"); err != nil || string(data) != "inside" {
		t.Fatalf("nested narrow read = %q, %v", data, err)
	}
	if err := nested.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if err := narrow.Close(ctx); err != nil {
		t.Fatal(err)
	}
	closedOperations := map[string]func() error{
		"canonical": func() error {
			_, err := narrow.Files().CanonicalPath(ctx, "input")
			return err
		},
		"read": func() error {
			_, err := narrow.Files().ReadFile(ctx, "input")
			return err
		},
		"write": func() error {
			return narrow.Files().WriteFile(ctx, "output", nil, 0o600)
		},
		"mkdir": func() error {
			return narrow.Files().MkdirAll(ctx, "dir", 0o755)
		},
		"readdir": func() error {
			_, err := narrow.Files().ReadDir(ctx, ".")
			return err
		},
		"stat": func() error {
			_, err := narrow.Files().Stat(ctx, "input")
			return err
		},
		"remove": func() error {
			return narrow.Files().Remove(ctx, "input")
		},
	}
	for name, operation := range closedOperations {
		if err := operation(); !env.HasCode(err, env.CodeClosed) {
			t.Fatalf("closed narrow %s = %v", name, err)
		}
	}
	if data, err := parent.ReadFile(ctx, "/children/one/input"); err != nil || string(data) != "inside" {
		t.Fatalf("closing view closed parent: %q, %v", data, err)
	}

	withShell, err := parent.Narrow(ctx, env.NarrowRequest{Root: "/children/one", Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = withShell.Close(context.Background()) }()
	shell, ok := withShell.Shell()
	if !ok {
		t.Fatal("explicit narrowed shell unavailable")
	}
	if result, err := shell.Exec(ctx, env.Command{Name: "mock"}); err != nil || string(result.Stdout) != "ok" {
		t.Fatalf("narrow shell = %#v, %v", result, err)
	}
	if commandDir != "/children/one" {
		t.Fatalf("narrow shell dir = %q", commandDir)
	}
	shellNarrower, ok := withShell.(env.Narrower)
	if !ok {
		t.Fatal("shell-enabled memory environment lost nested narrowing")
	}
	nestedShell, err := shellNarrower.Narrow(ctx, env.NarrowRequest{Root: ".", Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := nestedShell.Shell(); !ok {
		t.Fatal("nested memory narrowing discarded explicitly retained shell access")
	}
	if err := nestedShell.Close(ctx); err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := withShell.Close(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled narrow close = %v", err)
	}
	if err := withShell.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := shell.Exec(ctx, env.Command{Name: "mock"}); !env.HasCode(err, env.CodeClosed) {
		t.Fatalf("closed narrow shell = %v", err)
	}
}

func TestNarrowValidation(t *testing.T) {
	t.Parallel()
	parent, err := New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = parent.Close(context.Background()) }()
	if _, err := parent.Narrow(context.Background(), env.NarrowRequest{}); !env.HasCode(err, env.CodeInvalid) {
		t.Fatalf("empty narrow root = %v", err)
	}
	if _, err := parent.Narrow(context.Background(), env.NarrowRequest{Root: "/missing"}); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("missing narrow root = %v", err)
	}
	if err := parent.WriteFile(context.Background(), "/file", nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := parent.Narrow(context.Background(), env.NarrowRequest{Root: "/file"}); !env.HasCode(err, env.CodeNotDirectory) {
		t.Fatalf("file narrow root = %v", err)
	}
}
