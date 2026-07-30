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
