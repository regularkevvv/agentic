package local

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/env/envtest"
)

func TestEnvironmentConformance(t *testing.T) {
	t.Parallel()
	envtest.Run(t, func(t *testing.T) env.Lease {
		environment, err := New(Config{Root: t.TempDir()})
		if err != nil {
			t.Fatal(err)
		}
		return environment
	})
}

func TestCanonicalPathsAndDisclosure(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "work"), 0o755); err != nil {
		t.Fatal(err)
	}
	local, err := New(Config{Root: root, Cwd: "work"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = local.Close(context.Background()) }()
	ctx := context.Background()
	if err := local.MkdirAll(ctx, "nested", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := local.WriteFile(ctx, "nested/file.txt", []byte("hello"), 0o600); err != nil {
		t.Fatal(err)
	}
	canonical, err := local.CanonicalPath(ctx, "nested/../nested/file.txt")
	canonicalRoot, canonicalErr := filepath.EvalSymlinks(root)
	if canonicalErr != nil {
		t.Fatal(canonicalErr)
	}
	want := filepath.Join(canonicalRoot, "work", "nested", "file.txt")
	if err != nil || canonical.Scheme != "file" || canonical.ID != want {
		t.Fatalf("CanonicalPath = %#v, %v", canonical, err)
	}
	if _, err := local.CanonicalPath(ctx, "../../escape"); !env.HasCode(err, env.CodeEscaped) {
		t.Fatalf("escape error = %v", err)
	}
	if !strings.Contains(local.String(), "not an OS sandbox") {
		t.Fatalf("String did not disclose boundary: %s", local)
	}
}

func TestSymlinkPoliciesStayWithinRoot(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret"), []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, "inside"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "inside", "file"), []byte("ok"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(outside, "secret"), filepath.Join(root, "outside-link")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("inside/file", filepath.Join(root, "inside-link")); err != nil {
		t.Fatal(err)
	}
	withinRoot, err := New(Config{Root: root, Symlinks: SymlinkWithinRoot})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = withinRoot.Close(context.Background()) }()
	if data, err := withinRoot.ReadFile(context.Background(), "inside-link"); err != nil || string(data) != "ok" {
		t.Fatalf("internal symlink = %q, %v", data, err)
	}
	if _, err := withinRoot.ReadFile(context.Background(), "outside-link"); !env.HasCode(err, env.CodeEscaped) {
		t.Fatalf("outside symlink error = %v", err)
	}

	deny, err := New(Config{Root: root, Symlinks: SymlinkDeny})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = deny.Close(context.Background()) }()
	if _, err := deny.ReadFile(context.Background(), "inside-link"); !env.HasCode(err, env.CodeEscaped) {
		t.Fatalf("denied symlink error = %v", err)
	}
}

func TestShellIsOrdinaryHostExecution(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	local, err := New(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = local.Close(context.Background()) }()
	shell, ok := local.Shell()
	if !ok {
		t.Fatal("local shell unavailable")
	}
	result, err := shell.Exec(context.Background(), env.Command{Name: "/bin/sh", Args: []string{"-c", "pwd; exit 7"}})
	canonicalRoot, canonicalErr := filepath.EvalSymlinks(root)
	if canonicalErr != nil {
		t.Fatal(canonicalErr)
	}
	if err != nil || result.ExitCode != 7 || strings.TrimSpace(string(result.Stdout)) != canonicalRoot {
		t.Fatalf("Exec = %#v, %v", result, err)
	}
}

func TestFactoryProvisionsDistinctLeases(t *testing.T) {
	t.Parallel()
	factory, err := NewFactory(Config{Root: t.TempDir()})
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
	if first == second {
		t.Fatal("factory reused one environment lease")
	}
}

func TestLocalConstructionFilesystemAndShellFailures(t *testing.T) {
	t.Parallel()
	if _, err := New(Config{Root: "relative"}); !env.HasCode(err, env.CodeInvalid) {
		t.Fatalf("relative root = %v", err)
	}
	if _, err := New(Config{Root: filepath.Join(t.TempDir(), "missing")}); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("missing root = %v", err)
	}
	root := t.TempDir()
	fileRoot := filepath.Join(root, "file-root")
	if err := os.WriteFile(fileRoot, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(Config{Root: fileRoot}); !env.HasCode(err, env.CodeNotDirectory) {
		t.Fatalf("file root = %v", err)
	}
	if _, err := New(Config{Root: root, Cwd: "missing"}); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("missing cwd = %v", err)
	}
	if _, err := New(Config{Root: root, Cwd: "file-root"}); !env.HasCode(err, env.CodeIO) {
		t.Fatalf("file cwd = %v", err)
	}

	local, err := New(Config{Root: root, Symlinks: SymlinkDeny})
	if err != nil {
		t.Fatal(err)
	}
	if local.Root() == "" {
		t.Fatal("canonical Root is empty")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := local.CanonicalPath(ctx, "."); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled canonicalization = %v", err)
	}
	if err := local.Close(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled close = %v", err)
	}
	if _, err := local.ReadFile(context.Background(), "missing"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("read missing = %v", err)
	}
	if err := local.WriteFile(context.Background(), "missing/file", nil, 0o600); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("write missing parent = %v", err)
	}
	if err := local.WriteFile(context.Background(), ".", nil, 0o600); err == nil {
		t.Fatal("write directory succeeded")
	}
	if err := local.MkdirAll(context.Background(), "file-root/child", 0o755); err == nil {
		t.Fatal("mkdir through file succeeded")
	}
	if _, err := local.ReadDir(context.Background(), "missing"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("readdir missing = %v", err)
	}
	if _, err := local.ReadDir(context.Background(), "file-root"); err == nil {
		t.Fatal("readdir file succeeded")
	}
	if _, err := local.Stat(context.Background(), "missing"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("stat missing = %v", err)
	}
	if err := local.Remove(context.Background(), "missing"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("remove missing = %v", err)
	}
	if _, err := local.Exec(context.Background(), env.Command{}); !env.HasCode(err, env.CodeInvalid) {
		t.Fatalf("empty command = %v", err)
	}
	if _, err := local.Exec(context.Background(), env.Command{Name: "echo", Dir: "../escape"}); !env.HasCode(err, env.CodeEscaped) {
		t.Fatalf("escaped command dir = %v", err)
	}
	if _, err := local.Exec(context.Background(), env.Command{Name: "definitely-not-an-agentic-command"}); err == nil {
		t.Fatal("missing command succeeded")
	}
	result, err := local.Exec(context.Background(), env.Command{
		Name:  "/bin/sh",
		Args:  []string{"-c", "read value; printf '%s:%s' \"$value\" \"$TEST_VALUE\""},
		Env:   []string{"TEST_VALUE=env"},
		Stdin: []byte("stdin\n"),
	})
	if err != nil || string(result.Stdout) != "stdin:env" {
		t.Fatalf("stdin/env command = %#v, %v", result, err)
	}
	if err := local.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := local.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	for name, operation := range map[string]func() error{
		"canonical": func() error { _, err := local.CanonicalPath(context.Background(), "."); return err },
		"read":      func() error { _, err := local.ReadFile(context.Background(), "."); return err },
		"write":     func() error { return local.WriteFile(context.Background(), "file", nil, 0o600) },
		"mkdir":     func() error { return local.MkdirAll(context.Background(), "dir", 0o755) },
		"readdir":   func() error { _, err := local.ReadDir(context.Background(), "."); return err },
		"stat":      func() error { _, err := local.Stat(context.Background(), "."); return err },
		"remove":    func() error { return local.Remove(context.Background(), "file") },
	} {
		if err := operation(); !env.HasCode(err, env.CodeClosed) {
			t.Fatalf("closed %s = %v", name, err)
		}
	}
}

func TestLocalFactoryCancellationAndAbsolutePaths(t *testing.T) {
	t.Parallel()
	if _, err := NewFactory(Config{Root: "relative"}); !env.HasCode(err, env.CodeInvalid) {
		t.Fatalf("relative factory = %v", err)
	}
	root := t.TempDir()
	factory, err := NewFactory(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := factory.Open(ctx, "session"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled factory = %v", err)
	}
	local, err := New(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = local.Close(context.Background()) }()
	canonicalRoot, _ := filepath.EvalSymlinks(root)
	future := filepath.Join(canonicalRoot, "future", "file")
	resource, err := local.CanonicalPath(context.Background(), future)
	if err != nil || resource.ID != future {
		t.Fatalf("future absolute = %#v, %v", resource, err)
	}
	if _, err := local.CanonicalPath(context.Background(), filepath.Dir(canonicalRoot)); !env.HasCode(err, env.CodeEscaped) {
		t.Fatalf("outside absolute = %v", err)
	}
}

func TestNarrowedLocalEnvironmentHasIndependentLease(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	childRoot := filepath.Join(root, "child")
	if err := os.Mkdir(childRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(childRoot, "inside"), []byte("inside"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "outside"), []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	parent, err := New(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = parent.Close(context.Background()) }()
	narrow, err := parent.Narrow(context.Background(), env.NarrowRequest{Root: "child"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := narrow.Shell(); ok {
		t.Fatal("local narrow enabled shell without an explicit request")
	}
	if data, err := narrow.Files().ReadFile(context.Background(), "inside"); err != nil || string(data) != "inside" {
		t.Fatalf("narrow local read = %q, %v", data, err)
	}
	if _, err := narrow.Files().ReadFile(context.Background(), "../outside"); !env.HasCode(err, env.CodeEscaped) {
		t.Fatalf("narrow local traversal = %v", err)
	}
	nestedNarrower, ok := narrow.(env.Narrower)
	if !ok {
		t.Fatal("narrowed local environment lost nested narrowing")
	}
	nested, err := nestedNarrower.Narrow(context.Background(), env.NarrowRequest{Root: ".", Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := nested.Shell(); ok {
		t.Fatal("nested local narrowing broadened disabled shell access")
	}
	if data, err := nested.Files().ReadFile(context.Background(), "inside"); err != nil || string(data) != "inside" {
		t.Fatalf("nested local read = %q, %v", data, err)
	}
	if err := nested.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := narrow.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if data, err := parent.ReadFile(context.Background(), "child/inside"); err != nil || string(data) != "inside" {
		t.Fatalf("narrow close affected parent: %q, %v", data, err)
	}

	withShell, err := parent.Narrow(context.Background(), env.NarrowRequest{Root: "child", Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = withShell.Close(context.Background()) }()
	shell, ok := withShell.Shell()
	if !ok {
		t.Fatal("explicit local narrow shell unavailable")
	}
	result, err := shell.Exec(context.Background(), env.Command{Name: "/bin/sh", Args: []string{"-c", "pwd"}})
	canonicalChild, canonicalErr := filepath.EvalSymlinks(childRoot)
	if canonicalErr != nil {
		t.Fatal(canonicalErr)
	}
	if err != nil || strings.TrimSpace(string(result.Stdout)) != canonicalChild {
		t.Fatalf("narrow local shell = %#v, %v", result, err)
	}
	shellNarrower, ok := withShell.(env.Narrower)
	if !ok {
		t.Fatal("shell-enabled local environment lost nested narrowing")
	}
	nestedShell, err := shellNarrower.Narrow(context.Background(), env.NarrowRequest{Root: ".", Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := nestedShell.Shell(); !ok {
		t.Fatal("nested local narrowing discarded explicitly retained shell access")
	}
	if err := nestedShell.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestNarrowedLocalValidation(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	parent, err := New(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = parent.Close(context.Background()) }()
	if _, err := parent.Narrow(context.Background(), env.NarrowRequest{}); !env.HasCode(err, env.CodeInvalid) {
		t.Fatalf("empty narrow root = %v", err)
	}
	if _, err := parent.Narrow(context.Background(), env.NarrowRequest{Root: "missing"}); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("missing narrow root = %v", err)
	}
	file := filepath.Join(root, "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := parent.Narrow(context.Background(), env.NarrowRequest{Root: "file"}); !env.HasCode(err, env.CodeNotDirectory) {
		t.Fatalf("file narrow root = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := parent.Narrow(canceled, env.NarrowRequest{Root: "."}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled narrow = %v", err)
	}
}
