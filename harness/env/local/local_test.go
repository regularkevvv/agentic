package local

import (
	"context"
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
