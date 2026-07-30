package local

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

func TestCanonicalizationConstructionAndHelperEdges(t *testing.T) {
	root := t.TempDir()
	if _, err := New(Config{Root: root, Cwd: t.TempDir()}); !harnessenv.HasCode(err, harnessenv.CodeEscaped) {
		t.Fatalf("outside cwd = %v", err)
	}

	if err := os.Symlink("b", filepath.Join(root, "a")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("a", filepath.Join(root, "b")); err != nil {
		t.Fatal(err)
	}
	local, err := New(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = local.Close(context.Background()) }()
	if _, err := local.CanonicalPath(context.Background(), "a"); err == nil {
		t.Fatal("symlink loop canonicalized")
	}
	if rel, err := local.relative(""); err != nil || rel != "." {
		t.Fatalf("empty relative path = %q, %v", rel, err)
	}
}

func TestConstructionRejectsUnreadableRoot(t *testing.T) {
	root := t.TempDir()
	if err := os.Chmod(root, 0); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Chmod(root, 0o700) }()
	if _, err := New(Config{Root: root}); err == nil {
		t.Skip("current process can open a mode-000 directory")
	}
}
