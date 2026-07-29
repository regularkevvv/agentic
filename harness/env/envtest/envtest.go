// Package envtest provides reusable filesystem-environment conformance tests.
package envtest

import (
	"context"
	"io/fs"
	"testing"

	"github.com/regularkevvv/agentic/harness/env"
)

type LeaseFactory func(*testing.T) env.Lease

func Run(t *testing.T, factory LeaseFactory) {
	t.Helper()
	t.Run("filesystem_lifecycle", func(t *testing.T) {
		lease := factory(t)
		if lease == nil {
			t.Fatal("factory returned nil lease")
		}
		files := lease.Files()
		if files == nil {
			t.Fatal("lease returned nil filesystem")
		}
		ctx := context.Background()
		if err := files.MkdirAll(ctx, "dir/sub", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := files.WriteFile(ctx, "dir/sub/file", []byte("value"), 0o640); err != nil {
			t.Fatal(err)
		}
		data, err := files.ReadFile(ctx, "dir/sub/file")
		if err != nil || string(data) != "value" {
			t.Fatalf("ReadFile = %q, %v", data, err)
		}
		data[0] = 'X'
		reloaded, _ := files.ReadFile(ctx, "dir/sub/file")
		if string(reloaded) != "value" {
			t.Fatal("ReadFile returned aliased data")
		}
		resource, err := files.CanonicalPath(ctx, "dir/../dir/sub/file")
		if err != nil || !resource.Valid() {
			t.Fatalf("CanonicalPath = %#v, %v", resource, err)
		}
		info, err := files.Stat(ctx, "dir/sub/file")
		if err != nil || info.Size != 5 || info.IsDir || info.Mode.Perm() != fs.FileMode(0o640) {
			t.Fatalf("Stat = %#v, %v", info, err)
		}
		entries, err := files.ReadDir(ctx, "dir/sub")
		if err != nil || len(entries) != 1 || entries[0].Name != "file" {
			t.Fatalf("ReadDir = %#v, %v", entries, err)
		}
		if err := files.Remove(ctx, "dir/sub/file"); err != nil {
			t.Fatal(err)
		}
		if _, err := files.ReadFile(ctx, "dir/sub/file"); !env.HasCode(err, env.CodeNotFound) {
			t.Fatalf("missing error = %v", err)
		}
		if err := lease.Close(ctx); err != nil {
			t.Fatal(err)
		}
		if err := lease.Close(ctx); err != nil {
			t.Fatalf("second Close: %v", err)
		}
		if _, err := files.Stat(ctx, "."); !env.HasCode(err, env.CodeClosed) {
			t.Fatalf("closed error = %v", err)
		}
	})
}
