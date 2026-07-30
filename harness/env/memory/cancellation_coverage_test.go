package memory

import (
	"context"
	"errors"
	"testing"
)

func TestEveryFilesystemOperationPropagatesCanceledCanonicalization(t *testing.T) {
	environment, err := New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	checks := []func() error{
		func() error { _, err := environment.ReadFile(ctx, "file"); return err },
		func() error { return environment.WriteFile(ctx, "file", nil, 0o600) },
		func() error { return environment.MkdirAll(ctx, "dir", 0o700) },
		func() error { _, err := environment.ReadDir(ctx, "."); return err },
		func() error { _, err := environment.Stat(ctx, "."); return err },
		func() error { return environment.Remove(ctx, "file") },
	}
	for index, check := range checks {
		if err := check(); !errors.Is(err, context.Canceled) {
			t.Fatalf("operation %d error = %v", index, err)
		}
	}
}

func TestDirectoryOrderingAndInvalidInternalBase(t *testing.T) {
	environment, err := New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := environment.WriteFile(context.Background(), "/z", nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := environment.WriteFile(context.Background(), "/a", nil, 0o600); err != nil {
		t.Fatal(err)
	}
	entries, err := environment.ReadDir(context.Background(), "/")
	if err != nil || len(entries) != 2 || entries[0].Name != "a" || entries[1].Name != "z" {
		t.Fatalf("ordered entries = %#v, %v", entries, err)
	}
	if _, err := memoryPath("relative", "child"); err == nil {
		t.Fatal("relative internal base was accepted")
	}
}
