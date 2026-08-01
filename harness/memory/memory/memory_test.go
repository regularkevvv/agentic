package memory

import (
	"context"
	"errors"
	"reflect"
	"testing"

	memorycore "github.com/regularkevvv/agentic/harness/memory"
	"github.com/regularkevvv/agentic/harness/memory/memorytest"
)

func TestConformance(t *testing.T) {
	memorytest.Run(t, func(*testing.T) memorytest.Store { return New() })
}

func TestValidationCancellationAndEmptyState(t *testing.T) {
	store := New()
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	for name, operation := range map[string]func() error{
		"read": func() error {
			_, err := store.Read(canceled, "scope", "main", memorycore.ReadOptions{MaxBytes: 1})
			return err
		},
		"mutate": func() error {
			_, err := store.Mutate(canceled, "scope", memorycore.Mutation{Path: "main", Kind: memorycore.MutationReplace, IdempotencyKey: "id", Fingerprint: "fp"})
			return err
		},
		"search": func() error {
			_, err := store.Search(canceled, "scope", memorycore.SearchOptions{Query: "x", Limit: 1, MaxBytes: 1})
			return err
		},
	} {
		t.Run("canceled "+name, func(t *testing.T) {
			if err := operation(); !errors.Is(err, context.Canceled) {
				t.Fatalf("error = %v", err)
			}
		})
	}
	if _, err := store.Read(context.Background(), "", "main", memorycore.ReadOptions{MaxBytes: 1}); err == nil {
		t.Fatal("invalid read scope succeeded")
	}
	if _, err := store.Read(context.Background(), "scope", "../bad", memorycore.ReadOptions{MaxBytes: 1}); err == nil {
		t.Fatal("invalid read path succeeded")
	}
	if _, err := store.List(context.Background(), "", memorycore.ListOptions{Limit: 1}); err == nil {
		t.Fatal("invalid list scope succeeded")
	}
	if _, err := store.List(context.Background(), "scope", memorycore.ListOptions{}); err == nil {
		t.Fatal("invalid list options succeeded")
	}
	if _, err := store.Mutate(context.Background(), "", memorycore.Mutation{}); err == nil {
		t.Fatal("invalid mutation scope succeeded")
	}
	if _, err := store.Search(context.Background(), "", memorycore.SearchOptions{}); err == nil {
		t.Fatal("invalid search scope succeeded")
	}
	paths, err := store.List(context.Background(), "scope", memorycore.ListOptions{Limit: 2})
	if err != nil || len(paths) != 0 {
		t.Fatalf("empty list = %v, %v", paths, err)
	}
	result, err := store.Search(context.Background(), "scope", memorycore.SearchOptions{Query: "x", Limit: 1, MaxBytes: 1})
	if err != nil || len(result.Matches) != 0 {
		t.Fatalf("empty search = %#v, %v", result, err)
	}
}

func TestPrefixPathSearchAndConflictFrontiers(t *testing.T) {
	store := New()
	ctx := context.Background()
	scope := memorycore.Scope("scope")
	create := func(path, content, key string) memorycore.MutationResult {
		t.Helper()
		result, err := store.Mutate(ctx, scope, memorycore.Mutation{
			Path: path, Kind: memorycore.MutationReplace, Content: []byte(content),
			IdempotencyKey: key, Fingerprint: key,
		})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	main := create("notes", "plain", "notes")
	create("notes/a", "first needle", "a")
	create("notes/b", "second needle", "b")
	create("other", "unrelated", "other")

	paths, err := store.List(ctx, scope, memorycore.ListOptions{Prefix: "notes", Limit: 10})
	if err != nil || !reflect.DeepEqual(paths, []string{"notes", "notes/a", "notes/b"}) {
		t.Fatalf("prefix list = %v, %v", paths, err)
	}
	all, err := store.List(ctx, scope, memorycore.ListOptions{Limit: 10})
	if err != nil || len(all) != 4 {
		t.Fatalf("all paths = %v, %v", all, err)
	}
	byPath, err := store.Search(ctx, scope, memorycore.SearchOptions{Query: "notes/a", Limit: 1, MaxBytes: 100})
	if err != nil || len(byPath.Matches) != 1 || byPath.Matches[0].Path != "notes/a" {
		t.Fatalf("path search = %#v, %v", byPath, err)
	}
	limited, err := store.Search(ctx, scope, memorycore.SearchOptions{Query: "needle", Limit: 1, MaxBytes: 100})
	if err != nil || len(limited.Matches) != 1 {
		t.Fatalf("match limit = %#v, %v", limited, err)
	}
	truncated, err := store.Search(ctx, scope, memorycore.SearchOptions{Query: "needle", Limit: 2, MaxBytes: 3})
	if err != nil || len(truncated.Matches) != 1 || len(truncated.Matches[0].Content) != 3 {
		t.Fatalf("byte-limited search = %#v, %v", truncated, err)
	}

	if _, err := store.Mutate(ctx, scope, memorycore.Mutation{
		Path: "notes", Kind: memorycore.MutationReplace, Content: []byte("again"),
		IdempotencyKey: "duplicate", Fingerprint: "duplicate",
	}); !errors.Is(err, memorycore.ErrConflict) {
		t.Fatalf("create conflict = %v", err)
	}
	if _, err := store.Mutate(ctx, scope, memorycore.Mutation{
		Path: "missing", Kind: memorycore.MutationDelete,
		IdempotencyKey: "delete-missing", Fingerprint: "delete-missing",
	}); !errors.Is(err, memorycore.ErrConflict) {
		t.Fatalf("delete missing = %v", err)
	}
	appended, err := store.Mutate(ctx, scope, memorycore.Mutation{
		Path: "fresh", Kind: memorycore.MutationAppend, Content: []byte("new"),
		IdempotencyKey: "append-new", Fingerprint: "append-new",
	})
	if err != nil || appended.Bytes != 3 {
		t.Fatalf("append new = %#v, %v", appended, err)
	}
	if main.Version == "" {
		t.Fatal("version was not generated")
	}
}
