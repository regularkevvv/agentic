// Package memorytest provides the shared adapter conformance suite.
package memorytest

import (
	"context"
	"errors"
	"reflect"
	"testing"

	memorycore "github.com/regularkevvv/agentic/harness/memory"
)

type Store interface {
	memorycore.Store
	memorycore.Searcher
}

func Run(t *testing.T, factory func(*testing.T) Store) {
	t.Helper()
	t.Run("CAS-idempotency-and-bounds", func(t *testing.T) {
		store := factory(t)
		ctx := context.Background()
		scope := memorycore.Scope("tenant-a")
		created, err := store.Mutate(ctx, scope, memorycore.Mutation{
			Path: "notes/main", Kind: memorycore.MutationReplace, Content: []byte("alpha"),
			IdempotencyKey: "create", Fingerprint: "fp-create",
		})
		if err != nil || created.Version == "" || created.Bytes != 5 {
			t.Fatalf("create = %#v, %v", created, err)
		}
		replayed, err := store.Mutate(ctx, scope, memorycore.Mutation{
			Path: "notes/main", Kind: memorycore.MutationReplace, Content: []byte("alpha"),
			IdempotencyKey: "create", Fingerprint: "fp-create",
		})
		if err != nil || !reflect.DeepEqual(replayed, created) {
			t.Fatalf("replay = %#v, %v", replayed, err)
		}
		_, err = store.Mutate(ctx, scope, memorycore.Mutation{
			Path: "notes/main", Kind: memorycore.MutationReplace, Content: []byte("other"),
			IdempotencyKey: "create", Fingerprint: "different",
		})
		if !errors.Is(err, memorycore.ErrIdempotencyConflict) {
			t.Fatalf("idempotency conflict = %v", err)
		}
		_, err = store.Mutate(ctx, scope, memorycore.Mutation{
			Path: "notes/main", Kind: memorycore.MutationAppend, Content: []byte("!"),
			ExpectedVersion: "stale", IdempotencyKey: "stale", Fingerprint: "fp-stale",
		})
		if !errors.Is(err, memorycore.ErrConflict) {
			t.Fatalf("stale CAS = %v", err)
		}
		appended, err := store.Mutate(ctx, scope, memorycore.Mutation{
			Path: "notes/main", Kind: memorycore.MutationAppend, Content: []byte(" beta"),
			ExpectedVersion: created.Version, IdempotencyKey: "append", Fingerprint: "fp-append",
		})
		if err != nil || appended.Version == created.Version || appended.Bytes != 10 {
			t.Fatalf("append = %#v, %v", appended, err)
		}
		file, err := store.Read(ctx, scope, "notes/main", memorycore.ReadOptions{MaxBytes: 10})
		if err != nil || string(file.Content) != "alpha beta" || file.Version != appended.Version {
			t.Fatalf("read = %#v, %v", file, err)
		}
		file.Content[0] = 'X'
		again, _ := store.Read(ctx, scope, "notes/main", memorycore.ReadOptions{MaxBytes: 10})
		if string(again.Content) != "alpha beta" {
			t.Fatal("read leaked mutable content")
		}
		if _, err := store.Read(ctx, scope, "notes/main", memorycore.ReadOptions{MaxBytes: 9}); !errors.Is(err, memorycore.ErrLimitExceeded) {
			t.Fatalf("read bound = %v", err)
		}
		if _, err := store.Read(ctx, memorycore.Scope("tenant-b"), "notes/main", memorycore.ReadOptions{MaxBytes: 20}); !errors.Is(err, memorycore.ErrNotFound) {
			t.Fatalf("scope isolation = %v", err)
		}
	})

	t.Run("list-search-delete", func(t *testing.T) {
		store := factory(t)
		ctx := context.Background()
		scope := memorycore.Scope("tenant-list")
		for index, value := range []struct{ path, content string }{
			{"zeta", "last"}, {"notes/b", "needle second"}, {"notes/a", "needle first"},
		} {
			_, err := store.Mutate(ctx, scope, memorycore.Mutation{
				Path: value.path, Kind: memorycore.MutationReplace, Content: []byte(value.content),
				IdempotencyKey: value.path, Fingerprint: value.path,
			})
			if err != nil {
				t.Fatalf("create %d: %v", index, err)
			}
		}
		paths, err := store.List(ctx, scope, memorycore.ListOptions{Prefix: "notes", Limit: 1})
		if err != nil || !reflect.DeepEqual(paths, []string{"notes/a"}) {
			t.Fatalf("list = %v, %v", paths, err)
		}
		search, err := store.Search(ctx, scope, memorycore.SearchOptions{Query: "needle", Limit: 2, MaxBytes: 12})
		if err != nil || len(search.Matches) != 1 || search.Matches[0].Path != "notes/a" || len(search.Matches[0].Content) != 12 {
			t.Fatalf("search = %#v, %v", search, err)
		}
		file, _ := store.Read(ctx, scope, "notes/a", memorycore.ReadOptions{MaxBytes: 100})
		deleted, err := store.Mutate(ctx, scope, memorycore.Mutation{
			Path: "notes/a", Kind: memorycore.MutationDelete, ExpectedVersion: file.Version,
			IdempotencyKey: "delete", Fingerprint: "fp-delete",
		})
		if err != nil || !deleted.Deleted {
			t.Fatalf("delete = %#v, %v", deleted, err)
		}
		if _, err := store.Read(ctx, scope, "notes/a", memorycore.ReadOptions{MaxBytes: 100}); !errors.Is(err, memorycore.ErrNotFound) {
			t.Fatalf("deleted read = %v", err)
		}
	})

	t.Run("cancellation-and-validation", func(t *testing.T) {
		store := factory(t)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := store.List(ctx, "scope", memorycore.ListOptions{Limit: 1}); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled list = %v", err)
		}
		if _, err := store.Search(context.Background(), "scope", memorycore.SearchOptions{}); err == nil {
			t.Fatal("invalid search succeeded")
		}
		if _, err := store.Mutate(context.Background(), "scope", memorycore.Mutation{}); err == nil {
			t.Fatal("invalid mutation succeeded")
		}
	})
}
