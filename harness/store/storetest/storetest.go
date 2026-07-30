// Package storetest provides a reusable conformance suite for journal adapters.
package storetest

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/regularkevvv/agentic/harness/store"
)

type Factory func(*testing.T) store.Repository

func Run(t *testing.T, factory Factory) {
	t.Helper()
	t.Run("sequence_clone_and_conflict", func(t *testing.T) {
		repository := factory(t)
		journal, created, err := repository.Create(context.Background(), "session_1", pending("created", "one"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = journal.Close(context.Background()) })
		if len(created.Entries) != 1 || created.Cursor.Seq != 1 || created.Entries[0].ParentID != "" || created.Cursor.EntryID == "" {
			t.Fatalf("creation commit = %#v", created)
		}
		created.Entries[0].Payload[0] = 'X'

		appended, err := journal.Append(context.Background(), created.Cursor, pending("one", "1"), pending("two", "2"))
		if err != nil {
			t.Fatal(err)
		}
		if appended.Cursor.Seq != 3 || appended.Entries[0].ParentID != created.Cursor.EntryID ||
			appended.Entries[1].ParentID != appended.Entries[0].ID {
			t.Fatalf("append commit = %#v", appended)
		}
		if _, err := journal.Append(context.Background(), created.Cursor, pending("stale", "x")); !errors.Is(err, store.ErrConflict) {
			t.Fatalf("stale append error = %v", err)
		}
		loaded, err := journal.Load(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if len(loaded.Entries) != 3 || string(loaded.Entries[0].Payload) != "one" {
			t.Fatalf("loaded = %#v", loaded)
		}
		loaded.Entries[0].Payload[0] = 'X'
		reloaded, _ := journal.Load(context.Background())
		if string(reloaded.Entries[0].Payload) != "one" {
			t.Fatal("Load returned aliased payload storage")
		}
	})

	t.Run("exclusive_lease_and_reopen", func(t *testing.T) {
		repository := factory(t)
		journal, _, err := repository.Create(context.Background(), "lease", pending("created", ""))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := repository.Open(context.Background(), "lease"); !errors.Is(err, store.ErrSessionOpen) {
			t.Fatalf("second Open error = %v", err)
		}
		if err := journal.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
		if err := journal.Close(context.Background()); err != nil {
			t.Fatalf("second Close: %v", err)
		}
		reopened, err := repository.Open(context.Background(), "lease")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := journal.Load(context.Background()); !errors.Is(err, store.ErrJournalClosed) {
			t.Fatalf("closed journal error = %v", err)
		}
		if err := reopened.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("validation_and_cancellation", func(t *testing.T) {
		repository := factory(t)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, _, err := repository.Create(ctx, "cancelled", pending("created", "")); !errors.Is(err, context.Canceled) {
			t.Fatalf("Create error = %v", err)
		}
		if _, _, err := repository.Create(context.Background(), "../escape"); !errors.Is(err, store.ErrInvalidSessionID) {
			t.Fatalf("invalid ID error = %v", err)
		}
		if _, err := repository.Open(context.Background(), "missing"); !errors.Is(err, store.ErrSessionNotFound) {
			t.Fatalf("missing error = %v", err)
		}
		journal, commit, err := repository.Create(context.Background(), "invalid_entry", pending("created", ""))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = journal.Close(context.Background()) })
		if _, err := journal.Append(context.Background(), commit.Cursor, store.PendingEntry{}); !errors.Is(err, store.ErrCorruptLog) {
			t.Fatalf("empty kind error = %v", err)
		}
	})

	t.Run("serialized_single_writer", func(t *testing.T) {
		repository := factory(t)
		journal, commit, err := repository.Create(context.Background(), "race", pending("created", ""))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = journal.Close(context.Background()) })
		const writers = 24
		cursor := commit.Cursor
		var client sync.Mutex
		var wg sync.WaitGroup
		wg.Add(writers)
		for i := 0; i < writers; i++ {
			go func(value int) {
				defer wg.Done()
				client.Lock()
				defer client.Unlock()
				next, appendErr := journal.Append(context.Background(), cursor, pending("message", fmt.Sprint(value)))
				if appendErr != nil {
					t.Errorf("append %d: %v", value, appendErr)
					return
				}
				cursor = next.Cursor
			}(i)
		}
		wg.Wait()
		loaded, err := journal.Load(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if len(loaded.Entries) != writers+1 || loaded.Cursor.Seq != writers+1 {
			t.Fatalf("loaded %d entries at %d", len(loaded.Entries), loaded.Cursor.Seq)
		}
	})

	t.Run("concurrent_stale_writers_conflict", func(t *testing.T) {
		repository := factory(t)
		journal, commit, err := repository.Create(context.Background(), "contended", pending("created", ""))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = journal.Close(context.Background()) })
		const writers = 24
		start := make(chan struct{})
		results := make(chan error, writers)
		var wg sync.WaitGroup
		wg.Add(writers)
		for i := 0; i < writers; i++ {
			go func(value int) {
				defer wg.Done()
				<-start
				_, appendErr := journal.Append(context.Background(), commit.Cursor, pending("message", fmt.Sprint(value)))
				results <- appendErr
			}(i)
		}
		close(start)
		wg.Wait()
		close(results)

		successes, conflicts := 0, 0
		for appendErr := range results {
			switch {
			case appendErr == nil:
				successes++
			case errors.Is(appendErr, store.ErrConflict):
				conflicts++
			default:
				t.Fatalf("contended append error = %v", appendErr)
			}
		}
		if successes != 1 || conflicts != writers-1 {
			t.Fatalf("contended appends succeeded=%d conflicted=%d", successes, conflicts)
		}
		loaded, err := journal.Load(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if len(loaded.Entries) != 2 || loaded.Cursor.Seq != 2 {
			t.Fatalf("contended journal = %#v", loaded)
		}
	})
}

func pending(kind, payload string) store.PendingEntry {
	return store.PendingEntry{Kind: kind, Payload: []byte(payload), Durability: store.DurabilitySync}
}
