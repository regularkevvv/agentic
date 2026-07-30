// Package artifacttest provides a reusable conformance suite for artifact
// stores.
package artifacttest

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/regularkevvv/agentic/harness/artifact"
)

type Factory func(*testing.T) artifact.Store

func Run(t *testing.T, factory Factory) {
	t.Helper()
	t.Run("idempotent_and_session_scoped", func(t *testing.T) {
		storage := factory(t)
		ctx := context.Background()
		handle, err := storage.Put(ctx, "one", "call-1", []byte("complete"))
		if err != nil {
			t.Fatal(err)
		}
		again, err := storage.Put(ctx, "one", "call-1", []byte("complete"))
		if err != nil || again != handle {
			t.Fatalf("idempotent Put = %q, %v", again, err)
		}
		if _, err := storage.Put(ctx, "one", "call-1", []byte("changed")); !errors.Is(err, artifact.ErrArtifactConflict) {
			t.Fatalf("conflict error = %v", err)
		}
		if _, err := storage.Get(ctx, "two", handle); !errors.Is(err, artifact.ErrArtifactNotFound) {
			t.Fatalf("cross-session Get error = %v", err)
		}
		data, err := storage.Get(ctx, "one", handle)
		if err != nil || string(data) != "complete" {
			t.Fatalf("Get = %q, %v", data, err)
		}
		data[0] = 'X'
		reloaded, _ := storage.Get(ctx, "one", handle)
		if string(reloaded) != "complete" {
			t.Fatal("Get returned aliased bytes")
		}
	})

	t.Run("validation", func(t *testing.T) {
		storage := factory(t)
		if _, err := storage.Put(context.Background(), "../bad", "key", nil); !errors.Is(err, artifact.ErrInvalidSessionID) {
			t.Fatalf("session error = %v", err)
		}
		if _, err := storage.Get(context.Background(), "ok", "path/to/file"); !errors.Is(err, artifact.ErrInvalidHandle) {
			t.Fatalf("handle error = %v", err)
		}
	})

	t.Run("concurrent_idempotence", func(t *testing.T) {
		storage := factory(t)
		const writers = 24
		handles := make(chan artifact.Handle, writers)
		failures := make(chan error, writers)
		var wg sync.WaitGroup
		wg.Add(writers)
		for i := 0; i < writers; i++ {
			go func() {
				defer wg.Done()
				handle, err := storage.Put(context.Background(), "session", "same-call", []byte("complete"))
				if err != nil {
					failures <- err
					return
				}
				handles <- handle
			}()
		}
		wg.Wait()
		close(handles)
		close(failures)
		for err := range failures {
			t.Errorf("concurrent Put: %v", err)
		}
		var expected artifact.Handle
		for handle := range handles {
			if expected == "" {
				expected = handle
			}
			if handle != expected {
				t.Fatalf("concurrent handles = %q and %q", expected, handle)
			}
		}
	})
}
