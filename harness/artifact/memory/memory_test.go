package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/harness/artifact"
	"github.com/regularkevvv/agentic/harness/artifact/artifacttest"
)

func TestStoreConformance(t *testing.T) {
	t.Parallel()
	artifacttest.Run(t, func(*testing.T) artifact.Store { return New() })
}

func TestStoreValidationCancellationAndDefensiveCopies(t *testing.T) {
	t.Parallel()
	store := New()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Put(ctx, "session", "key", nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Put = %v", err)
	}
	if _, err := store.Put(context.Background(), "bad/session", "key", nil); err == nil {
		t.Fatal("invalid session Put succeeded")
	}
	if _, err := store.Put(context.Background(), "session", "", nil); err == nil {
		t.Fatal("empty key Put succeeded")
	}
	data := []byte("value")
	handle, err := store.Put(context.Background(), "session", "key", data)
	if err != nil {
		t.Fatal(err)
	}
	data[0] = 'X'
	got, err := store.Get(context.Background(), "session", handle)
	if err != nil || string(got) != "value" {
		t.Fatalf("stored data = %q, %v", got, err)
	}
	got[0] = 'X'
	again, _ := store.Get(context.Background(), "session", handle)
	if string(again) != "value" {
		t.Fatal("Get exposed mutable storage")
	}
	if _, err := store.Put(context.Background(), "session", "key", []byte("conflict")); !errors.Is(err, artifact.ErrArtifactConflict) {
		t.Fatalf("conflict = %v", err)
	}
	if _, err := store.Get(ctx, "session", handle); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Get = %v", err)
	}
	if _, err := store.Get(context.Background(), "bad/session", handle); err == nil {
		t.Fatal("invalid session Get succeeded")
	}
	if _, err := store.Get(context.Background(), "session", "bad"); err == nil {
		t.Fatal("invalid handle Get succeeded")
	}
	if store.Count("session") != 1 {
		t.Fatalf("Count = %d", store.Count("session"))
	}
}
