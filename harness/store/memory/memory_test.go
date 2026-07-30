package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/harness/store"
	"github.com/regularkevvv/agentic/harness/store/storetest"
)

func TestRepositoryConformance(t *testing.T) {
	t.Parallel()
	storetest.Run(t, func(*testing.T) store.Repository { return New() })
}

func TestRepositoryValidationCancellationAndCorruptPending(t *testing.T) {
	t.Parallel()
	repository := New()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := repository.Create(ctx, "session"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Create = %v", err)
	}
	if _, _, err := repository.Create(context.Background(), "bad/session"); err == nil {
		t.Fatal("invalid Create succeeded")
	}
	if _, _, err := repository.Create(context.Background(), "empty", store.PendingEntry{}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("empty kind = %v", err)
	}
	if _, _, err := repository.Create(context.Background(), "durability", store.PendingEntry{
		Kind:       "kind",
		Durability: store.DurabilitySync + 1,
	}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("invalid durability = %v", err)
	}
	journal, commit, err := repository.Create(context.Background(), "session", store.PendingEntry{Kind: "created"})
	if err != nil {
		t.Fatal(err)
	}
	if journal.SessionID() != "session" {
		t.Fatalf("session ID = %q", journal.SessionID())
	}
	if _, err := repository.Open(ctx, "session"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Open = %v", err)
	}
	if _, err := journal.Load(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Load = %v", err)
	}
	if _, err := journal.Append(ctx, commit.Cursor, store.PendingEntry{Kind: "next"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Append = %v", err)
	}
	if _, err := journal.Append(context.Background(), commit.Cursor, store.PendingEntry{}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("corrupt Append = %v", err)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := journal.Load(context.Background()); !errors.Is(err, store.ErrJournalClosed) {
		t.Fatalf("closed Load = %v", err)
	}
	if _, err := journal.Append(context.Background(), commit.Cursor, store.PendingEntry{Kind: "next"}); !errors.Is(err, store.ErrJournalClosed) {
		t.Fatalf("closed Append = %v", err)
	}
}
