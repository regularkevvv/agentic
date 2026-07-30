package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/harness/store"
)

func TestRepositoryDuplicateInvalidOpenAndEmptyTail(t *testing.T) {
	repository := New()
	journal, _, err := repository.Create(context.Background(), "session")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = journal.Close(context.Background()) })

	if _, _, err := repository.Create(context.Background(), "session"); !errors.Is(err, store.ErrSessionExists) {
		t.Fatalf("duplicate Create = %v", err)
	}
	if _, err := repository.Open(context.Background(), "bad/session"); err == nil {
		t.Fatal("invalid Open succeeded")
	}
	if cursor := tail(nil); !cursor.Equal(store.Cursor{}) {
		t.Fatalf("empty tail = %#v", cursor)
	}
}
