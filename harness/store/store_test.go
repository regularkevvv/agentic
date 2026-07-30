package store

import (
	"errors"
	"strings"
	"testing"
)

func TestCursorValidationClonesConflictAndCommit(t *testing.T) {
	t.Parallel()
	zero := Cursor{}
	if !zero.IsZero() || !zero.Equal(Cursor{}) || zero.Equal(Cursor{Seq: 1}) {
		t.Fatal("cursor equality changed")
	}
	entry := Entry{Seq: 1, ID: "entry", Payload: []byte("value")}
	if !entry.Cursor().Equal(Cursor{Seq: 1, EntryID: "entry"}) {
		t.Fatalf("entry cursor = %#v", entry.Cursor())
	}
	conflict := &ConflictError{Expected: Cursor{Seq: 1, EntryID: "a"}, Actual: Cursor{Seq: 2, EntryID: "b"}}
	if !errors.Is(conflict, ErrConflict) || !strings.Contains(conflict.Error(), "expected (1") {
		t.Fatalf("conflict = %v", conflict)
	}
	for _, id := range []string{"", "bad/session", strings.Repeat("a", 129)} {
		if err := ValidateSessionID(id); !errors.Is(err, ErrInvalidSessionID) {
			t.Fatalf("session %q = %v", id, err)
		}
	}
	if err := ValidateSessionID("Valid_session-1"); err != nil {
		t.Fatal(err)
	}

	pending := []PendingEntry{{Kind: "kind", Payload: []byte("pending")}}
	clonedPending := ClonePending(pending)
	pending[0].Payload[0] = 'X'
	if string(clonedPending[0].Payload) != "pending" || ClonePending(nil) != nil {
		t.Fatalf("pending clone = %#v", clonedPending)
	}
	entries := []Entry{entry}
	clonedEntries := CloneEntries(entries)
	entries[0].Payload[0] = 'X'
	if string(clonedEntries[0].Payload) != "value" || CloneEntries(nil) != nil {
		t.Fatalf("entry clone = %#v", clonedEntries)
	}
	fallback := Cursor{Seq: 9, EntryID: "fallback"}
	empty := NewCommit(nil, fallback)
	if !empty.Cursor.Equal(fallback) || empty.Entries != nil {
		t.Fatalf("empty commit = %#v", empty)
	}
	commit := NewCommit(clonedEntries, fallback)
	if !commit.Cursor.Equal(entry.Cursor()) || len(commit.Entries) != 1 {
		t.Fatalf("commit = %#v", commit)
	}
}
