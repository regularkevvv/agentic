package jsonl

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/regularkevvv/agentic/harness/store"
	"github.com/regularkevvv/agentic/harness/store/storetest"
)

func TestRepositoryConformance(t *testing.T) {
	t.Parallel()
	storetest.Run(t, func(t *testing.T) store.Repository {
		repository, err := New(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		return repository
	})
}

func TestRecoversOnlyTrailingPartialLine(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	root := t.TempDir()
	repository, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	journal, created, err := repository.Create(ctx, "recover", syncEntry("created", nil))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = journal.Close(context.Background()) })
	appended, err := journal.Append(ctx, created.Cursor, syncEntry("message", []byte("complete")))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "recover.jsonl")
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString(`{"schema":1,"seq":3`); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	loaded, err := journal.Load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Entries) != 2 || !loaded.Cursor.Equal(appended.Cursor) {
		t.Fatalf("loaded = %#v", loaded)
	}
	sidecars, err := filepath.Glob(path + ".partial-*")
	if err != nil || len(sidecars) != 1 {
		t.Fatalf("sidecars = %v, %v", sidecars, err)
	}
	partial, err := os.ReadFile(sidecars[0])
	if err != nil {
		t.Fatal(err)
	}
	if string(partial) != `{"schema":1,"seq":3` {
		t.Fatalf("partial = %q", partial)
	}
	after, err := journal.Append(ctx, loaded.Cursor, syncEntry("message", []byte("after")))
	if err != nil || after.Cursor.Seq != 3 {
		t.Fatalf("post-recovery append = %#v, %v", after, err)
	}
}

func TestRejectsCorruptCompleteLine(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	root := t.TempDir()
	repository, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	journal, _, err := repository.Create(ctx, "corrupt", syncEntry("created", nil))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = journal.Close(context.Background()) })
	path := filepath.Join(root, "corrupt.jsonl")
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString("not-json\n"); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := journal.Load(ctx); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("Load error = %v", err)
	}
	if matches, _ := filepath.Glob(path + ".partial-*"); len(matches) != 0 {
		t.Fatalf("complete corruption created sidecars: %v", matches)
	}
}

func TestIndependentRepositoriesRefuseSecondOwner(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	first, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	second, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	journal, _, err := first.Create(context.Background(), "owned", syncEntry("created", nil))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = journal.Close(context.Background()) }()
	if _, err := second.Open(context.Background(), "owned"); !errors.Is(err, store.ErrSessionOpen) {
		t.Fatalf("second owner error = %v", err)
	}
}

func syncEntry(kind string, payload []byte) store.PendingEntry {
	return store.PendingEntry{Kind: kind, Payload: payload, Durability: store.DurabilitySync}
}
