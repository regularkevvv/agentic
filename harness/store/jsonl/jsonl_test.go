package jsonl

import (
	"context"
	"encoding/json"
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

func TestRepositoryValidationCancellationClosedJournalAndHelpers(t *testing.T) {
	t.Parallel()
	if _, err := New(""); err == nil {
		t.Fatal("empty root succeeded")
	}
	parent := t.TempDir()
	rootFile := filepath.Join(parent, "root-file")
	if err := os.WriteFile(rootFile, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(rootFile); err == nil {
		t.Fatal("file root succeeded")
	}
	repository, err := New(filepath.Join(parent, "journals"))
	if err != nil {
		t.Fatal(err)
	}
	if repository.Root() == "" {
		t.Fatal("canonical Root is empty")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := repository.Create(ctx, "session"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Create = %v", err)
	}
	if _, _, err := repository.Create(context.Background(), "bad/session"); err == nil {
		t.Fatal("invalid Create succeeded")
	}
	if _, _, err := repository.Create(context.Background(), "empty", store.PendingEntry{}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("empty pending = %v", err)
	}
	if _, _, err := repository.Create(context.Background(), "durability", store.PendingEntry{
		Kind:       "kind",
		Durability: store.DurabilitySync + 1,
	}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("bad durability = %v", err)
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
	if _, err := repository.Open(context.Background(), "missing"); !errors.Is(err, store.ErrSessionNotFound) {
		t.Fatalf("missing Open = %v", err)
	}
	if !tail(nil).IsZero() || anySync(nil) || !anySync([]store.Entry{{Durability: store.DurabilitySync}}) {
		t.Fatal("journal helper result changed")
	}
}

func TestRejectsEveryInvalidCompleteChainShape(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	repository, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	base := diskEntry{Schema: store.CurrentSchema, Seq: 1, ID: "one", Kind: "kind"}
	cases := map[string]diskEntry{
		"schema":     func() diskEntry { value := base; value.Schema = 2; return value }(),
		"sequence":   func() diskEntry { value := base; value.Seq = 2; return value }(),
		"id":         func() diskEntry { value := base; value.ID = ""; return value }(),
		"kind":       func() diskEntry { value := base; value.Kind = ""; return value }(),
		"parent":     func() diskEntry { value := base; value.ParentID = "wrong"; return value }(),
		"durability": func() diskEntry { value := base; value.Durability = store.DurabilitySync + 1; return value }(),
	}
	for name, entry := range cases {
		entry := entry
		t.Run(name, func(t *testing.T) {
			encoded, err := json.Marshal(entry)
			if err != nil {
				t.Fatal(err)
			}
			sessionID := "invalid_" + name
			if err := os.WriteFile(filepath.Join(root, sessionID+".jsonl"), append(encoded, '\n'), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := repository.Open(context.Background(), sessionID); !errors.Is(err, store.ErrCorruptLog) {
				t.Fatalf("invalid chain = %v", err)
			}
		})
	}
	emptyID := "empty_log"
	if err := os.WriteFile(filepath.Join(root, emptyID+".jsonl"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	journal, err := repository.Open(context.Background(), emptyID)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := journal.Load(context.Background())
	if err != nil || len(loaded.Entries) != 0 || !loaded.Cursor.IsZero() {
		t.Fatalf("empty load = %#v, %v", loaded, err)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestLockAndDirectoryHelpers(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if _, err := acquireFileLock(filepath.Join(root, "missing", "lock")); err == nil {
		t.Fatal("lock in missing directory succeeded")
	}
	lock, err := acquireFileLock(filepath.Join(root, "lock"))
	if err != nil {
		t.Fatal(err)
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	var nilLock *fileLock
	if err := nilLock.Close(); err != nil {
		t.Fatal(err)
	}
	if err := syncDirectory(filepath.Join(root, "missing")); err == nil {
		t.Fatal("sync missing directory succeeded")
	}
	file := filepath.Join(root, "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := syncDirectory(file); err != nil {
		t.Fatalf("sync file = %v", err)
	}
}
