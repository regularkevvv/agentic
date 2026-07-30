package jsonl

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/harness/store"
)

func TestCreateOpenOwnershipAndFilesystemFailureEdges(t *testing.T) {
	root := t.TempDir()
	repository, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := repository.Open(context.Background(), "bad/session"); err == nil {
		t.Fatal("invalid Open succeeded")
	}
	journal, _, err := repository.Create(context.Background(), "open", syncEntry("created", nil))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = journal.Close(context.Background()) }()
	if _, _, err := repository.Create(context.Background(), "open"); !errors.Is(err, store.ErrSessionOpen) {
		t.Fatalf("in-process duplicate Create = %v", err)
	}

	second, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := second.Create(context.Background(), "open"); !errors.Is(err, store.ErrSessionOpen) {
		t.Fatalf("cross-process-style duplicate Create = %v", err)
	}

	if err := os.WriteFile(filepath.Join(root, "exists.jsonl"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := repository.Create(context.Background(), "exists"); !errors.Is(err, store.ErrSessionExists) {
		t.Fatalf("existing journal Create = %v", err)
	}

	if err := os.Mkdir(filepath.Join(root, "badlock.lock"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, _, err := repository.Create(context.Background(), "badlock"); err == nil ||
		errors.Is(err, store.ErrSessionOpen) {
		t.Fatalf("non-lock Create error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "badopen.jsonl"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, "badopen.lock"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := repository.Open(context.Background(), "badopen"); err == nil ||
		errors.Is(err, store.ErrSessionOpen) {
		t.Fatalf("non-lock Open error = %v", err)
	}
}

func TestJournalReadAppendCloseAndWriteFailureEdges(t *testing.T) {
	root := t.TempDir()
	repository, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	journal, commit, err := repository.Create(context.Background(), "append", syncEntry("created", nil))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "append.jsonl")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, err := journal.Append(context.Background(), commit.Cursor, syncEntry("next", nil)); !errors.Is(err, store.ErrSessionNotFound) {
		t.Fatalf("append after removal = %v", err)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}

	directoryID := "directory"
	if err := os.Mkdir(filepath.Join(root, directoryID+".jsonl"), 0o700); err != nil {
		t.Fatal(err)
	}
	direct := &Journal{repository: repository, sessionID: directoryID}
	if _, err := direct.Load(context.Background()); err == nil ||
		errors.Is(err, store.ErrSessionNotFound) {
		t.Fatalf("directory load error = %v", err)
	}

	nilLease := &Journal{
		repository: repository,
		sessionID:  "nil-lease",
	}
	repository.open[nilLease.sessionID] = nilLease
	if err := nilLease.Close(context.Background()); err != nil {
		t.Fatalf("nil lease close = %v", err)
	}

	file, err := os.CreateTemp(t.TempDir(), "closed")
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if err := writeEntries(file, nil); err != nil {
		t.Fatalf("empty write = %v", err)
	}
	if err := writeEntries(file, []store.Entry{{Schema: 1, Seq: 1, ID: "id", Kind: "kind"}}); err == nil {
		t.Fatal("write to closed file succeeded")
	}
}

func TestPartialRecoverySidecarAndJournalOpenFailures(t *testing.T) {
	repository := &Repository{root: filepath.Join(t.TempDir(), "missing"), open: make(map[string]*Journal)}
	journal := &Journal{repository: repository, sessionID: "session"}
	if _, err := journal.recoverPartial(
		filepath.Join(repository.root, "session.jsonl"),
		[]byte("partial"),
	); err == nil || !strings.Contains(err.Error(), "preserve partial") {
		t.Fatalf("sidecar write error = %v", err)
	}

	root := t.TempDir()
	repository = &Repository{root: root, open: make(map[string]*Journal)}
	path := filepath.Join(root, "journal-directory")
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	journal = &Journal{repository: repository, sessionID: "session"}
	if _, err := journal.recoverPartial(path, []byte("partial")); err == nil ||
		!strings.Contains(err.Error(), "tail recovery") {
		t.Fatalf("journal tail open error = %v", err)
	}
}

func TestMalformedLogAndPureJournalHelperEdges(t *testing.T) {
	root := t.TempDir()
	repository, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "corrupt.jsonl"), []byte("{\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := repository.Open(context.Background(), "corrupt"); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("malformed log Open = %v", err)
	}
	if cursor := tail(nil); !cursor.Equal(store.Cursor{}) {
		t.Fatalf("empty tail = %#v", cursor)
	}
	if _, err := sequence(store.Cursor{}, []store.PendingEntry{{
		Kind:       "kind",
		Durability: store.DurabilitySync + 1,
	}}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("invalid durability sequence = %v", err)
	}
	if err := (*fileLock)(nil).Close(); err != nil {
		t.Fatalf("nil file lock close = %v", err)
	}
	if runtime.GOOS != "windows" {
		if err := syncDirectory("/dev/null"); err == nil {
			t.Fatal("syncing /dev/null as a journal directory succeeded")
		}
	}
}

func TestCreateReportsJournalPermissionFailureAfterLock(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix directory permissions required")
	}
	root := t.TempDir()
	repository, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(repository.lockPath("permission"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(root, 0o500); err != nil {
		t.Fatal(err)
	}
	_, _, createErr := repository.Create(context.Background(), "permission")
	if err := os.Chmod(root, 0o700); err != nil {
		t.Fatal(err)
	}
	if createErr == nil {
		t.Skip("current process can create files in a mode-0500 directory")
	}
	if errors.Is(createErr, store.ErrSessionOpen) || errors.Is(createErr, store.ErrSessionExists) {
		t.Fatalf("permission failure misclassified = %v", createErr)
	}
}
