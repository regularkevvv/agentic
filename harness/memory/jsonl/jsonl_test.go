package jsonl

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	memorycore "github.com/regularkevvv/agentic/harness/memory"
	"github.com/regularkevvv/agentic/harness/memory/memorytest"
)

func TestConformance(t *testing.T) {
	memorytest.Run(t, func(t *testing.T) memorytest.Store {
		store, err := New(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		return store
	})
}

func TestRestartTailRepairAndCorruption(t *testing.T) {
	root := t.TempDir()
	store, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	scope := memorycore.Scope("restart")
	created, err := store.Mutate(context.Background(), scope, memorycore.Mutation{
		Path: "main", Kind: memorycore.MutationReplace, Content: []byte("durable"),
		IdempotencyKey: "one", Fingerprint: "one",
	})
	if err != nil {
		t.Fatal(err)
	}
	path := store.path(scope)
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString(`{"partial":`); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	reopened, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := reopened.Read(context.Background(), scope, "main", memorycore.ReadOptions{MaxBytes: 100})
	if err != nil || string(loaded.Content) != "durable" || loaded.Version != created.Version {
		t.Fatalf("recovered = %#v, %v", loaded, err)
	}
	matches, _ := filepath.Glob(path + ".partial-*")
	if len(matches) != 1 {
		t.Fatalf("diagnostic sidecars = %v", matches)
	}
	data, _ := os.ReadFile(path)
	if !strings.HasSuffix(string(data), "\n") || strings.Contains(string(data), "partial") {
		t.Fatalf("repaired log = %q", data)
	}
	if err := os.WriteFile(path, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := reopened.List(context.Background(), scope, memorycore.ListOptions{Limit: 1}); !errors.Is(err, ErrCorruptLog) {
		t.Fatalf("complete corruption = %v", err)
	}
}

func TestCrossInstanceCASSerializes(t *testing.T) {
	root := t.TempDir()
	first, _ := New(root)
	second, _ := New(root)
	scope := memorycore.Scope("race")
	stores := []*Store{first, second}
	errorsSeen := make(chan error, 2)
	var wait sync.WaitGroup
	for index, store := range stores {
		wait.Add(1)
		go func(index int, store *Store) {
			defer wait.Done()
			_, err := store.Mutate(context.Background(), scope, memorycore.Mutation{
				Path: "main", Kind: memorycore.MutationReplace, Content: []byte{byte('a' + index)},
				IdempotencyKey: string(rune('a' + index)), Fingerprint: string(rune('a' + index)),
			})
			errorsSeen <- err
		}(index, store)
	}
	wait.Wait()
	close(errorsSeen)
	successes, conflicts := 0, 0
	for err := range errorsSeen {
		if err == nil {
			successes++
		} else if errors.Is(err, memorycore.ErrConflict) {
			conflicts++
		} else {
			t.Fatal(err)
		}
	}
	if successes != 1 || conflicts != 1 {
		t.Fatalf("successes=%d conflicts=%d", successes, conflicts)
	}
}

func TestValidationCancellationQueriesAndConflicts(t *testing.T) {
	if _, err := New(""); err == nil {
		t.Fatal("empty root succeeded")
	}
	parent := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(parent, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(filepath.Join(parent, "child")); err == nil {
		t.Fatal("root below file succeeded")
	}
	store, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if store.Root() == "" || !filepath.IsAbs(store.Root()) {
		t.Fatalf("root = %q", store.Root())
	}
	ctx := context.Background()
	if _, err := store.Read(ctx, "", "main", memorycore.ReadOptions{MaxBytes: 1}); err == nil {
		t.Fatal("invalid read scope succeeded")
	}
	if _, err := store.Read(ctx, "scope", "../bad", memorycore.ReadOptions{MaxBytes: 1}); err == nil {
		t.Fatal("invalid read path succeeded")
	}
	if _, err := store.List(ctx, "", memorycore.ListOptions{Limit: 1}); err == nil {
		t.Fatal("invalid list scope succeeded")
	}
	if _, err := store.List(ctx, "scope", memorycore.ListOptions{}); err == nil {
		t.Fatal("invalid list options succeeded")
	}
	if _, err := store.Search(ctx, "", memorycore.SearchOptions{}); err == nil {
		t.Fatal("invalid search scope succeeded")
	}
	if _, err := store.Search(ctx, "scope", memorycore.SearchOptions{}); err == nil {
		t.Fatal("invalid search options succeeded")
	}
	if _, err := store.Mutate(ctx, "", memorycore.Mutation{}); err == nil {
		t.Fatal("invalid mutation scope succeeded")
	}
	if _, err := store.Mutate(ctx, "scope", memorycore.Mutation{}); err == nil {
		t.Fatal("invalid mutation succeeded")
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := store.List(canceled, "scope", memorycore.ListOptions{Limit: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled list = %v", err)
	}
	if _, err := store.Search(canceled, "scope", memorycore.SearchOptions{Query: "x", Limit: 1, MaxBytes: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled search = %v", err)
	}
	if _, err := store.Read(canceled, "scope", "main", memorycore.ReadOptions{MaxBytes: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled read = %v", err)
	}

	scope := memorycore.Scope("queries")
	create := func(path, content, key string) memorycore.MutationResult {
		t.Helper()
		result, err := store.Mutate(ctx, scope, memorycore.Mutation{
			Path: path, Kind: memorycore.MutationReplace, Content: []byte(content),
			IdempotencyKey: key, Fingerprint: key,
		})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	main := create("notes", "plain", "notes")
	create("notes/a", "first needle", "a")
	create("notes/b", "second needle", "b")
	create("other", "unrelated", "other")
	paths, err := store.List(ctx, scope, memorycore.ListOptions{Prefix: "notes", Limit: 10})
	if err != nil || !reflect.DeepEqual(paths, []string{"notes", "notes/a", "notes/b"}) {
		t.Fatalf("prefix list = %v, %v", paths, err)
	}
	all, err := store.List(ctx, scope, memorycore.ListOptions{Limit: 10})
	if err != nil || len(all) != 4 {
		t.Fatalf("all list = %v, %v", all, err)
	}
	match, err := store.Search(ctx, scope, memorycore.SearchOptions{Query: "notes/a", Limit: 1, MaxBytes: 100})
	if err != nil || len(match.Matches) != 1 || match.Matches[0].Path != "notes/a" {
		t.Fatalf("path search = %#v, %v", match, err)
	}
	limited, err := store.Search(ctx, scope, memorycore.SearchOptions{Query: "needle", Limit: 1, MaxBytes: 100})
	if err != nil || len(limited.Matches) != 1 {
		t.Fatalf("limited search = %#v, %v", limited, err)
	}
	truncated, err := store.Search(ctx, scope, memorycore.SearchOptions{Query: "needle", Limit: 2, MaxBytes: 3})
	if err != nil || len(truncated.Matches) != 1 || len(truncated.Matches[0].Content) != 3 {
		t.Fatalf("byte-limited search = %#v, %v", truncated, err)
	}
	if _, err := store.Read(ctx, scope, "missing", memorycore.ReadOptions{MaxBytes: 1}); !errors.Is(err, memorycore.ErrNotFound) {
		t.Fatalf("missing read = %v", err)
	}
	if _, err := store.Read(ctx, scope, "notes", memorycore.ReadOptions{MaxBytes: 4}); !errors.Is(err, memorycore.ErrLimitExceeded) {
		t.Fatalf("bounded read = %v", err)
	}
	if _, err := store.Mutate(ctx, scope, memorycore.Mutation{
		Path: "notes", Kind: memorycore.MutationReplace, IdempotencyKey: "exists", Fingerprint: "exists",
	}); !errors.Is(err, memorycore.ErrConflict) {
		t.Fatalf("create conflict = %v", err)
	}
	if _, err := store.Mutate(ctx, scope, memorycore.Mutation{
		Path: "notes", Kind: memorycore.MutationReplace, ExpectedVersion: "stale", IdempotencyKey: "stale", Fingerprint: "stale",
	}); !errors.Is(err, memorycore.ErrConflict) {
		t.Fatalf("version conflict = %v", err)
	}
	if _, err := store.Mutate(ctx, scope, memorycore.Mutation{
		Path: "missing", Kind: memorycore.MutationDelete, IdempotencyKey: "delete-missing", Fingerprint: "delete-missing",
	}); !errors.Is(err, memorycore.ErrConflict) {
		t.Fatalf("delete missing = %v", err)
	}
	appended, err := store.Mutate(ctx, scope, memorycore.Mutation{
		Path: "fresh", Kind: memorycore.MutationAppend, Content: []byte("new"), IdempotencyKey: "append-new", Fingerprint: "append-new",
	})
	if err != nil || appended.Bytes != 3 || main.Version == "" {
		t.Fatalf("append new = %#v, %v", appended, err)
	}
}

func TestLockCancellationAndDirectFailurePaths(t *testing.T) {
	store, _ := New(t.TempDir())
	scope := memorycore.Scope("locked")
	held, err := acquireFileLock(store.lockPath(scope))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.acquire(ctx, scope); !errors.Is(err, context.Canceled) {
		t.Fatalf("lock cancellation = %v", err)
	}
	if err := held.Close(); err != nil {
		t.Fatal(err)
	}
	if err := held.Close(); err != nil {
		t.Fatalf("repeated close = %v", err)
	}
	var nilLock *fileLock
	if err := nilLock.Close(); err != nil {
		t.Fatalf("nil close = %v", err)
	}

	bad := &Store{root: filepath.Join(t.TempDir(), "missing", "root")}
	if _, err := bad.acquire(context.Background(), "scope"); err == nil {
		t.Fatal("lock in missing root succeeded")
	}
	if err := syncDirectory(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("sync missing directory succeeded")
	}
	if _, err := bad.recoverPartial(filepath.Join(t.TempDir(), "missing", "log"), []byte("partial")); err == nil {
		t.Fatal("partial sidecar in missing directory succeeded")
	}
	root := t.TempDir()
	missingRoot := &Store{root: filepath.Join(root, "missing")}
	path := filepath.Join(root, "log")
	if _, err := missingRoot.recoverPartial(path, []byte("partial")); err == nil {
		t.Fatal("partial recovery with unsyncable root succeeded")
	}
	if _, err := store.recoverPartial(filepath.Join(store.root, "absent"), []byte("partial")); err == nil {
		t.Fatal("partial recovery without source log succeeded")
	}
	if data, err := store.recoverPartial(filepath.Join(store.root, "unused"), nil); err != nil || data != nil {
		t.Fatalf("empty recovery = %q, %v", data, err)
	}
	if data, err := store.recoverPartial(filepath.Join(store.root, "unused"), []byte("ok\n")); err != nil || string(data) != "ok\n" {
		t.Fatalf("complete recovery = %q, %v", data, err)
	}
	appendPath := store.path("append-error")
	if err := os.Mkdir(appendPath, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := store.append("append-error", record{}); err == nil {
		t.Fatal("append to directory succeeded")
	}

	for name, operation := range map[string]func(context.Context) error{
		"read": func(ctx context.Context) error {
			_, err := store.Read(ctx, "blocked-read", "main", memorycore.ReadOptions{MaxBytes: 1})
			return err
		},
		"mutate": func(ctx context.Context) error {
			_, err := store.Mutate(ctx, "blocked-mutate", memorycore.Mutation{
				Path: "main", Kind: memorycore.MutationReplace, IdempotencyKey: "id", Fingerprint: "fp",
			})
			return err
		},
	} {
		t.Run(name, func(t *testing.T) {
			scope := memorycore.Scope("blocked-" + name)
			lock, err := acquireFileLock(store.lockPath(scope))
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
			defer cancel()
			if err := operation(ctx); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("blocked operation = %v", err)
			}
			if err := lock.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCorruptRecordFrontiers(t *testing.T) {
	scope := memorycore.Scope("scope")
	mutation := memorycore.Mutation{
		Path: "main", Kind: memorycore.MutationReplace, Content: []byte("value"),
		IdempotencyKey: "id", Fingerprint: "fp",
	}
	version := makeVersion(1, mutation.Path, mutation.Content)
	valid := record{
		Version: 1, Scope: scope, Sequence: 1, Mutation: mutation,
		Result: memorycore.MutationResult{Path: "main", Version: version, Bytes: 5},
		File:   &memorycore.File{Path: "main", Version: version, Content: []byte("value")},
	}
	staleMutation := mutation
	staleMutation.IdempotencyKey = "stale"
	staleMutation.Fingerprint = "stale"
	staleMutation.ExpectedVersion = "stale"
	staleVersion := makeVersion(2, staleMutation.Path, staleMutation.Content)
	for name, records := range map[string][]record{
		"wrong version":        {{Version: 2, Scope: scope, Sequence: 1, Mutation: mutation, Result: valid.Result, File: valid.File}},
		"missing file":         {{Version: 1, Scope: scope, Sequence: 1, Mutation: mutation, Result: valid.Result}},
		"invalid file":         {{Version: 1, Scope: scope, Sequence: 1, Mutation: mutation, Result: valid.Result, File: &memorycore.File{Path: "other", Version: version}}},
		"wrong result version": {{Version: 1, Scope: scope, Sequence: 1, Mutation: mutation, Result: memorycore.MutationResult{Path: "main", Version: "wrong", Bytes: 5}, File: valid.File}},
		"wrong result bytes":   {{Version: 1, Scope: scope, Sequence: 1, Mutation: mutation, Result: memorycore.MutationResult{Path: "main", Version: version, Bytes: 4}, File: valid.File}},
		"wrong file content":   {{Version: 1, Scope: scope, Sequence: 1, Mutation: mutation, Result: valid.Result, File: &memorycore.File{Path: "main", Version: version, Content: []byte("other")}}},
		"repeated op":          {valid, {Version: 1, Scope: scope, Sequence: 2, Mutation: mutation, Result: valid.Result, File: valid.File}},
		"wrong sequence":       {{Version: 1, Scope: scope, Sequence: 2, Mutation: mutation, Result: valid.Result, File: valid.File}},
		"deleted with file":    {{Version: 1, Scope: scope, Sequence: 1, Mutation: mutation, Result: memorycore.MutationResult{Path: "main", Version: version, Deleted: true}, File: valid.File}},
		"stale CAS":            {valid, {Version: 1, Scope: scope, Sequence: 2, Mutation: staleMutation, Result: memorycore.MutationResult{Path: "main", Version: staleVersion, Bytes: 5}, File: &memorycore.File{Path: "main", Version: staleVersion, Content: []byte("value")}}},
	} {
		t.Run(name, func(t *testing.T) {
			store, _ := New(t.TempDir())
			file, err := os.Create(store.path(scope))
			if err != nil {
				t.Fatal(err)
			}
			encoder := json.NewEncoder(file)
			for _, current := range records {
				if err := encoder.Encode(current); err != nil {
					t.Fatal(err)
				}
			}
			if err := file.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err := store.List(context.Background(), scope, memorycore.ListOptions{Limit: 10}); !errors.Is(err, ErrCorruptLog) {
				t.Fatalf("corrupt load = %v", err)
			}
		})
	}
	store, _ := New(t.TempDir())
	if err := os.WriteFile(store.path(scope), []byte("not-json\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.List(context.Background(), scope, memorycore.ListOptions{Limit: 1}); !errors.Is(err, ErrCorruptLog) {
		t.Fatalf("malformed JSON = %v", err)
	}
	if _, err := store.Mutate(context.Background(), scope, memorycore.Mutation{
		Path: "main", Kind: memorycore.MutationReplace, IdempotencyKey: "new", Fingerprint: "new",
	}); !errors.Is(err, ErrCorruptLog) {
		t.Fatalf("mutation over corrupt log = %v", err)
	}

	directoryLog, _ := New(t.TempDir())
	if err := os.Mkdir(directoryLog.path("directory"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := directoryLog.List(context.Background(), "directory", memorycore.ListOptions{Limit: 1}); err == nil {
		t.Fatal("directory log read succeeded")
	}
}

func TestMutationReportsAppendFailure(t *testing.T) {
	root := t.TempDir()
	store, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	scope := memorycore.Scope("read-only")
	lock, err := acquireFileLock(store.lockPath(scope))
	if err != nil {
		t.Fatal(err)
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(root, 0o500); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Chmod(root, 0o700) }()
	_, err = store.Mutate(context.Background(), scope, memorycore.Mutation{
		Path: "main", Kind: memorycore.MutationReplace, IdempotencyKey: "id", Fingerprint: "fp",
	})
	if err == nil {
		t.Fatal("mutation in read-only root succeeded")
	}
}
