package file

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/regularkevvv/agentic/harness/artifact"
	"github.com/regularkevvv/agentic/harness/artifact/artifacttest"
)

func TestStoreConformance(t *testing.T) {
	t.Parallel()
	artifacttest.Run(t, func(t *testing.T) artifact.Store {
		storage, err := New(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		return storage
	})
}

func TestSurvivesReopen(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	storage, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := storage.Put(context.Background(), "session", "call", []byte("durable"))
	if err != nil {
		t.Fatal(err)
	}
	reopened, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	again, err := reopened.Put(context.Background(), "session", "call", []byte("durable"))
	if err != nil || again != handle {
		t.Fatalf("reopened Put = %q, %v", again, err)
	}
}

func TestStoreValidationConflictsAndCorruptMetadata(t *testing.T) {
	t.Parallel()
	if _, err := New(""); err == nil {
		t.Fatal("empty root succeeded")
	}
	parent := t.TempDir()
	rootFile := filepath.Join(parent, "root-file")
	if err := os.WriteFile(rootFile, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(rootFile); err == nil {
		t.Fatal("file root succeeded")
	}

	store, err := New(filepath.Join(parent, "artifacts"))
	if err != nil {
		t.Fatal(err)
	}
	if store.Root() == "" {
		t.Fatal("canonical Root is empty")
	}
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
	handle, err := store.Put(context.Background(), "session", "key", []byte("value"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Put(context.Background(), "session", "key", []byte("different")); !errors.Is(err, artifact.ErrArtifactConflict) {
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
	validMissing := artifact.Handle("art_" + hex.EncodeToString(make([]byte, 32)))
	if _, err := store.Get(context.Background(), "session", validMissing); !errors.Is(err, artifact.ErrArtifactNotFound) {
		t.Fatalf("missing artifact = %v", err)
	}

	directory := filepath.Join(store.Root(), "corrupt")
	if err := os.Mkdir(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	keyDigest := hashKey("key")
	metadataPath := filepath.Join(directory, hex.EncodeToString(keyDigest[:])+".json")
	if err := os.WriteFile(metadataPath, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Put(context.Background(), "corrupt", "key", []byte("value")); err == nil {
		t.Fatal("corrupt metadata succeeded")
	}
	badMetadata, _ := json.Marshal(fileMetadata{Handle: "bad", Digest: "digest", Size: 5})
	if err := os.WriteFile(metadataPath, badMetadata, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Put(context.Background(), "corrupt", "key", []byte("value")); err == nil {
		t.Fatal("invalid metadata handle succeeded")
	}
	otherDigest := hashKey("not-value")
	goodMetadata, _ := json.Marshal(fileMetadata{Handle: handle, Digest: hex.EncodeToString(otherDigest[:]), Size: 5})
	if err := os.WriteFile(metadataPath, goodMetadata, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Put(context.Background(), "corrupt", "key", []byte("value")); !errors.Is(err, artifact.ErrArtifactConflict) {
		t.Fatalf("metadata conflict = %v", err)
	}
}

func TestFileHelpersAndSessionDirectoryFailure(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "blocked"), []byte("file"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Put(context.Background(), "blocked", "key", []byte("value")); err == nil {
		t.Fatal("file session directory succeeded")
	}
	if err := writeAtomic(filepath.Join(root, "missing"), filepath.Join(root, "missing", "target"), nil); err == nil {
		t.Fatal("writeAtomic missing directory succeeded")
	}
	if err := syncDir(filepath.Join(root, "missing")); err == nil {
		t.Fatal("syncDir missing path succeeded")
	}
	file := filepath.Join(root, "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := syncDir(file); err != nil {
		t.Fatalf("syncDir file = %v", err)
	}
}
