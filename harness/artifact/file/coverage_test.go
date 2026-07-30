package file

import (
	"context"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/harness/artifact"
)

func TestArtifactFilesystemIntegrityAndAtomicFailureEdges(t *testing.T) {
	root := t.TempDir()
	store, err := New(root)
	if err != nil {
		t.Fatal(err)
	}

	handle, err := store.Put(context.Background(), "mismatch", "key", []byte("value"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(root, "mismatch", handle.String()+".blob")); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Put(context.Background(), "mismatch", "key", []byte("value")); err == nil ||
		!strings.Contains(err.Error(), "metadata/data mismatch") {
		t.Fatalf("missing blob mismatch = %v", err)
	}

	metadataDirectory := filepath.Join(root, "metadata-directory")
	if err := os.Mkdir(metadataDirectory, 0o700); err != nil {
		t.Fatal(err)
	}
	keyDigest := hashKey("key")
	if err := os.Mkdir(filepath.Join(metadataDirectory, hex.EncodeToString(keyDigest[:])+".json"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Put(context.Background(), "metadata-directory", "key", []byte("value")); err == nil ||
		!strings.Contains(err.Error(), "read artifact metadata") {
		t.Fatalf("metadata directory error = %v", err)
	}

	validHandle := artifact.Handle("art_" + hex.EncodeToString(make([]byte, 32)))
	blobDirectory := filepath.Join(root, "blob-directory")
	if err := os.Mkdir(blobDirectory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(blobDirectory, validHandle.String()+".blob"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Get(context.Background(), "blob-directory", validHandle); err == nil ||
		!strings.Contains(err.Error(), "read artifact") {
		t.Fatalf("blob directory error = %v", err)
	}

	atomicDirectory := t.TempDir()
	target := filepath.Join(atomicDirectory, "target")
	if err := os.Mkdir(target, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(target, "child"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := writeAtomic(atomicDirectory, target, []byte("value")); err == nil ||
		!strings.Contains(err.Error(), "commit artifact") {
		t.Fatalf("atomic rename error = %v", err)
	}

	if runtime.GOOS != "windows" {
		if err := syncDir("/dev/null"); err == nil {
			t.Fatal("syncing /dev/null as an artifact directory succeeded")
		}
	}
}
