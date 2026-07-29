package file

import (
	"context"
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
