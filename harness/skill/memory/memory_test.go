package memory

import (
	"context"
	"errors"
	"sync"
	"testing"

	skillcore "github.com/regularkevvv/agentic/harness/skill"
)

func TestSourceLifecycleBoundsAndRace(t *testing.T) {
	source, err := New(Config{MaxInstructionBytes: 8, MaxDescriptionBytes: 16, MaxResources: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := source.Put("one", skillcore.Skill{Name: "beta", Description: "second", Instructions: "12345678"}); err != nil {
		t.Fatal(err)
	}
	if err := source.Put("one", skillcore.Skill{Name: "alpha", Description: "first", Instructions: "guide", Resources: []string{"a"}}); err != nil {
		t.Fatal(err)
	}
	if err := source.Put("two", skillcore.Skill{Name: "alpha", Description: "isolated", Instructions: "other"}); err != nil {
		t.Fatal(err)
	}
	values, err := source.List(context.Background(), "one", 1)
	if err != nil || len(values) != 1 || values[0].Name != "alpha" {
		t.Fatalf("list = %#v, %v", values, err)
	}
	value, err := source.Read(context.Background(), "one", "alpha", 5)
	if err != nil || value.Description != "first" {
		t.Fatalf("read = %#v, %v", value, err)
	}
	value.Resources[0] = "mutated"
	again, _ := source.Read(context.Background(), "one", "alpha", 5)
	if again.Resources[0] != "a" {
		t.Fatal("read leaked mutable resources")
	}
	if _, err := source.Read(context.Background(), "one", "beta", 7); !errors.Is(err, skillcore.ErrLimitExceeded) {
		t.Fatalf("read bound = %v", err)
	}
	if err := source.Delete("one", "alpha"); err != nil {
		t.Fatal(err)
	}
	if _, err := source.Read(context.Background(), "one", "alpha", 8); !errors.Is(err, skillcore.ErrNotFound) {
		t.Fatalf("deleted read = %v", err)
	}
	if isolated, _ := source.Read(context.Background(), "two", "alpha", 8); isolated.Description != "isolated" {
		t.Fatalf("scope isolation = %#v", isolated)
	}

	var wait sync.WaitGroup
	for index := 0; index < 20; index++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_ = source.Put("race", skillcore.Skill{Name: "same", Description: "safe", Instructions: "value"})
			_, _ = source.List(context.Background(), "race", 10)
		}()
	}
	wait.Wait()
}

func TestSourceValidationAndCancellation(t *testing.T) {
	if _, err := New(Config{MaxResources: -1}); err == nil {
		t.Fatal("negative config succeeded")
	}
	source, _ := New(Config{})
	if err := source.Put("", skillcore.Skill{}); err == nil {
		t.Fatal("invalid put succeeded")
	}
	if err := source.Delete("scope", "missing"); !errors.Is(err, skillcore.ErrNotFound) {
		t.Fatalf("missing delete = %v", err)
	}
	if err := source.Put("scope", skillcore.Skill{Name: "present", Description: "value", Instructions: "body"}); err != nil {
		t.Fatal(err)
	}
	if err := source.Delete("scope", "other"); !errors.Is(err, skillcore.ErrNotFound) {
		t.Fatalf("missing delete in existing scope = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := source.List(ctx, "scope", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled list = %v", err)
	}
	if _, err := source.Read(ctx, "scope", "name", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled read = %v", err)
	}
	if _, err := source.List(context.Background(), "scope", 0); err == nil {
		t.Fatal("unbounded list succeeded")
	}
	if err := source.Put("scope", skillcore.Skill{Name: "bad/name", Description: "bad", Instructions: "bad"}); err == nil {
		t.Fatal("invalid skill put succeeded")
	}
	if err := source.Delete("", "name"); err == nil {
		t.Fatal("invalid delete scope succeeded")
	}
	if err := source.Delete("scope", "../bad"); err == nil {
		t.Fatal("invalid delete name succeeded")
	}
	if _, err := source.List(context.Background(), "", 1); err == nil {
		t.Fatal("invalid list scope succeeded")
	}
	values, err := source.List(context.Background(), "empty", 1)
	if err != nil || len(values) != 0 {
		t.Fatalf("empty list = %#v, %v", values, err)
	}
	if _, err := source.Read(context.Background(), "", "name", 1); err == nil {
		t.Fatal("invalid read scope succeeded")
	}
	if _, err := source.Read(context.Background(), "scope", "../bad", 1); err == nil {
		t.Fatal("invalid read name succeeded")
	}
	if _, err := source.Read(context.Background(), "scope", "name", 0); err == nil {
		t.Fatal("invalid read bound succeeded")
	}
	if _, err := source.Read(context.Background(), "scope", "missing", 1); !errors.Is(err, skillcore.ErrNotFound) {
		t.Fatalf("missing read = %v", err)
	}
}
