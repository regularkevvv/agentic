package spill

import (
	"context"
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/artifact"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
)

func TestPreservesFullOutputAndReturnsOpaquePreviewOnce(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	storage := artifactmemory.New()
	processor, err := NewProcessor(storage, "session", Config{Threshold: 16, Head: 5, Tail: 5})
	if err != nil {
		t.Fatal(err)
	}
	content := "αβγ---middle---ωxyz"
	call := agentic.ToolUse{ID: "call-1", Name: "large"}
	base := agentic.ToolExecutionResult{ToolUseID: call.ID, ToolName: call.Name, Content: content, IsError: true, Error: errors.New("truth")}
	projected, err := processor.Process(ctx, call, base)
	if err != nil {
		t.Fatal(err)
	}
	visible, ok := projected.Content.(string)
	if !ok || !strings.Contains(visible, "harness artifact art_") || strings.Contains(visible, "middle") {
		t.Fatalf("visible content = %q", visible)
	}
	if !projected.IsError || projected.Error != base.Error || projected.ToolUseID != base.ToolUseID || projected.ToolName != base.ToolName {
		t.Fatalf("processor changed result truth: %#v", projected)
	}
	parts := strings.SplitN(visible, ";", 2)
	handle := artifact.Handle(strings.TrimPrefix(parts[0], "[harness artifact "))
	full, err := storage.Get(ctx, "session", handle)
	if err != nil || string(full) != content {
		t.Fatalf("stored full output = %q, %v", full, err)
	}
	second, err := processor.Process(ctx, call, base)
	if err != nil || second.Content != projected.Content || storage.Count("session") != 1 {
		t.Fatalf("second spill = %#v, count=%d, err=%v", second, storage.Count("session"), err)
	}
	if _, err := storage.Get(ctx, "other", handle); !errors.Is(err, artifact.ErrArtifactNotFound) {
		t.Fatalf("cross-session read = %v", err)
	}
}

func TestLeavesSmallCanonicalResultInline(t *testing.T) {
	t.Parallel()
	processor, err := NewProcessor(artifactmemory.New(), "session", Config{Threshold: 64, Head: 8, Tail: 8})
	if err != nil {
		t.Fatal(err)
	}
	call := agentic.ToolUse{ID: "call", Name: "tool"}
	result, err := processor.Process(context.Background(), call, agentic.ToolExecutionResult{Content: map[string]int{"value": 1}})
	if err != nil || result.Content != `{"value":1}` {
		t.Fatalf("inline result = %#v, %v", result.Content, err)
	}
}

func TestFactoryAndValidation(t *testing.T) {
	t.Parallel()
	storage := artifactmemory.New()
	factory, err := NewFactory(storage, Config{Threshold: 64, Head: 8, Tail: 8})
	if err != nil {
		t.Fatal(err)
	}
	if processor, err := factory.Open(context.Background(), "session"); err != nil || processor == nil {
		t.Fatalf("Open = %#v, %v", processor, err)
	}
	if _, err := NewFactory(storage, Config{Threshold: 3, Head: 2, Tail: 2}); err == nil {
		t.Fatal("invalid spill limits succeeded")
	}
	if err := (Config{Disabled: true, Threshold: 1}).Validate(); err == nil {
		t.Fatal("disabled spill with limits succeeded")
	}
	disabled, err := NewProcessor(storage, "session", Config{Disabled: true})
	if err != nil {
		t.Fatal(err)
	}
	content := strings.Repeat("unbounded", 100)
	result, err := disabled.Process(context.Background(), agentic.ToolUse{ID: "disabled", Name: "tool"}, agentic.ToolExecutionResult{Content: content})
	if err != nil || result.Content != content || storage.Count("session") != 0 {
		t.Fatalf("disabled result=%#v count=%d err=%v", result.Content, storage.Count("session"), err)
	}
}

type failingStore struct {
	err error
}

func (s failingStore) Put(context.Context, string, string, []byte) (artifact.Handle, error) {
	return "", s.err
}

func (s failingStore) Get(context.Context, string, artifact.Handle) ([]byte, error) {
	return nil, s.err
}

func TestFactoryProcessorCancellationDefaultsAndStoreFailure(t *testing.T) {
	t.Parallel()
	if _, err := NewFactory(nil, Config{}); err == nil {
		t.Fatal("nil factory store succeeded")
	}
	if _, err := NewProcessor(nil, "session", Config{}); err == nil {
		t.Fatal("nil processor store succeeded")
	}
	if _, err := NewProcessor(artifactmemory.New(), "bad/session", Config{}); err == nil {
		t.Fatal("invalid processor session succeeded")
	}
	if _, err := NewProcessor(artifactmemory.New(), "session", Config{
		Threshold: 3,
		Head:      2,
		Tail:      2,
	}); err == nil {
		t.Fatal("invalid processor spill limits succeeded")
	}
	if err := (Config{}).Validate(); err != nil {
		t.Fatalf("default config = %v", err)
	}
	factory, err := NewFactory(artifactmemory.New(), Config{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := factory.Open(ctx, "session"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Open = %v", err)
	}

	wantErr := errors.New("storage")
	processor, err := NewProcessor(failingStore{err: wantErr}, "session", Config{Threshold: 4, Head: 1, Tail: 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := processor.Process(context.Background(), agentic.ToolUse{ID: "call"}, agentic.ToolExecutionResult{
		Content: "oversized",
	}); !errors.Is(err, wantErr) {
		t.Fatalf("spill failure = %v", err)
	}
	if got := utf8Head([]byte("ok"), 10); got != "ok" {
		t.Fatalf("full head = %q", got)
	}
	if got := utf8Tail([]byte("ok"), 10); got != "ok" {
		t.Fatalf("full tail = %q", got)
	}
	if got := utf8Head([]byte("éx"), 1); got != "" {
		t.Fatalf("split head = %q", got)
	}
	if got := utf8Tail([]byte("xé"), 1); got != "" {
		t.Fatalf("split tail = %q", got)
	}
}
