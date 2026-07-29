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
}
