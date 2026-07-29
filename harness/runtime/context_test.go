package runtime

import (
	"context"
	"testing"

	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
)

func TestToolRuntimeContextRoundTrip(t *testing.T) {
	t.Parallel()
	if _, ok := FromContext(context.Background()); ok {
		t.Fatal("empty context contained ToolRuntime")
	}
	environment, err := envmemory.New("/workspace", nil)
	if err != nil {
		t.Fatal(err)
	}
	updates := make(chan ToolUpdate, 1)
	expected := ToolRuntime{Environment: environment, SessionID: "session", Emit: func(update ToolUpdate) { updates <- update }}
	ctx := WithContext(context.Background(), expected)
	got, ok := FromContext(ctx)
	if !ok || got.Environment != environment || got.SessionID != expected.SessionID || got.Emit == nil {
		t.Fatalf("FromContext = %#v, %v", got, ok)
	}
	got.Emit(ToolUpdate{Kind: "progress"})
	if update := <-updates; update.Kind != "progress" {
		t.Fatalf("update = %#v", update)
	}
}
