package core

import (
	"context"
	"testing"
)

func TestToolExecutionStateContextRoundTrip(t *testing.T) {
	if _, ok := ToolExecutionStateFromContext(context.Background()); ok {
		t.Fatal("unexpected state on background context")
	}
	state := ToolExecutionState{Messages: []Message{NewTextMessage(RoleUser, "history")}}
	ctx := WithToolExecutionState(context.Background(), state)
	got, ok := ToolExecutionStateFromContext(ctx)
	if !ok || len(got.Messages) != 1 || got.Messages[0].GetTextContent() != "history" {
		t.Fatalf("unexpected state %#v, %v", got, ok)
	}
}

func TestModelRetryError(t *testing.T) {
	err := &ModelRetry{Message: "try again"}
	if err.Error() != "try again" {
		t.Fatalf("unexpected error %q", err.Error())
	}
}
