package postgres

// The scripted provider must replay from request history, not from how many
// abandoned network attempts happened in an earlier worker.
import (
	"context"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

func TestFixtureModelRetriesAnUncommittedRequest(t *testing.T) {
	gate := make(chan struct{})
	m := &model{readFile: true, gate: gate}
	req := agentic.ChatRequest{Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "first")}}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := m.Request(ctx, &req); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled request=%v", err)
	}
	close(gate)
	reply, err := m.Request(t.Context(), &req)
	if err != nil {
		t.Fatal(err)
	}
	calls := reply.Message.GetToolUses()
	if len(calls) != 1 || calls[0].ID != "read-fixture" {
		t.Fatalf("retry skipped the tool: %+v", reply)
	}
}
