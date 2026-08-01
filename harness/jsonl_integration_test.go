package harness

import (
	"context"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	storejsonl "github.com/regularkevvv/agentic/harness/store/jsonl"
)

func TestJSONLRuntimeRestoresAndDrainsOpaqueQueuedInputOnce(t *testing.T) {
	root := t.TempDir()
	firstRepository, err := storejsonl.New(root)
	if err != nil {
		t.Fatal(err)
	}
	config := runtimeConfig(t)
	config.Sessions = firstRepository
	config.Codec = prefixedCodec{base: jsoncodec.New()}
	firstRuntime, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	first, err := firstRuntime.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := first.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "durable"))
	if err != nil {
		t.Fatal(err)
	}
	sessionID := first.ID()
	if err := first.Close(context.Background()); err != nil {
		t.Fatal(err)
	}

	secondRepository, err := storejsonl.New(root)
	if err != nil {
		t.Fatal(err)
	}
	config.Sessions = secondRepository
	secondRuntime, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	second, err := secondRuntime.ResumeSession(context.Background(), sessionID)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := second.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Pending) != 1 || snapshot.Pending[0].ID != receipt.ID {
		t.Fatalf("restored queue = %#v", snapshot.Pending)
	}
	if _, err := second.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); err != nil {
		t.Fatal(err)
	}
	if err := second.Close(context.Background()); err != nil {
		t.Fatal(err)
	}

	thirdRepository, err := storejsonl.New(root)
	if err != nil {
		t.Fatal(err)
	}
	config.Sessions = thirdRepository
	thirdRuntime, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	third, err := thirdRuntime.ResumeSession(context.Background(), sessionID)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = third.Close(context.Background()) }()
	snapshot, err = third.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Pending) != 0 {
		t.Fatalf("queue drained more than once: %#v", snapshot.Pending)
	}
	count := 0
	for _, message := range snapshot.Messages {
		if message.GetTextContent() == "durable" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("durable queued message count = %d", count)
	}
}
