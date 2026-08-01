package session

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestSessionPersistsDurableContextAndRederivesEphemeralTail(t *testing.T) {
	repository := storememory.New()
	model := &scriptedModel{steps: []modelStep{
		{
			message: agentic.NewToolUseMessage(agentic.ToolUse{
				ID:   "noop-1",
				Name: "noop",
			}),
		},
		textStep("first done"),
		textStep("second done"),
	}}
	agent := agentic.NewAgent("", model)
	agentic.AddTool(agent, func(context.Context, struct{}) (string, error) {
		return "ok", nil
	}, agentic.AutoToolName("noop"), agentic.AutoToolDescription("No-op tool"))
	policy, err := contextpolicy.New(contextpolicy.Config{}, []contextpolicy.Transform{
		contextpolicy.TransformFunc(func(_ context.Context, value *contextpolicy.TransformContext) error {
			found := false
			for _, message := range value.Durable.Messages {
				if message.GetTextContent() == "<durable-context/>" {
					found = true
				}
			}
			if !found {
				value.Durable.Messages = append(
					value.Durable.Messages,
					agentic.NewTextMessage(agentic.RoleUser, "<durable-context/>"),
				)
			}
			*value.Ephemeral = append(
				*value.Ephemeral,
				agentic.NewTextMessage(agentic.RoleUser, "<ephemeral-context/>"),
			)
			return nil
		}),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	config := sessionConfig(t, agent, repository, artifactmemory.New(), spill.Config{})
	config.Context = policy
	first, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "first")); err != nil {
		t.Fatal(err)
	}
	calls := model.Calls()
	if len(calls) != 2 {
		t.Fatalf("model calls = %d", len(calls))
	}
	for index, call := range calls {
		if countMessageText(call.Messages, "<durable-context/>") != 1 ||
			countMessageText(call.Messages, "<ephemeral-context/>") != 1 ||
			call.Messages[len(call.Messages)-1].GetTextContent() != "<ephemeral-context/>" {
			t.Fatalf("call %d context = %#v", index, call.Messages)
		}
	}
	snapshot, err := first.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if countMessageText(snapshot.Messages, "<durable-context/>") != 0 ||
		countMessageText(snapshot.Messages, "<ephemeral-context/>") != 0 {
		t.Fatalf("provider-only context leaked into transcript: %#v", snapshot.Messages)
	}
	loaded, err := first.journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	durableEntries := 0
	for _, entry := range loaded.Entries {
		if entry.Kind == kindContextMessage {
			durableEntries++
		}
	}
	if durableEntries != 1 {
		t.Fatalf("durable context entries = %d", durableEntries)
	}
	if err := first.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := recovered.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "second")); err != nil {
		t.Fatal(err)
	}
	calls = model.Calls()
	if len(calls) != 3 ||
		countMessageText(calls[2].Messages, "<durable-context/>") != 1 ||
		countMessageText(calls[2].Messages, "<ephemeral-context/>") != 1 {
		t.Fatalf("recovered context = %#v", calls)
	}
}

type countingCompactor struct {
	calls atomic.Int32
}

func (c *countingCompactor) Summarize(context.Context, []agentic.Message) (agentic.Message, error) {
	c.calls.Add(1)
	return agentic.NewTextMessage(agentic.RoleUser, "<persisted-summary/>"), nil
}

func TestSessionReappliesDurableCompactionAfterRestart(t *testing.T) {
	repository := storememory.New()
	long := strings.Repeat("history-", 125)
	model := &scriptedModel{steps: []modelStep{
		textStep(long + "one"),
		textStep(long + "two"),
		textStep(long + "three"),
		textStep("four"),
	}}
	agent := agentic.NewAgent("", model)
	compactor := &countingCompactor{}
	policy, err := contextpolicy.New(contextpolicy.Config{
		ContextWindowTokens: 300,
		TriggerPercent:      70,
		TargetPercent:       50,
		RecentMessages:      1,
		MessageOverhead:     1,
		PartOverhead:        1,
		ToolOverhead:        1,
		Counter: contextpolicy.TokenCounterFunc(func(_ context.Context, value []byte) (int, error) {
			count := len(value) / 10
			if count == 0 {
				count = 1
			}
			return count, nil
		}),
	}, nil, compactor)
	if err != nil {
		t.Fatal(err)
	}
	config := sessionConfig(t, agent, repository, artifactmemory.New(), spill.Config{})
	config.Context = policy
	first, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	for _, prompt := range []string{"one", "two", "three"} {
		if _, err := first.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, prompt)); err != nil {
			t.Fatal(err)
		}
	}
	if compactor.calls.Load() != 1 {
		t.Fatalf("compactor calls before restart = %d", compactor.calls.Load())
	}
	loaded, err := first.journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	compactions := 0
	for _, entry := range loaded.Entries {
		if entry.Kind == kindCompaction {
			compactions++
		}
	}
	if compactions != 1 {
		t.Fatalf("compaction entries = %d", compactions)
	}
	if err := first.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := recovered.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "four")); err != nil {
		t.Fatal(err)
	}
	if compactor.calls.Load() != 1 {
		t.Fatalf("persisted compaction was recomputed: calls=%d", compactor.calls.Load())
	}
	calls := model.Calls()
	if len(calls) != 4 || countMessageText(calls[3].Messages, "<persisted-summary/>") != 1 {
		t.Fatalf("recovered provider view = %#v", calls)
	}
}

func countMessageText(messages []agentic.Message, text string) int {
	count := 0
	for _, message := range messages {
		if message.GetTextContent() == text {
			count++
		}
	}
	return count
}
