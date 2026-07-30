package session

import (
	"context"
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/repair"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func agenticEntry(t *testing.T, kind string, typ agentic.EventType, payload any) store.PendingEntry {
	t.Helper()
	payloadCodec := jsoncodec.New()
	encoded, err := codec.Encode(payloadCodec, payload)
	if err != nil {
		t.Fatal(err)
	}
	entry, err := pending(payloadCodec, kind, event.Record{
		Nature:  agentic.EventAuthoritative,
		Type:    typ,
		Source:  "agentic",
		Payload: encoded,
	})
	if err != nil {
		t.Fatal(err)
	}
	return entry
}

func crashedConfig(t *testing.T, calls []agentic.ToolUse, point string) (Config[string], *countingDriver, *storememory.Repository) {
	t.Helper()
	driver := &countingDriver{}
	repository := storememory.New()
	config := sessionConfig(t, driver, repository, artifactmemory.New(), spill.Config{})
	seed, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	prompt := agentic.NewTextMessage(agentic.RoleUser, "crash test")
	assistant := agentic.NewToolUseMessage(calls...)
	batch := newEntryBatch(config.Codec, 6)
	batch.Add(kindRunOpened, runOpenedPayload{ID: "crashed", Mode: "start"})
	batch.Add(kindMessage, messagePayload{Message: prompt, Source: "prompt"})
	entries, err := batch.Result()
	if err != nil {
		t.Fatal(err)
	}
	entries = append(entries,
		agenticEntry(t, kindAssistantCommitted, agentic.EventTypeAssistantCommitted, event.AssistantPayload{Message: assistant}),
		agenticEntry(t, kindToolBatchPlanned, agentic.EventTypeToolBatchPlanned, event.ToolBatchPayload{Calls: calls}),
	)
	if point == "started" || point == "result" {
		entries = append(entries, agenticEntry(t, kindToolStarted, agentic.EventTypeToolStarted, event.ToolStartedPayload{Call: calls[0], Attempt: 1}))
	}
	if point == "result" {
		entries = append(entries, agenticEntry(t, kindToolResult, agentic.EventTypeToolResultCommitted, event.ToolResultPayload{
			ToolUseID: calls[0].ID,
			ToolName:  calls[0].Name,
			Content:   "committed",
		}))
	}
	commit, err := seed.journal.Append(context.Background(), seed.cursor, entries...)
	if err != nil {
		t.Fatal(err)
	}
	seed.cursor = commit.Cursor
	if err := seed.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	return config, driver, repository
}

func TestRecoveryBeforeToolStartRepairsThenContinues(t *testing.T) {
	call := agentic.ToolUse{ID: "before", Name: "effect", Input: map[string]any{"value": 1}}
	config, driver, _ := crashedConfig(t, []agentic.ToolUse{call}, "planned")
	session, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.WaitForIdle(context.Background()); err != nil {
		t.Fatal(err)
	}
	if driver.Count() != 1 || driver.Last().Mode != agentic.DriveContinue {
		t.Fatalf("recovery drives = %#v", driver.drives)
	}
	history := driver.Last().History
	if open, err := repair.InspectFrontier(history); err != nil || len(open) != 0 {
		t.Fatalf("recovery history frontier = %#v, %v", open, err)
	}
	results := history[len(history)-1].GetToolResults()
	if len(results) != 1 || results[0].ToolUseID != call.ID || !results[0].IsError || !stringsContains(results[0].Content, "abandoned before it started") {
		t.Fatalf("synthetic result = %#v", results)
	}
}

func TestRecoveryAfterToolStartSuspendsIndeterminateWithoutRepeating(t *testing.T) {
	one := agentic.ToolUse{ID: "started", Name: "effect"}
	two := agentic.ToolUse{ID: "planned", Name: "effect"}
	config, driver, _ := crashedConfig(t, []agentic.ToolUse{one, two}, "started")
	session, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != Suspended || snapshot.Suspension == nil || snapshot.Suspension.Kind != "harness.recovery.indeterminate" || driver.Count() != 0 {
		t.Fatalf("snapshot=%#v drives=%d", snapshot, driver.Count())
	}
	// Reopening the same unresolved crash frontier is idempotent and still does
	// not invoke the driver or tool path.
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	again, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if again.State() != Suspended || driver.Count() != 0 {
		t.Fatalf("second recovery state=%s drives=%d", again.State(), driver.Count())
	}
	loaded, loadErr := again.journal.Load(context.Background())
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	suspensions := 0
	for _, entry := range loaded.Entries {
		if entry.Kind == kindRecoverySuspension {
			suspensions++
		}
	}
	if suspensions != 1 {
		t.Fatalf("recovery suspension entries = %d", suspensions)
	}
	next, err := again.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "after interrupt"))
	if err != nil {
		t.Fatal(err)
	}
	if err := again.Interrupt(context.Background()); err != nil {
		t.Fatal(err)
	}
	snapshot, _ = again.Snapshot(context.Background())
	if snapshot.State != Idle || len(snapshot.Pending) != 1 || snapshot.Pending[0].ID != next.ID {
		t.Fatalf("post-interrupt snapshot = %#v", snapshot)
	}
	if open, err := repair.InspectFrontier(snapshot.Messages); err != nil || len(open) != 0 {
		t.Fatalf("interrupt frontier = %#v, %v", open, err)
	}
	lastTwo := snapshot.Messages[len(snapshot.Messages)-2:]
	if lastTwo[0].GetToolResults()[0].ToolUseID != one.ID || lastTwo[1].GetToolResults()[0].ToolUseID != two.ID {
		t.Fatalf("interrupt did not repair source order: %#v", lastTwo)
	}
}

func TestRecoveryAfterToolResultContinuesWithoutRepeating(t *testing.T) {
	call := agentic.ToolUse{ID: "complete", Name: "effect"}
	config, driver, _ := crashedConfig(t, []agentic.ToolUse{call}, "result")
	session, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.WaitForIdle(context.Background()); err != nil {
		t.Fatal(err)
	}
	if driver.Count() != 1 {
		t.Fatalf("drives = %d", driver.Count())
	}
	history := driver.Last().History
	results := history[len(history)-1].GetToolResults()
	if len(results) != 1 || results[0].Content != "committed" || results[0].IsError {
		t.Fatalf("committed result = %#v", results)
	}
	loaded, loadErr := session.journal.Load(context.Background())
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	for _, entry := range loaded.Entries {
		if entry.Kind == kindRecoverySuspension {
			t.Fatal("completed tool was marked indeterminate")
		}
	}
}

func TestRecoveryCompletesDurablyDrainedInputExactlyOnce(t *testing.T) {
	driver := &countingDriver{}
	repository := storememory.New()
	config := sessionConfig(t, driver, repository, artifactmemory.New(), spill.Config{})
	seed, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	prompt := agentic.NewTextMessage(agentic.RoleUser, "prompt")
	queued := QueueEntry{ID: "queued", Kind: QueueSteer, Message: agentic.NewTextMessage(agentic.RoleUser, "durable steer")}
	batch := newEntryBatch(config.Codec, 4)
	batch.Add(kindRunOpened, runOpenedPayload{ID: "crashed", Mode: "start"})
	batch.Add(kindMessage, messagePayload{Message: prompt, Source: "prompt"})
	batch.Add(kindQueueAccepted, queueMutationPayload{ID: queued.ID, Entry: &queued})
	batch.Add(kindQueueDrained, queueMutationPayload{ID: queued.ID})
	entries, err := batch.Result()
	if err != nil {
		t.Fatal(err)
	}
	commit, err := seed.journal.Append(context.Background(), seed.cursor, entries...)
	if err != nil {
		t.Fatal(err)
	}
	seed.cursor = commit.Cursor
	if err := seed.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if err := recovered.WaitForIdle(context.Background()); err != nil {
		t.Fatal(err)
	}
	if driver.Count() != 1 {
		t.Fatalf("drives = %d", driver.Count())
	}
	history := driver.Last().History
	if len(history) != 2 || history[0].GetTextContent() != "prompt" || history[1].GetTextContent() != "durable steer" {
		t.Fatalf("continued history = %#v", history)
	}
	if err := recovered.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	again, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := again.Snapshot(context.Background())
	count := 0
	for _, message := range snapshot.Messages {
		if message.GetTextContent() == "durable steer" {
			count++
		}
	}
	if count != 1 || len(snapshot.Pending) != 0 {
		t.Fatalf("recovered steer count=%d pending=%#v", count, snapshot.Pending)
	}
}

func TestRecoveryRejectsCorruptDuplicateResultFrontier(t *testing.T) {
	call := agentic.ToolUse{ID: "duplicate", Name: "effect"}
	config, _, repository := crashedConfig(t, []agentic.ToolUse{call}, "result")
	duplicate := agenticEntry(t, kindToolResult, agentic.EventTypeToolResultCommitted, event.ToolResultPayload{ToolUseID: call.ID, ToolName: call.Name, Content: "twice"})
	journal, err := repository.Open(context.Background(), config.ID)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := journal.Append(context.Background(), loaded.Cursor, duplicate); err != nil {
		t.Fatal(err)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := Recover(context.Background(), config); !errors.Is(err, agentic.ErrTranscriptInvalid) {
		t.Fatalf("duplicate recovery error = %v", err)
	}
}

func stringsContains(value, substring string) bool {
	return strings.Contains(value, substring)
}
