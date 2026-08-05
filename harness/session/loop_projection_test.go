package session

// Projection tests (plan S2): a replay covering every durable journal kind,
// malformed-payload rejection through the configured codec, privacy of
// system/instructions/thinking content, and preview projection semantics.

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func loopTestEntry(t *testing.T, payloadCodec codec.Codec, seq uint64, kind string, payload any) store.Entry {
	t.Helper()
	encoded, err := codec.Encode(payloadCodec, payload)
	if err != nil {
		t.Fatal(err)
	}
	return store.Entry{Schema: 1, Seq: seq, ID: fmt.Sprintf("id-%d", seq), Kind: kind, Payload: encoded}
}

func loopAgenticEntry(t *testing.T, payloadCodec codec.Codec, seq uint64, kind string, eventType agentic.EventType, payload any) store.Entry {
	t.Helper()
	encoded, err := codec.Encode(payloadCodec, payload)
	if err != nil {
		t.Fatal(err)
	}
	record := event.Record{Nature: agentic.EventAuthoritative, Type: eventType, Source: "agentic", Payload: encoded}
	return loopTestEntry(t, payloadCodec, seq, kind, record)
}

func projectLoopEntries(t *testing.T, payloadCodec codec.Codec, entries []store.Entry) []sessionloop.Event {
	t.Helper()
	records, err := loopRecords(payloadCodec, entries)
	if err != nil {
		t.Fatal(err)
	}
	projector := newLoopProjector("session-under-test", payloadCodec, nil)
	var events []sessionloop.Event
	for _, record := range records {
		projected, applyErr := projector.apply(context.Background(), record)
		if applyErr != nil {
			t.Fatalf("apply %d: %v", record.Cursor, applyErr)
		}
		events = append(events, projected...)
	}
	return events
}

// TestLoopProjectionCoversEveryJournalKind replays a synthetic journal
// containing every persisted kind and freezes the deliberate mapping of the
// doc-comment table: each kind either projects to its documented event or
// projects to nothing.
func TestLoopProjectionCoversEveryJournalKind(t *testing.T) {
	payloadCodec := jsoncodec.New()
	user := agentic.NewTextMessage(agentic.RoleUser, "prompt text")
	queuedMessage := agentic.NewTextMessage(agentic.RoleUser, "queued text")
	assistant := agentic.Message{Role: agentic.RoleAssistant, Content: []agentic.Part{
		{Type: agentic.ContentText, Text: "answer"},
		{Type: agentic.ContentToolUse, ToolUse: &agentic.ToolUse{ID: "call-1", Name: "lookup", Input: map[string]any{"q": "x"}}},
		{Type: agentic.ContentThinking, Thinking: &agentic.ThinkingBlock{Text: "hidden"}},
	}}
	suspension := agentic.Suspension{ID: "susp-1", Kind: "custom.kind", Payload: json.RawMessage(`{}`)}
	recoverySusp := agentic.Suspension{ID: "rec-1", Kind: "harness.recovery.indeterminate", Payload: json.RawMessage(`{"version":1,"calls":[]}`)}
	seq := uint64(0)
	next := func() uint64 { seq++; return seq }
	entries := []store.Entry{
		loopTestEntry(t, payloadCodec, next(), kindSessionCreated, sessionCreatedPayload{}),
		loopTestEntry(t, payloadCodec, next(), kindMessage, messagePayload{Message: user, Source: "initial_history"}),
		loopTestEntry(t, payloadCodec, next(), kindMessage, messagePayload{
			Message: agentic.NewTextMessage(agentic.RoleSystem, "system history"), Source: "initial_history"}),
		loopTestEntry(t, payloadCodec, next(), kindQueueAccepted, queueMutationPayload{ID: "q1", Entry: &QueueEntry{ID: "q1", Kind: QueueNextTurn, Message: queuedMessage}}),
		loopTestEntry(t, payloadCodec, next(), kindRunOpened, runOpenedPayload{ID: "run-1", Mode: "start"}),
		loopTestEntry(t, payloadCodec, next(), kindQueueDrained, queueMutationPayload{ID: "q1"}),
		loopTestEntry(t, payloadCodec, next(), kindMessage, messagePayload{Message: queuedMessage, Source: string(QueueNextTurn), QueueID: "q1"}),
		loopTestEntry(t, payloadCodec, next(), kindMessage, messagePayload{Message: user, Source: "prompt"}),
		loopAgenticEntry(t, payloadCodec, next(), kindAssistantCommitted, agentic.EventTypeAssistantCommitted, event.AssistantPayload{Message: assistant}),
		loopAgenticEntry(t, payloadCodec, next(), kindToolBatchPlanned, agentic.EventTypeToolBatchPlanned, event.ToolBatchPayload{}),
		loopAgenticEntry(t, payloadCodec, next(), kindToolStarted, agentic.EventTypeToolStarted, event.ToolStartedPayload{}),
		loopAgenticEntry(t, payloadCodec, next(), kindToolResult, agentic.EventTypeToolResultCommitted, event.ToolResultPayload{
			ToolUseID: "call-1", ToolName: "lookup", Content: `{"found":true}`}),
		loopTestEntry(t, payloadCodec, next(), kindQueueAccepted, queueMutationPayload{ID: "q2", Entry: &QueueEntry{ID: "q2", Kind: QueueSteer, Message: queuedMessage}}),
		loopTestEntry(t, payloadCodec, next(), kindQueueAccepted, queueMutationPayload{ID: "q3", Entry: &QueueEntry{ID: "q3", Kind: QueueFollowUp, Message: queuedMessage}}),
		loopAgenticEntry(t, payloadCodec, next(), kindMessagesInjected, agentic.EventTypeTurnMessagesInjected, event.MessagesPayload{
			Messages: []agentic.Message{queuedMessage, queuedMessage}, QueueIDs: []string{"q2", "q3"}}),
		loopAgenticEntry(t, payloadCodec, next(), kindTurnStarted, agentic.EventTypeTurnStarted, struct{}{}),
		loopAgenticEntry(t, payloadCodec, next(), kindTurnEnded, agentic.EventTypeTurnEnded, event.TurnEndedPayload{}),
		loopAgenticEntry(t, payloadCodec, next(), kindRunSuspended, agentic.EventTypeRunSuspended, event.SuspensionPayload{Suspension: suspension}),
		loopTestEntry(t, payloadCodec, next(), kindResolutionAccepted, resolutionAcceptedPayload{SuspensionID: "susp-1"}),
		loopAgenticEntry(t, payloadCodec, next(), kindRunStarted, agentic.EventTypeRunStarted, struct{}{}),
		loopAgenticEntry(t, payloadCodec, next(), kindToolResult, agentic.EventTypeToolResultCommitted, event.ToolResultPayload{
			ToolUseID: "call-1", ToolName: "lookup", Content: "plain text result", IsError: true}),
		loopAgenticEntry(t, payloadCodec, next(), kindRunError, agentic.EventTypeRunError, event.RunErrorPayload{Error: "late"}),
		loopAgenticEntry(t, payloadCodec, next(), kindRunCompleted, agentic.EventTypeRunCompleted, event.RunCompletedPayload{}),
		loopAgenticEntry(t, payloadCodec, next(), kindRunInterrupted, agentic.EventTypeRunInterrupted, struct{}{}),
		loopAgenticEntry(t, payloadCodec, next(), kindRunEnded, agentic.EventTypeRunEnded, event.RunEndedPayload{}),
		loopAgenticEntry(t, payloadCodec, next(), kindOutputValidated, agentic.EventTypeOutputValidated, event.OutputPayload{}),
		loopTestEntry(t, payloadCodec, next(), kindUsageCommitted, usagePayload{Session: agentic.Usage{TotalTokens: 42}}),
		loopTestEntry(t, payloadCodec, next(), kindSystemMessage, messagePayload{
			Message: agentic.NewTextMessage(agentic.RoleSystem, "driver system"), Source: "driver_system"}),
		loopTestEntry(t, payloadCodec, next(), kindQueueAccepted, queueMutationPayload{ID: "q4", Entry: &QueueEntry{ID: "q4", Kind: QueueNextTurn, Message: queuedMessage}}),
		loopTestEntry(t, payloadCodec, next(), kindQueueCancelled, queueMutationPayload{ID: "q4", Reason: "run ended"}),
		loopTestEntry(t, payloadCodec, next(), kindRunClosed, runClosedPayload{ID: "run-1", Status: agentic.ExecutionStopped, Error: "budget stop"}),
		loopTestEntry(t, payloadCodec, next(), kindInterruptMarker, interruptMarkerPayload{Message: "marker"}),
		loopTestEntry(t, payloadCodec, next(), kindRepair, messagePayload{Message: user, Source: "interrupt_repair"}),
		loopTestEntry(t, payloadCodec, next(), kindContextMessage, contextMessagePayload{Message: user}),
		loopTestEntry(t, payloadCodec, next(), kindCompaction, compactionPayload{}),
		loopTestEntry(t, payloadCodec, next(), kindChildUsage, childUsagePayload{Session: agentic.Usage{TotalTokens: 43}}),
		loopTestEntry(t, payloadCodec, next(), kindChildEvent, event.Record{Cursor: 1, Nature: agentic.EventAuthoritative, SessionID: "child"}),
		loopTestEntry(t, payloadCodec, next(), kindBranchMoved, struct{}{}),
		loopTestEntry(t, payloadCodec, next(), kindRuntimeOperation, harnessruntime.Operation{ID: "op", Kind: "k", Phase: "p"}),
		loopTestEntry(t, payloadCodec, next(), kindRecoverySuspension, event.SuspensionPayload{Suspension: recoverySusp}),
		loopTestEntry(t, payloadCodec, next(), kindFault, struct{ Error string }{Error: "fault text"}),
		loopTestEntry(t, payloadCodec, next(), kindRecovered, struct{ State string }{State: "continue"}),
		// Indeterminate-resume batch: buffered entries attribute to the NEW run.
		loopTestEntry(t, payloadCodec, next(), kindResolutionAccepted, resolutionAcceptedPayload{SuspensionID: "rec-1"}),
		loopTestEntry(t, payloadCodec, next(), kindMessage, messagePayload{Message: user, Source: "recovery_resolution"}),
		loopTestEntry(t, payloadCodec, next(), kindMessage, messagePayload{Message: user, Source: "resume_prompt"}),
		loopTestEntry(t, payloadCodec, next(), kindRunOpened, runOpenedPayload{ID: "run-2", Mode: "continue", Recovery: true}),
		loopTestEntry(t, payloadCodec, next(), kindRunClosed, runClosedPayload{ID: "run-2", Status: agentic.ExecutionCompleted}),
		loopTestEntry(t, payloadCodec, next(), kindRecovered, struct{ State string }{State: "idle"}),
	}

	events := projectLoopEntries(t, payloadCodec, entries)
	var trace []string
	for _, projected := range events {
		switch projected.Kind {
		case sessionloop.EventEntryCommitted:
			trace = append(trace, fmt.Sprintf("entry:%s:%s:%s", projected.Entry.Origin, projected.Entry.Role, projected.Entry.RunID))
		case sessionloop.EventRunStarted:
			trace = append(trace, "run.started:"+string(projected.RunID))
		case sessionloop.EventRunSettled:
			trace = append(trace, fmt.Sprintf("run.settled:%s:%s", projected.Outcome.RunID, projected.Outcome.Kind))
		case sessionloop.EventRunSuspended:
			trace = append(trace, fmt.Sprintf("run.suspended:%s:%s", projected.Suspension.ID, projected.Suspension.Kind))
		case sessionloop.EventQueueAccepted, sessionloop.EventQueueDrained, sessionloop.EventQueueCancelled:
			trace = append(trace, fmt.Sprintf("%s:%s:%s", projected.Kind, projected.Queue.ID, projected.Queue.Kind))
		case sessionloop.EventUsage:
			trace = append(trace, fmt.Sprintf("usage:%d", projected.Usage.TotalTokens))
		case sessionloop.EventCommandAccepted:
			trace = append(trace, "command.accepted:"+string(projected.RunID))
		case sessionloop.EventSessionState:
			trace = append(trace, "session.state:"+string(projected.State))
		default:
			trace = append(trace, string(projected.Kind))
		}
	}
	want := []string{
		"entry:start:user:",
		"queue.accepted:q1:next_turn",
		"run.started:run-1",
		"queue.drained:q1:next_turn",
		"entry:next_turn:user:run-1",
		"entry:start:user:run-1",
		"entry:assistant:assistant:run-1",
		"entry:tool:tool:run-1",
		"queue.accepted:q2:steer",
		"queue.accepted:q3:follow_up",
		"entry:steer:user:run-1",
		"entry:follow_up:user:run-1",
		"run.suspended:susp-1:custom.kind",
		"command.accepted:run-1",
		"entry:tool:tool:run-1",
		"usage:42",
		"queue.accepted:q4:next_turn",
		"queue.cancelled:q4:next_turn",
		"run.settled:run-1:failed",
		"agentic.transcript.compaction",
		"usage:43",
		"run.suspended:rec-1:harness.recovery.indeterminate",
		"session.state:faulted",
		"session.state:running",
		"command.accepted:",
		"entry:start:user:run-2",
		"entry:start:user:run-2",
		"run.started:run-2",
		"run.settled:run-2:completed",
		"session.state:idle",
	}
	if fmt.Sprint(trace) != fmt.Sprint(want) {
		t.Fatalf("projection trace:\ngot  %v\nwant %v", trace, want)
	}

	// Detail assertions: thinking is excluded, tool data is codec-independent
	// JSON, tool-result data appears only for JSON content.
	var assistantEntry, jsonTool, plainTool *sessionloop.Entry
	toolSeen := 0
	for index := range events {
		if events[index].Kind != sessionloop.EventEntryCommitted {
			continue
		}
		entry := events[index].Entry
		if entry.Origin == sessionloop.OriginAssistant {
			assistantEntry = entry
		}
		if entry.Origin == sessionloop.OriginTool {
			toolSeen++
			if toolSeen == 1 {
				jsonTool = entry
			} else {
				plainTool = entry
			}
		}
	}
	if assistantEntry == nil || len(assistantEntry.Content) != 2 ||
		assistantEntry.Content[0].Kind != sessionloop.BlockText ||
		assistantEntry.Content[1].Kind != sessionloop.BlockToolCall ||
		string(assistantEntry.Content[1].ToolCall.Data) != `{"q":"x"}` {
		t.Fatalf("assistant entry = %#v", assistantEntry)
	}
	if jsonTool == nil || len(jsonTool.Content) != 1 ||
		string(jsonTool.Content[0].Data) != `{"found":true}` ||
		jsonTool.Content[0].ToolResult.CallID != "call-1" || jsonTool.Content[0].ToolResult.IsError {
		t.Fatalf("json tool entry = %#v", jsonTool)
	}
	if plainTool == nil || plainTool.Content[0].Data != nil ||
		plainTool.Content[0].Text != "plain text result" || !plainTool.Content[0].ToolResult.IsError {
		t.Fatalf("plain tool entry = %#v", plainTool)
	}
}

func TestLoopOutcomeKindMapping(t *testing.T) {
	cases := map[agentic.ExecutionStatus]sessionloop.RunOutcomeKind{
		agentic.ExecutionCompleted:   sessionloop.RunCompleted,
		agentic.ExecutionInterrupted: sessionloop.RunInterrupted,
		agentic.ExecutionFailed:      sessionloop.RunFailed,
		agentic.ExecutionStopped:     sessionloop.RunFailed,
		agentic.ExecutionSuspended:   sessionloop.RunFailed,
	}
	for status, want := range cases {
		if got := loopOutcomeKind(status); got != want {
			t.Fatalf("loopOutcomeKind(%v) = %v, want %v", status, got, want)
		}
	}
}

func TestLoopProjectionRejectsMalformedPayloads(t *testing.T) {
	payloadCodec := jsoncodec.New()
	malformed := []byte("{")
	harnessKinds := []string{
		kindRunOpened, kindRunClosed, kindMessage, kindQueueAccepted, kindQueueDrained,
		kindQueueCancelled, kindUsageCommitted, kindChildUsage, kindResolutionAccepted,
		kindRecoverySuspension, kindRecovered, kindFault,
	}
	for _, kind := range harnessKinds {
		projector := newLoopProjector("session-under-test", payloadCodec, nil)
		record := ownRecord(store.Entry{Seq: 1, ID: "id-1", Kind: kind, Payload: malformed}, ownNature(kind))
		if _, err := projector.apply(context.Background(), record); err == nil {
			t.Fatalf("malformed %s payload was accepted", kind)
		}
	}

	// Outer decode failures surface from the journal-to-record conversion.
	if _, err := loopRecords(payloadCodec, []store.Entry{
		{Schema: 1, Seq: 1, ID: "id-1", Kind: kindAssistantCommitted, Payload: malformed},
	}); err == nil {
		t.Fatal("malformed agentic outer payload was accepted")
	}

	// Inner (double-decode) failures surface from apply.
	agenticTypes := map[agentic.EventType]string{
		agentic.EventTypeAssistantCommitted:   kindAssistantCommitted,
		agentic.EventTypeToolResultCommitted:  kindToolResult,
		agentic.EventTypeTurnMessagesInjected: kindMessagesInjected,
		agentic.EventTypeRunSuspended:         kindRunSuspended,
	}
	for eventType := range agenticTypes {
		projector := newLoopProjector("session-under-test", payloadCodec, nil)
		record := event.Record{Cursor: 1, Nature: agentic.EventAuthoritative, Type: eventType, Source: "agentic", Payload: malformed}
		if _, err := projector.apply(context.Background(), record); err == nil {
			t.Fatalf("malformed inner payload for type %d was accepted", eventType)
		}
	}

	// queue.accepted without its entry payload is corrupt.
	projector := newLoopProjector("session-under-test", payloadCodec, nil)
	entry := loopTestEntry(t, payloadCodec, 1, kindQueueAccepted, queueMutationPayload{ID: "q1"})
	record := ownRecord(entry, ownNature(kindQueueAccepted))
	if _, err := projector.apply(context.Background(), record); err == nil {
		t.Fatal("queue.accepted without entry was accepted")
	}
}

// systemDriver inserts a leading system message so the session persists a
// message.system entry through the legacy projection-validation path.
type systemDriver struct {
	system string
}

func (d *systemDriver) Run(ctx context.Context, prompt string, options ...agentic.RunOption) (*agentic.Result[string], error) {
	message := agentic.NewTextMessage(agentic.RoleUser, prompt)
	execution, err := d.Drive(ctx, agentic.DriveInput{Mode: agentic.DriveStart, Prompt: &message}, options...)
	if execution == nil {
		return nil, err
	}
	return execution.Result, err
}

func (d *systemDriver) Drive(_ context.Context, input agentic.DriveInput, _ ...agentic.RunOption) (*agentic.Execution[string], error) {
	messages := []agentic.Message{agentic.NewTextMessage(agentic.RoleSystem, d.system)}
	messages = append(messages, cloneMessages(input.History)...)
	if input.Prompt != nil {
		messages = append(messages, cloneMessages([]agentic.Message{*input.Prompt})[0])
	}
	return &agentic.Execution[string]{Status: agentic.ExecutionCompleted, Result: &agentic.Result[string]{Messages: messages}}, nil
}

func (d *systemDriver) Resume(context.Context, agentic.ResumeInput, ...agentic.RunOption) (*agentic.Execution[string], error) {
	return nil, fmt.Errorf("unexpected Resume")
}

// TestLoopProjectionPrivacy proves system instructions, injected context,
// thinking text, and exchange instructions never reach projected events,
// snapshot entries, or their JSON serialization (plan §8.5/§13).
func TestLoopProjectionPrivacy(t *testing.T) {
	secrets := []string{
		"SYSTEM-SECRET", "CONTEXT-SECRET", "THINKING-SECRET", "HISTORY-SECRET", "INSTRUCTIONS-SECRET",
	}

	// Part 1: fabricated journal covering every privacy-excluded shape.
	payloadCodec := jsoncodec.New()
	assistant := agentic.Message{Role: agentic.RoleAssistant, Content: []agentic.Part{
		{Type: agentic.ContentThinking, Thinking: &agentic.ThinkingBlock{Text: "THINKING-SECRET"}},
		{Type: agentic.ContentText, Text: "visible answer"},
	}}
	entries := []store.Entry{
		loopTestEntry(t, payloadCodec, 1, kindSessionCreated, sessionCreatedPayload{}),
		loopTestEntry(t, payloadCodec, 2, kindMessage, messagePayload{
			Message: agentic.NewTextMessage(agentic.RoleSystem, "HISTORY-SECRET"), Source: "initial_history"}),
		loopTestEntry(t, payloadCodec, 3, kindRunOpened, runOpenedPayload{ID: "run-1", Mode: "start", Instructions: "INSTRUCTIONS-SECRET"}),
		loopTestEntry(t, payloadCodec, 4, kindSystemMessage, messagePayload{
			Message: agentic.NewTextMessage(agentic.RoleSystem, "SYSTEM-SECRET"), Source: "driver_system"}),
		loopTestEntry(t, payloadCodec, 5, kindContextMessage, contextMessagePayload{
			Message: agentic.NewTextMessage(agentic.RoleUser, "CONTEXT-SECRET")}),
		loopAgenticEntry(t, payloadCodec, 6, kindAssistantCommitted, agentic.EventTypeAssistantCommitted, event.AssistantPayload{Message: assistant}),
		loopTestEntry(t, payloadCodec, 7, kindRunClosed, runClosedPayload{ID: "run-1", Status: agentic.ExecutionCompleted}),
	}
	events := projectLoopEntries(t, payloadCodec, entries)
	serialized, err := json.Marshal(events)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range secrets {
		if strings.Contains(string(serialized), secret) {
			t.Fatalf("projected events leaked %q: %s", secret, serialized)
		}
	}
	if !strings.Contains(string(serialized), "visible answer") {
		t.Fatal("projection lost the visible assistant text")
	}

	// Part 2: a real session with exchange instructions and a driver-inserted
	// system message, observed through the live view stream and snapshot.
	driver := &systemDriver{system: "SYSTEM-SECRET"}
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	config.Instructions = harnessruntime.ExchangeInstructionProviderFunc(
		func(context.Context, harnessruntime.ExchangeContext) (string, error) {
			return "INSTRUCTIONS-SECRET", nil
		})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	view, err := NewLoopView(session, LoopConfig[string]{CloseRoot: session.Close})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = view.Close(context.Background()) })
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 128})
	receipt := loopDispatch(t, view, sessionloopStartCommand("visible prompt"))
	_, seen := awaitLoopSettled(t, stream, receipt.RunID)
	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	liveSerialized, err := json.Marshal(struct {
		Events   []sessionloop.Event
		Snapshot sessionloop.Snapshot
	}{seen, snapshot})
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range secrets {
		if strings.Contains(string(liveSerialized), secret) {
			t.Fatalf("live view leaked %q", secret)
		}
	}
}

func TestLoopProjectionPreviews(t *testing.T) {
	payloadCodec := jsoncodec.New()
	projector := newLoopProjector("session-under-test", payloadCodec, nil)

	// Establish a durable position first.
	opened := ownRecord(loopTestEntry(t, payloadCodec, 7, kindRunOpened, runOpenedPayload{ID: "run-1"}), agentic.EventAuthoritative)
	if _, err := projector.apply(context.Background(), opened); err != nil {
		t.Fatal(err)
	}
	encode := func(value any) []byte {
		encoded, err := codec.Encode(payloadCodec, value)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	previews := []event.Record{
		{Cursor: 7, Nature: agentic.EventPreview, Type: agentic.EventTypeTextPreview, Source: "agentic",
			Payload: encode(struct{ Delta string }{"hel"})},
		{Cursor: 7, Nature: agentic.EventPreview, Type: agentic.EventTypeThinkingPreview, Source: "agentic",
			Payload: encode(struct{ Delta string }{"thinking-delta"})},
		{Cursor: 7, Nature: agentic.EventPreview, Type: agentic.EventTypeToolCallPreview, Source: "agentic",
			Payload: encode(event.ToolBatchPayload{Calls: []agentic.ToolUse{{ID: "call-9", Name: "lookup"}}})},
		{Cursor: 7, Nature: agentic.EventPreview, Type: agentic.EventTypeToolArgumentPreview, Source: "agentic",
			Payload: encode(struct {
				ToolCallID string
				Delta      string
			}{"call-9", `{"arg`})},
		{Cursor: 7, Nature: agentic.EventPreview, Source: "tool", Name: "progress", Payload: encode(struct{}{})},
	}
	var projected []sessionloop.Event
	for _, record := range previews {
		events, err := projector.apply(context.Background(), record)
		if err != nil {
			t.Fatal(err)
		}
		projected = append(projected, events...)
	}
	if len(projected) != 5 {
		t.Fatalf("projected %d preview events, want 5", len(projected))
	}
	wantKinds := []sessionloop.PreviewKind{
		sessionloop.PreviewText, sessionloop.PreviewThinking,
		sessionloop.PreviewTool, sessionloop.PreviewTool, sessionloop.PreviewTool,
	}
	for index, preview := range projected {
		if preview.Nature != sessionloop.EventPreview || preview.Kind != sessionloop.EventPreviewDelta {
			t.Fatalf("preview %d shape = %#v", index, preview)
		}
		if preview.Ordinal != uint64(index+1) {
			t.Fatalf("preview %d ordinal = %d, want stream-local %d", index, preview.Ordinal, index+1)
		}
		if preview.Position.Sequence != 7 || preview.Position.IsZero() {
			t.Fatalf("preview %d position = %#v, must repeat the latest durable position", index, preview.Position)
		}
		if preview.Preview.Kind != wantKinds[index] {
			t.Fatalf("preview %d kind = %s, want %s", index, preview.Preview.Kind, wantKinds[index])
		}
	}
	if projected[0].Preview.Text != "hel" || projected[3].Preview.ToolCallID != "call-9" ||
		projected[2].Preview.ToolCallID != "call-9" {
		t.Fatalf("preview payloads = %#v", projected)
	}
	if projected[4].Preview.Text != "progress" {
		t.Fatalf("tool-update preview lost its identity: %#v", projected[4].Preview)
	}

	// An unknown preview shape is dropped (previews are lossy) and its loss
	// count carries onto the next projected event.
	dropped := event.Record{Cursor: 7, Nature: agentic.EventPreview, Type: agentic.EventType(250),
		Source: "agentic", Payload: encode(struct{}{}), Dropped: event.EventsDropped{Preview: 3}}
	if events, err := projector.apply(context.Background(), dropped); err != nil || len(events) != 0 {
		t.Fatalf("unknown preview projected %d events err=%v", len(events), err)
	}
	carrier := event.Record{Cursor: 7, Nature: agentic.EventPreview, Type: agentic.EventTypeTextPreview,
		Source: "agentic", Payload: encode(struct{ Delta string }{"lo"})}
	events, err := projector.apply(context.Background(), carrier)
	if err != nil || len(events) != 1 {
		t.Fatalf("carrier projected %d events err=%v", len(events), err)
	}
	if events[0].Dropped != 3 {
		t.Fatalf("carrier Dropped = %d, want the accumulated 3", events[0].Dropped)
	}

	// Child-session records are ignored entirely.
	child := event.Record{Cursor: 8, Nature: agentic.EventAuthoritative, SessionID: "someone-else",
		Source: "harness", Name: kindRunOpened, Payload: encode(runOpenedPayload{ID: "child-run"})}
	if events, err := projector.apply(context.Background(), child); err != nil || len(events) != 0 {
		t.Fatalf("child record projected %d events err=%v", len(events), err)
	}

	// A drained queue ID this fold never saw degrades to the bare ID.
	unknownDrain := ownRecord(loopTestEntry(t, payloadCodec, 9, kindQueueDrained, queueMutationPayload{ID: "mystery"}), agentic.EventAuthoritative)
	events, err = projector.apply(context.Background(), unknownDrain)
	if err != nil || len(events) != 1 || events[0].Queue.ID != "mystery" || events[0].Queue.Kind != "" {
		t.Fatalf("unknown drain projection = %#v err=%v", events, err)
	}
}
