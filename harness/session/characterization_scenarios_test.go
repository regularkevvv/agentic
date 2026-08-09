package session

// Characterization scenarios 1-6, 8, 9, and context cancellation from the
// sessionloop plan (section 8.2). Every test drives the public surface only
// and freezes the observable contract with committed goldens and exact
// assertions.

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/permission"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

// Scenario 1: a tool-free prompt freezes the journal, the authoritative event
// sequence, the surrounding snapshots, and the durable frontier visible at
// the exact moment Driver.Drive is invoked.
func TestCharacterizationToolFreePromptFreezesJournalEventsAndSnapshot(t *testing.T) {
	driver := &countingDriver{}
	config := characterizationConfig(t, driver, storememory.New())
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	before, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if before.Cursor != 1 || before.State != Idle || before.RunID != "" ||
		len(before.Messages) != 0 || len(before.Pending) != 0 || before.Suspension != nil ||
		before.Usage.TotalTokens != 0 || before.Usage.Requests != 0 {
		t.Fatalf("pre-prompt snapshot = %#v", before)
	}

	inspected := false
	driver.before = func(input agentic.DriveInput) error {
		loaded, loadErr := session.journal.Load(context.Background())
		if loadErr != nil {
			return loadErr
		}
		snapshot, snapshotErr := session.Snapshot(context.Background())
		if snapshotErr != nil {
			return snapshotErr
		}
		// The durable frontier at Drive time is exactly the acceptance batch:
		// session.created + run.opened + prompt, already at the snapshot cursor.
		kinds := journalKinds(loaded.Entries)
		if len(kinds) != 3 || kinds[0] != kindSessionCreated || kinds[1] != kindRunOpened || kinds[2] != kindMessage {
			return fmt.Errorf("durable frontier kinds at Drive = %v", kinds)
		}
		if loaded.Cursor.Seq != snapshot.Cursor {
			return fmt.Errorf("driver observed journal cursor %d while snapshot was at %d", loaded.Cursor.Seq, snapshot.Cursor)
		}
		if snapshot.State != Running || snapshot.RunID != "run_c1" {
			return fmt.Errorf("mid-drive snapshot = %#v", snapshot)
		}
		opened, decodeErr := decodePayload[runOpenedPayload](config.Codec, loaded.Entries[1])
		if decodeErr != nil {
			return decodeErr
		}
		if opened.ID != "run_c1" || opened.Mode != "start" || opened.Recovery || opened.Instructions != "" {
			return fmt.Errorf("run.opened payload at Drive = %#v", opened)
		}
		prompt, decodeErr := decodePayload[messagePayload](config.Codec, loaded.Entries[2])
		if decodeErr != nil {
			return decodeErr
		}
		if prompt.Source != "prompt" || prompt.Message.GetTextContent() != "hello characterization" {
			return fmt.Errorf("prompt payload at Drive = %#v", prompt)
		}
		if len(input.History) != 0 || input.Prompt == nil || input.Prompt.GetTextContent() != "hello characterization" {
			return fmt.Errorf("drive input = %#v", input)
		}
		inspected = true
		return nil
	}

	execution, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "hello characterization"))
	if err != nil {
		t.Fatalf("Prompt error = %v", err)
	}
	if execution == nil || execution.Status != agentic.ExecutionCompleted {
		t.Fatalf("execution = %#v", execution)
	}
	if !inspected || driver.Count() != 1 {
		t.Fatalf("inspected=%t drives=%d", inspected, driver.Count())
	}

	after, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if after.Cursor != 4 || after.State != Idle || after.RunID != "" ||
		len(after.Messages) != 1 || after.Messages[0].GetTextContent() != "hello characterization" ||
		len(after.Pending) != 0 || after.Suspension != nil ||
		after.Usage.TotalTokens != 0 || after.Usage.Requests != 0 {
		t.Fatalf("post-prompt snapshot = %#v", after)
	}

	entries := loadJournalEntries(t, session)
	compareGolden(t, "tool_free_prompt.golden.json",
		marshalGolden(t, normalizeJournal(t, config.Codec, config.ID, entries)))
	records := replayDurableRecords(t, session, len(entries))
	compareGolden(t, "tool_free_prompt.events.golden.json",
		marshalGolden(t, normalizeEvents(records)))
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

type characterizationToolInput struct {
	Value string `json:"value"`
}

// Scenario 2: a two-tool exchange over the real agent loop with handlers
// supplied through config.Toolsets freezes the ordering of
// tool_batch_planned/tool_started/tool_result/assistant_committed and the
// full journal and event sequence.
func TestCharacterizationMultiToolExchange(t *testing.T) {
	calls := []agentic.ToolUse{
		{ID: "tool-1", Name: "effect", Input: map[string]any{"value": "one"}},
		{ID: "tool-2", Name: "effect", Input: map[string]any{"value": "two"}},
	}
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(calls...), usage: agentic.Usage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10, Requests: 1}},
		textStep("done"),
	}}
	var handled atomic.Int32
	tool, handler := agentic.MustToolPlain("effect", "Apply a deterministic effect",
		func(input characterizationToolInput) (string, error) {
			handled.Add(1)
			return "applied " + input.Value, nil
		})
	config := characterizationConfig(t, agentic.NewAgent("", model), storememory.New())
	config.Toolsets = []agentic.Toolset{agentic.NewToolset().Add(tool, handler)}
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "run tools"))
	if err != nil || execution.Status != agentic.ExecutionCompleted {
		t.Fatalf("execution=%#v err=%v", execution, err)
	}
	if handled.Load() != 2 {
		t.Fatalf("toolset handler ran %d times", handled.Load())
	}

	entries := loadJournalEntries(t, session)
	var toolOrder []string
	for _, entry := range entries {
		switch entry.Kind {
		case kindAssistantCommitted, kindToolBatchPlanned, kindToolStarted, kindToolResult:
			toolOrder = append(toolOrder, entry.Kind)
		}
	}
	wantOrder := []string{
		kindAssistantCommitted, kindToolBatchPlanned,
		kindToolStarted, kindToolStarted,
		kindToolResult, kindToolResult,
		kindAssistantCommitted,
	}
	if fmt.Sprint(toolOrder) != fmt.Sprint(wantOrder) {
		t.Fatalf("tool entry order = %v, want %v", toolOrder, wantOrder)
	}
	compareGolden(t, "multi_tool.golden.json",
		marshalGolden(t, normalizeJournal(t, config.Codec, config.ID, entries)))
	records := replayDurableRecords(t, session, len(entries))
	compareGolden(t, "multi_tool.events.golden.json",
		marshalGolden(t, normalizeEvents(records)))
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// Scenario 3: a steer accepted mid-turn drains at the next turn boundary
// ahead of a follow-up, and the follow-up drains only at a later valid
// candidate boundary with no queued steer. Receipts, the accepted/drained
// sequence, and injected-message ordering are frozen.
func TestCharacterizationSteerAndFollowUpTiming(t *testing.T) {
	enteredFirst := make(chan struct{})
	releaseFirst := make(chan struct{})
	enteredSecond := make(chan struct{})
	releaseSecond := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "first"), usage: agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}, entered: enteredFirst, release: releaseFirst},
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "second"), usage: agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}, entered: enteredSecond, release: releaseSecond},
		textStep("final"),
	}}
	config := characterizationConfig(t, agentic.NewAgent("", model), storememory.New())
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	done := promptAsync(session, "hello")
	awaitSignal(t, enteredFirst, "first model turn")

	steer, err := session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "steer please"))
	if err != nil {
		t.Fatal(err)
	}
	follow, err := session.FollowUp(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "follow please"))
	if err != nil {
		t.Fatal(err)
	}
	if steer.ID != "queue_c1" || steer.Kind != QueueSteer ||
		follow.ID != "queue_c2" || follow.Kind != QueueFollowUp ||
		follow.Cursor <= steer.Cursor {
		t.Fatalf("receipts = %#v, %#v", steer, follow)
	}
	close(releaseFirst)

	// At the first boundary only the steer drained even though the candidate
	// was valid; the follow-up must still be pending during the second turn.
	awaitSignal(t, enteredSecond, "second model turn")
	mid, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if mid.State != Running || mid.RunID != "run_c1" ||
		len(mid.Pending) != 1 || mid.Pending[0].ID != follow.ID {
		t.Fatalf("mid-run snapshot = %#v", mid)
	}
	close(releaseSecond)

	outcome := awaitPrompt(t, done, "prompt settlement")
	if outcome.err != nil || outcome.execution.Status != agentic.ExecutionCompleted {
		t.Fatalf("execution=%#v err=%v", outcome.execution, outcome.err)
	}

	entries := loadJournalEntries(t, session)
	acceptedCursorByID := map[string]uint64{}
	var queueTrace []string
	for _, entry := range entries {
		switch entry.Kind {
		case kindQueueAccepted, kindQueueDrained:
			payload, decodeErr := decodePayload[queueMutationPayload](config.Codec, entry)
			if decodeErr != nil {
				t.Fatal(decodeErr)
			}
			queueTrace = append(queueTrace, entry.Kind+":"+payload.ID)
			if entry.Kind == kindQueueAccepted {
				acceptedCursorByID[payload.ID] = entry.Seq
			}
		}
	}
	wantTrace := []string{
		"queue.accepted:queue_c1", "queue.accepted:queue_c2",
		"queue.drained:queue_c1", "queue.drained:queue_c2",
	}
	if fmt.Sprint(queueTrace) != fmt.Sprint(wantTrace) {
		t.Fatalf("queue trace = %v, want %v", queueTrace, wantTrace)
	}
	if acceptedCursorByID[steer.ID] != steer.Cursor || acceptedCursorByID[follow.ID] != follow.Cursor {
		t.Fatalf("receipt cursors %d/%d do not match accepted sequences %v", steer.Cursor, follow.Cursor, acceptedCursorByID)
	}

	final, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var texts []string
	for _, message := range final.Messages {
		texts = append(texts, message.GetTextContent())
	}
	wantTexts := []string{"hello", "first", "steer please", "second", "follow please", "final"}
	if fmt.Sprint(texts) != fmt.Sprint(wantTexts) {
		t.Fatalf("injected-message ordering = %v, want %v", texts, wantTexts)
	}
	compareGolden(t, "steer_follow_up.golden.json",
		marshalGolden(t, normalizeJournal(t, config.Codec, config.ID, entries)))
	records := replayDurableRecords(t, session, len(entries))
	compareGolden(t, "steer_follow_up.events.golden.json",
		marshalGolden(t, normalizeEvents(records)))
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// Scenario 4: a queued next_turn input survives interruption of the running
// prompt and is drained exactly once inside the next run.opened batch.
func TestCharacterizationNextTurnSurvivesInterruption(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{}) // never closed: the run is interrupted
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "never delivered"), entered: entered, release: release},
		textStep("after interrupt"),
	}}
	config := characterizationConfig(t, agentic.NewAgent("", model), storememory.New())
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	done := promptAsync(session, "interrupted prompt")
	awaitSignal(t, entered, "model entry")
	next, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "queued next"))
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Interrupt(context.Background()); err != nil {
		t.Fatal(err)
	}
	outcome := awaitPrompt(t, done, "interrupted prompt settlement")
	if !errors.Is(outcome.err, context.Canceled) {
		t.Fatalf("interrupted Prompt error = %v", outcome.err)
	}
	interrupted, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if interrupted.State != Idle || len(interrupted.Pending) != 1 || interrupted.Pending[0].ID != next.ID {
		t.Fatalf("next_turn did not survive interruption: %#v", interrupted)
	}
	cursorBeforeSecond := interrupted.Cursor

	execution, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "second prompt"))
	if err != nil || execution.Status != agentic.ExecutionCompleted {
		t.Fatalf("execution=%#v err=%v", execution, err)
	}

	entries := loadJournalEntries(t, session)
	// The run.opened acceptance batch of the second prompt drains the queued
	// next_turn exactly once, ahead of the prompt message.
	var batchTrace []string
	for _, entry := range entries {
		if entry.Seq <= cursorBeforeSecond || len(batchTrace) == 4 {
			continue
		}
		switch entry.Kind {
		case kindRunOpened:
			batchTrace = append(batchTrace, kindRunOpened)
		case kindQueueDrained:
			payload, decodeErr := decodePayload[queueMutationPayload](config.Codec, entry)
			if decodeErr != nil {
				t.Fatal(decodeErr)
			}
			batchTrace = append(batchTrace, kindQueueDrained+":"+payload.ID)
		case kindMessage:
			payload, decodeErr := decodePayload[messagePayload](config.Codec, entry)
			if decodeErr != nil {
				t.Fatal(decodeErr)
			}
			batchTrace = append(batchTrace, kindMessage+":"+payload.Source+":"+payload.Message.GetTextContent())
		}
	}
	wantBatch := []string{
		kindRunOpened,
		kindQueueDrained + ":" + next.ID,
		kindMessage + ":next_turn:queued next",
		kindMessage + ":prompt:second prompt",
	}
	if fmt.Sprint(batchTrace) != fmt.Sprint(wantBatch) {
		t.Fatalf("run.opened batch composition = %v, want %v", batchTrace, wantBatch)
	}
	if countEntries(entries, kindQueueDrained) != 1 {
		t.Fatalf("next_turn drained %d times", countEntries(entries, kindQueueDrained))
	}
	compareGolden(t, "next_turn_interrupt.golden.json",
		marshalGolden(t, normalizeJournal(t, config.Codec, config.ID, entries)))
	records := replayDurableRecords(t, session, len(entries))
	compareGolden(t, "next_turn_interrupt.events.golden.json",
		marshalGolden(t, normalizeEvents(records)))
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// Scenario 5: a permission-gated suspension resumes with the exchange
// instructions resolved exactly once and the same run identity across
// suspend/resume; the resolution.accepted payload and the durable state
// trace (run.opened -> run_suspended -> resolution.accepted -> run.closed)
// are frozen.
func TestCharacterizationSuspensionResumeStableInstructions(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "gate-1", Name: "danger", Input: map[string]any{"value": "x"}})},
		textStep("finished"),
	}}
	agent := agentic.NewAgent("base", model)
	agentic.AddTool(agent,
		func(context.Context, resumeToolInput) (string, error) { return "gated ok", nil },
		agentic.AutoToolName("danger"),
		agentic.AutoToolDescription("Perform a gated action"),
	)
	policy, err := permission.New(permission.DecisionDeny,
		permission.Rule{Pattern: "tool/danger/**", Decision: permission.DecisionAsk},
	)
	if err != nil {
		t.Fatal(err)
	}
	permissionCapability, err := permission.NewCapability(policy)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(permissionCapability)
	if err != nil {
		t.Fatal(err)
	}
	var resolutions atomic.Int32
	config := characterizationConfig(t, agent, storememory.New())
	config.ToolGate = plan.ToolGate()
	config.Context = plan.ContextPolicy()
	config.Instructions = harnessruntime.ExchangeInstructionProviderFunc(
		func(context.Context, harnessruntime.ExchangeContext) (string, error) {
			return fmt.Sprintf("instr-%d", resolutions.Add(1)), nil
		})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}

	execution, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "start"))
	if err != nil {
		t.Fatal(err)
	}
	if execution.Status != agentic.ExecutionSuspended || execution.Suspension == nil || session.State() != Suspended {
		t.Fatalf("execution=%#v state=%s", execution, session.State())
	}
	suspended, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if suspended.RunID != "run_c1" || suspended.Suspension == nil {
		t.Fatalf("suspended snapshot = %#v", suspended)
	}

	resumed, err := session.Resume(context.Background(), ResumeRequest{
		SuspensionID: execution.Suspension.ID,
		Resolutions:  []ToolResolution{{CallID: "gate-1", Action: ResolutionApprove}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resumed.Status != agentic.ExecutionCompleted || session.State() != Idle {
		t.Fatalf("resumed execution=%#v state=%s", resumed, session.State())
	}
	if resolutions.Load() != 1 {
		t.Fatalf("instruction provider resolved %d times, want exactly once for the run", resolutions.Load())
	}

	entries := loadJournalEntries(t, session)
	if countEntries(entries, kindRunOpened) != 1 || countEntries(entries, kindRunClosed) != 1 ||
		countEntries(entries, kindResolutionAccepted) != 1 {
		t.Fatalf("run/resolution entry counts = %v", journalKinds(entries))
	}
	opened, err := decodePayload[runOpenedPayload](config.Codec, entries[firstEntryIndex(entries, kindRunOpened)])
	if err != nil {
		t.Fatal(err)
	}
	if opened.ID != "run_c1" || opened.Instructions != "instr-1" {
		t.Fatalf("run.opened payload = %#v", opened)
	}
	closed, err := decodePayload[runClosedPayload](config.Codec, entries[firstEntryIndex(entries, kindRunClosed)])
	if err != nil {
		t.Fatal(err)
	}
	if closed.ID != opened.ID || closed.Status != agentic.ExecutionCompleted || closed.Error != "" {
		t.Fatalf("run identity changed across suspend/resume: opened=%#v closed=%#v", opened, closed)
	}
	resolution, err := decodePayload[resolutionAcceptedPayload](config.Codec, entries[firstEntryIndex(entries, kindResolutionAccepted)])
	if err != nil {
		t.Fatal(err)
	}
	if resolution.SuspensionID != execution.Suspension.ID ||
		len(resolution.Request.Resolutions) != 1 ||
		resolution.Request.Resolutions[0].CallID != "gate-1" ||
		resolution.Request.Resolutions[0].Action != ResolutionApprove ||
		resolution.Request.Prompt != nil {
		t.Fatalf("resolution.accepted payload = %#v", resolution)
	}

	// Durable state trace Idle->Running->Suspended->Running->Idle: run.opened
	// (Running), agentic.run_suspended (Suspended), resolution.accepted
	// (Running again), run.closed (Idle) in strictly this order.
	openedIndex := firstEntryIndex(entries, kindRunOpened)
	suspendedIndex := firstEntryIndex(entries, kindRunSuspended)
	resolutionIndex := firstEntryIndex(entries, kindResolutionAccepted)
	closedIndex := firstEntryIndex(entries, kindRunClosed)
	if openedIndex < 0 || suspendedIndex < 0 || resolutionIndex < 0 || closedIndex < 0 ||
		!(openedIndex < suspendedIndex && suspendedIndex < resolutionIndex && resolutionIndex < closedIndex) {
		t.Fatalf("state trace indexes opened=%d suspended=%d resolution=%d closed=%d in %v",
			openedIndex, suspendedIndex, resolutionIndex, closedIndex, journalKinds(entries))
	}

	// The resumed exchange reuses the cached instructions verbatim.
	calls := model.Calls()
	if len(calls) != 2 {
		t.Fatalf("model calls = %d", len(calls))
	}
	system := requestSystem(calls[1])
	if !containsInstruction(system, "instr-1") || containsInstruction(system, "instr-2") {
		t.Fatalf("resumed system instructions = %q", system)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func containsInstruction(system, instruction string) bool {
	return len(system) > 0 && stringsContains(system, instruction)
}

// lateResultModel blocks in Request until released and deliberately ignores
// context cancellation so an in-flight interrupt races a successful model
// response.
type lateResultModel struct {
	once    sync.Once
	entered chan struct{}
	release chan struct{}
	message agentic.Message
}

func (m *lateResultModel) Name() string { return "test:late-result" }

func (m *lateResultModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	m.once.Do(func() { close(m.entered) })
	<-m.release
	return &agentic.ChatResponse{
		Model:           m.Name(),
		Message:         m.message,
		Usage:           agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		FinishReason:    agentic.FinishReasonStop,
		RawFinishReason: string(agentic.FinishReasonStop),
	}, nil
}

// Scenario 6: a model result delivered after interruption began is still
// committed durably BEFORE interrupt.marker and run.closed; the execution
// error is context.Canceled and the session settles Idle.
func TestCharacterizationInterruptDeliversLateResultBeforeSettlement(t *testing.T) {
	model := &lateResultModel{
		entered: make(chan struct{}),
		release: make(chan struct{}),
		message: agentic.NewTextMessage(agentic.RoleAssistant, "late answer"),
	}
	config := characterizationConfig(t, agentic.NewAgent("", model), storememory.New())
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	done := promptAsync(session, "interrupt me")
	awaitSignal(t, model.entered, "model entry")

	interruptDone := make(chan error, 1)
	go func() {
		interruptDone <- session.Interrupt(context.Background())
	}()
	// Release the model only after the interrupt is observably in flight.
	awaitState(t, session, Interrupting)
	close(model.release)

	if err := awaitErr(t, interruptDone, "interrupt settlement"); err != nil {
		t.Fatalf("Interrupt error = %v", err)
	}
	outcome := awaitPrompt(t, done, "prompt settlement")
	if !errors.Is(outcome.err, context.Canceled) {
		t.Fatalf("Prompt error = %v, want context.Canceled", outcome.err)
	}
	if session.State() != Idle {
		t.Fatalf("state = %s", session.State())
	}

	entries := loadJournalEntries(t, session)
	assistantIndex := firstEntryIndex(entries, kindAssistantCommitted)
	markerIndex := firstEntryIndex(entries, kindInterruptMarker)
	closedIndex := firstEntryIndex(entries, kindRunClosed)
	if assistantIndex < 0 || markerIndex < 0 || closedIndex < 0 ||
		!(assistantIndex < markerIndex && markerIndex < closedIndex) {
		t.Fatalf("late result ordering assistant=%d marker=%d closed=%d in %v",
			assistantIndex, markerIndex, closedIndex, journalKinds(entries))
	}
	committedRecord, err := decodePayload[event.Record](config.Codec, entries[assistantIndex])
	if err != nil {
		t.Fatal(err)
	}
	committed, err := event.Decode[event.AssistantPayload](config.Codec, committedRecord)
	if err != nil {
		t.Fatal(err)
	}
	if committed.Message.GetTextContent() != "late answer" {
		t.Fatalf("late assistant commit = %#v", committed.Message)
	}
	closed, err := decodePayload[runClosedPayload](config.Codec, entries[closedIndex])
	if err != nil {
		t.Fatal(err)
	}
	if closed.ID != "run_c1" || closed.Status != agentic.ExecutionInterrupted || closed.Error != context.Canceled.Error() {
		t.Fatalf("run.closed payload = %#v", closed)
	}
	compareGolden(t, "late_result_interrupt.golden.json",
		marshalGolden(t, normalizeJournal(t, config.Codec, config.ID, entries)))
	records := replayDurableRecords(t, session, len(entries))
	compareGolden(t, "late_result_interrupt.events.golden.json",
		marshalGolden(t, normalizeEvents(records)))
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// Scenario 8: preview overflow is lossy and counted on the next delivered
// record; authoritative overflow terminally disconnects the subscriber with
// ErrSubscriberLagged and closed channels; resubscribing replays the missed
// durable suffix exactly.
func TestCharacterizationPreviewLossAndSubscriberLag(t *testing.T) {
	driver := &countingDriver{}
	config := characterizationConfig(t, driver, storememory.New())
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}

	// Preview loss: buffer of one, three previews published, none drained.
	previews := session.Subscribe(SubscribeOptions{AfterCursor: 1, Buffer: 1, Preview: true})
	for i := 0; i < 3; i++ {
		if err := session.Emit(context.Background(), previewEvent{turn: 0}); err != nil {
			t.Fatal(err)
		}
	}
	first := <-previews.Events
	if first.Nature != agentic.EventPreview || !first.Dropped.Empty() {
		t.Fatalf("first preview = %#v", first)
	}
	if err := session.Emit(context.Background(), previewEvent{turn: 0}); err != nil {
		t.Fatal(err)
	}
	fourth := <-previews.Events
	if fourth.Nature != agentic.EventPreview || fourth.Dropped.Preview != 2 {
		t.Fatalf("preview after loss = %#v, want Dropped.Preview=2", fourth)
	}
	previews.Close()

	// Authoritative overflow: the prompt commits three durable records while
	// the subscriber holds a buffer of one and never drains.
	lagged := session.Subscribe(SubscribeOptions{AfterCursor: 1, Buffer: 1})
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "one")); err != nil {
		t.Fatal(err)
	}
	delivered := 0
	for range lagged.Events {
		delivered++
	}
	var lag *event.ErrSubscriberLagged
	if err := <-lagged.Err; !errors.As(err, &lag) {
		t.Fatalf("lag error = %v", err)
	}
	if delivered != 1 || lag.LastCursor != 2 {
		t.Fatalf("delivered=%d lastCursor=%d, want 1 delivered and lag at cursor 2", delivered, lag.LastCursor)
	}
	if _, open := <-lagged.Events; open {
		t.Fatal("events channel stayed open after authoritative lag")
	}
	if _, open := <-lagged.Err; open {
		t.Fatal("error channel stayed open after authoritative lag")
	}

	// Snapshot plus resubscribe from the lagged cursor replays the missed
	// durable suffix exactly.
	snapshot, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	entries := loadJournalEntries(t, session)
	replayed := session.Subscribe(SubscribeOptions{AfterCursor: lag.LastCursor, Buffer: 16})
	var suffix []event.Record
	for len(suffix) == 0 || suffix[len(suffix)-1].Cursor < snapshot.Cursor {
		record, open := <-replayed.Events
		if !open {
			t.Fatalf("replay closed after %d records", len(suffix))
		}
		suffix = append(suffix, record)
	}
	if len(suffix) != 2 ||
		suffix[0].Cursor != 3 || suffix[0].Name != entries[2].Kind ||
		suffix[1].Cursor != 4 || suffix[1].Name != entries[3].Kind {
		t.Fatalf("replayed suffix = %#v", normalizeEvents(suffix))
	}
	replayed.Close()

	// Resubscribing after the snapshot cursor replays nothing and receives
	// only records committed later.
	resumed := session.Subscribe(SubscribeOptions{AfterCursor: snapshot.Cursor, Buffer: 16})
	defer resumed.Close()
	receipt, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "later"))
	if err != nil {
		t.Fatal(err)
	}
	next := <-resumed.Events
	if next.Cursor != receipt.Cursor || next.Cursor <= snapshot.Cursor || next.Name != kindQueueAccepted {
		t.Fatalf("post-snapshot record = %#v", next)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// Scenario 9: an append failure on write-ahead queue acceptance returns the
// error without faulting, while an append failure on a mid-run durable emit
// faults the session permanently for every subsequent public call.
func TestCharacterizationJournalAppendConflictFaults(t *testing.T) {
	writeErr := errors.New("characterization append failure")

	t.Run("QueueAcceptanceFailureDoesNotFault", func(t *testing.T) {
		repository := &failingRepository{base: storememory.New()}
		driver := &countingDriver{}
		config := characterizationConfig(t, driver, repository)
		session, err := New(context.Background(), config)
		if err != nil {
			t.Fatal(err)
		}
		repository.fail(kindQueueAccepted, writeErr)
		if _, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "rejected")); !errors.Is(err, writeErr) || errors.Is(err, ErrSessionFaulted) {
			t.Fatalf("queue acceptance error = %v", err)
		}
		snapshot, err := session.Snapshot(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if snapshot.State != Idle || len(snapshot.Pending) != 0 {
			t.Fatalf("failed acceptance mutated state: %#v", snapshot)
		}
		// The session stays usable; the failed attempt consumed queue_c1, so
		// the next acceptance carries queue_c2.
		receipt, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "accepted"))
		if err != nil || receipt.ID != "queue_c2" {
			t.Fatalf("post-failure receipt = %#v, %v", receipt, err)
		}
		if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "still works")); err != nil {
			t.Fatal(err)
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("MidRunEmitFailureFaults", func(t *testing.T) {
		repository := &failingRepository{base: storememory.New()}
		model := &scriptedModel{steps: []modelStep{textStep("will fault")}}
		config := characterizationConfig(t, agentic.NewAgent("", model), repository)
		session, err := New(context.Background(), config)
		if err != nil {
			t.Fatal(err)
		}
		repository.fail(kindAssistantCommitted, writeErr)
		if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "fault run")); !errors.Is(err, ErrSessionFaulted) || !errors.Is(err, writeErr) {
			t.Fatalf("faulting Prompt error = %v", err)
		}
		if session.State() != Faulted {
			t.Fatalf("state = %s", session.State())
		}
		var fault *FaultError
		if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "after fault")); !errors.As(err, &fault) || fault.SessionID != config.ID {
			t.Fatalf("post-fault Prompt error = %v", err)
		}
		if _, err := session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "after fault")); !errors.Is(err, ErrSessionFaulted) {
			t.Fatalf("post-fault Steer error = %v", err)
		}
		if err := session.WaitForIdle(context.Background()); !errors.Is(err, ErrSessionFaulted) {
			t.Fatalf("post-fault WaitForIdle error = %v", err)
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})
}

// Context cancellation characterization: pre-canceled acceptance leaves the
// journal untouched, cancellation during model execution or during tool
// execution settles exactly like an interrupt (with the canceled tool's error
// result durably committed before settlement), a canceled Snapshot returns
// the context error, and Close with a pre-canceled context reports the
// context error while still transitioning the session to Closed.
func TestCharacterizationContextCancellation(t *testing.T) {
	t.Run("PreCanceledPromptLeavesJournalUnchanged", func(t *testing.T) {
		driver := &countingDriver{}
		config := characterizationConfig(t, driver, storememory.New())
		session, err := New(context.Background(), config)
		if err != nil {
			t.Fatal(err)
		}
		before := marshalGolden(t, normalizeJournal(t, config.Codec, config.ID, loadJournalEntries(t, session)))
		canceled, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := session.Prompt(canceled, agentic.NewTextMessage(agentic.RoleUser, "never")); !errors.Is(err, context.Canceled) {
			t.Fatalf("pre-canceled Prompt error = %v", err)
		}
		after := marshalGolden(t, normalizeJournal(t, config.Codec, config.ID, loadJournalEntries(t, session)))
		if len(before) != len(after) || !goldenBytesEqual(before, after) {
			t.Fatalf("journal changed on pre-canceled Prompt:\nbefore %s\nafter %s", before, after)
		}
		if session.State() != Idle || driver.Count() != 0 {
			t.Fatalf("state=%s drives=%d", session.State(), driver.Count())
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("CancelDuringModelExecutionSettlesAsInterrupt", func(t *testing.T) {
		entered := make(chan struct{})
		release := make(chan struct{}) // never closed: cancellation ends the call
		model := &scriptedModel{steps: []modelStep{
			{message: agentic.NewTextMessage(agentic.RoleAssistant, "never"), entered: entered, release: release},
		}}
		config := characterizationConfig(t, agentic.NewAgent("", model), storememory.New())
		session, err := New(context.Background(), config)
		if err != nil {
			t.Fatal(err)
		}
		runCtx, cancel := context.WithCancel(context.Background())
		defer cancel()
		done := make(chan promptOutcome, 1)
		go func() {
			execution, promptErr := session.Prompt(runCtx, agentic.NewTextMessage(agentic.RoleUser, "cancel me"))
			done <- promptOutcome{execution: execution, err: promptErr}
		}()
		awaitSignal(t, entered, "model entry")
		cancel()
		outcome := awaitPrompt(t, done, "canceled prompt settlement")
		if !errors.Is(outcome.err, context.Canceled) {
			t.Fatalf("canceled Prompt error = %v", outcome.err)
		}
		if err := session.WaitForIdle(context.Background()); err != nil {
			t.Fatal(err)
		}
		// Settlement shape is identical to a deliberate interrupt: the journal
		// tail is interrupt.marker then run.closed(interrupted, context canceled).
		entries := loadJournalEntries(t, session)
		kinds := journalKinds(entries)
		if len(kinds) < 2 || kinds[len(kinds)-2] != kindInterruptMarker || kinds[len(kinds)-1] != kindRunClosed {
			t.Fatalf("settlement tail = %v", kinds)
		}
		closed, err := decodePayload[runClosedPayload](config.Codec, entries[len(entries)-1])
		if err != nil {
			t.Fatal(err)
		}
		if closed.ID != "run_c1" || closed.Status != agentic.ExecutionInterrupted || closed.Error != context.Canceled.Error() {
			t.Fatalf("run.closed payload = %#v", closed)
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("CancelDuringToolExecutionCommitsErrorResultThenSettlesAsInterrupt", func(t *testing.T) {
		entered := make(chan struct{})
		model := &scriptedModel{steps: []modelStep{
			{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "tool-1", Name: "block", Input: map[string]any{"value": "x"}})},
			textStep("never"),
		}}
		tool, handler := agentic.MustToolWithContext("block", "Block until the run context is canceled",
			func(toolCtx context.Context, input characterizationToolInput) (string, error) {
				select {
				case <-entered:
				default:
					close(entered)
				}
				<-toolCtx.Done()
				return "", toolCtx.Err()
			})
		config := characterizationConfig(t, agentic.NewAgent("", model), storememory.New())
		config.Toolsets = []agentic.Toolset{agentic.NewToolset().Add(tool, handler)}
		session, err := New(context.Background(), config)
		if err != nil {
			t.Fatal(err)
		}
		runCtx, cancel := context.WithCancel(context.Background())
		defer cancel()
		done := make(chan promptOutcome, 1)
		go func() {
			execution, promptErr := session.Prompt(runCtx, agentic.NewTextMessage(agentic.RoleUser, "cancel my tool"))
			done <- promptOutcome{execution: execution, err: promptErr}
		}()
		awaitSignal(t, entered, "tool handler entry")
		cancel()
		outcome := awaitPrompt(t, done, "canceled tool settlement")
		if !errors.Is(outcome.err, context.Canceled) {
			t.Fatalf("canceled Prompt error = %v", outcome.err)
		}
		if outcome.execution == nil || outcome.execution.Status != agentic.ExecutionInterrupted {
			t.Fatalf("execution = %#v", outcome.execution)
		}
		if err := session.WaitForIdle(context.Background()); err != nil {
			t.Fatal(err)
		}
		if session.State() != Idle {
			t.Fatalf("state = %s", session.State())
		}
		// The canceled tool's error result is durably committed BEFORE the
		// interrupt settlement: the full journal kind sequence is frozen, with
		// agentic.tool_result ahead of interrupt.marker and run.closed.
		entries := loadJournalEntries(t, session)
		kinds := journalKinds(entries)
		wantKinds := []string{
			kindSessionCreated, kindRunOpened, kindMessage,
			kindRunStarted, kindTurnStarted,
			kindAssistantCommitted, kindToolBatchPlanned,
			kindToolStarted, kindToolResult,
			kindTurnEnded, kindRunInterrupted, kindRunEnded,
			kindInterruptMarker, kindRunClosed,
		}
		if fmt.Sprint(kinds) != fmt.Sprint(wantKinds) {
			t.Fatalf("journal kinds = %v, want %v", kinds, wantKinds)
		}
		closed, err := decodePayload[runClosedPayload](config.Codec, entries[len(entries)-1])
		if err != nil {
			t.Fatal(err)
		}
		if closed.ID != "run_c1" || closed.Status != agentic.ExecutionInterrupted || closed.Error != context.Canceled.Error() {
			t.Fatalf("run.closed payload = %#v", closed)
		}
		// The scripted second step never ran: the model was called exactly once.
		if calls := model.Calls(); len(calls) != 1 {
			t.Fatalf("model called %d times", len(calls))
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("CanceledSnapshotReturnsContextError", func(t *testing.T) {
		config := characterizationConfig(t, &countingDriver{}, storememory.New())
		session, err := New(context.Background(), config)
		if err != nil {
			t.Fatal(err)
		}
		canceled, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := session.Snapshot(canceled); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled Snapshot error = %v", err)
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("PreCanceledCloseReturnsContextErrorYetTransitionsToClosed", func(t *testing.T) {
		config := characterizationConfig(t, &countingDriver{}, storememory.New())
		session, err := New(context.Background(), config)
		if err != nil {
			t.Fatal(err)
		}
		canceled, cancel := context.WithCancel(context.Background())
		cancel()
		// Frozen behavior: Close with a pre-canceled context surfaces the
		// context error from cleanup, but the state machine still transitions
		// to Closed and never reopens.
		if err := session.Close(canceled); !errors.Is(err, context.Canceled) {
			t.Fatalf("pre-canceled Close error = %v", err)
		}
		if session.State() != Closed {
			t.Fatalf("state after pre-canceled Close = %s", session.State())
		}
		// Cleanup is retried, not skipped: a repeated canceled Close fails the
		// same way, and a live-context Close then completes the release and is
		// idempotently nil afterwards.
		if err := session.Close(canceled); !errors.Is(err, context.Canceled) {
			t.Fatalf("repeated pre-canceled Close error = %v", err)
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatalf("live-context Close after canceled Close = %v", err)
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatalf("idempotent Close after recovery = %v", err)
		}
	})
}
