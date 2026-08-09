package session

// LoopView behavior tests: protocol lifecycle, projection replay equality
// (plan S2 exit gate), legacy interop, error mapping, capability honesty,
// and stream semantics.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/permission"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

const loopTestWait = 10 * time.Second

func loopTestContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), loopTestWait)
	t.Cleanup(cancel)
	return ctx
}

func sessionloopTextInput(text string) *sessionloop.Input {
	return &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: text}}}
}

func sessionloopStartCommand(text string) sessionloop.Command {
	return sessionloop.Command{Kind: sessionloop.CommandStart, Input: sessionloopTextInput(text)}
}

func newLoopViewForTest(
	t *testing.T,
	driver agentic.Driver[string],
	repository store.Repository,
	mutate func(*Config[string], *LoopConfig[string]),
) (*LoopView[string], *Session[string]) {
	t.Helper()
	config := sessionConfig(t, driver, repository, artifactmemory.New(), spill.Config{})
	loopConfig := LoopConfig[string]{}
	if mutate != nil {
		mutate(&config, &loopConfig)
	}
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if loopConfig.CloseRoot == nil {
		loopConfig.CloseRoot = session.Close
	}
	view, err := NewLoopView(session, loopConfig)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = view.Close(context.Background()) })
	return view, session
}

func loopSubscribe(t *testing.T, view *LoopView[string], options sessionloop.SubscribeOptions) sessionloop.Stream {
	t.Helper()
	stream, err := view.Subscribe(loopTestContext(t), options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	return stream
}

func loopDispatch(t *testing.T, view *LoopView[string], command sessionloop.Command) sessionloop.Receipt {
	t.Helper()
	receipt, err := view.Dispatch(loopTestContext(t), command)
	if err != nil {
		t.Fatalf("Dispatch(%s) failed: %v", command.Kind, err)
	}
	return receipt
}

func loopNextEvent(t *testing.T, stream sessionloop.Stream) sessionloop.Event {
	t.Helper()
	event, err := stream.Next(loopTestContext(t))
	if err != nil {
		t.Fatalf("Stream.Next failed: %v", err)
	}
	return event
}

func awaitLoopKind(t *testing.T, stream sessionloop.Stream, kind sessionloop.EventKind) (sessionloop.Event, []sessionloop.Event) {
	t.Helper()
	var seen []sessionloop.Event
	for {
		event := loopNextEvent(t, stream)
		seen = append(seen, event)
		if event.Kind == kind {
			return event, seen
		}
	}
}

func awaitLoopSettled(t *testing.T, stream sessionloop.Stream, runID sessionloop.RunID) (sessionloop.Event, []sessionloop.Event) {
	t.Helper()
	var seen []sessionloop.Event
	for {
		event := loopNextEvent(t, stream)
		seen = append(seen, event)
		if event.Kind == sessionloop.EventRunSettled && event.RunID == runID {
			return event, seen
		}
	}
}

func loopCommittedEntries(events []sessionloop.Event) []sessionloop.Entry {
	var entries []sessionloop.Entry
	for _, event := range events {
		if event.Kind == sessionloop.EventEntryCommitted && event.Entry != nil {
			entries = append(entries, *event.Entry)
		}
	}
	return entries
}

func TestLoopViewStartLifecycle(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	if !view.Capabilities().Supports(sessionloop.CapabilityDurableAcceptance) ||
		view.Capabilities().Supports(sessionloop.CapabilityIdempotentDispatch) ||
		view.Capabilities().Supports(sessionloop.CapabilityStructuredOutput) {
		t.Fatalf("capabilities = %v", view.Capabilities())
	}
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{})

	command := sessionloopStartCommand("hello loop")
	command.ID = "cmd-start-1"
	receipt := loopDispatch(t, view, command)
	if receipt.CommandID != "cmd-start-1" || receipt.SessionID != view.ID() || receipt.RunID == "" ||
		receipt.Guarantee != sessionloop.AcceptanceDurable ||
		receipt.Position.Sequence == 0 || receipt.Position.Token == "" {
		t.Fatalf("start receipt = %#v", receipt)
	}

	started, _ := awaitLoopKind(t, stream, sessionloop.EventRunStarted)
	if started.RunID != receipt.RunID || started.State != sessionloop.StateRunning ||
		started.CommandID != "cmd-start-1" || started.Ordinal != 0 || started.Position.IsZero() {
		t.Fatalf("run.started = %#v", started)
	}
	settled, seen := awaitLoopSettled(t, stream, receipt.RunID)
	if settled.Outcome == nil || settled.Outcome.Kind != sessionloop.RunCompleted ||
		settled.Outcome.RunID != receipt.RunID || settled.CommandID != "cmd-start-1" {
		t.Fatalf("run.settled = %#v", settled)
	}
	entries := loopCommittedEntries(seen)
	if len(entries) != 1 || entries[0].Origin != sessionloop.OriginStart ||
		entries[0].Role != sessionloop.RoleUser || entries[0].CommandID != "cmd-start-1" ||
		entries[0].RunID != receipt.RunID || len(entries[0].Blocks) != 1 ||
		entries[0].Blocks[0].Text != "hello loop" {
		t.Fatalf("committed entries = %#v", entries)
	}

	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != sessionloop.StateIdle || snapshot.ActiveRunID != "" ||
		snapshot.SessionID != view.ID() || snapshot.Position.Sequence == 0 ||
		snapshot.Position.Token == "" || !reflect.DeepEqual(snapshot.Entries, entries) {
		t.Fatalf("snapshot = %#v", snapshot)
	}
	if err := view.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

type loopToolInput struct {
	Value string `json:"value"`
}

// TestLoopViewProjectionReplayEquality runs tools + steer + suspension +
// resume + settlement through the view and proves snapshot entries equal the
// full-replay folded entries (plan S2 exit gate: snapshot and replay reach
// the same normalized state; multi-user steering keeps order and run
// attribution).
func TestLoopViewProjectionReplayEquality(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{
			message: agentic.NewTextMessage(agentic.RoleAssistant, "thinking aloud"),
			usage:   agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15, Requests: 1},
			entered: entered,
			release: release,
		},
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "gate-1", Name: "danger", Input: map[string]any{"value": "x"}})},
		textStep("finished"),
	}}
	agent := agentic.NewAgent("base", model)
	agentic.AddTool(agent,
		func(context.Context, loopToolInput) (string, error) { return "gated ok", nil },
		agentic.AutoToolName("danger"),
		agentic.AutoToolDescription("Perform a gated action"),
	)
	policy, err := permission.New(permission.DecisionDeny,
		permission.Rule{Pattern: "tool/danger/**", Decision: permission.DecisionAsk})
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
	view, _ := newLoopViewForTest(t, agent, storememory.New(), func(config *Config[string], _ *LoopConfig[string]) {
		config.ToolGate = plan.ToolGate()
		config.Context = plan.ContextPolicy()
	})
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 256})

	start := sessionloopStartCommand("begin")
	start.ID = "cmd-start"
	receipt := loopDispatch(t, view, start)
	awaitSignal(t, entered, "first model turn")

	steer := sessionloop.Command{
		ID:    "cmd-steer",
		Kind:  sessionloop.CommandSteer,
		RunID: receipt.RunID,
		Input: sessionloopTextInput("steer this run"),
	}
	steerReceipt := loopDispatch(t, view, steer)
	if steerReceipt.QueueID == "" || steerReceipt.Guarantee != sessionloop.AcceptanceDurable {
		t.Fatalf("steer receipt = %#v", steerReceipt)
	}
	close(release)

	suspended, prefix := awaitLoopKind(t, stream, sessionloop.EventRunSuspended)
	if suspended.RunID != receipt.RunID || suspended.State != sessionloop.StateSuspended ||
		suspended.Suspension == nil || suspended.Suspension.ID == "" {
		t.Fatalf("run.suspended = %#v", suspended)
	}
	if len(suspended.Suspension.Decisions) != 1 ||
		suspended.Suspension.Decisions[0].ID != "gate-1" ||
		suspended.Suspension.Decisions[0].Name != "danger" {
		t.Fatalf("default suspension projection = %#v", suspended.Suspension)
	}
	queueAccepted := false
	for _, event := range prefix {
		if event.Kind == sessionloop.EventQueueAccepted && event.Queue != nil &&
			event.Queue.ID == steerReceipt.QueueID && event.Queue.Kind == sessionloop.CommandSteer &&
			event.CommandID == "cmd-steer" {
			queueAccepted = true
		}
	}
	if !queueAccepted {
		t.Fatalf("no queue.accepted event for %q in %#v", steerReceipt.QueueID, prefix)
	}

	midSnapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if midSnapshot.State != sessionloop.StateSuspended || midSnapshot.Suspension == nil ||
		midSnapshot.Suspension.ID != suspended.Suspension.ID {
		t.Fatalf("suspended snapshot = %#v", midSnapshot)
	}

	resolve := sessionloop.Command{
		ID:    "cmd-resolve",
		Kind:  sessionloop.CommandResolve,
		RunID: receipt.RunID,
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspended.Suspension.ID,
			Decisions: []sessionloop.ResolutionDecision{
				{ID: "gate-1", Action: sessionloop.ResolutionApprove},
			},
		},
	}
	resolveReceipt := loopDispatch(t, view, resolve)
	if resolveReceipt.RunID != receipt.RunID || resolveReceipt.Guarantee != sessionloop.AcceptanceDurable ||
		resolveReceipt.Position.Sequence == 0 {
		t.Fatalf("resolve receipt = %#v", resolveReceipt)
	}

	settled, suffix := awaitLoopSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome = %#v", settled.Outcome)
	}
	commandAccepted := false
	for _, event := range suffix {
		if event.Kind == sessionloop.EventCommandAccepted {
			if event.CommandID != "cmd-resolve" || event.RunID != receipt.RunID {
				t.Fatalf("command.accepted = %#v", event)
			}
			commandAccepted = true
		}
	}
	if !commandAccepted {
		t.Fatal("no command.accepted event for the resolve command")
	}

	streamed := loopCommittedEntries(append(prefix, suffix...))
	texts := []string{}
	origins := []sessionloop.EntryOrigin{}
	for _, entry := range streamed {
		origins = append(origins, entry.Origin)
		if len(entry.Blocks) > 0 && entry.Blocks[0].Kind == sessionloop.EntryBlockText {
			texts = append(texts, entry.Blocks[0].Text)
		}
	}
	wantOrigins := []sessionloop.EntryOrigin{
		sessionloop.OriginStart, sessionloop.OriginAssistant, sessionloop.OriginSteer,
		sessionloop.OriginAssistant, sessionloop.OriginTool, sessionloop.OriginAssistant,
	}
	if fmt.Sprint(origins) != fmt.Sprint(wantOrigins) {
		t.Fatalf("entry origins = %v, want %v", origins, wantOrigins)
	}
	for _, entry := range streamed {
		if entry.RunID != receipt.RunID {
			t.Fatalf("entry lost run attribution: %#v", entry)
		}
	}
	var toolCall *sessionloop.EntryToolCall
	var toolResult *sessionloop.EntryToolResult
	for _, entry := range streamed {
		for _, block := range entry.Blocks {
			if block.Kind == sessionloop.EntryBlockToolCall {
				toolCall = block.ToolCall
			}
			if block.Kind == sessionloop.EntryBlockToolResult {
				toolResult = block.ToolResult
			}
		}
	}
	if toolCall == nil || toolCall.CallID != "gate-1" || toolCall.Name != "danger" ||
		!json.Valid(toolCall.Data) {
		t.Fatalf("tool call block = %#v", toolCall)
	}
	if toolResult == nil || toolResult.CallID != "gate-1" || toolResult.IsError {
		t.Fatalf("tool result block = %#v", toolResult)
	}

	// Snapshot entries and a full replay fold must agree exactly.
	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(snapshot.Entries, streamed) {
		t.Fatalf("snapshot entries diverge from streamed entries:\nsnapshot %#v\nstreamed %#v",
			snapshot.Entries, streamed)
	}
	replay := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 256})
	_, replayed := awaitLoopSettled(t, replay, receipt.RunID)
	if !reflect.DeepEqual(loopCommittedEntries(replayed), streamed) {
		t.Fatal("replayed entries diverge from live entries")
	}
	_ = texts
	if err := view.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// TestLoopViewLegacyInterop proves a legacy blocking Prompt is visible on a
// concurrently subscribed view stream with correct lifecycle and EMPTY
// command IDs (legacy calls carry no protocol attribution).
func TestLoopViewLegacyInterop(t *testing.T) {
	driver := &countingDriver{}
	view, session := newLoopViewForTest(t, driver, storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{})

	execution, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "legacy prompt"))
	if err != nil || execution.Status != agentic.ExecutionCompleted {
		t.Fatalf("legacy prompt execution=%#v err=%v", execution, err)
	}

	started, _ := awaitLoopKind(t, stream, sessionloop.EventRunStarted)
	if started.RunID == "" || started.CommandID != "" {
		t.Fatalf("legacy run.started = %#v", started)
	}
	settled, seen := awaitLoopSettled(t, stream, started.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted || settled.CommandID != "" {
		t.Fatalf("legacy run.settled = %#v", settled)
	}
	entries := loopCommittedEntries(seen)
	if len(entries) != 1 || entries[0].CommandID != "" || entries[0].RunID != started.RunID ||
		entries[0].Origin != sessionloop.OriginStart {
		t.Fatalf("legacy entries = %#v", entries)
	}
}

func TestMapLoopErrorPreservesBothIdentities(t *testing.T) {
	pairs := []struct {
		inner error
		outer error
	}{
		{ErrSessionBusy, sessionloop.ErrSessionBusy},
		{ErrRunClosing, sessionloop.ErrCommandConflict},
		{ErrNotRunning, sessionloop.ErrNotRunning},
		{ErrSessionSuspended, sessionloop.ErrSuspended},
		{&FaultError{SessionID: "s", Cause: errors.New("boom")}, sessionloop.ErrSessionFaulted},
		{ErrSessionClosed, sessionloop.ErrSessionClosed},
		{store.ErrSessionOpen, sessionloop.ErrSessionOpen},
		{ErrInvalidMessage, sessionloop.ErrInvalidCommand},
		{fmt.Errorf("%w: detail", ErrInvalidResumeRequest), sessionloop.ErrInvalidCommand},
	}
	for _, pair := range pairs {
		mapped := mapLoopError(pair.inner)
		if !errors.Is(mapped, pair.outer) {
			t.Fatalf("mapLoopError(%v) = %v, missing portable identity %v", pair.inner, mapped, pair.outer)
		}
		concrete := error(pair.inner)
		if !errors.Is(mapped, concrete) {
			var fault *FaultError
			if !errors.As(mapped, &fault) {
				t.Fatalf("mapLoopError(%v) = %v, lost the concrete cause", pair.inner, mapped)
			}
		}
	}
	budget := &BudgetError{Cause: errors.New("limit")}
	if mapped := mapLoopError(budget); mapped != error(budget) {
		t.Fatalf("BudgetError must pass through unwrapped, got %v", mapped)
	}
	plain := errors.New("plain")
	if mapped := mapLoopError(plain); mapped != plain {
		t.Fatalf("unknown errors must pass through, got %v", mapped)
	}
	if mapLoopError(nil) != nil {
		t.Fatal("nil must map to nil")
	}
}

func TestLoopViewDispatchValidationAndUnsupportedInput(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)

	if _, err := view.Dispatch(loopTestContext(t), sessionloop.Command{Kind: sessionloop.CommandStart}); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatalf("structurally invalid command err = %v", err)
	}
	withKey := sessionloopStartCommand("keyed")
	withKey.IdempotencyKey = "key-1"
	if _, err := view.Dispatch(loopTestContext(t), withKey); !errors.Is(err, sessionloop.ErrUnsupported) {
		t.Fatalf("idempotency key err = %v, want ErrUnsupported", err)
	}
	dataStart := sessionloop.Command{Kind: sessionloop.CommandStart, Input: &sessionloop.Input{
		Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockData, Data: json.RawMessage(`{"a":1}`)}},
	}}
	if _, err := view.Dispatch(loopTestContext(t), dataStart); !errors.Is(err, sessionloop.ErrUnsupported) {
		t.Fatalf("data block err = %v, want ErrUnsupported", err)
	}
	empty := sessionloop.Command{Kind: sessionloop.CommandStart, Input: &sessionloop.Input{}}
	if _, err := view.Dispatch(loopTestContext(t), empty); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatalf("empty input err = %v, want ErrInvalidCommand", err)
	}
	// Meta stays correlation-only: dispatch succeeds and content is text only.
	meta := sessionloopStartCommand("meta carried")
	meta.Input.Meta = map[string]string{"trace": "abc"}
	receipt := loopDispatch(t, view, meta)
	if receipt.RunID == "" {
		t.Fatalf("meta-carrying start receipt = %#v", receipt)
	}
}

func TestLoopViewStaleRunAndInterruptReceipt(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "blocked"), entered: entered, release: release},
		textStep("second done"),
	}}
	view, _ := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 128})

	receipt := loopDispatch(t, view, sessionloopStartCommand("interrupt me"))
	awaitSignal(t, entered, "model entered")

	if _, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandSteer, RunID: "run-other", Input: sessionloopTextInput("stale"),
	}); !errors.Is(err, sessionloop.ErrStaleRun) {
		t.Fatalf("stale steer err = %v", err)
	}
	if _, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandInterrupt, RunID: "run-other",
	}); !errors.Is(err, sessionloop.ErrStaleRun) {
		t.Fatalf("stale interrupt err = %v", err)
	}

	interrupt := sessionloop.Command{ID: "cmd-int", Kind: sessionloop.CommandInterrupt, RunID: receipt.RunID}
	interruptReceipt := loopDispatch(t, view, interrupt)
	if interruptReceipt.Guarantee != sessionloop.AcceptanceAccepted ||
		!interruptReceipt.Position.IsZero() || interruptReceipt.RunID != receipt.RunID {
		t.Fatalf("interrupt receipt = %#v", interruptReceipt)
	}
	close(release)
	settled, _ := awaitLoopSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunInterrupted {
		t.Fatalf("outcome = %#v", settled.Outcome)
	}

	// The session is idle again; a second run works and the settled run stays
	// rejected for targeted commands.
	if _, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandInterrupt, RunID: receipt.RunID,
	}); !errors.Is(err, sessionloop.ErrStaleRun) {
		t.Fatalf("interrupt after settle err = %v", err)
	}
	second := loopDispatch(t, view, sessionloopStartCommand("second run"))
	settled, _ = awaitLoopSettled(t, stream, second.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("second outcome = %#v", settled.Outcome)
	}
}

func TestLoopViewQueueEventsAndPendingProjection(t *testing.T) {
	driver := &countingDriver{}
	view, _ := newLoopViewForTest(t, driver, storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 128})

	queued := sessionloop.Command{ID: "cmd-q1", Kind: sessionloop.CommandNextTurn, Input: sessionloopTextInput("queued for later")}
	receipt := loopDispatch(t, view, queued)
	if receipt.QueueID == "" || receipt.RunID != "" || receipt.Position.Sequence == 0 {
		t.Fatalf("next_turn receipt = %#v", receipt)
	}
	accepted, _ := awaitLoopKind(t, stream, sessionloop.EventQueueAccepted)
	if accepted.Queue == nil || accepted.Queue.ID != receipt.QueueID ||
		accepted.Queue.Kind != sessionloop.CommandNextTurn || accepted.CommandID != "cmd-q1" ||
		len(accepted.Queue.Blocks) != 1 || accepted.Queue.Blocks[0].Text != "queued for later" {
		t.Fatalf("queue.accepted = %#v", accepted)
	}

	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Pending) != 1 || snapshot.Pending[0].ID != receipt.QueueID ||
		snapshot.Pending[0].Kind != sessionloop.CommandNextTurn ||
		snapshot.Pending[0].CommandID != "cmd-q1" ||
		snapshot.Pending[0].Position.Sequence != receipt.Position.Sequence {
		t.Fatalf("pending projection = %#v", snapshot.Pending)
	}

	startReceipt := loopDispatch(t, view, sessionloopStartCommand("drain now"))
	drained, seen := awaitLoopKind(t, stream, sessionloop.EventQueueDrained)
	if drained.Queue == nil || drained.Queue.ID != receipt.QueueID ||
		drained.Queue.Kind != sessionloop.CommandNextTurn || drained.CommandID != "cmd-q1" {
		t.Fatalf("queue.drained = %#v", drained)
	}
	if seen[len(seen)-2].Kind != sessionloop.EventRunStarted {
		t.Fatalf("queue.drained did not directly follow run.started: %#v", seen)
	}
	settled, tail := awaitLoopSettled(t, stream, startReceipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome = %#v", settled.Outcome)
	}
	entries := loopCommittedEntries(tail)
	if len(entries) != 2 || entries[0].Origin != sessionloop.OriginNextTurn ||
		entries[0].CommandID != "cmd-q1" || entries[1].Origin != sessionloop.OriginStart {
		t.Fatalf("drained entries = %#v", entries)
	}
}

func TestLoopViewStructuredOutput(t *testing.T) {
	driver := &countingDriver{}
	view, _ := newLoopViewForTest(t, driver, storememory.New(), func(_ *Config[string], loopConfig *LoopConfig[string]) {
		loopConfig.OutputProjector = func(output string) (json.RawMessage, error) {
			return json.Marshal(map[string]string{"output": output})
		}
	})
	if !view.Capabilities().Supports(sessionloop.CapabilityStructuredOutput) {
		t.Fatalf("capabilities = %v", view.Capabilities())
	}
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{})
	receipt := loopDispatch(t, view, sessionloopStartCommand("produce output"))
	settled, _ := awaitLoopSettled(t, stream, receipt.RunID)
	if string(settled.Outcome.Output) != `{"output":""}` {
		t.Fatalf("structured output = %q", settled.Outcome.Output)
	}
	// A replay through the same live view still carries the captured output.
	replay := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64})
	replayedSettled, _ := awaitLoopSettled(t, replay, receipt.RunID)
	if string(replayedSettled.Outcome.Output) != `{"output":""}` {
		t.Fatalf("replayed structured output = %q", replayedSettled.Outcome.Output)
	}
	// A pure journal replay (no live host) never invents output.
	inner := view.inner
	loaded, err := inner.journalRef().Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	records, err := loopRecords(inner.codecRef(), loaded.Entries)
	if err != nil {
		t.Fatal(err)
	}
	projector := newLoopProjector(inner.ID(), inner.codecRef(), nil)
	for _, record := range records {
		events, applyErr := projector.apply(context.Background(), record)
		if applyErr != nil {
			t.Fatal(applyErr)
		}
		for _, event := range events {
			if event.Kind == sessionloop.EventRunSettled && len(event.Outcome.Output) != 0 {
				t.Fatalf("pure replay leaked structured output: %#v", event.Outcome)
			}
		}
	}
}

func TestLoopViewSubscribePositions(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{})
	receipt := loopDispatch(t, view, sessionloopStartCommand("first"))
	_, seen := awaitLoopSettled(t, stream, receipt.RunID)

	if _, err := view.Subscribe(loopTestContext(t), sessionloop.SubscribeOptions{
		After: sessionloop.Position{Sequence: 999},
	}); !errors.Is(err, sessionloop.ErrUnknownPosition) {
		t.Fatalf("beyond-history subscribe err = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := view.Subscribe(canceled, sessionloop.SubscribeOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled subscribe err = %v", err)
	}

	middle := seen[0].Position
	var want []sessionloop.Event
	for _, event := range seen {
		if event.Position.Sequence > middle.Sequence {
			want = append(want, event)
		}
	}
	replay := loopSubscribe(t, view, sessionloop.SubscribeOptions{After: middle})
	for index, expected := range want {
		got := loopNextEvent(t, replay)
		if !reflect.DeepEqual(got, expected) {
			t.Fatalf("replay diverged at %d:\ngot  %#v\nwant %#v", index, got, expected)
		}
	}
}

func TestLoopViewCloseInterruptsAndMemoizes(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "held"), entered: entered, release: release},
	}}
	view, session := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64})

	receipt := loopDispatch(t, view, sessionloopStartCommand("close me"))
	awaitSignal(t, entered, "model entered")

	if err := view.Close(context.Background()); err != nil {
		t.Fatalf("Close = %v", err)
	}
	if err := view.Close(context.Background()); err != nil {
		t.Fatalf("second Close = %v", err)
	}
	if session.State() != Closed {
		t.Fatalf("inner state after Close = %s", session.State())
	}
	if _, err := view.Dispatch(loopTestContext(t), sessionloopStartCommand("late")); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatalf("dispatch after close err = %v", err)
	}
	if _, err := view.Snapshot(loopTestContext(t)); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatalf("snapshot after close err = %v", err)
	}
	if _, err := view.Subscribe(loopTestContext(t), sessionloop.SubscribeOptions{}); !errors.Is(err, sessionloop.ErrSessionClosed) {
		t.Fatalf("subscribe after close err = %v", err)
	}

	// The interrupted settlement reached the stream before it terminated.
	sawSettled := false
	for {
		event, err := stream.Next(loopTestContext(t))
		if err != nil {
			if !errors.Is(err, io.EOF) {
				t.Fatalf("stream after close ended with %v", err)
			}
			break
		}
		if event.Kind == sessionloop.EventRunSettled && event.RunID == receipt.RunID {
			if event.Outcome.Kind != sessionloop.RunInterrupted {
				t.Fatalf("outcome = %#v", event.Outcome)
			}
			sawSettled = true
		}
	}
	if !sawSettled {
		t.Fatal("interrupted settlement never reached the stream")
	}
	select {
	case <-release:
	default:
		close(release)
	}
}

func TestLoopViewDispatchErrorMappingIntegration(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "busy"), entered: entered, release: release},
	}}
	view, session := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 64})

	receipt := loopDispatch(t, view, sessionloopStartCommand("hold the session"))
	awaitSignal(t, entered, "model entered")

	if _, err := view.Dispatch(loopTestContext(t), sessionloopStartCommand("second start")); !errors.Is(err, sessionloop.ErrSessionBusy) || !errors.Is(err, ErrSessionBusy) {
		t.Fatalf("busy start err = %v", err)
	}
	// ErrRunClosing -> ErrCommandConflict at the closing boundary.
	session.mu.Lock()
	priorState := session.state
	session.transitionLocked(Closing)
	session.mu.Unlock()
	if _, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandSteer, RunID: receipt.RunID, Input: sessionloopTextInput("too late"),
	}); !errors.Is(err, sessionloop.ErrCommandConflict) || !errors.Is(err, ErrRunClosing) {
		t.Fatalf("closing steer err = %v", err)
	}
	session.mu.Lock()
	session.transitionLocked(priorState)
	session.mu.Unlock()

	close(release)
	awaitLoopSettled(t, stream, receipt.RunID)
}

func TestLoopViewSuspendedErrorMapping(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "gate-2", Name: "danger", Input: map[string]any{"value": "y"}})},
		textStep("after resume"),
	}}
	agent := agentic.NewAgent("", model)
	agentic.AddTool(agent,
		func(context.Context, loopToolInput) (string, error) { return "ok", nil },
		agentic.AutoToolName("danger"),
		agentic.AutoToolDescription("gated"),
	)
	policy, err := permission.New(permission.DecisionDeny,
		permission.Rule{Pattern: "tool/danger/**", Decision: permission.DecisionAsk})
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
	view, _ := newLoopViewForTest(t, agent, storememory.New(), func(config *Config[string], _ *LoopConfig[string]) {
		config.ToolGate = plan.ToolGate()
		config.Context = plan.ContextPolicy()
	})
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 128})
	receipt := loopDispatch(t, view, sessionloopStartCommand("suspend me"))
	suspended, _ := awaitLoopKind(t, stream, sessionloop.EventRunSuspended)

	// A steer while suspended maps ErrSessionSuspended -> ErrSuspended.
	if _, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandSteer, RunID: receipt.RunID, Input: sessionloopTextInput("nudge"),
	}); !errors.Is(err, sessionloop.ErrSuspended) || !errors.Is(err, ErrSessionSuspended) {
		t.Fatalf("suspended steer err = %v", err)
	}
	// A start while suspended is also ErrSuspended.
	if _, err := view.Dispatch(loopTestContext(t), sessionloopStartCommand("start over")); !errors.Is(err, sessionloop.ErrSuspended) {
		t.Fatalf("suspended start err = %v", err)
	}
	// A wrong suspension ID maps ErrInvalidResumeRequest -> ErrInvalidCommand
	// without consuming the suspension.
	if _, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind:  sessionloop.CommandResolve,
		RunID: receipt.RunID,
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspended.Suspension.ID + "-wrong",
			Decisions:    []sessionloop.ResolutionDecision{{ID: "gate-2", Action: sessionloop.ResolutionApprove}},
		},
	}); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatalf("wrong suspension resolve err = %v", err)
	}
	// External-result data must be one JSON value at conversion time.
	if _, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind:  sessionloop.CommandResolve,
		RunID: receipt.RunID,
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspended.Suspension.ID,
			Decisions: []sessionloop.ResolutionDecision{
				{ID: "gate-2", Action: sessionloop.ResolutionExternalResult, Data: json.RawMessage(`{"bad"`)},
			},
		},
	}); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatalf("malformed external result err = %v", err)
	}

	resolve := sessionloop.Command{
		Kind:  sessionloop.CommandResolve,
		RunID: receipt.RunID,
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspended.Suspension.ID,
			Decisions:    []sessionloop.ResolutionDecision{{ID: "gate-2", Action: sessionloop.ResolutionApprove}},
		},
	}
	loopDispatch(t, view, resolve)
	settled, _ := awaitLoopSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome after resume = %#v", settled.Outcome)
	}
}

// TestLoopViewClosePreservesDurableSuspension proves Close treats a
// suspension as a durable pause (law L11): closing a Suspended session must
// not interrupt it, the suspension survives close/reopen, and the reopened
// session resolves it to completion.
func TestLoopViewClosePreservesDurableSuspension(t *testing.T) {
	repository := storememory.New()
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "gate-d", Name: "danger", Input: map[string]any{"value": "x"}})},
	}}
	agent := agentic.NewAgent("", model)
	agentic.AddTool(agent,
		func(context.Context, loopToolInput) (string, error) { return "ok", nil },
		agentic.AutoToolName("danger"), agentic.AutoToolDescription("gated"))
	config := sessionConfig[string](t, agent, repository, artifactmemory.New(), spill.Config{})
	config.ToolGate = suspendingGate{}
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	view, err := NewLoopView(session, LoopConfig[string]{CloseRoot: session.Close})
	if err != nil {
		t.Fatal(err)
	}
	receipt := loopDispatch(t, view, sessionloopStartCommand("pause durably"))
	awaitState(t, session, Suspended)
	snapshot, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Suspension == nil {
		t.Fatalf("suspended snapshot carries no suspension: %#v", snapshot)
	}
	suspensionID := snapshot.Suspension.ID

	if err := view.Close(context.Background()); err != nil {
		t.Fatalf("Close of a suspended session = %v", err)
	}
	if session.State() != Closed {
		t.Fatalf("state after close = %s", session.State())
	}

	// Reopen: the suspension survived; Close settled nothing.
	recoveredModel := &scriptedModel{steps: []modelStep{textStep("resumed after reopen")}}
	recoveredAgent := agentic.NewAgent("", recoveredModel)
	agentic.AddTool(recoveredAgent,
		func(context.Context, loopToolInput) (string, error) { return "ok", nil },
		agentic.AutoToolName("danger"), agentic.AutoToolDescription("gated"))
	recoveredConfig := sessionConfig[string](t, recoveredAgent, repository, artifactmemory.New(), spill.Config{})
	recoveredConfig.ID = config.ID
	recoveredConfig.ToolGate = suspendingGate{}
	recovered, err := Recover(context.Background(), recoveredConfig)
	if err != nil {
		t.Fatalf("Recover of the suspended session = %v", err)
	}
	if recovered.State() != Suspended {
		t.Fatalf("reopened state = %s, want suspended (close destroyed the durable pause)", recovered.State())
	}
	reopened, err := recovered.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if reopened.Suspension == nil || reopened.Suspension.ID != suspensionID ||
		reopened.RunID != string(receipt.RunID) {
		t.Fatalf("reopened suspension = %#v, want the original %q on run %q",
			reopened.Suspension, suspensionID, receipt.RunID)
	}
	entries := loadJournalEntries(t, recovered)
	if countEntries(entries, kindRunClosed) != 0 {
		t.Fatalf("Close settled the suspended run: %v", journalKinds(entries))
	}

	execution, err := recovered.Resume(context.Background(), ResumeRequest{
		SuspensionID: suspensionID,
		Resolutions:  []ToolResolution{{CallID: "gate-d", Action: ResolutionApprove}},
	})
	if err != nil || execution.Status != agentic.ExecutionCompleted {
		t.Fatalf("resolve after reopen execution=%#v err=%v", execution, err)
	}
	if err := recovered.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// resumeBounceDriver fails every Resume with a validation-class error before
// any RunStarted event, reproducing the silent resume-validation bounce
// (finishExecution moves Running back to Suspended with no journal write).
type resumeBounceDriver struct {
	agentic.Driver[string]
}

func (d *resumeBounceDriver) Resume(context.Context, agentic.ResumeInput, ...agentic.RunOption) (*agentic.Execution[string], error) {
	return nil, fmt.Errorf("scripted resume validation failure: %w", agentic.ErrDriveInput)
}

// TestLoopViewResolveBounceEmitsZeroPositionSessionState proves the bounce
// is visible to protocol consumers: every live view-owned stream receives an
// authoritative session.state suspended event with a ZERO position (law L6:
// zero position = not replayable, the lawful seam for live-only facts), and
// the durable suspension is untouched.
func TestLoopViewResolveBounceEmitsZeroPositionSessionState(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "gate-b", Name: "danger", Input: map[string]any{"value": "y"}})},
		textStep("never reached"),
	}}
	agent := agentic.NewAgent("", model)
	agentic.AddTool(agent,
		func(context.Context, loopToolInput) (string, error) { return "ok", nil },
		agentic.AutoToolName("danger"), agentic.AutoToolDescription("gated"))
	view, session := newLoopViewForTest(t, &resumeBounceDriver{Driver: agent}, storememory.New(),
		func(config *Config[string], _ *LoopConfig[string]) {
			config.ToolGate = suspendingGate{}
		})
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 128})
	receipt := loopDispatch(t, view, sessionloopStartCommand("suspend then bounce"))
	suspended, _ := awaitLoopKind(t, stream, sessionloop.EventRunSuspended)

	resolve := sessionloop.Command{
		ID:    "cmd-bounce",
		Kind:  sessionloop.CommandResolve,
		RunID: receipt.RunID,
		Resolution: &sessionloop.Resolution{
			SuspensionID: suspended.Suspension.ID,
			Decisions:    []sessionloop.ResolutionDecision{{ID: "gate-b", Action: sessionloop.ResolutionApprove}},
		},
	}
	loopDispatch(t, view, resolve)

	bounced, _ := awaitLoopKind(t, stream, sessionloop.EventSessionState)
	if !bounced.Position.IsZero() {
		t.Fatalf("bounce announcement position = %#v, want zero (live-only, not replayable)", bounced.Position)
	}
	if bounced.Nature != sessionloop.EventAuthoritative || bounced.State != sessionloop.StateSuspended ||
		bounced.RunID != receipt.RunID || bounced.CommandID != "cmd-bounce" ||
		bounced.SessionID != view.ID() {
		t.Fatalf("bounce announcement = %#v", bounced)
	}

	// The durable suspension is untouched by the bounced resolve.
	if session.State() != Suspended {
		t.Fatalf("state after bounce = %s, want suspended", session.State())
	}
	snapshot, err := view.Snapshot(loopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != sessionloop.StateSuspended || snapshot.Suspension == nil ||
		snapshot.Suspension.ID != suspended.Suspension.ID {
		t.Fatalf("snapshot after bounce = %#v", snapshot)
	}

	// The zero-position event is live-only: a fresh replay never carries it.
	replay := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 128})
	for {
		event := loopNextEvent(t, replay)
		if event.Kind == sessionloop.EventSessionState {
			t.Fatalf("live-only bounce announcement leaked into replay: %#v", event)
		}
		if event.Position.Sequence >= snapshot.Position.Sequence {
			break
		}
	}
}

func TestDefaultSuspensionProjectorUnknownKind(t *testing.T) {
	projected, err := defaultSuspensionProjector(agentic.Suspension{
		ID:      "susp-x",
		Kind:    "custom.kind",
		Payload: json.RawMessage(`{"opaque":true}`),
	})
	if err != nil {
		t.Fatalf("unknown kind err = %v", err)
	}
	if projected.ID != "susp-x" || projected.Kind != "custom.kind" ||
		projected.Description == "" || len(projected.Decisions) != 0 {
		t.Fatalf("unknown-kind projection = %#v", projected)
	}
}

func TestNewLoopViewValidation(t *testing.T) {
	if _, err := NewLoopView[string](nil, LoopConfig[string]{CloseRoot: func(context.Context) error { return nil }}); err == nil {
		t.Fatal("nil inner session accepted")
	}
	driver := &countingDriver{}
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(context.Background()) }()
	if _, err := NewLoopView(session, LoopConfig[string]{}); err == nil {
		t.Fatal("missing CloseRoot accepted")
	}
	view, err := NewLoopView(session, LoopConfig[string]{
		CloseRoot:    session.Close,
		capabilities: sessionloop.NewCapabilities(sessionloop.CapabilityDurableAcceptance),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Capabilities()) != 1 {
		t.Fatalf("capability override ignored: %v", view.Capabilities())
	}
	if _, err := view.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandSteer, RunID: "run-1", Input: sessionloopTextInput("x"),
	}); !errors.Is(err, sessionloop.ErrUnsupported) {
		t.Fatalf("unadvertised steer err = %v", err)
	}
}
