package session

// Focused branch coverage for the sessionloop projection and view: unit
// tests for conversion helpers, error arms, and stream terminal behavior.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestLoopResumeRequestConversion(t *testing.T) {
	command := sessionloop.Command{
		Kind:  sessionloop.CommandResolve,
		RunID: "run-1",
		Input: sessionloopTextInput("continue with this"),
		Resolution: &sessionloop.Resolution{
			SuspensionID: "susp-1",
			Decisions: []sessionloop.ResolutionDecision{
				{ID: "c1", Action: sessionloop.ResolutionApprove},
				{ID: "c2", Action: sessionloop.ResolutionDeny, Reason: "too risky"},
				{ID: "c3", Action: sessionloop.ResolutionExternalResult, Data: json.RawMessage(`{"answer":42}`)},
				{ID: "c4", Action: sessionloop.ResolutionExternalResult},
			},
		},
	}
	request, err := loopResumeRequest(command)
	if err != nil {
		t.Fatal(err)
	}
	if request.SuspensionID != "susp-1" || len(request.Resolutions) != 4 {
		t.Fatalf("request = %#v", request)
	}
	if request.Resolutions[0].Action != ResolutionApprove ||
		request.Resolutions[1].Action != ResolutionDeny || request.Resolutions[1].Reason != "too risky" ||
		request.Resolutions[2].Action != ResolutionExternalResult ||
		fmt.Sprint(request.Resolutions[2].Result) != "map[answer:42]" ||
		request.Resolutions[3].Result != nil {
		t.Fatalf("resolutions = %#v", request.Resolutions)
	}
	if request.Prompt == nil || request.Prompt.GetTextContent() != "continue with this" {
		t.Fatalf("prompt = %#v", request.Prompt)
	}

	bad := command.Clone()
	bad.Resolution.Decisions[2].Data = json.RawMessage(`{`)
	if _, err := loopResumeRequest(bad); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatalf("malformed data err = %v", err)
	}
	badInput := command.Clone()
	badInput.Input = &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockData, Data: json.RawMessage(`1`)}}}
	if _, err := loopResumeRequest(badInput); !errors.Is(err, sessionloop.ErrUnsupported) {
		t.Fatalf("unsupported continuation err = %v", err)
	}
}

func TestLoopContentProjectionAndRoleHelpers(t *testing.T) {
	message := agentic.Message{Role: agentic.RoleAssistant, Content: []agentic.Part{
		{Type: agentic.ContentText, Text: ""},
		{Type: agentic.ContentText, Text: "kept"},
		{Type: agentic.ContentToolUse},
		{Type: agentic.ContentToolUse, ToolUse: &agentic.ToolUse{ID: "c1", Name: "t", Input: map[string]any{"k": "v"}}},
		{Type: agentic.ContentToolResult},
		{Type: agentic.ContentToolResult, ToolResult: &agentic.ToolResult{ToolUseID: "c1", Name: "t", Content: `[1,2]`}},
		{Type: agentic.ContentToolResult, ToolResult: &agentic.ToolResult{ToolUseID: "c2", Name: "t", Content: "not json", IsError: true}},
		{Type: agentic.ContentThinking, Thinking: &agentic.ThinkingBlock{Text: "secret"}},
	}}
	blocks := projectMessageToEntryBlocks(message)
	if len(blocks) != 4 {
		t.Fatalf("blocks = %#v", blocks)
	}
	if blocks[0].Text != "kept" || blocks[1].ToolCall.CallID != "c1" ||
		string(blocks[2].Data) != `[1,2]` || blocks[3].Data != nil || !blocks[3].ToolResult.IsError {
		t.Fatalf("blocks = %#v", blocks)
	}

	if loopRole(agentic.RoleAssistant) != sessionloop.RoleAssistant ||
		loopRole(agentic.RoleTool) != sessionloop.RoleTool ||
		loopRole(agentic.RoleUser) != sessionloop.RoleUser ||
		loopRole(agentic.RoleSystem) != sessionloop.RoleUser {
		t.Fatal("loopRole mapping broken")
	}
	if loopCommandKind(QueueSteer) != sessionloop.CommandSteer ||
		loopCommandKind(QueueFollowUp) != sessionloop.CommandFollowUp ||
		loopCommandKind(QueueNextTurn) != sessionloop.CommandNextTurn ||
		loopCommandKind(QueueKind("custom")) != sessionloop.CommandKind("custom") {
		t.Fatal("loopCommandKind mapping broken")
	}
	if cloneLoopInputBlocks(nil) != nil {
		t.Fatal("cloneLoopInputBlocks(nil) != nil")
	}
	inputBlocks := projectMessageToInputBlocks(agentic.NewTextMessage(agentic.RoleUser, "kept"))
	cloned := cloneLoopInputBlocks(inputBlocks)
	cloned[0].Text = "mutated"
	if inputBlocks[0].Text != "kept" {
		t.Fatal("cloneLoopInputBlocks shares memory")
	}
}

func TestLoopProjectionAuxiliaryBranches(t *testing.T) {
	payloadCodec := jsoncodec.New()
	encode := func(value any) []byte {
		encoded, err := codec.Encode(payloadCodec, value)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}

	// Unknown message sources default to a start-origin entry.
	projector := newLoopProjector("sess", payloadCodec, nil)
	events, err := projector.apply(context.Background(), event.Record{
		Cursor: 2, Nature: agentic.EventAuthoritative, Source: "harness", Name: kindMessage,
		Payload: encode(messagePayload{Message: agentic.NewTextMessage(agentic.RoleUser, "odd"), Source: "unexpected"}),
	})
	if err != nil || len(events) != 1 || events[0].Entry.Origin != sessionloop.OriginStart {
		t.Fatalf("unknown-source message = %#v err=%v", events, err)
	}

	// Recovery-injected messages route through the same projection.
	events, err = projector.apply(context.Background(), event.Record{
		Cursor: 3, Nature: agentic.EventAuthoritative, Source: "harness.recovery",
		Type:    agentic.EventTypeTurnMessagesInjected,
		Payload: encode(event.MessagesPayload{Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "reinjected")}, QueueIDs: []string{"q-lost"}}),
	})
	if err != nil || len(events) != 1 || events[0].Entry.Origin != sessionloop.OriginSteer {
		t.Fatalf("recovery injection = %#v err=%v", events, err)
	}

	// A failing suspension projector propagates from both suspension kinds.
	failing := newLoopProjector("sess", payloadCodec, func(agentic.Suspension) (sessionloop.Suspension, error) {
		return sessionloop.Suspension{}, errors.New("projector down")
	})
	if _, err := failing.apply(context.Background(), event.Record{
		Cursor: 4, Nature: agentic.EventAuthoritative, Source: "agentic", Type: agentic.EventTypeRunSuspended,
		Payload: encode(event.SuspensionPayload{Suspension: agentic.Suspension{ID: "s"}}),
	}); err == nil || err.Error() != "projector down" {
		t.Fatalf("run_suspended projector err = %v", err)
	}
	if _, err := failing.apply(context.Background(), event.Record{
		Cursor: 5, Nature: agentic.EventAuthoritative, Source: "harness", Name: kindRecoverySuspension,
		Payload: encode(event.SuspensionPayload{Suspension: agentic.Suspension{ID: "s"}}),
	}); err == nil {
		t.Fatal("recovery.suspension projector error swallowed")
	}

	// Malformed preview payloads are dropped, never fatal (previews are lossy).
	lossy := newLoopProjector("sess", payloadCodec, nil)
	for _, eventType := range []agentic.EventType{
		agentic.EventTypeTextPreview, agentic.EventTypeThinkingPreview,
		agentic.EventTypeToolCallPreview, agentic.EventTypeToolArgumentPreview,
	} {
		events, applyErr := lossy.apply(context.Background(), event.Record{
			Cursor: 6, Nature: agentic.EventPreview, Source: "agentic", Type: eventType, Payload: []byte("{"),
		})
		if applyErr != nil || len(events) != 0 {
			t.Fatalf("malformed preview type %d: events=%d err=%v", eventType, len(events), applyErr)
		}
	}
	events, err = lossy.apply(context.Background(), event.Record{
		Cursor: 6, Nature: agentic.EventPreview, Source: "agentic", Type: agentic.EventTypeToolCallPreview,
		Payload: encode(event.ToolBatchPayload{}),
	})
	if err != nil || len(events) != 0 {
		t.Fatalf("empty tool-call preview: events=%d err=%v", len(events), err)
	}
}

type failingSecondIDs struct {
	calls int
}

func (f *failingSecondIDs) New(prefix string) (string, error) {
	f.calls++
	if prefix == "cmd" {
		return "", errors.New("cmd id generator down")
	}
	return fmt.Sprintf("%s-%d", prefix, f.calls), nil
}

func TestLoopViewDispatchBranchCoverage(t *testing.T) {
	// Generated command IDs surface generator failures.
	failingIDs := &failingSecondIDs{}
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), func(config *Config[string], _ *LoopConfig[string]) {
		config.IDs = failingIDs
	})
	if _, err := view.Dispatch(loopTestContext(t), sessionloopStartCommand("needs id")); err == nil {
		t.Fatal("command ID generator failure swallowed")
	}

	// Queue dispatch rejects non-text blocks before acceptance.
	plain, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	if _, err := plain.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandNextTurn,
		Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{
			{Kind: sessionloop.InputBlockData, Data: json.RawMessage(`1`)},
		}},
	}); !errors.Is(err, sessionloop.ErrUnsupported) {
		t.Fatalf("queue data block err = %v", err)
	}

	// Interrupt maps a faulted session onto ErrSessionFaulted.
	repository := newHookRepository()
	model := &scriptedModel{steps: []modelStep{textStep("faults")}}
	faulted, session := newLoopViewForTest(t, agentic.NewAgent("", model), repository, nil)
	boom := errors.New("append down")
	repository.journal().set(func(entries []store.PendingEntry) error {
		if batchHasKind(entries, kindAssistantCommitted) {
			return boom
		}
		return nil
	}, nil)
	receipt := loopDispatch(t, faulted, sessionloopStartCommand("fault run"))
	awaitState(t, session, Faulted)
	if _, err := faulted.Dispatch(loopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandInterrupt, RunID: receipt.RunID,
	}); !errors.Is(err, sessionloop.ErrSessionFaulted) {
		t.Fatalf("faulted interrupt err = %v", err)
	}
	if snapshot, err := faulted.Snapshot(loopTestContext(t)); err != nil || snapshot.State != sessionloop.StateFaulted {
		t.Fatalf("faulted snapshot = %#v err=%v", snapshot, err)
	}
	if err := faulted.Close(context.Background()); err != nil {
		t.Fatalf("Close of faulted view = %v", err)
	}
}

func TestLoopViewSnapshotBranchCoverage(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	receipt := loopDispatch(t, view, sessionloopStartCommand("snapshot me"))
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	_ = receipt
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := view.Snapshot(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled snapshot err = %v", err)
	}

	// A suspension projector failure propagates from Snapshot.
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "g", Name: "danger", Input: map[string]any{}})},
		textStep("done"),
	}}
	agent := agentic.NewAgent("", model)
	agentic.AddTool(agent,
		func(context.Context, loopToolInput) (string, error) { return "ok", nil },
		agentic.AutoToolName("danger"), agentic.AutoToolDescription("gated"))
	suspendedView, session := newLoopViewForTest(t, agent, storememory.New(), func(config *Config[string], loopConfig *LoopConfig[string]) {
		gate := suspendingGate{}
		config.ToolGate = gate
		loopConfig.SuspensionProjector = func(agentic.Suspension) (sessionloop.Suspension, error) {
			return sessionloop.Suspension{}, errors.New("cannot project")
		}
	})
	loopDispatch(t, suspendedView, sessionloopStartCommand("pause"))
	awaitState(t, session, Suspended)
	if _, err := suspendedView.Snapshot(loopTestContext(t)); err == nil || err.Error() != "cannot project" {
		t.Fatalf("suspension projector snapshot err = %v", err)
	}
}

// suspendingGate defers every regular batch with a minimal valid deferral
// envelope so tests can reach Suspended without the permission capability.
type suspendingGate struct{}

func (suspendingGate) EvaluateBatch(_ context.Context, calls []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
	ids := make([]string, len(calls))
	dispositions := make([]agentic.ToolDisposition, len(calls))
	for index, call := range calls {
		ids[index] = call.ID
		dispositions[index] = agentic.ToolDisposition{Kind: agentic.ToolDispositionSuspend}
	}
	payload, err := json.Marshal(struct {
		Version               int      `json:"version"`
		RequiredResolutionIDs []string `json:"required_resolution_ids"`
	}{Version: 1, RequiredResolutionIDs: ids})
	if err != nil {
		return agentic.ToolBatchDecision{}, err
	}
	return agentic.ToolBatchDecision{
		Calls:    dispositions,
		Deferral: &agentic.ToolDeferral{Kind: "harness.permission.v1", Payload: payload},
	}, nil
}

func TestLoopStreamTerminalBranches(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	stream := loopSubscribe(t, view, sessionloop.SubscribeOptions{})

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := stream.Next(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Next err = %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Next(loopTestContext(t)); !errors.Is(err, io.EOF) {
		t.Fatalf("Next after Close err = %v", err)
	}

	// The terminal error repeats once set.
	lagging := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 1})
	loopDispatch(t, view, sessionloopStartCommand("one"))
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	loopDispatch(t, view, sessionloopStartCommand("two"))
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	var terminal error
	for {
		if _, err := lagging.Next(loopTestContext(t)); err != nil {
			terminal = err
			break
		}
	}
	if !errors.Is(terminal, sessionloop.ErrLagged) {
		t.Fatalf("terminal err = %v", terminal)
	}
	if _, err := lagging.Next(loopTestContext(t)); !errors.Is(err, sessionloop.ErrLagged) {
		t.Fatalf("repeated terminal err = %v", err)
	}

	// A projection failure inside Next surfaces to the caller: corrupt the
	// stream by publishing a malformed durable record directly.
	broken := loopSubscribe(t, view, sessionloop.SubscribeOptions{})
	view.inner.bus.PublishDurable(event.Record{
		Cursor: view.inner.currentCursor().Seq + 1, Nature: agentic.EventAuthoritative,
		Source: "harness", Name: kindRunOpened, Payload: []byte("{"),
	})
	sawError := false
	for {
		_, err := broken.Next(loopTestContext(t))
		if err == nil {
			continue
		}
		if !errors.Is(err, io.EOF) {
			sawError = true
		}
		break
	}
	if !sawError {
		t.Fatal("projection decode failure never surfaced from Next")
	}
}

func TestLoopViewCloseBranchCoverage(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "held"), entered: entered, release: release},
	}}
	view, _ := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), nil)
	loopDispatch(t, view, sessionloopStartCommand("cancel close"))
	awaitSignal(t, entered, "model entered")

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := view.Close(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Close err = %v", err)
	}
	// A canceled close is not memoized; a real close still succeeds.
	close(release)
	if err := view.Close(context.Background()); err != nil {
		t.Fatalf("Close after canceled attempt = %v", err)
	}
}

func TestLoopViewResolveTargetsSuspendedRunAfterLegacyStart(t *testing.T) {
	// A legacy-started run resolved through the view records the resolve
	// command as the run's attribution (there was no start command).
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "g2", Name: "danger", Input: map[string]any{}})},
		textStep("done"),
	}}
	agent := agentic.NewAgent("", model)
	agentic.AddTool(agent,
		func(context.Context, loopToolInput) (string, error) { return "ok", nil },
		agentic.AutoToolName("danger"), agentic.AutoToolDescription("gated"))
	view, session := newLoopViewForTest(t, agent, storememory.New(), func(config *Config[string], _ *LoopConfig[string]) {
		config.ToolGate = suspendingGate{}
	})
	promptDone := promptAsync(session, "legacy suspend")
	awaitState(t, session, Suspended)
	outcome := awaitPrompt(t, promptDone, "legacy prompt suspension")
	if outcome.err != nil || outcome.execution.Status != agentic.ExecutionSuspended {
		t.Fatalf("legacy suspension outcome=%#v err=%v", outcome.execution, outcome.err)
	}
	snapshot, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	resolve := sessionloop.Command{
		ID:    "cmd-late-resolve",
		Kind:  sessionloop.CommandResolve,
		RunID: sessionloop.RunID(snapshot.RunID),
		Resolution: &sessionloop.Resolution{
			SuspensionID: snapshot.Suspension.ID,
			Decisions:    []sessionloop.ResolutionDecision{{ID: "g2", Action: sessionloop.ResolutionApprove}},
		},
	}
	receipt, err := view.Dispatch(loopTestContext(t), resolve)
	if err != nil {
		t.Fatalf("resolve of legacy run = %v", err)
	}
	if err := session.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	if view.commandForRun(string(receipt.RunID)) != "cmd-late-resolve" {
		t.Fatalf("resolve attribution = %q", view.commandForRun(string(receipt.RunID)))
	}
}

func TestMapSessionLoopHelpersInHost(t *testing.T) {
	// projectedOutput honors context cancellation while a run is in flight.
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewTextMessage(agentic.RoleAssistant, "held"), entered: entered, release: release},
	}}
	view, _ := newLoopViewForTest(t, agentic.NewAgent("", model), storememory.New(), func(_ *Config[string], loopConfig *LoopConfig[string]) {
		loopConfig.OutputProjector = func(output string) (json.RawMessage, error) { return json.Marshal(output) }
	})
	receipt := loopDispatch(t, view, sessionloopStartCommand("wait for me"))
	awaitSignal(t, entered, "model entered")
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if output := view.projectedOutput(canceled, string(receipt.RunID)); output != nil {
		t.Fatalf("canceled wait yielded output %q", output)
	}
	close(release)
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
}

// TestLoopStreamInjectAfterTerminal proves the injection side-channel never
// resurrects a closed or terminally failed stream: injected live-only events
// are dropped once the stream is closed or lagged.
func TestLoopStreamInjectAfterTerminal(t *testing.T) {
	view, _ := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	closed := loopSubscribe(t, view, sessionloop.SubscribeOptions{})
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	closed.(*loopStream).inject(sessionloop.Event{Kind: sessionloop.EventSessionState})
	if _, err := closed.Next(loopTestContext(t)); !errors.Is(err, io.EOF) {
		t.Fatalf("Next after close+inject = %v, want io.EOF", err)
	}

	lagging := loopSubscribe(t, view, sessionloop.SubscribeOptions{Buffer: 1})
	loopDispatch(t, view, sessionloopStartCommand("one"))
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	loopDispatch(t, view, sessionloopStartCommand("two"))
	if err := view.inner.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := lagging.Next(loopTestContext(t)); err != nil {
			if !errors.Is(err, sessionloop.ErrLagged) {
				t.Fatalf("terminal err = %v, want ErrLagged", err)
			}
			break
		}
	}
	lagging.(*loopStream).inject(sessionloop.Event{Kind: sessionloop.EventSessionState})
	if _, err := lagging.Next(loopTestContext(t)); !errors.Is(err, sessionloop.ErrLagged) {
		t.Fatalf("Next after lag+inject = %v, want the terminal ErrLagged", err)
	}
}

// TestAcceptWithCursorStaleTargetFastFail covers the first locked stale
// check: a target naming a foreign run fails before ID generation.
func TestAcceptWithCursorStaleTargetFastFail(t *testing.T) {
	driver := &countingDriver{}
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(context.Background()) }()
	accepted, err := session.prepareStart(context.Background(),
		agentic.NewTextMessage(agentic.RoleUser, "hold the run"), context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := session.acceptWithCursor(context.Background(), QueueSteer,
		agentic.NewTextMessage(agentic.RoleUser, "steer"), "run-other"); !errors.Is(err, errStaleRunTarget) {
		t.Fatalf("stale-target acceptance = %v, want errStaleRunTarget", err)
	}
	if _, err := session.requestInterrupt(context.Background(), accepted.runID); err != nil {
		t.Fatal(err)
	}
	if err := session.finishInterrupt(&agentic.Execution[string]{Status: agentic.ExecutionInterrupted}); err != nil {
		t.Fatal(err)
	}
}

func TestLoopViewProjectionRejectsCorruptJournal(t *testing.T) {
	view, session := newLoopViewForTest(t, &countingDriver{}, storememory.New(), nil)
	receipt := loopDispatch(t, view, sessionloopStartCommand("seed history"))
	if err := session.WaitForIdle(loopTestContext(t)); err != nil {
		t.Fatal(err)
	}
	_ = receipt

	// Corrupt harness-kind payload: record conversion succeeds, apply fails.
	commit, err := session.journal.Append(context.Background(), session.currentCursor(), store.PendingEntry{
		Kind: kindRunOpened, Payload: []byte("{"), Durability: store.DurabilitySync,
	})
	if err != nil {
		t.Fatal(err)
	}
	session.mu.Lock()
	session.cursor = commit.Cursor
	session.mu.Unlock()
	if _, err := view.Snapshot(loopTestContext(t)); err == nil {
		t.Fatal("snapshot over a corrupt harness payload succeeded")
	}
	if _, err := view.Subscribe(loopTestContext(t), sessionloop.SubscribeOptions{
		After: sessionloop.Position{Sequence: commit.Cursor.Seq},
	}); err == nil {
		t.Fatal("seeded subscribe over a corrupt harness payload succeeded")
	}

	// Corrupt agentic-kind payload: the double decode fails during record
	// conversion for snapshot and seeded subscribe alike.
	commit, err = session.journal.Append(context.Background(), session.currentCursor(), store.PendingEntry{
		Kind: kindAssistantCommitted, Payload: []byte("{"), Durability: store.DurabilitySync,
	})
	if err != nil {
		t.Fatal(err)
	}
	session.mu.Lock()
	session.cursor = commit.Cursor
	session.mu.Unlock()
	if _, err := view.Snapshot(loopTestContext(t)); err == nil {
		t.Fatal("snapshot over a corrupt agentic payload succeeded")
	}
	if _, err := view.Subscribe(loopTestContext(t), sessionloop.SubscribeOptions{
		After: sessionloop.Position{Sequence: commit.Cursor.Seq},
	}); err == nil {
		t.Fatal("seeded subscribe over a corrupt agentic payload succeeded")
	}
}

func TestLoopPreviewBeforeAnyDurablePosition(t *testing.T) {
	payloadCodec := jsoncodec.New()
	projector := newLoopProjector("sess", payloadCodec, nil)
	payload, err := codec.Encode(payloadCodec, struct{ Delta string }{"early"})
	if err != nil {
		t.Fatal(err)
	}
	events, err := projector.apply(context.Background(), event.Record{
		Cursor: 5, Nature: agentic.EventPreview, Source: "agentic",
		Type: agentic.EventTypeTextPreview, Payload: payload,
	})
	if err != nil || len(events) != 1 {
		t.Fatalf("events=%d err=%v", len(events), err)
	}
	if events[0].Position.Sequence != 5 {
		t.Fatalf("early preview position = %#v, want the record cursor fallback", events[0].Position)
	}
}
