package agentic

import (
	"context"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/internal/testutil"
)

type phase1Toolset struct {
	tools    []Tool
	handlers []ToolHandler
}

func (s phase1Toolset) ToolsAndHandlers() ([]Tool, []ToolHandler) {
	return s.tools, s.handlers
}

type phase1Registry struct {
	tools []Tool
}

func (r phase1Registry) Register(Tool, ToolHandler) error { return nil }
func (r phase1Registry) Get(string) (ToolHandler, bool)   { return nil, false }
func (r phase1Registry) Execute(_ context.Context, call ToolUse, _ any) (ToolExecutionResult, error) {
	return makeToolResultError(call, "phase1 registry", nil), nil
}
func (r phase1Registry) ExecuteBatch(context.Context, []ToolUse, any) ([]ToolExecutionResult, error) {
	return nil, nil
}
func (r phase1Registry) Tools() []Tool   { return r.tools }
func (r phase1Registry) Has(string) bool { return false }
func (r phase1Registry) Count() int      { return len(r.tools) }

func TestPhase1TranscriptValidationAndDefensiveCopies(t *testing.T) {
	call := ToolUse{ID: "call-1", Name: "tool", Input: map[string]any{"value": "x"}}
	valid := []Message{NewToolUseMessage(call), NewToolResultMessageFor("call-1", "tool", "ok", false)}
	frontier, err := inspectTranscript(valid)
	if err != nil || len(frontier) != 0 {
		t.Fatalf("valid transcript = %#v, %v", frontier, err)
	}

	cases := []struct {
		name     string
		messages []Message
	}{
		{
			name:     "tool call outside assistant",
			messages: []Message{{Role: RoleUser, Content: []Part{{Type: ContentToolUse, ToolUse: &call}}}},
		},
		{
			name:     "duplicate call IDs",
			messages: []Message{NewToolUseMessage(call, call)},
		},
		{
			name:     "tool result outside tool message",
			messages: []Message{NewToolUseMessage(call), {Role: RoleUser, Content: NewToolResultMessageFor("call-1", "tool", "x", false).Content}},
		},
		{
			name:     "orphan result",
			messages: []Message{NewToolResultMessageFor("call-1", "tool", "x", false)},
		},
		{
			name:     "unknown result",
			messages: []Message{NewToolUseMessage(call), NewToolResultMessageFor("other", "tool", "x", false)},
		},
		{
			name: "duplicate result",
			messages: []Message{NewToolUseMessage(call), {
				Role: RoleTool,
				Content: append(NewToolResultMessageFor("call-1", "tool", "x", false).Content,
					NewToolResultMessageFor("call-1", "tool", "y", false).Content...),
			}},
		},
		{
			name:     "name mismatch",
			messages: []Message{NewToolUseMessage(call), NewToolResultMessageFor("call-1", "other", "x", false)},
		},
		{
			name:     "new frontier while open",
			messages: []Message{NewToolUseMessage(call), NewToolUseMessage(ToolUse{ID: "call-2", Name: "other"})},
		},
		{
			name:     "user while open",
			messages: []Message{NewToolUseMessage(call), NewTextMessage(RoleUser, "too early")},
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if _, err := inspectTranscript(test.messages); !errors.Is(err, ErrTranscriptInvalid) {
				t.Fatalf("inspectTranscript error = %v, want ErrTranscriptInvalid", err)
			}
		})
	}

	open, err := inspectTranscript([]Message{NewToolUseMessage(call)})
	if err != nil || len(open) != 1 || open[0].ID != "call-1" {
		t.Fatalf("open frontier = %#v, %v", open, err)
	}
	open[0].Input["value"] = "mutated"
	if call.Input["value"] != "x" {
		t.Fatalf("frontier copy mutated source: %#v", call)
	}

	nonJSONCall := ToolUse{ID: "non-json", Name: "tool", Input: map[string]any{"fn": func() {}}}
	if got := cloneToolUses([]ToolUse{nonJSONCall}); len(got) != 1 {
		t.Fatalf("fallback clone calls = %#v", got)
	}
	nonJSONMessage := NewToolUseMessage(nonJSONCall)
	if got := cloneMessages([]Message{nonJSONMessage}); len(got) != 1 {
		t.Fatalf("fallback clone messages = %#v", got)
	}

	calls, results := transcriptToolState(valid)
	if len(calls) != 1 || len(results) != 1 || results[0].ToolName != "tool" {
		t.Fatalf("transcript state = %#v / %#v", calls, results)
	}
}

func TestPhase1GateAndResultValidationBranches(t *testing.T) {
	calls := []ToolUse{{ID: "one", Name: "tool"}}
	validResult := ToolExecutionResult{ToolUseID: "one", ToolName: "tool", Content: "ok"}
	cases := []struct {
		name      string
		decision  ToolBatchDecision
		suspended bool
		wantErr   bool
	}{
		{"execute", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionExecute}}}, false, false},
		{"return", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionReturn, Result: &validResult}}}, false, false},
		{"suspend", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionSuspend}}, Deferral: &ToolDeferral{Kind: "approval"}}, true, false},
		{"wrong cardinality", ToolBatchDecision{}, false, true},
		{"execute has result", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionExecute, Result: &validResult}}}, false, true},
		{"return missing result", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionReturn}}}, false, true},
		{"wrong result identity", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionReturn, Result: &ToolExecutionResult{ToolUseID: "wrong", ToolName: "tool"}}}}, false, true},
		{"suspend missing deferral", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionSuspend}}}, false, true},
		{"suspend continuation", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionSuspend, Continue: true}}, Deferral: &ToolDeferral{Kind: "approval"}}, false, true},
		{"deferral without suspend", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionExecute}}, Deferral: &ToolDeferral{Kind: "approval"}}, false, true},
		{"invalid kind", ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionInvalid}}}, false, true},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			suspended, err := validateToolBatchDecision(calls, test.decision)
			if test.wantErr {
				if err == nil {
					t.Fatal("expected validation error")
				}
				return
			}
			if err != nil || suspended != test.suspended {
				t.Fatalf("validate decision = %v, %v", suspended, err)
			}
		})
	}
	if err := validateToolResultIdentity(calls[0], ToolExecutionResult{ToolUseID: "one", ToolName: "other"}); err == nil {
		t.Fatal("expected name mismatch")
	}

	gate := ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) {
		return ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionExecute}}}, nil
	})
	if decision, err := gate.EvaluateBatch(context.Background(), calls); err != nil || decision.Calls[0].Kind != ToolDispositionExecute {
		t.Fatalf("ToolGateFunc = %#v, %v", decision, err)
	}
	processor := ToolResultProcessorFunc(func(_ context.Context, _ ToolUse, result ToolExecutionResult) (ToolExecutionResult, error) {
		result.Content = "changed"
		return result, nil
	})
	if result, err := processor.Process(context.Background(), calls[0], validResult); err != nil || result.Content != "changed" {
		t.Fatalf("ToolResultProcessorFunc = %#v, %v", result, err)
	}
}

func TestPhase1ToolResultProcessorRejectsUnsafeProjection(t *testing.T) {
	call := ToolUse{ID: "call", Name: "tool"}
	base := ToolExecutionResult{ToolUseID: "call", ToolName: "tool", Content: "ok"}
	newState := func(processor ToolResultProcessor) *loopState {
		return &loopState{ctx: context.Background(), options: &runOptions{toolResultProcessor: processor}}
	}
	c := &agentCore{}

	if result, err := c.projectToolResult(context.Background(), newState(nil), call, base); err != nil || result.Content != "ok" {
		t.Fatalf("no processor = %#v, %v", result, err)
	}
	for _, test := range []struct {
		name      string
		base      ToolExecutionResult
		processor ToolResultProcessor
	}{
		{
			name: "processor error",
			base: base,
			processor: ToolResultProcessorFunc(func(context.Context, ToolUse, ToolExecutionResult) (ToolExecutionResult, error) {
				return ToolExecutionResult{}, errors.New("boom")
			}),
		},
		{
			name: "identity changed",
			base: base,
			processor: ToolResultProcessorFunc(func(_ context.Context, _ ToolUse, result ToolExecutionResult) (ToolExecutionResult, error) {
				result.ToolName = "other"
				return result, nil
			}),
		},
		{
			name: "error became success",
			base: ToolExecutionResult{ToolUseID: "call", ToolName: "tool", IsError: true, Error: errors.New("tool failed")},
			processor: ToolResultProcessorFunc(func(_ context.Context, _ ToolUse, result ToolExecutionResult) (ToolExecutionResult, error) {
				result.IsError = false
				return result, nil
			}),
		},
		{
			name: "error removed",
			base: ToolExecutionResult{ToolUseID: "call", ToolName: "tool", IsError: true, Error: errors.New("tool failed")},
			processor: ToolResultProcessorFunc(func(_ context.Context, _ ToolUse, result ToolExecutionResult) (ToolExecutionResult, error) {
				result.Error = nil
				return result, nil
			}),
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			result, err := c.projectToolResult(context.Background(), newState(test.processor), call, test.base)
			if err == nil || !result.IsError {
				t.Fatalf("unsafe projection = %#v, %v", result, err)
			}
		})
	}
	if result, err := c.projectToolResult(context.Background(), newState(nil), call, ToolExecutionResult{ToolUseID: "wrong", ToolName: "tool"}); err == nil || !result.IsError {
		t.Fatalf("invalid input result identity = %#v, %v", result, err)
	}
}

func TestPhase1RegistryAndDriverHelpers(t *testing.T) {
	empty, err := buildExecutionToolRegistry(nil, nil)
	if err != nil || empty.Count() != 0 || len(empty.Tools()) != 0 {
		t.Fatalf("empty run registry = %#v, %v", empty, err)
	}
	if _, ok := empty.Get("missing"); ok {
		t.Fatal("empty registry unexpectedly found handler")
	}
	if _, ok := empty.Tool("missing"); ok {
		t.Fatal("empty registry unexpectedly found definition")
	}
	unknown, err := empty.Execute(context.Background(), ToolUse{ID: "missing", Name: "missing"}, nil)
	if err != nil || !unknown.IsError {
		t.Fatalf("empty registry execute = %#v, %v", unknown, err)
	}

	definition, handler := MustToolPlain("tool", "tool", func(phase1ToolInput) (string, error) { return "ok", nil })
	if _, err := buildExecutionToolRegistry(nil, []Toolset{nil}); err == nil {
		t.Fatal("expected nil toolset error")
	}
	if _, err := buildExecutionToolRegistry(nil, []Toolset{phase1Toolset{tools: []Tool{definition}, handlers: nil}}); err == nil {
		t.Fatal("expected mismatched toolset error")
	}
	if _, err := buildExecutionToolRegistry(nil, []Toolset{phase1Toolset{tools: []Tool{definition}, handlers: []ToolHandler{&noopToolHandler{name: "other"}}}}); err == nil {
		t.Fatal("expected handler-name registration error")
	}
	if _, err := buildExecutionToolRegistry(NewRegistry(), []Toolset{phase1Toolset{tools: []Tool{definition}, handlers: []ToolHandler{handler}}, phase1Toolset{tools: []Tool{definition}, handlers: []ToolHandler{handler}}}); err == nil {
		t.Fatal("expected duplicate overlay error")
	}
	if _, err := buildExecutionToolRegistry(phase1Registry{tools: []Tool{{}}}, nil); err == nil {
		t.Fatal("expected empty base tool name error")
	}
	base := phase1Registry{}
	routed, err := buildExecutionToolRegistry(base, nil)
	if err != nil || !routed.hasExecutor {
		t.Fatalf("custom base registry = %#v, %v", routed, err)
	}
	if result, err := routed.Execute(context.Background(), ToolUse{ID: "base", Name: "dynamic"}, nil); err != nil || !result.IsError {
		t.Fatalf("dynamic base dispatch = %#v, %v", result, err)
	}

	for _, status := range []ExecutionStatus{ExecutionSuspended, ExecutionStopped, ExecutionInterrupted, ExecutionFailed, ExecutionCompleted} {
		err := executionError(&Execution[string]{Status: status}, nil)
		if status == ExecutionCompleted && err != nil {
			t.Fatalf("completed execution error = %v", err)
		}
		if status != ExecutionCompleted && err == nil {
			t.Fatalf("status %v did not map to an error", status)
		}
	}
	if id := newSuspensionID(); id == "" {
		t.Fatal("empty suspension ID")
	}
	baseEvent := newEventBase(EventLifecycle, EventTypeRunEnded, 4)
	if baseEvent.Nature() != EventLifecycle || baseEvent.Type() != EventTypeRunEnded || baseEvent.TurnIndex() != 4 {
		t.Fatalf("event base = %#v", baseEvent)
	}
}

func TestPhase1TurnArbitrationAndStreamTransportErrors(t *testing.T) {
	tool, handler := MustToolPlain("tool", "tool", func(phase1ToolInput) (string, error) { return "ok", nil })
	model := &testutil.StubModel{NameValue: "model", Response: &ChatResponse{Message: NewTextMessage(RoleAssistant, "done"), FinishReason: FinishReasonStop}}
	agent := NewAgent("system", model).AddTool(tool, handler)

	stopped, err := agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: messagePointer(NewTextMessage(RoleUser, "run"))}, WithRunTurnHook(func(context.Context, Turn) (TurnDecision, error) {
		return TurnDecision{Action: TurnStop}, nil
	}))
	if err != nil || stopped.Status != ExecutionCompleted {
		t.Fatalf("TurnStop with valid candidate = %#v, %v", stopped, err)
	}

	model.Response = &ChatResponse{Message: NewToolUseMessage(ToolUse{ID: "tool-1", Name: "tool", Input: map[string]any{"value": "x"}}), FinishReason: FinishReasonToolCalls}
	stopped, err = agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: messagePointer(NewTextMessage(RoleUser, "run"))}, WithRunTurnHook(func(context.Context, Turn) (TurnDecision, error) {
		return TurnDecision{Action: TurnStop}, nil
	}))
	if err != nil || stopped.Status != ExecutionStopped {
		t.Fatalf("TurnStop without candidate = %#v, %v", stopped, err)
	}

	model.Response = &ChatResponse{Message: NewTextMessage(RoleAssistant, "done"), FinishReason: FinishReasonStop}
	if execution, err := agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: messagePointer(NewTextMessage(RoleUser, "run"))}, WithRunTurnHook(func(context.Context, Turn) (TurnDecision, error) {
		return TurnDecision{Action: TurnContinue}, nil
	})); err == nil || execution.Status != ExecutionFailed {
		t.Fatalf("empty TurnContinue = %#v, %v", execution, err)
	}
	if execution, err := agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: messagePointer(NewTextMessage(RoleUser, "run"))}, WithRunEventSink(EventSinkFunc(func(_ context.Context, event Event) error {
		if event.Type() == EventTypeTurnEnded {
			return errors.New("sink failed")
		}
		return nil
	}))); err == nil || execution.Status != ExecutionFailed {
		t.Fatalf("turn end sink error = %#v, %v", execution, err)
	}

	streamState := &loopState{ctx: context.Background(), options: &runOptions{eventSink: EventSinkFunc(func(context.Context, Event) error { return nil })}}
	streamModel := testutil.NewScriptedStream(
		StreamEvent{Type: StreamEventThinkingDelta, Delta: "think", Signature: "sig", ProviderName: "provider", ThinkingID: "id"},
		StreamEvent{Type: StreamEventTextDelta, Delta: "text"},
		StreamEvent{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "stream-call", Name: "tool"}},
		StreamEvent{Type: StreamEventToolCallDelta, ToolCallID: "stream-call", Delta: `{"value":"x"}`},
		StreamEvent{Type: StreamEventDone, Usage: &Usage{TotalTokens: 3}, FinishReason: FinishReasonToolCalls},
	)
	message, usage, finish, err := agent.core.consumeStream(streamState, streamModel)
	if err != nil || message.GetThinkingContent() != "think" || message.GetTextContent() != "text" || usage.TotalTokens != 3 || finish != FinishReasonToolCalls {
		t.Fatalf("consume stream = %#v, %#v, %q, %v", message, usage, finish, err)
	}
	badArgs := testutil.NewScriptedStream(
		StreamEvent{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "bad", Name: "tool"}},
		StreamEvent{Type: StreamEventToolCallDelta, ToolCallID: "bad", Delta: "{"},
	)
	if _, _, _, err := agent.core.consumeStream(streamState, badArgs); err == nil {
		t.Fatal("expected malformed streamed arguments error")
	}
	streamErr := errors.New("stream error")
	if _, _, _, err := agent.core.consumeStream(streamState, testutil.NewScriptedStream(StreamEvent{Type: StreamEventError, Error: streamErr})); !errors.Is(err, streamErr) {
		t.Fatalf("stream error = %v", err)
	}
}

func messagePointer(message Message) *Message { return &message }

func TestPhase1ResumeRetryStateAndPayloadValidation(t *testing.T) {
	tool, handler := MustToolPlain("retry", "retry", func(phase1ToolInput) (string, error) { return "", Retry("again") })
	model := &testutil.StubModel{NameValue: "model", Response: &ChatResponse{Message: NewToolUseMessage(ToolUse{ID: "retry-1", Name: "retry", Input: map[string]any{"value": "x"}}), FinishReason: FinishReasonToolCalls}}
	agent := NewAgent("system", model, WithRetries(RetryConfig{MaxRetries: 1})).AddTool(tool, handler)
	prompt := NewTextMessage(RoleUser, "run")
	suspended, err := agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt}, WithRunToolGate(ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) {
		return ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionSuspend}}, Deferral: &ToolDeferral{Kind: "retry"}}, nil
	})))
	if err != nil {
		t.Fatalf("suspend retry = %v", err)
	}
	// A resume retry attempts one follow-up request. Make it terminal so the
	// test observes the restored retry counters rather than looping forever.
	model.Response = &ChatResponse{Message: NewTextMessage(RoleAssistant, "done"), FinishReason: FinishReasonStop}
	completed, err := agent.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: *suspended.Suspension,
		Decisions: []ToolResumeDecision{{CallID: "retry-1", Action: ToolResumeExecute}},
	})
	if err != nil || completed.Result.Retries != 1 || completed.Result.Output != "done" {
		t.Fatalf("resume retry = %#v, %v", completed, err)
	}
	payload, err := decodeToolSuspension(Suspension{Payload: []byte(`{"Version":99}`)})
	if err == nil || payload.Version != 0 || !errors.Is(err, ErrSuspensionVersion) {
		t.Fatalf("unknown payload version = %#v, %v", payload, err)
	}
	if _, err := decodeToolSuspension(Suspension{Payload: []byte("not json")}); !errors.Is(err, ErrSuspensionVersion) {
		t.Fatalf("malformed payload error = %v", err)
	}
	if counts := copyRetryCounts(map[string]int{"retry": 1}); counts["retry"] != 1 {
		t.Fatalf("copied retry counts = %#v", counts)
	}
	if copyRetryCounts(nil) != nil {
		t.Fatal("nil retry counts did not stay nil")
	}
}

func TestPhase1DriverAndTurnValidationErrorBranches(t *testing.T) {
	call := ToolUse{ID: "open", Name: "tool"}
	prompt := NewTextMessage(RoleUser, "prompt")
	if err := validateDriveInput(DriveInput{Mode: DriveStart, Prompt: &prompt}, []Message{NewToolUseMessage(call)}); !errors.Is(err, ErrTranscriptInvalid) {
		t.Fatalf("open DriveStart = %v", err)
	}
	if err := validateDriveInput(DriveInput{Mode: DriveMode(99)}, nil); !errors.Is(err, ErrDriveInput) {
		t.Fatalf("unknown drive mode = %v", err)
	}
	if err := validateDriveInput(DriveInput{Mode: DriveContinue}, nil); !errors.Is(err, ErrDriveInput) {
		t.Fatalf("empty continue = %v", err)
	}
	if err := validateDriveInput(DriveInput{Mode: DriveContinue, History: []Message{NewTextMessage(RoleSystem, "system")}}, []Message{NewTextMessage(RoleSystem, "system")}); !errors.Is(err, ErrDriveInput) {
		t.Fatalf("system-only continue = %v", err)
	}

	for _, decision := range []TurnDecision{
		{Action: TurnDefault, Inject: []Message{NewTextMessage(RoleUser, "not allowed")}},
		{Action: TurnStop, Inject: []Message{NewTextMessage(RoleUser, "not allowed")}},
		{Action: TurnContinue},
		{Action: TurnContinue, Inject: []Message{NewTextMessage(RoleAssistant, "wrong role")}},
		{Action: TurnAction(99)},
	} {
		if err := validateTurnDecision(decision); !errors.Is(err, ErrTurnDecision) {
			t.Fatalf("turn decision %#v error = %v", decision, err)
		}
	}

	if !sameToolCalls([]ToolUse{{ID: "a", Input: map[string]any{"x": 1}}}, []ToolUse{{ID: "a", Input: map[string]any{"x": 1}}}) {
		t.Fatal("equal calls were not equal")
	}
	if sameToolCalls([]ToolUse{{ID: "a"}}, []ToolUse{{ID: "b"}}) || sameToolCalls([]ToolUse{{ID: "a"}}, nil) {
		t.Fatal("different calls were equal")
	}
	if sameToolCalls([]ToolUse{{ID: "a", Input: map[string]any{"bad": func() {}}}}, []ToolUse{{ID: "a", Input: map[string]any{"bad": func() {}}}}) {
		t.Fatal("non-JSON calls were equal")
	}
}

func TestPhase1ExecutionRegistryAndLegacyStreamAdapterBoundaries(t *testing.T) {
	var nilRegistry *executionToolRegistry
	if nilRegistry.Count() != 0 || nilRegistry.Tools() != nil {
		t.Fatalf("nil registry accessors = %d/%#v", nilRegistry.Count(), nilRegistry.Tools())
	}
	if _, ok := nilRegistry.Get("missing"); ok {
		t.Fatal("nil registry unexpectedly returned a handler")
	}
	if _, ok := nilRegistry.Tool("missing"); ok {
		t.Fatal("nil registry unexpectedly returned a tool")
	}
	missing := ToolUse{ID: "missing", Name: "missing"}
	result, err := nilRegistry.Execute(context.Background(), missing, nil)
	if err != nil || !result.IsError || result.ToolUseID != "missing" {
		t.Fatalf("nil registry execution = %#v, %v", result, err)
	}

	agent := NewAgent("system", &testutil.StubModel{NameValue: "model"})
	state, err := agent.core.prepareLoop(context.Background(), "prompt", dependencyEnvelope{})
	if err != nil || state.messages[len(state.messages)-1].GetTextContent() != "prompt" {
		t.Fatalf("prepare loop = %#v, %v", state, err)
	}
	if _, err := agent.core.prepareLoopForResume(context.Background(), []Message{NewTextMessage(RoleUser, "history")}, dependencyEnvelope{}, WithMessages(NewTextMessage(RoleUser, "extra"))); !errors.Is(err, ErrDriveInput) {
		t.Fatalf("resume WithMessages error = %v", err)
	}
	if _, err := agent.core.prepareLoopForResume(context.Background(), nil, dependencyEnvelope{}); !errors.Is(err, ErrDriveInput) {
		t.Fatalf("empty resume history error = %v", err)
	}

	ch := make(chan StreamEvent, 16)
	legacy := newLegacyStreamSink(ch)
	base := func(kind EventType) eventBase { return newEventBase(EventPreview, kind, 3) }
	call := ToolUse{ID: "call-1", Name: "tool", Input: map[string]any{"value": "x"}}
	for _, event := range []Event{
		&TextPreviewEvent{eventBase: base(EventTypeTextPreview), Delta: "preview"},
		&ThinkingPreviewEvent{eventBase: base(EventTypeThinkingPreview), Delta: "think", Signature: "sig", ProviderName: "provider", ThinkingID: "thinking"},
		&ToolCallPreviewEvent{eventBase: base(EventTypeToolCallPreview), Call: call},
		&ToolArgumentPreviewEvent{eventBase: base(EventTypeToolArgumentPreview), ToolCallID: call.ID, Delta: `{"value":"x"}`},
		&AssistantCommittedEvent{eventBase: newEventBase(EventAuthoritative, EventTypeAssistantCommitted, 4), Message: NewTextMessage(RoleAssistant, "committed")},
		&ToolStartedEvent{eventBase: base(EventTypeToolStarted), Call: call, Attempt: 1},
		&ToolResultCommittedEvent{eventBase: base(EventTypeToolResultCommitted), Result: ToolExecutionResult{ToolUseID: call.ID, ToolName: call.Name, Content: "ok"}},
		&RunCompletedEvent{eventBase: base(EventTypeRunCompleted), Usage: Usage{TotalTokens: 7}, FinishReason: FinishReasonStop},
	} {
		if err := legacy.Emit(context.Background(), event); err != nil {
			t.Fatalf("legacy Emit(%T): %v", event, err)
		}
	}
	if len(ch) != 7 { // duplicate ToolStarted must not duplicate the call-start event.
		t.Fatalf("legacy stream events = %d, want 7", len(ch))
	}
	last := StreamEvent{}
	for len(ch) > 0 {
		last = <-ch
	}
	if last.Type != StreamEventDone || last.Usage == nil || last.Usage.TotalTokens != 7 {
		t.Fatalf("legacy terminal event = %#v", last)
	}

	terminal := newLegacyStreamSink(make(chan StreamEvent, 2))
	if err := terminal.Emit(context.Background(), &RunSuspendedEvent{eventBase: base(EventTypeRunSuspended)}); err != nil {
		t.Fatalf("legacy suspended event: %v", err)
	}
	terminal.sendErrorIfNeeded(errors.New("duplicate error"))
	if len(terminal.ch) != 1 {
		t.Fatalf("legacy terminal emitted %d errors", len(terminal.ch))
	}

	firstCalls, secondCalls := 0, 0
	fanout := fanoutEventSink{
		first:  EventSinkFunc(func(context.Context, Event) error { firstCalls++; return nil }),
		second: EventSinkFunc(func(context.Context, Event) error { secondCalls++; return nil }),
	}
	if err := fanout.Emit(context.Background(), &RunStartedEvent{eventBase: base(EventTypeRunStarted)}); err != nil || firstCalls != 1 || secondCalls != 1 {
		t.Fatalf("fanout success = %v, calls %d/%d", err, firstCalls, secondCalls)
	}
	fanout.first = EventSinkFunc(func(context.Context, Event) error { return errors.New("first failed") })
	if err := fanout.Emit(context.Background(), &RunStartedEvent{eventBase: base(EventTypeRunStarted)}); err == nil || secondCalls != 1 {
		t.Fatalf("fanout first failure = %v, second calls %d", err, secondCalls)
	}
	if err := (fanoutEventSink{}).Emit(context.Background(), &RunStartedEvent{eventBase: base(EventTypeRunStarted)}); err != nil {
		t.Fatalf("empty fanout = %v", err)
	}
}

func TestPhase1SchedulerAndRegularTurnFailureBranches(t *testing.T) {
	definition, handler := MustToolPlain("tool", "tool", func(phase1ToolInput) (string, error) { return "ok", nil })
	registry, err := buildExecutionToolRegistry(nil, []Toolset{phase1Toolset{tools: []Tool{definition}, handlers: []ToolHandler{handler}}})
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	call := ToolUse{ID: "call", Name: "tool", Input: map[string]any{"value": "x"}}
	c := NewAgent("system", &testutil.StubModel{NameValue: "model"}).core

	newState := func(sink EventSink) *loopState {
		return &loopState{
			ctx:      context.Background(),
			options:  &runOptions{eventSink: sink},
			registry: registry,
			messages: []Message{NewToolUseMessage(call)},
		}
	}
	if _, _, err := c.executeAdmittedTools(newState(EventSinkFunc(func(_ context.Context, event Event) error {
		if event.Type() == EventTypeToolStarted {
			return errors.New("start event failed")
		}
		return nil
	})), []ToolUse{call}, nil); err == nil {
		t.Fatal("expected ToolStarted sink error")
	}

	broken := &streamRegistryStub{executeErr: errors.New("registry failed")}
	brokenRegistry, err := buildExecutionToolRegistry(broken, nil)
	if err != nil {
		t.Fatalf("broken registry: %v", err)
	}
	brokenState := newState(nil)
	brokenState.registry = brokenRegistry
	results, _, err := c.executeAdmittedTools(brokenState, []ToolUse{call}, nil)
	if err == nil || len(results) != 1 || !results[0].IsError {
		t.Fatalf("broken scheduler = %#v, %v", results, err)
	}

	for _, test := range []struct {
		name  string
		gate  ToolGate
		state func(*loopState)
	}{
		{
			name: "planning sink error",
			gate: allowAllToolGate{},
			state: func(state *loopState) {
				state.options.eventSink = EventSinkFunc(func(_ context.Context, event Event) error {
					if event.Type() == EventTypeToolBatchPlanned {
						return errors.New("planned failed")
					}
					return nil
				})
			},
		},
		{
			name: "gate error",
			gate: ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) {
				return ToolBatchDecision{}, errors.New("gate failed")
			}),
			state: func(*loopState) {},
		},
		{
			name:  "invalid gate decision",
			gate:  ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) { return ToolBatchDecision{}, nil }),
			state: func(*loopState) {},
		},
		{
			name:  "tool limit",
			gate:  allowAllToolGate{},
			state: func(state *loopState) { state.usageLimits = &UsageLimits{MaxToolCalls: IntPtr(0)} },
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			state := newState(nil)
			state.options.toolGate = test.gate
			test.state(state)
			outcome := c.executeRegularTurn(state, []ToolUse{call})
			if outcome.fatal == nil || !outcome.results[call.ID].IsError {
				t.Fatalf("regular failure outcome = %#v", outcome)
			}
		})
	}

	noTools := newState(nil)
	noTools.registry, _ = buildExecutionToolRegistry(nil, nil)
	noTools.options.toolGate = allowAllToolGate{}
	if outcome := c.executeRegularTurn(noTools, []ToolUse{call}); outcome.fatal == nil || !outcome.results[call.ID].IsError {
		t.Fatalf("missing registry outcome = %#v", outcome)
	}

	state := newState(nil)
	if err := state.commitToolResults([]ToolExecutionResult{{ToolUseID: "call", ToolName: "tool", Content: "ok"}}); err != nil {
		t.Fatalf("commit results: %v", err)
	}
	state.options.eventSink = EventSinkFunc(func(context.Context, Event) error { return errors.New("commit event failed") })
	if err := state.commitToolResults([]ToolExecutionResult{{ToolUseID: "call-2", ToolName: "tool", Content: "ok"}}); err == nil {
		t.Fatal("expected tool-result event error")
	}
}

func TestPhase1LifecycleFailureAndSnapshotBranches(t *testing.T) {
	c := &agentCore{}
	newState := func(sink EventSink) *loopState {
		return &loopState{ctx: context.Background(), options: &runOptions{eventSink: sink}, messages: []Message{NewTextMessage(RoleUser, "prompt")}}
	}
	completed, _, err := completedExecution(c, newState(nil), "done")
	if err != nil || completed.Status != ExecutionCompleted {
		t.Fatalf("completed execution = %#v, %v", completed, err)
	}
	if execution, _, err := completedExecution(c, newState(EventSinkFunc(func(_ context.Context, event Event) error {
		if event.Type() == EventTypeRunCompleted {
			return errors.New("completed sink")
		}
		return nil
	})), "done"); err == nil || execution.Status != ExecutionFailed {
		t.Fatalf("completed sink failure = %#v, %v", execution, err)
	}
	if execution, _, err := completedExecution(c, newState(EventSinkFunc(func(_ context.Context, event Event) error {
		if event.Type() == EventTypeRunEnded {
			return errors.New("ended sink")
		}
		return nil
	})), "done"); err == nil || execution.Status != ExecutionFailed {
		t.Fatalf("ended sink failure = %#v, %v", execution, err)
	}
	if execution, _, err := stoppedExecution[string](c, newState(EventSinkFunc(func(context.Context, Event) error { return errors.New("stop sink") }))); err == nil || execution.Status != ExecutionFailed {
		t.Fatalf("stopped sink failure = %#v, %v", execution, err)
	}

	suspension := Suspension{ID: "s", Payload: []byte("payload")}
	execution := &Execution[string]{Status: ExecutionSuspended, Result: &Result[string]{Messages: []Message{NewTextMessage(RoleUser, "prompt")}, ToolCalls: []ToolUse{{ID: "call"}}, ToolResults: []ToolExecutionResult{{ToolUseID: "call"}}}, Suspension: &suspension}
	snapshot := executionSnapshot(execution)
	snapshot.Messages[0].Content[0].Text = "changed"
	snapshot.Suspension.Payload[0] = 'X'
	if execution.Result.Messages[0].GetTextContent() != "prompt" || string(execution.Suspension.Payload) != "payload" {
		t.Fatalf("execution snapshot mutated execution: %#v", execution)
	}
	if zero := executionSnapshot[string](nil); zero.Status != 0 || len(zero.Messages) != 0 {
		t.Fatalf("nil execution snapshot = %#v", zero)
	}
}
