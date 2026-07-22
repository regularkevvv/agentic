package agentic

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/testutil"
	providertest "github.com/regularkevvv/agentic/provider/test"
)

type phase1RunnerOnly struct{}

func (phase1RunnerOnly) Run(context.Context, string, ...RunOption) (*Result[string], error) {
	return nil, nil
}

type phase1ToolInput struct {
	Value string `json:"value" validate:"required"`
}

type phase1Output struct {
	Value string `json:"value" validate:"required"`
}

type phase1Deps struct {
	Prefix string
}

func TestDriverRequiresExplicitCapabilityAndValidatesInputs(t *testing.T) {
	if _, err := RequireDriver[string](phase1RunnerOnly{}); !errors.Is(err, ErrDriverRequired) {
		t.Fatalf("RequireDriver runner-only error = %v, want ErrDriverRequired", err)
	}
	var nilAgent *Agent
	if _, err := RequireDriver[string](nilAgent); !errors.Is(err, ErrDriverRequired) {
		t.Fatalf("RequireDriver nil agent error = %v, want ErrDriverRequired", err)
	}

	model := providertest.NewTestModel(
		providertest.ModelResponse{Text: "started"},
		providertest.ModelResponse{Text: "continued"},
	)
	agent := NewAgent("system", model)
	driver, err := RequireDriver[string](agent)
	if err != nil {
		t.Fatalf("RequireDriver: %v", err)
	}

	startPrompt := NewTextMessage(RoleUser, "start")
	started, err := driver.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &startPrompt})
	if err != nil || started.Status != ExecutionCompleted || started.Result.Output != "started" {
		t.Fatalf("DriveStart = %#v, %v", started, err)
	}

	history := []Message{
		NewTextMessage(RoleUser, "earlier"),
		NewTextMessage(RoleAssistant, "earlier answer"),
		NewTextMessage(RoleUser, "continue without a synthetic prompt"),
	}
	continued, err := driver.Drive(context.Background(), DriveInput{Mode: DriveContinue, History: history})
	if err != nil || continued.Status != ExecutionCompleted || continued.Result.Output != "continued" {
		t.Fatalf("DriveContinue = %#v, %v", continued, err)
	}
	requests := model.Calls()
	if len(requests) != 2 {
		t.Fatalf("model requests = %d, want 2", len(requests))
	}
	lastRequest := requests[1]
	if len(lastRequest.Messages) != len(history)+1 { // system + supplied history
		t.Fatalf("continue request messages = %#v", lastRequest.Messages)
	}
	if got := lastRequest.Messages[len(lastRequest.Messages)-1].GetTextContent(); got != "continue without a synthetic prompt" {
		t.Fatalf("continue request ended with %q, want supplied history frontier", got)
	}

	badPrompt := NewTextMessage(RoleAssistant, "not user")
	if _, err := driver.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &badPrompt}); !errors.Is(err, ErrDriveInput) {
		t.Fatalf("bad DriveStart error = %v, want ErrDriveInput", err)
	}
	if _, err := driver.Drive(context.Background(), DriveInput{Mode: DriveContinue, History: history, Prompt: &startPrompt}); !errors.Is(err, ErrDriveInput) {
		t.Fatalf("DriveContinue prompt error = %v, want ErrDriveInput", err)
	}
	if _, err := driver.Drive(context.Background(), DriveInput{Mode: DriveContinue, History: []Message{NewTextMessage(RoleAssistant, "bad frontier")}}); !errors.Is(err, ErrDriveInput) {
		t.Fatalf("DriveContinue frontier error = %v, want ErrDriveInput", err)
	}
}

func TestDriverSuspendsResumesAndNeverAddsImplicitPrompt(t *testing.T) {
	tool, handler := MustToolPlain("echo", "echo", func(input phase1ToolInput) (string, error) {
		return "handled:" + input.Value, nil
	})
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "echo-1", Name: "echo", Input: map[string]any{"value": "one"}}}},
		providertest.ModelResponse{Text: "after resume"},
	)
	agent := NewAgent("system", model).AddTool(tool, handler)
	driver, err := RequireDriver[string](agent)
	if err != nil {
		t.Fatalf("RequireDriver: %v", err)
	}
	prompt := NewTextMessage(RoleUser, "run")
	suspended, err := driver.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt}, WithRunToolGate(ToolGateFunc(func(_ context.Context, calls []ToolUse) (ToolBatchDecision, error) {
		if len(calls) != 1 || calls[0].ID != "echo-1" {
			t.Fatalf("gate calls = %#v", calls)
		}
		return ToolBatchDecision{
			Calls:    []ToolDisposition{{Kind: ToolDispositionSuspend}},
			Deferral: &ToolDeferral{Kind: "approval", Payload: []byte(`{"request":"echo"}`)},
		}, nil
	})))
	if err != nil || suspended.Status != ExecutionSuspended || suspended.Suspension == nil {
		t.Fatalf("suspended execution = %#v, %v", suspended, err)
	}
	if len(suspended.Result.ToolResults) != 0 {
		t.Fatalf("suspension committed tool results: %#v", suspended.Result.ToolResults)
	}
	if got := suspended.Result.Messages[len(suspended.Result.Messages)-1].Role; got != RoleAssistant {
		t.Fatalf("suspension frontier role = %q, want assistant", got)
	}

	if _, err := driver.Resume(context.Background(), ResumeInput{
		History:    suspended.Result.Messages,
		Suspension: *suspended.Suspension,
	}); !errors.Is(err, ErrResumeDecision) {
		t.Fatalf("missing resume decisions error = %v, want ErrResumeDecision", err)
	}

	completed, err := driver.Resume(context.Background(), ResumeInput{
		History:    suspended.Result.Messages,
		Suspension: *suspended.Suspension,
		Decisions: []ToolResumeDecision{{
			CallID: "echo-1",
			Action: ToolResumeExecute,
		}},
	})
	if err != nil || completed.Status != ExecutionCompleted || completed.Result.Output != "after resume" {
		t.Fatalf("Resume = %#v, %v", completed, err)
	}
	if len(completed.Result.ToolCalls) != 1 || len(completed.Result.ToolResults) != 1 {
		t.Fatalf("resume calls/results = %d/%d", len(completed.Result.ToolCalls), len(completed.Result.ToolResults))
	}
	requests := model.Calls()
	if len(requests) != 2 {
		t.Fatalf("model requests = %d, want 2", len(requests))
	}
	for _, message := range requests[1].Messages {
		if message.Role == RoleUser && message.GetTextContent() == "" {
			t.Fatalf("resume inserted an empty user prompt: %#v", requests[1].Messages)
		}
	}
	last := requests[1].Messages[len(requests[1].Messages)-1]
	if last.Role != RoleTool || last.GetToolResults()[0].Content != "handled:one" {
		t.Fatalf("resume request frontier = %#v, want paired tool result", last)
	}
}

func TestDriverResumeRejectsStaleAndInvalidDecisionsBeforeEffects(t *testing.T) {
	var calls atomic.Int32
	tool, handler := MustToolPlain("required", "required", func(input phase1ToolInput) (string, error) {
		calls.Add(1)
		return input.Value, nil
	})
	model := providertest.NewTestModel(providertest.ModelResponse{ToolCalls: []ToolUse{{
		ID: "required-1", Name: "required", Input: map[string]any{"value": "original"},
	}}})
	agent := NewAgent("system", model).AddTool(tool, handler)
	prompt := NewTextMessage(RoleUser, "run")
	suspended, err := agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt}, WithRunToolGate(ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) {
		return ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionSuspend}}, Deferral: &ToolDeferral{Kind: "approval"}}, nil
	})))
	if err != nil || suspended.Status != ExecutionSuspended {
		t.Fatalf("suspend = %#v, %v", suspended, err)
	}

	stale := *suspended.Suspension
	stale.FrontierHash = "v1:wrong"
	if _, err := agent.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: stale,
		Decisions: []ToolResumeDecision{{CallID: "required-1", Action: ToolResumeExecute}},
	}); !errors.Is(err, ErrSuspensionMismatch) {
		t.Fatalf("stale resume error = %v, want ErrSuspensionMismatch", err)
	}

	tamperedID := *suspended.Suspension
	tamperedID.ID = "another-suspension"
	if _, err := agent.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: tamperedID,
		Decisions: []ToolResumeDecision{{CallID: "required-1", Action: ToolResumeExecute}},
	}); !errors.Is(err, ErrSuspensionMismatch) {
		t.Fatalf("tampered suspension ID error = %v, want ErrSuspensionMismatch", err)
	}

	if _, err := agent.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: *suspended.Suspension,
		Decisions: []ToolResumeDecision{{CallID: "required-1", Action: ToolResumeExecute, Input: map[string]any{"value": 42}}},
	}); !errors.Is(err, ErrResumeDecision) {
		t.Fatalf("invalid override error = %v, want ErrResumeDecision", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("handler ran before a valid resume decision: %d", calls.Load())
	}

	if _, err := agent.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: *suspended.Suspension,
		Decisions: []ToolResumeDecision{{CallID: "required-1", Action: ToolResumeReturn, Result: &ToolExecutionResult{
			ToolUseID: "required-1", ToolName: "required", Content: "operator result",
		}}}}, WithRunMaxIterations(9)); !errors.Is(err, ErrSuspensionMismatch) {
		t.Fatalf("configuration mismatch error = %v, want ErrSuspensionMismatch", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("handler ran across rejected resumptions: %d", calls.Load())
	}
}

func TestDriverResumeRejectsMalformedStateBeforeModelOrToolEffects(t *testing.T) {
	model := providertest.NewTestModel()
	agent := NewAgent("", model)
	call := ToolUse{ID: "open-1", Name: "pending", Input: map[string]any{"value": "x"}}
	openHistory := []Message{NewToolUseMessage(call)}
	assistantPrompt := NewTextMessage(RoleAssistant, "not a user follow-up")
	if execution, err := agent.Resume(context.Background(), ResumeInput{History: openHistory, Prompt: &assistantPrompt}); execution != nil || !errors.Is(err, ErrDriveInput) {
		t.Fatalf("non-user resume prompt = %#v, %v", execution, err)
	}
	if execution, err := agent.Resume(context.Background(), ResumeInput{History: []Message{NewTextMessage(RoleUser, "closed")}}); execution == nil || execution.Status != ExecutionFailed || !errors.Is(err, ErrSuspensionMismatch) {
		t.Fatalf("closed resume history = %#v, %v", execution, err)
	}
	if execution, err := agent.Resume(context.Background(), ResumeInput{History: openHistory}); execution == nil || execution.Status != ExecutionFailed || !errors.Is(err, ErrSuspensionMismatch) {
		t.Fatalf("missing suspension identity = %#v, %v", execution, err)
	}
	badPayload := Suspension{
		ID:           "suspension-1",
		FrontierHash: frontierHash(openHistory, []ToolUse{call}),
		Payload:      []byte("not json"),
	}
	if execution, err := agent.Resume(context.Background(), ResumeInput{History: openHistory, Suspension: badPayload}); execution == nil || execution.Status != ExecutionFailed || !errors.Is(err, ErrSuspensionVersion) {
		t.Fatalf("malformed suspension payload = %#v, %v", execution, err)
	}
	if calls := model.Calls(); len(calls) != 0 {
		t.Fatalf("malformed resume performed model work: %#v", calls)
	}
}

func TestDriverResumeCanReturnExternalResult(t *testing.T) {
	var calls atomic.Int32
	tool, handler := MustToolPlain("external", "external", func(input phase1ToolInput) (string, error) {
		calls.Add(1)
		return input.Value, nil
	})
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "external-1", Name: "external", Input: map[string]any{"value": "model"}}}},
		providertest.ModelResponse{Text: "external complete"},
	)
	agent := NewAgent("system", model).AddTool(tool, handler)
	prompt := NewTextMessage(RoleUser, "run")
	suspended, err := agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt}, WithRunToolGate(ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) {
		return ToolBatchDecision{Calls: []ToolDisposition{{Kind: ToolDispositionSuspend}}, Deferral: &ToolDeferral{Kind: "operator"}}, nil
	})))
	if err != nil {
		t.Fatalf("Drive: %v", err)
	}
	completed, err := agent.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: *suspended.Suspension,
		Decisions: []ToolResumeDecision{{CallID: "external-1", Action: ToolResumeReturn, Result: &ToolExecutionResult{
			ToolUseID: "external-1", ToolName: "external", Content: "fulfilled elsewhere",
		}}},
	})
	if err != nil || completed.Result.Output != "external complete" {
		t.Fatalf("external resume = %#v, %v", completed, err)
	}
	if calls.Load() != 0 {
		t.Fatalf("external return ran handler %d times", calls.Load())
	}
	if got := completed.Result.ToolResults[0].Content; got != "fulfilled elsewhere" {
		t.Fatalf("external result = %#v", completed.Result.ToolResults[0])
	}
}

func TestTurnHookRunsAfterValidationAndControlsContinuation(t *testing.T) {
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "out-1", Name: "__output__", Input: map[string]any{"value": "first"}}}},
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "out-2", Name: "__output__", Input: map[string]any{"value": "final"}}}},
	)
	agent := NewTypedAgent[phase1Output]("system", model, "result")
	var turns []Turn
	var events []EventType
	result, err := agent.Run(context.Background(), "run",
		WithRunTurnHook(func(_ context.Context, turn Turn) (TurnDecision, error) {
			turns = append(turns, turn)
			if len(turns) == 1 {
				if !turn.Candidate.Valid() || turn.Candidate.Source != CompletionOutputTool {
					t.Fatalf("first hook candidate = %#v", turn.Candidate)
				}
				return TurnDecision{Action: TurnContinue, Inject: []Message{NewTextMessage(RoleUser, "keep going")}}, nil
			}
			return TurnDecision{}, nil
		}),
		WithRunEventSink(EventSinkFunc(func(_ context.Context, event Event) error {
			events = append(events, event.Type())
			return nil
		})),
	)
	if err != nil || result.Output.Value != "final" {
		t.Fatalf("hook-controlled run = %#v, %v", result, err)
	}
	if len(turns) != 2 {
		t.Fatalf("turn hooks = %d, want 2", len(turns))
	}
	requests := model.Calls()
	if len(requests) != 2 {
		t.Fatalf("model requests = %d", len(requests))
	}
	if got := requests[1].Messages[len(requests[1].Messages)-1].GetTextContent(); got != "keep going" {
		t.Fatalf("continuation injection = %#v", requests[1].Messages)
	}
	if !containsEventType(events, EventTypeOutputValidated) || !containsEventType(events, EventTypeTurnMessagesInjected) || !containsEventType(events, EventTypeRunCompleted) {
		t.Fatalf("canonical events missing expected lifecycle: %#v", events)
	}
}

func TestInvalidTypedOutputRetriesBeforeHookSeesCandidate(t *testing.T) {
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "bad", Name: "__output__", Input: map[string]any{"value": ""}}}},
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "good", Name: "__output__", Input: map[string]any{"value": "good"}}}},
	)
	agent := NewTypedAgent[phase1Output]("system", model, "result", WithMaxValidationRetries(1))
	var candidates []CompletionCandidate
	result, err := agent.Run(context.Background(), "run", WithRunTurnHook(func(_ context.Context, turn Turn) (TurnDecision, error) {
		candidates = append(candidates, turn.Candidate)
		return TurnDecision{}, nil
	}))
	if err != nil || result.Output.Value != "good" {
		t.Fatalf("retry result = %#v, %v", result, err)
	}
	if len(candidates) != 2 || candidates[0].Valid() || !candidates[1].Valid() {
		t.Fatalf("hook candidates = %#v", candidates)
	}
	requests := model.Calls()
	if len(requests) != 2 || requests[1].Messages[len(requests[1].Messages)-1].Role != RoleTool {
		t.Fatalf("validation retry did not preserve paired output frontier: %#v", requests)
	}
}

func TestPerRunOverlayProcessorAndToolCallContext(t *testing.T) {
	var seen ToolCallContext
	overlayTool, overlayHandler := MustToolWithContext("overlay", "overlay", func(ctx context.Context, input phase1ToolInput) (string, error) {
		var ok bool
		seen, ok = CurrentToolCall(ctx)
		if !ok {
			return "", errors.New("missing ToolCallContext")
		}
		return "raw:" + input.Value, nil
	})
	set := NewToolset().Add(overlayTool, overlayHandler)
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "overlay-1", Name: "overlay", Input: map[string]any{"value": "x"}}}},
		providertest.ModelResponse{Text: "done"},
	)
	agent := NewAgent("system", model)
	result, err := agent.Run(context.Background(), "run",
		WithRunToolsets(set),
		WithRunToolResultProcessor(ToolResultProcessorFunc(func(_ context.Context, call ToolUse, result ToolExecutionResult) (ToolExecutionResult, error) {
			if call.ID != "overlay-1" {
				t.Fatalf("processor call = %#v", call)
			}
			result.Content = "projected"
			return result, nil
		})),
	)
	if err != nil || result.Output != "done" {
		t.Fatalf("overlay run = %#v, %v", result, err)
	}
	if seen.ID != "overlay-1" || seen.Name != "overlay" || seen.Attempt != 1 {
		t.Fatalf("ToolCallContext = %#v", seen)
	}
	if result.ToolResults[0].Content != "projected" {
		t.Fatalf("projected result = %#v", result.ToolResults[0])
	}
	if agent.core.registry != nil {
		t.Fatalf("per-run overlay mutated the shared registry")
	}
	requests := model.Calls()
	if len(requests[0].Tools) != 1 || requests[0].Tools[0].Function.Name != "overlay" {
		t.Fatalf("overlay was absent from request: %#v", requests[0].Tools)
	}
	last := requests[1].Messages[len(requests[1].Messages)-1]
	if last.GetToolResults()[0].Content != "projected" {
		t.Fatalf("model did not receive projected result: %#v", requests[1].Messages)
	}
}

func TestToolGateReturnContinuesBeforeTypedOutputIsAccepted(t *testing.T) {
	var calls atomic.Int32
	tool, handler := MustToolPlain("regular", "regular", func(input phase1ToolInput) (string, error) {
		calls.Add(1)
		return input.Value, nil
	})
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{
			{ID: "regular-1", Name: "regular", Input: map[string]any{"value": "x"}},
			{ID: "out-1", Name: "__output__", Input: map[string]any{"value": "too early"}},
		}},
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "out-2", Name: "__output__", Input: map[string]any{"value": "final"}}}},
	)
	agent := NewTypedAgent[phase1Output]("system", model, "result").AddTool(tool, handler)
	result, err := agent.Run(context.Background(), "run", WithRunToolGate(ToolGateFunc(func(_ context.Context, calls []ToolUse) (ToolBatchDecision, error) {
		return ToolBatchDecision{Calls: []ToolDisposition{{
			Kind:     ToolDispositionReturn,
			Continue: true,
			Result:   &ToolExecutionResult{ToolUseID: calls[0].ID, ToolName: calls[0].Name, Content: "deferred"},
		}}}, nil
	})))
	if err != nil || result.Output.Value != "final" {
		t.Fatalf("gate continuation result = %#v, %v", result, err)
	}
	if calls.Load() != 0 {
		t.Fatalf("gate return executed handler %d times", calls.Load())
	}
	if len(result.ToolResults) != 3 || !result.ToolResults[1].IsError {
		t.Fatalf("expected returned regular result and discarded first output, got %#v", result.ToolResults)
	}
}

func TestToolResultProcessorFailurePairsResultAndSkipsTurnHook(t *testing.T) {
	tool, handler := MustToolPlain("effect", "effect", func(input phase1ToolInput) (string, error) {
		return input.Value, nil
	})
	model := providertest.NewTestModel(providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "effect-1", Name: "effect", Input: map[string]any{"value": "x"}}}})
	agent := NewAgent("system", model).AddTool(tool, handler)
	hookCalled := false
	prompt := NewTextMessage(RoleUser, "run")
	execution, err := agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt},
		WithRunToolResultProcessor(ToolResultProcessorFunc(func(context.Context, ToolUse, ToolExecutionResult) (ToolExecutionResult, error) {
			return ToolExecutionResult{}, errors.New("projection failed")
		})),
		WithRunTurnHook(func(context.Context, Turn) (TurnDecision, error) {
			hookCalled = true
			return TurnDecision{}, nil
		}),
	)
	if err == nil || execution.Status != ExecutionFailed {
		t.Fatalf("processor failure execution = %#v, %v", execution, err)
	}
	if hookCalled {
		t.Fatal("turn hook ran after processor failure")
	}
	if len(execution.Result.ToolResults) != 1 || !execution.Result.ToolResults[0].IsError {
		t.Fatalf("processor failure left an unpaired frontier: %#v", execution.Result)
	}
}

func TestTruncatedTurnNeverExecutesPartialToolCalls(t *testing.T) {
	var calls atomic.Int32
	tool, handler := MustToolPlain("partial", "partial", func(input phase1ToolInput) (string, error) {
		calls.Add(1)
		return input.Value, nil
	})
	response := &ChatResponse{
		Message:         NewToolUseMessage(ToolUse{ID: "partial-1", Name: "partial", Input: map[string]any{"value": "x"}}),
		FinishReason:    FinishReasonLength,
		RawFinishReason: "length",
	}
	blocking := NewAgent("system", &testutil.StubModel{NameValue: "truncated", Response: response}).AddTool(tool, handler)
	prompt := NewTextMessage(RoleUser, "run")
	execution, err := blocking.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt})
	if execution == nil || execution.Status != ExecutionFailed || !IsProviderError(err) {
		t.Fatalf("truncated blocking execution = %#v, %v", execution, err)
	}
	if execution.Result.FinishReason != FinishReasonLength || len(execution.Result.ToolCalls) != 0 || calls.Load() != 0 {
		t.Fatalf("truncated blocking state = %#v, handler calls %d", execution.Result, calls.Load())
	}

	streaming := NewAgent("system", &testutil.ScriptedStreamModel{Streams: [][]StreamEvent{{
		{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "partial-2", Name: "partial"}},
		{Type: StreamEventToolCallDelta, ToolCallID: "partial-2", Delta: `{"value":"x"}`},
		{Type: StreamEventDone, FinishReason: FinishReasonLength},
	}}}).AddTool(tool, handler)
	stream, err := streaming.RunStream(context.Background(), "run")
	if err != nil || !IsProviderError(stream.Wait()) {
		t.Fatalf("truncated stream = %#v, %v", stream, err)
	}
	snapshot, ok := stream.Snapshot()
	if !ok || snapshot.Status != ExecutionFailed || len(snapshot.ToolCalls) != 0 || calls.Load() != 0 {
		t.Fatalf("truncated stream snapshot = %#v, handler calls %d", snapshot, calls.Load())
	}
}

func TestCancellationGraceDetachesNonCooperativeTool(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	tool, handler := MustToolWithContext("block", "block", func(context.Context, phase1ToolInput) (string, error) {
		close(started)
		<-release
		return "late", nil
	})
	model := providertest.NewTestModel(providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "block-1", Name: "block", Input: map[string]any{"value": "x"}}}})
	agent := NewAgent("system", model).AddTool(tool, handler)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	prompt := NewTextMessage(RoleUser, "run")
	resultCh := make(chan struct {
		execution *Execution[string]
		err       error
	}, 1)
	go func() {
		execution, err := agent.Drive(ctx, DriveInput{Mode: DriveStart, Prompt: &prompt}, WithRunToolCancellationGrace(0))
		resultCh <- struct {
			execution *Execution[string]
			err       error
		}{execution, err}
	}()
	select {
	case <-started:
		cancel()
	case <-time.After(time.Second):
		t.Fatal("tool handler did not start")
	}
	select {
	case result := <-resultCh:
		if result.execution.Status != ExecutionInterrupted || !errors.Is(result.err, context.Canceled) {
			t.Fatalf("cancelled execution = %#v, %v", result.execution, result.err)
		}
		if len(result.execution.Result.ToolResults) != 1 || !result.execution.Result.ToolResults[0].IsError {
			t.Fatalf("cancelled tool result = %#v", result.execution.Result.ToolResults)
		}
	case <-time.After(time.Second):
		t.Fatal("execution did not respect cancellation grace")
	}
}

func TestTypedDriverResumeInjectsPromptAfterResolvedToolResults(t *testing.T) {
	tool, handler := MustToolPlain("typed-work", "typed work", func(input phase1ToolInput) (string, error) {
		return "handled:" + input.Value, nil
	})
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "typed-work-1", Name: "typed-work", Input: map[string]any{"value": "one"}}}},
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "typed-output-1", Name: "__output__", Input: map[string]any{"value": "done"}}}},
	)
	agent := NewTypedAgent[phase1Output]("system", model, "result").AddTool(tool, handler)
	prompt := NewTextMessage(RoleUser, "start")
	suspended, err := agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt}, WithRunToolGate(ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) {
		return ToolBatchDecision{
			Calls:    []ToolDisposition{{Kind: ToolDispositionSuspend}},
			Deferral: &ToolDeferral{Kind: "approval"},
		}, nil
	})))
	if err != nil || suspended.Status != ExecutionSuspended {
		t.Fatalf("typed suspension = %#v, %v", suspended, err)
	}

	followUp := NewTextMessage(RoleUser, "operator approved it")
	completed, err := agent.Resume(context.Background(), ResumeInput{
		History:    suspended.Result.Messages,
		Suspension: *suspended.Suspension,
		Decisions:  []ToolResumeDecision{{CallID: "typed-work-1", Action: ToolResumeExecute}},
		Prompt:     &followUp,
	})
	if err != nil || completed.Status != ExecutionCompleted || completed.Result.Output.Value != "done" {
		t.Fatalf("typed resume = %#v, %v", completed, err)
	}
	requests := model.Calls()
	if len(requests) != 2 {
		t.Fatalf("typed model requests = %d, want 2", len(requests))
	}
	last := requests[1].Messages[len(requests[1].Messages)-1]
	if last.Role != RoleUser || last.GetTextContent() != "operator approved it" {
		t.Fatalf("resume prompt was not appended after tool result: %#v", requests[1].Messages)
	}
}

func TestDependencyAndBoundDriversResolveDepsOnResume(t *testing.T) {
	tool, handler := MustToolWithDeps[phase1ToolInput, string, phase1Deps]("dep-work", "dependency work", func(ctx RunContext[phase1Deps], input phase1ToolInput) (string, error) {
		return ctx.Deps.Prefix + input.Value, nil
	})
	newModel := func() *providertest.TestModel {
		return providertest.NewTestModel(
			providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "dep-work-1", Name: "dep-work", Input: map[string]any{"value": "one"}}}},
			providertest.ModelResponse{Text: "finished"},
		)
	}
	suspend := WithRunToolGate(ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) {
		return ToolBatchDecision{
			Calls:    []ToolDisposition{{Kind: ToolDispositionSuspend}},
			Deferral: &ToolDeferral{Kind: "approval"},
		}, nil
	}))

	direct := NewAgentWithDeps[phase1Deps]("system", newModel()).AddTool(tool, handler)
	prompt := NewTextMessage(RoleUser, "run")
	suspended, err := direct.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt}, phase1Deps{Prefix: "direct:"}, suspend)
	if err != nil || suspended.Status != ExecutionSuspended {
		t.Fatalf("dependency suspension = %#v, %v", suspended, err)
	}
	completed, err := direct.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: *suspended.Suspension,
		Decisions: []ToolResumeDecision{{CallID: "dep-work-1", Action: ToolResumeExecute}},
	}, phase1Deps{Prefix: "direct:"})
	if err != nil || completed.Result.Output != "finished" || completed.Result.ToolResults[0].Content != "direct:one" {
		t.Fatalf("dependency resume = %#v, %v", completed, err)
	}

	providerCalls := 0
	bound := NewAgentWithDeps[phase1Deps]("system", newModel()).AddTool(tool, handler).BindProvider(func(context.Context) (phase1Deps, error) {
		providerCalls++
		return phase1Deps{Prefix: "bound:"}, nil
	})
	driver, err := RequireDriver[string](bound)
	if err != nil {
		t.Fatalf("RequireDriver bound runner: %v", err)
	}
	suspended, err = driver.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt}, suspend)
	if err != nil || suspended.Status != ExecutionSuspended {
		t.Fatalf("bound suspension = %#v, %v", suspended, err)
	}
	completed, err = driver.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: *suspended.Suspension,
		Decisions: []ToolResumeDecision{{CallID: "dep-work-1", Action: ToolResumeExecute}},
	})
	if err != nil || completed.Result.ToolResults[0].Content != "bound:one" {
		t.Fatalf("bound resume = %#v, %v", completed, err)
	}
	if providerCalls != 2 {
		t.Fatalf("dependency provider calls = %d, want one per Drive and Resume", providerCalls)
	}

	typedModel := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "typed-dep-1", Name: "dep-work", Input: map[string]any{"value": "two"}}}},
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "typed-dep-output", Name: "__output__", Input: map[string]any{"value": "typed done"}}}},
	)
	typed := NewTypedAgentWithDeps[phase1Output, phase1Deps]("system", typedModel, "result").AddTool(tool, handler)
	typedSuspended, err := typed.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt}, phase1Deps{Prefix: "typed:"}, suspend)
	if err != nil || typedSuspended.Status != ExecutionSuspended {
		t.Fatalf("typed dependency suspension = %#v, %v", typedSuspended, err)
	}
	completedTyped, err := typed.Resume(context.Background(), ResumeInput{
		History: typedSuspended.Result.Messages, Suspension: *typedSuspended.Suspension,
		Decisions: []ToolResumeDecision{{CallID: "typed-dep-1", Action: ToolResumeExecute}},
	}, phase1Deps{Prefix: "typed:"})
	if err != nil || completedTyped.Result.Output.Value != "typed done" || completedTyped.Result.ToolResults[0].Content != "typed:two" {
		t.Fatalf("typed dependency resume = %#v, %v", completedTyped, err)
	}
}

func TestStreamSnapshotCarriesExecutionAndProviderStreamDoesNot(t *testing.T) {
	model := &testutil.ScriptedStreamModel{Streams: [][]StreamEvent{{
		{Type: StreamEventTextDelta, Delta: "streamed"},
		{Type: StreamEventDone, Usage: &Usage{TotalTokens: 4}, FinishReason: FinishReasonStop},
	}}}
	agent := NewAgent("system", model)
	stream, err := agent.RunStream(context.Background(), "run")
	if err != nil {
		t.Fatalf("RunStream: %v", err)
	}
	if err := stream.Wait(); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	snapshot, ok := stream.Snapshot()
	if !ok || snapshot.Status != ExecutionCompleted || snapshot.Usage.TotalTokens != 4 || len(snapshot.Messages) == 0 {
		t.Fatalf("stream snapshot = %#v, %v", snapshot, ok)
	}
	providerStream := testutil.NewScriptedStream(StreamEvent{Type: StreamEventDone})
	if _, ok := providerStream.Snapshot(); ok {
		t.Fatal("provider stream unexpectedly owns an execution snapshot")
	}
}

func TestRunStreamProjectsSuspendedAndStoppedTerminalStatuses(t *testing.T) {
	tool, handler := MustToolPlain("stream-tool", "stream tool", func(input phase1ToolInput) (string, error) {
		return input.Value, nil
	})
	newAgent := func() *Agent {
		return NewAgent("system", providertest.NewTestModel(providertest.ModelResponse{ToolCalls: []ToolUse{{
			ID: "stream-tool-1", Name: "stream-tool", Input: map[string]any{"value": "x"},
		}}})).AddTool(tool, handler)
	}

	suspended, err := newAgent().RunStream(context.Background(), "run", WithRunToolGate(ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) {
		return ToolBatchDecision{
			Calls:    []ToolDisposition{{Kind: ToolDispositionSuspend}},
			Deferral: &ToolDeferral{Kind: "approval"},
		}, nil
	})))
	if err != nil || !errors.Is(suspended.Wait(), ErrExecutionSuspended) {
		t.Fatalf("suspended stream = %#v, %v", suspended, err)
	}
	if snapshot, ok := suspended.Snapshot(); !ok || snapshot.Status != ExecutionSuspended {
		t.Fatalf("suspended snapshot = %#v, %v", snapshot, ok)
	}

	stopped, err := newAgent().RunStream(context.Background(), "run", WithRunTurnHook(func(context.Context, Turn) (TurnDecision, error) {
		return TurnDecision{Action: TurnStop}, nil
	}))
	if err != nil || !errors.Is(stopped.Wait(), ErrExecutionStopped) {
		t.Fatalf("stopped stream = %#v, %v", stopped, err)
	}
	if snapshot, ok := stopped.Snapshot(); !ok || snapshot.Status != ExecutionStopped {
		t.Fatalf("stopped snapshot = %#v, %v", snapshot, ok)
	}
}

func containsEventType(events []EventType, want EventType) bool {
	for _, event := range events {
		if event == want {
			return true
		}
	}
	return false
}
