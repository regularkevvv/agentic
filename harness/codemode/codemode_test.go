package codemode

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/artifact"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/capability"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type fakeExecutor struct {
	mu      sync.Mutex
	start   Step
	resume  Step
	request Request
	results []CallResult
}

func (e *fakeExecutor) Start(_ context.Context, request Request) (Step, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.request = request
	return e.start, nil
}

func (e *fakeExecutor) Resume(_ context.Context, _ Checkpoint, results []CallResult) (Step, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.results = append([]CallResult(nil), results...)
	step := e.resume
	if step.Done && step.Output == nil && len(results) > 0 {
		step.Output = results[0].Content
	}
	return step, nil
}

type fakeCapture struct{ gate agentic.ToolGate }

func (c fakeCapture) History() []agentic.Message  { return nil }
func (c fakeCapture) Toolsets() []agentic.Toolset { return nil }
func (c fakeCapture) ToolGate() agentic.ToolGate  { return c.gate }
func (c fakeCapture) DelegationTools() []string   { return nil }
func (c fakeCapture) Scope() harnessruntime.Scope { return harnessruntime.Scope{} }
func (c fakeCapture) AcquireBudget(context.Context) (harnessruntime.BudgetLease, error) {
	return nil, nil
}
func (c fakeCapture) ProjectEvent(context.Context, event.Record) error { return nil }

type fakeOperations struct {
	mu          sync.Mutex
	records     []harnessruntime.Operation
	projections []agentic.ToolUse
}

func (o *fakeOperations) RecordOperation(_ context.Context, value harnessruntime.Operation) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	value.Payload = append([]byte(nil), value.Payload...)
	o.records = append(o.records, value)
	return nil
}

func (o *fakeOperations) ProjectToolResult(_ context.Context, call agentic.ToolUse, result agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error) {
	o.mu.Lock()
	o.projections = append(o.projections, call)
	o.mu.Unlock()
	return result, nil
}

func selectedCapability(t *testing.T, calls *atomic.Int32) capability.Capability {
	t.Helper()
	tool, handler, err := agentic.ToolWithContext(
		"selected_tool",
		"Selected test tool.",
		func(ctx context.Context, input struct {
			Value string `json:"value"`
		}) (map[string]any, error) {
			calls.Add(1)
			call, ok := agentic.CurrentToolCall(ctx)
			if !ok || call.ID != nestedCallID("outer-1", 0, "nested-1") || call.Name != "selected_tool" {
				return nil, errors.New("nested call context was not installed")
			}
			if _, ok := agentic.CurrentToolResume(ctx); ok {
				return nil, errors.New("nested call inherited outer resume context")
			}
			return map[string]any{"value": input.Value}, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return capability.Func{Name: "selected", Apply: func(registry *capability.Registry) error {
		return registry.AddToolset(agentic.NewToolset().Add(tool, handler))
	}}
}

func TestSelectedToolsAreHiddenAndExecuteThroughRuntimePorts(t *testing.T) {
	var selectedCalls atomic.Int32
	executor := &fakeExecutor{
		start:  Step{Checkpoint: []byte("checkpoint"), Calls: []Call{{ID: "nested-1", Name: "selected_tool", Input: map[string]any{"value": "ok"}}}, Stdout: "one"},
		resume: Step{Done: true, Stdout: " two"},
	}
	mode := New(Config{
		SelectedTools: []string{"selected_tool"}, Executor: executor,
		Order: capability.Ordering{After: []string{"selected"}},
	})
	plan, err := capability.Compile(selectedCapability(t, &selectedCalls), mode)
	if err != nil {
		t.Fatal(err)
	}
	tools := plan.Tools()
	if len(tools) != 1 || tools[0].Function.Name != defaultTool {
		t.Fatalf("model-visible tools = %#v", tools)
	}
	_, handlers := plan.Toolsets()[0].ToolsAndHandlers()
	operations := &fakeOperations{}
	gate := agentic.ToolGateFunc(func(_ context.Context, calls []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
		dispositions := make([]agentic.ToolDisposition, len(calls))
		for index := range dispositions {
			dispositions[index].Kind = agentic.ToolDispositionExecute
		}
		return agentic.ToolBatchDecision{Calls: dispositions}, nil
	})
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		SessionID: "session", Capture: fakeCapture{gate: gate}, Operations: operations,
	})
	ctx = agentic.WithToolCallContext(ctx, agentic.ToolCallContext{ID: "outer-1", Name: defaultTool, Attempt: 1})
	result, err := handlers[0].Execute(ctx, map[string]any{"code": "call selected"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	value, ok := result.(Result)
	if !ok || value.Stdout != "one two" || selectedCalls.Load() != 1 {
		t.Fatalf("result=%#v calls=%d", result, selectedCalls.Load())
	}
	executor.mu.Lock()
	request, results := executor.request, append([]CallResult(nil), executor.results...)
	executor.mu.Unlock()
	if len(request.Tools) != 1 || request.Tools[0].Name != "selected_tool" || len(results) != 1 || results[0].Name != "selected_tool" {
		t.Fatalf("request=%#v results=%#v", request, results)
	}
	operations.mu.Lock()
	phases := make([]string, len(operations.records))
	for index, record := range operations.records {
		phases[index] = record.Phase
	}
	projections := append([]agentic.ToolUse(nil), operations.projections...)
	operations.mu.Unlock()
	if strings.Join(phases, ",") != "planned,started,result,completed" {
		t.Fatalf("operation phases = %v", phases)
	}
	if len(projections) != 1 || projections[0].ID != nestedCallID("outer-1", 0, "nested-1") || results[0].ID != "nested-1" {
		t.Fatalf("host projections=%#v executor results=%#v", projections, results)
	}
}

func TestNestedHostIdentityNamespacingPreservesExecutorIDs(t *testing.T) {
	handler := testHandler(t, ExecutorFunc{})
	operations := &fakeOperations{}
	runtime := mustRuntime(runtimeContext(operations, allowGate()))

	for _, program := range []deferredProgram{
		{
			OuterCallID: "outer/one", Step: 0,
			Calls:        []Call{{ID: "call/shared", Name: "selected", Input: map[string]any{}}},
			Dispositions: []wireDisposition{{Kind: agentic.ToolDispositionExecute}},
		},
		{
			OuterCallID: "outer", Step: 1,
			Calls:        []Call{{ID: "call/shared", Name: "selected", Input: map[string]any{}}},
			Dispositions: []wireDisposition{{Kind: agentic.ToolDispositionExecute}},
		},
	} {
		results, err := handler.completeBatch(context.Background(), nil, runtime, program, nil)
		if err != nil || len(results) != 1 || results[0].ID != "call/shared" {
			t.Fatalf("executor result = %#v, %v", results, err)
		}
	}

	operations.mu.Lock()
	projections := append([]agentic.ToolUse(nil), operations.projections...)
	operations.mu.Unlock()
	if len(projections) != 2 || projections[0].ID == projections[1].ID ||
		projections[0].ID != nestedCallID("outer/one", 0, "call/shared") ||
		projections[1].ID != nestedCallID("outer", 1, "call/shared") {
		t.Fatalf("host projections = %#v", projections)
	}
}

func TestNestedGateReturnAndSuspensionPlanner(t *testing.T) {
	var selectedCalls atomic.Int32
	executor := &fakeExecutor{
		start:  Step{Checkpoint: []byte("checkpoint"), Calls: []Call{{ID: "nested-1", Name: "selected_tool", Input: map[string]any{"value": "ok"}}}},
		resume: Step{Done: true},
	}
	mode := New(Config{SelectedTools: []string{"selected_tool"}, Executor: executor, Order: capability.Ordering{After: []string{"selected"}}})
	plan, err := capability.Compile(selectedCapability(t, &selectedCalls), mode)
	if err != nil {
		t.Fatal(err)
	}
	_, handlers := plan.Toolsets()[0].ToolsAndHandlers()
	operations := &fakeOperations{}
	gate := agentic.ToolGateFunc(func(_ context.Context, calls []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
		return agentic.ToolBatchDecision{
			Calls: []agentic.ToolDisposition{{Kind: agentic.ToolDispositionReturn, Result: &agentic.ToolExecutionResult{
				ToolUseID: calls[0].ID, ToolName: calls[0].Name, Content: "denied", IsError: true, Error: errors.New("denied"),
			}}},
		}, nil
	})
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{SessionID: "session", Capture: fakeCapture{gate: gate}, Operations: operations})
	ctx = agentic.WithToolCallContext(ctx, agentic.ToolCallContext{ID: "outer", Name: defaultTool, Attempt: 1})
	if _, err := handlers[0].Execute(ctx, map[string]any{"code": "code"}, nil); err != nil {
		t.Fatal(err)
	}
	if selectedCalls.Load() != 0 {
		t.Fatal("gate-returned call executed")
	}

	executor.start = Step{Checkpoint: []byte("ask-checkpoint"), Calls: []Call{{ID: "nested-1", Name: "selected_tool", Input: map[string]any{"value": "ask"}}}}
	gate = agentic.ToolGateFunc(func(_ context.Context, _ []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
		return agentic.ToolBatchDecision{
			Calls:    []agentic.ToolDisposition{{Kind: agentic.ToolDispositionSuspend}},
			Deferral: &agentic.ToolDeferral{Kind: "test.ask", Payload: json.RawMessage(`{"why":"approval"}`)},
		}, nil
	})
	ctx = harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{SessionID: "session", Capture: fakeCapture{gate: gate}, Operations: operations})
	ctx = agentic.WithToolCallContext(ctx, agentic.ToolCallContext{ID: "outer", Name: defaultTool, Attempt: 1})
	value, err := handlers[0].Execute(ctx, map[string]any{"code": "code"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	suspension, ok := value.(agentic.ToolHandlerSuspension)
	if !ok || suspension.Deferral.Kind != DeferralKind {
		t.Fatalf("handler suspension = %#v", value)
	}
	rootID := "root-suspension"
	rootPayload, _ := json.Marshal(struct {
		Version           int
		SuspensionID      string
		HandlerSuspension bool
		Calls             []agentic.ToolUse
		ExecutableCallIDs []string
		Deferral          agentic.ToolDeferral
	}{1, rootID, true, []agentic.ToolUse{{ID: "outer", Name: defaultTool}}, []string{"outer"}, suspension.Deferral})
	decisions, err := plan.ResumePlanner().PlanResume(agentic.Suspension{ID: rootID, Kind: DeferralKind, Payload: rootPayload}, harnessruntime.ResumeRequest{
		SuspensionID: rootID,
		Resolutions:  []harnessruntime.ToolResolution{{CallID: "nested-1", Action: harnessruntime.ResolutionDeny, Reason: "no"}},
	})
	if err != nil || len(decisions) != 1 || decisions[0].CallID != "outer" || decisions[0].Action != agentic.ToolResumeExecute || len(decisions[0].Payload) == 0 {
		t.Fatalf("decisions = %#v, %v", decisions, err)
	}
	if _, err := plan.ResumePlanner().PlanResume(agentic.Suspension{ID: rootID, Kind: DeferralKind, Payload: rootPayload}, harnessruntime.ResumeRequest{SuspensionID: rootID}); !errors.Is(err, harnessruntime.ErrInvalidResumeRequest) {
		t.Fatalf("missing nested resolution = %v", err)
	}
}

type scriptedModel struct {
	mu    sync.Mutex
	calls int
}

func (m *scriptedModel) Name() string { return "test:codemode" }

func (m *scriptedModel) Request(_ context.Context, _ *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	m.mu.Lock()
	m.calls++
	call := m.calls
	m.mu.Unlock()
	if call == 1 {
		return &agentic.ChatResponse{
			Model: m.Name(), Message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "outer-1", Name: defaultTool, Input: map[string]any{"code": "program"}}),
			FinishReason: agentic.FinishReasonToolCalls, RawFinishReason: string(agentic.FinishReasonToolCalls),
		}, nil
	}
	return &agentic.ChatResponse{
		Model: m.Name(), Message: agentic.NewTextMessage(agentic.RoleAssistant, "done"),
		FinishReason: agentic.FinishReasonStop, RawFinishReason: string(agentic.FinishReasonStop),
	}, nil
}

func TestSessionNestedAskResumesExactlyOnceAndSpillsOnce(t *testing.T) {
	model := &scriptedModel{}
	agent := agentic.NewAgent("", model)
	var selectedCalls atomic.Int32
	large := strings.Repeat("x", 2048)
	tool, handler, err := agentic.ToolWithContext(
		"selected_tool", "selected",
		func(ctx context.Context, _ struct{}) (string, error) {
			selectedCalls.Add(1)
			if _, resumed := agentic.CurrentToolResume(ctx); resumed {
				return "", errors.New("selected tool inherited outer resume")
			}
			return large, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	selected := capability.Func{Name: "selected", Apply: func(registry *capability.Registry) error {
		return registry.AddToolset(agentic.NewToolset().Add(tool, handler))
	}}
	gate := capability.Func{Name: "ask", Order: capability.Ordering{After: []string{"selected"}}, Apply: func(registry *capability.Registry) error {
		return registry.AddToolGateMiddleware(capability.ToolGateMiddlewareFunc(func(
			_ context.Context,
			calls []agentic.ToolUse,
			current agentic.ToolBatchDecision,
		) (agentic.ToolBatchDecision, error) {
			if len(calls) == 1 && calls[0].Name == "selected_tool" {
				return agentic.ToolBatchDecision{
					Calls:    []agentic.ToolDisposition{{Kind: agentic.ToolDispositionSuspend}},
					Deferral: &agentic.ToolDeferral{Kind: "test.ask", Payload: json.RawMessage(`{"version":1}`)},
				}, nil
			}
			return current, nil
		}))
	}}
	executor := &fakeExecutor{
		start:  Step{Checkpoint: []byte("checkpoint"), Calls: []Call{{ID: "nested-1", Name: "selected_tool", Input: map[string]any{}}}},
		resume: Step{Done: true},
	}
	mode := New(Config{SelectedTools: []string{"selected_tool"}, Executor: executor, Order: capability.Ordering{After: []string{"selected", "ask"}}})
	artifacts := artifactmemory.New()
	processors, _ := spill.NewFactory(artifacts, spill.Config{Threshold: 512, Head: 16, Tail: 16})
	environments, _ := envmemory.NewFactory(envmemory.Config{Cwd: "/workspace"})
	runtime, err := harness.New(
		agent,
		harness.WithRuntime(harness.RuntimeConfig{
			Sessions: storememory.New(), Codec: jsoncodec.New(), Events: inproc.NewFactory(), Environments: environments,
			ResultProcessors: processors, Clock: system.NewClock(), IDs: system.NewIDs(), ToolCancellationGrace: time.Second,
		}),
		harness.WithCapabilities(selected, gate, mode),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	session, err := runtime.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	execution, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "run"))
	if err != nil || execution.Status != agentic.ExecutionSuspended || selectedCalls.Load() != 0 {
		t.Fatalf("suspended execution=%#v err=%v calls=%d", execution, err, selectedCalls.Load())
	}
	execution, err = session.Resume(context.Background(), harness.ResumeRequest{
		SuspensionID: execution.Suspension.ID,
		Resolutions:  []harness.ToolResolution{{CallID: "nested-1", Action: harness.ResolutionApprove}},
	})
	if err != nil || execution.Status != agentic.ExecutionCompleted || selectedCalls.Load() != 1 {
		t.Fatalf("resumed execution=%#v err=%v calls=%d", execution, err, selectedCalls.Load())
	}
	executor.mu.Lock()
	results := append([]CallResult(nil), executor.results...)
	executor.mu.Unlock()
	if len(results) != 1 {
		t.Fatalf("executor results = %#v", results)
	}
	projected, ok := results[0].Content.(string)
	if !ok || !strings.Contains(projected, "[harness artifact art_") {
		t.Fatalf("projected nested result = %#v", results[0].Content)
	}
	if artifacts.Count(session.ID()) != 1 {
		t.Fatalf("artifact count = %d", artifacts.Count(session.ID()))
	}
	start := strings.Index(projected, "art_")
	handle := artifact.Handle(projected[start : start+68])
	full, err := artifacts.Get(context.Background(), session.ID(), handle)
	if err != nil || string(full) != large {
		t.Fatalf("full artifact = %d bytes, %v", len(full), err)
	}
	if _, err := artifacts.Get(context.Background(), "other", handle); !errors.Is(err, artifact.ErrArtifactNotFound) {
		t.Fatalf("cross-session artifact read = %v", err)
	}
}

func TestCapabilityValidation(t *testing.T) {
	dummy := ExecutorFunc{
		StartFunc:  func(context.Context, Request) (Step, error) { return Step{Done: true}, nil },
		ResumeFunc: func(context.Context, Checkpoint, []CallResult) (Step, error) { return Step{Done: true}, nil },
	}
	for name, config := range map[string]Config{
		"missing executor": {},
		"bad tool name":    {Executor: dummy, ToolName: "bad-name", SelectedTools: []string{"one"}},
		"missing selected": {Executor: dummy},
		"bad limits":       {Executor: dummy, SelectedTools: []string{"one"}, Limits: Limits{MaxSteps: -1}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := capability.Compile(New(config)); err == nil {
				t.Fatal("invalid codemode config succeeded")
			}
		})
	}
}

func TestRestoreInlineContentPreservesShapeButNotTransformations(t *testing.T) {
	t.Parallel()
	original := agentic.ToolExecutionResult{
		ToolUseID: "nested",
		ToolName:  "selected_tool",
		Content:   map[string]int{"value": 40},
	}
	inline := original
	inline.Content = `{"value":40}`
	restored := restoreInlineContent(original, inline)
	if !reflect.DeepEqual(restored.Content, original.Content) {
		t.Fatalf("restored inline content = %#v", restored.Content)
	}

	transformed := original
	transformed.Content = "[harness artifact opaque]"
	if got := restoreInlineContent(original, transformed).Content; got != transformed.Content {
		t.Fatalf("transformed content = %#v", got)
	}
}
