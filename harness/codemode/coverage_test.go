package codemode

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type stagedOperations struct {
	records int
	failAt  int
}

func (o *stagedOperations) RecordOperation(context.Context, harnessruntime.Operation) error {
	o.records++
	if o.records == o.failAt {
		return errors.New("record failed")
	}
	return nil
}

func (*stagedOperations) ProjectToolResult(_ context.Context, _ agentic.ToolUse, result agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error) {
	return result, nil
}

type failingToolRegistry struct{}

func (failingToolRegistry) Register(agentic.Tool, agentic.ToolHandler) error { return nil }
func (failingToolRegistry) Get(string) (agentic.ToolHandler, bool)           { return nil, false }
func (failingToolRegistry) Execute(context.Context, agentic.ToolUse, any) (agentic.ToolExecutionResult, error) {
	return agentic.ToolExecutionResult{}, errors.New("registry failed")
}
func (failingToolRegistry) ExecuteBatch(context.Context, []agentic.ToolUse, any) ([]agentic.ToolExecutionResult, error) {
	return nil, errors.New("registry failed")
}
func (failingToolRegistry) Tools() []agentic.Tool { return nil }
func (failingToolRegistry) Has(string) bool       { return true }
func (failingToolRegistry) Count() int            { return 1 }

type plainHandler struct {
	name string
	err  error
}

func (h plainHandler) Name() string { return h.name }
func (h plainHandler) Execute(context.Context, map[string]any, any) (any, error) {
	return map[string]any{"ok": true}, h.err
}

type suspendableHandler struct{ plainHandler }

func (s suspendableHandler) MaySuspendToolExecution() bool { return true }

type errorOperations struct {
	recordErr  error
	projectErr error
}

func (o errorOperations) RecordOperation(context.Context, harnessruntime.Operation) error {
	return o.recordErr
}
func (o errorOperations) ProjectToolResult(_ context.Context, _ agentic.ToolUse, result agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error) {
	return result, o.projectErr
}

type projectionOperations struct {
	result func(agentic.ToolExecutionResult) agentic.ToolExecutionResult
}

func (projectionOperations) RecordOperation(context.Context, harnessruntime.Operation) error {
	return nil
}
func (o projectionOperations) ProjectToolResult(
	_ context.Context,
	_ agentic.ToolUse,
	result agentic.ToolExecutionResult,
) (agentic.ToolExecutionResult, error) {
	return o.result(result), nil
}

type gateCapture struct{ gate agentic.ToolGate }

func (c gateCapture) History() []agentic.Message  { return nil }
func (c gateCapture) Toolsets() []agentic.Toolset { return nil }
func (c gateCapture) ToolGate() agentic.ToolGate  { return c.gate }
func (c gateCapture) DelegationTools() []string   { return nil }
func (c gateCapture) Scope() harnessruntime.Scope { return harnessruntime.Scope{} }
func (c gateCapture) AcquireBudget(context.Context) (harnessruntime.BudgetLease, error) {
	return nil, nil
}
func (c gateCapture) ProjectEvent(context.Context, event.Record) error { return nil }

func testHandler(t *testing.T, executor Executor) *runCodeHandler {
	t.Helper()
	registry := agentic.NewRegistry()
	tool := agentic.Tool{Type: agentic.ToolTypeFunction, Function: agentic.Function{
		Name: "selected", Parameters: map[string]any{"type": "object"},
	}}
	if err := registry.Register(tool, plainHandler{name: "selected"}); err != nil {
		t.Fatal(err)
	}
	return &runCodeHandler{
		name: defaultTool, executor: executor, registry: registry,
		limits: Limits{}.withDefaults(), catalog: []Tool{{Name: "selected", Parameters: map[string]any{"type": "object"}}},
	}
}

func allowGate() agentic.ToolGate {
	return agentic.ToolGateFunc(func(_ context.Context, calls []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
		values := make([]agentic.ToolDisposition, len(calls))
		for index := range values {
			values[index].Kind = agentic.ToolDispositionExecute
		}
		return agentic.ToolBatchDecision{Calls: values}, nil
	})
}

func runtimeContext(operations harnessruntime.OperationRuntime, gate agentic.ToolGate) context.Context {
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		SessionID: "session", Capture: gateCapture{gate: gate}, Operations: operations,
	})
	return agentic.WithToolCallContext(ctx, agentic.ToolCallContext{ID: "outer", Name: defaultTool, Attempt: 1})
}

func TestExecutorFuncAndHandlerEntryValidation(t *testing.T) {
	startCalled, resumeCalled := false, false
	executor := ExecutorFunc{
		StartFunc: func(context.Context, Request) (Step, error) { startCalled = true; return Step{Done: true}, nil },
		ResumeFunc: func(context.Context, Checkpoint, []CallResult) (Step, error) {
			resumeCalled = true
			return Step{Done: true}, nil
		},
	}
	_, _ = executor.Start(context.Background(), Request{})
	_, _ = executor.Resume(context.Background(), nil, nil)
	if !startCalled || !resumeCalled {
		t.Fatal("executor function adapter did not delegate")
	}
	handler := testHandler(t, executor)
	if _, err := handler.Execute(context.Background(), map[string]any{"code": "x"}, nil); err == nil {
		t.Fatal("missing runtime succeeded")
	}
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{Capture: gateCapture{gate: allowGate()}, Operations: &fakeOperations{}})
	if _, err := handler.Execute(ctx, map[string]any{"code": "x"}, nil); err == nil {
		t.Fatal("missing outer call succeeded")
	}
	ctx = agentic.WithToolCallContext(ctx, agentic.ToolCallContext{ID: "outer", Name: defaultTool})
	if _, err := handler.Execute(ctx, map[string]any{}, nil); err == nil {
		t.Fatal("missing code succeeded")
	}
	handler.limits.MaxCodeBytes = 1
	if _, err := handler.Execute(ctx, map[string]any{"code": "too long"}, nil); err == nil {
		t.Fatal("oversized code succeeded")
	}
	failing := testHandler(t, ExecutorFunc{StartFunc: func(context.Context, Request) (Step, error) { return Step{}, errors.New("start failed") }})
	if _, err := failing.Execute(runtimeContext(&fakeOperations{}, allowGate()), map[string]any{"code": "x"}, nil); err == nil || !strings.Contains(err.Error(), "start") {
		t.Fatalf("start error = %v", err)
	}
}

func TestStepGateAndDriveValidationBranches(t *testing.T) {
	handler := testHandler(t, ExecutorFunc{})
	valid := Step{Checkpoint: []byte("cp"), Calls: []Call{{ID: "one", Name: "selected", Input: map[string]any{}}}}
	for name, step := range map[string]Step{
		"neither":            {},
		"both":               {Done: true, Calls: valid.Calls},
		"done checkpoint":    {Done: true, Checkpoint: []byte("cp")},
		"done output":        {Done: true, Output: make(chan int)},
		"nonterminal output": {Checkpoint: []byte("cp"), Calls: valid.Calls, Output: "bad"},
		"no checkpoint":      {Calls: valid.Calls},
		"duplicate":          {Checkpoint: []byte("cp"), Calls: []Call{{ID: "same", Name: "selected"}, {ID: "same", Name: "selected"}}},
		"unknown":            {Checkpoint: []byte("cp"), Calls: []Call{{ID: "one", Name: "unknown"}}},
		"bad input":          {Checkpoint: []byte("cp"), Calls: []Call{{ID: "one", Name: "selected", Input: map[string]any{"bad": make(chan int)}}}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := handler.validateStep(step); err == nil {
				t.Fatal("invalid step succeeded")
			}
		})
	}
	handler.limits.MaxCallsPerStep = 1
	if _, err := handler.validateStep(Step{Checkpoint: []byte("cp"), Calls: []Call{{ID: "a", Name: "selected"}, {ID: "b", Name: "selected"}}}); err == nil {
		t.Fatal("call limit succeeded")
	}
	handler = testHandler(t, ExecutorFunc{})
	calls := toToolUses(valid.Calls)
	invalidDecisions := []agentic.ToolBatchDecision{
		{},
		{Calls: []agentic.ToolDisposition{{Kind: agentic.ToolDispositionExecute, Result: &agentic.ToolExecutionResult{}}}},
		{Calls: []agentic.ToolDisposition{{Kind: agentic.ToolDispositionReturn}}},
		{Calls: []agentic.ToolDisposition{{Kind: agentic.ToolDispositionReturn, Result: &agentic.ToolExecutionResult{ToolUseID: "other", ToolName: "selected"}}}},
		{Calls: []agentic.ToolDisposition{{Kind: agentic.ToolDispositionSuspend, Continue: true}}, Deferral: &agentic.ToolDeferral{Kind: "ask"}},
		{Calls: []agentic.ToolDisposition{{Kind: agentic.ToolDispositionInvalid}}},
		{Calls: []agentic.ToolDisposition{{Kind: agentic.ToolDispositionSuspend}}},
		{Calls: []agentic.ToolDisposition{{Kind: agentic.ToolDispositionExecute}}, Deferral: &agentic.ToolDeferral{Kind: "ask"}},
	}
	for index, decision := range invalidDecisions {
		if _, _, err := handler.validateDecision(calls, decision); err == nil {
			t.Fatalf("invalid decision %d succeeded", index)
		}
	}

	stdoutHandler := testHandler(t, ExecutorFunc{})
	stdoutHandler.limits.MaxStdoutBytes = 1
	if _, err := stdoutHandler.drive(runtimeContext(&fakeOperations{}, allowGate()), nil, mustRuntime(runtimeContext(&fakeOperations{}, allowGate())), agentic.ToolCallContext{ID: "outer", Name: defaultTool}, 0, "", Step{Done: true, Stdout: "long"}); err == nil {
		t.Fatal("stdout limit succeeded")
	}
	limitHandler := testHandler(t, ExecutorFunc{})
	limitHandler.limits.MaxSteps = 1
	if _, err := limitHandler.drive(runtimeContext(&fakeOperations{}, allowGate()), nil, mustRuntime(runtimeContext(&fakeOperations{}, allowGate())), agentic.ToolCallContext{ID: "outer", Name: defaultTool}, 1, "", valid); err == nil {
		t.Fatal("step limit succeeded")
	}
	if _, err := handler.drive(runtimeContext(errorOperations{recordErr: errors.New("disk")}, allowGate()), nil, mustRuntime(runtimeContext(errorOperations{recordErr: errors.New("disk")}, allowGate())), agentic.ToolCallContext{ID: "outer", Name: defaultTool}, 0, "", valid); err == nil {
		t.Fatal("operation failure succeeded")
	}
	badGate := agentic.ToolGateFunc(func(context.Context, []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
		return agentic.ToolBatchDecision{}, errors.New("gate failed")
	})
	if _, err := handler.drive(runtimeContext(&fakeOperations{}, badGate), nil, mustRuntime(runtimeContext(&fakeOperations{}, badGate)), agentic.ToolCallContext{ID: "outer", Name: defaultTool}, 0, "", valid); err == nil {
		t.Fatal("gate error succeeded")
	}
}

func mustRuntime(ctx context.Context) harnessruntime.ToolRuntime {
	value, _ := harnessruntime.FromContext(ctx)
	return value
}

func TestCompleteBatchResumeFormsAndFailures(t *testing.T) {
	handler := testHandler(t, ExecutorFunc{})
	base := deferredProgram{
		Version: wireVersion, OuterCallID: "outer", Step: 0, Checkpoint: []byte("cp"),
		Calls: []Call{{ID: "one", Name: "selected", Input: map[string]any{}}},
	}
	runtime := mustRuntime(runtimeContext(&fakeOperations{}, allowGate()))
	base.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionSuspend}}
	forms := []resumeResolution{
		{CallID: "one", Action: agentic.ToolResumeExecute, OverrideArgs: map[string]any{"value": "override"}},
		{CallID: "one", Action: agentic.ToolResumeReturn, Result: &wireResult{ID: "one", Name: "selected", Content: "denied", IsError: true}},
	}
	for _, resolution := range forms {
		results, err := handler.completeBatch(context.Background(), nil, runtime, base, []resumeResolution{resolution})
		if err != nil || len(results) != 1 {
			t.Fatalf("resolution=%#v results=%#v err=%v", resolution, results, err)
		}
	}
	invalid := [][]resumeResolution{
		nil,
		{{CallID: "one", Action: agentic.ToolResumeInvalid}},
		{{CallID: "one", Action: agentic.ToolResumeReturn}},
		{{CallID: "one", Action: agentic.ToolResumeExecute}, {CallID: "one", Action: agentic.ToolResumeExecute}},
		{{CallID: "unknown", Action: agentic.ToolResumeExecute}},
	}
	for index, resolutions := range invalid {
		if _, err := handler.completeBatch(context.Background(), nil, runtime, base, resolutions); err == nil {
			t.Fatalf("invalid resolutions %d succeeded", index)
		}
	}
	base.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionInvalid}}
	if _, err := handler.completeBatch(context.Background(), nil, runtime, base, nil); err == nil {
		t.Fatal("invalid disposition succeeded")
	}
	base.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionExecute}}
	failingRuntime := mustRuntime(runtimeContext(errorOperations{projectErr: errors.New("project")}, allowGate()))
	if _, err := handler.completeBatch(context.Background(), nil, failingRuntime, base, nil); err == nil {
		t.Fatal("projection error succeeded")
	}

	base.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionReturn, Result: &wireResult{ID: "wrong", Name: "selected"}}}
	if _, err := handler.completeBatch(context.Background(), nil, runtime, base, nil); err == nil {
		t.Fatal("gate result with wrong identity succeeded")
	}
	base.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionSuspend}}
	if _, err := handler.completeBatch(context.Background(), nil, runtime, base, []resumeResolution{{
		CallID: "one", Action: agentic.ToolResumeReturn, Result: &wireResult{ID: "wrong", Name: "selected"},
	}}); err == nil {
		t.Fatal("resume result with wrong identity succeeded")
	}

	base.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionExecute}}
	wrongIdentity := projectionOperations{result: func(result agentic.ToolExecutionResult) agentic.ToolExecutionResult {
		result.ToolUseID = "wrong"
		return result
	}}
	if _, err := handler.completeBatch(context.Background(), nil, mustRuntime(runtimeContext(wrongIdentity, allowGate())), base, nil); err == nil {
		t.Fatal("projection identity change succeeded")
	}
	unencodable := projectionOperations{result: func(result agentic.ToolExecutionResult) agentic.ToolExecutionResult {
		result.Content = make(chan int)
		return result
	}}
	if _, err := handler.completeBatch(context.Background(), nil, mustRuntime(runtimeContext(unencodable, allowGate())), base, nil); err == nil {
		t.Fatal("unencodable projection succeeded")
	}
}

func TestResumeAndWireValidationBranches(t *testing.T) {
	handler := testHandler(t, ExecutorFunc{ResumeFunc: func(context.Context, Checkpoint, []CallResult) (Step, error) { return Step{Done: true}, nil }})
	runtime := mustRuntime(runtimeContext(&fakeOperations{}, allowGate()))
	outer := agentic.ToolCallContext{ID: "outer", Name: defaultTool}
	if _, err := handler.resume(context.Background(), nil, runtime, outer, agentic.ToolResumeContext{Deferral: agentic.ToolDeferral{Kind: "other"}}); err == nil {
		t.Fatal("wrong deferral kind succeeded")
	}
	if _, err := handler.resume(context.Background(), nil, runtime, outer, agentic.ToolResumeContext{Deferral: agentic.ToolDeferral{Kind: DeferralKind, Payload: []byte("{")}}); err == nil {
		t.Fatal("malformed deferral succeeded")
	}
	program := deferredProgram{
		Version: wireVersion, OuterCallID: "outer", Step: 0, Checkpoint: []byte("cp"),
		Calls:        []Call{{ID: "one", Name: "selected", Input: map[string]any{}}},
		Dispositions: []wireDisposition{{Kind: agentic.ToolDispositionSuspend}},
	}
	encoded, _ := json.Marshal(program)
	if _, err := handler.resume(context.Background(), nil, runtime, outer, agentic.ToolResumeContext{Deferral: agentic.ToolDeferral{Kind: DeferralKind, Payload: encoded}, Payload: []byte("{")}); err == nil {
		t.Fatal("malformed resume payload succeeded")
	}
	badResume, _ := json.Marshal(resumePayload{Version: 2, OuterCallID: "outer"})
	if _, err := handler.resume(context.Background(), nil, runtime, outer, agentic.ToolResumeContext{Deferral: agentic.ToolDeferral{Kind: DeferralKind, Payload: encoded}, Payload: badResume}); err == nil {
		t.Fatal("wrong resume version succeeded")
	}
	program.Dispositions[0].Kind = agentic.ToolDispositionExecute
	if err := handler.validateProgram(program, "outer"); err == nil {
		t.Fatal("program without suspension succeeded")
	}
	program.Dispositions[0].Kind = agentic.ToolDispositionSuspend
	program.Calls[0].ID = ""
	if err := handler.validateProgram(program, "outer"); err == nil {
		t.Fatal("program with invalid calls succeeded")
	}
	if _, err := fromWireResult(nil); err == nil {
		t.Fatal("nil wire result succeeded")
	}
	if cloneMap(map[string]any{"bad": make(chan int)}) != nil {
		t.Fatal("uncloneable map succeeded")
	}
	if _, err := encodeBounded(strings.Repeat("x", 5), 1, "value"); err == nil {
		t.Fatal("oversized encoding succeeded")
	}
	if _, err := cloneJSON(make(chan int), 100, "value"); err == nil {
		t.Fatal("uncloneable JSON succeeded")
	}
}

func validPlannerSuspension(t *testing.T) (agentic.Suspension, deferredProgram) {
	t.Helper()
	program := deferredProgram{
		Version: wireVersion, OuterCallID: "outer", Step: 0, Checkpoint: []byte("cp"),
		Calls:        []Call{{ID: "one", Name: "selected", Input: map[string]any{}}},
		Dispositions: []wireDisposition{{Kind: agentic.ToolDispositionSuspend}},
	}
	deferral, _ := json.Marshal(program)
	payload, _ := json.Marshal(struct {
		Version           int
		SuspensionID      string
		HandlerSuspension bool
		Calls             []agentic.ToolUse
		ExecutableCallIDs []string
		Deferral          agentic.ToolDeferral
	}{1, "s", true, []agentic.ToolUse{{ID: "outer", Name: defaultTool}}, []string{"outer"}, agentic.ToolDeferral{Kind: DeferralKind, Payload: deferral}})
	return agentic.Suspension{ID: "s", Kind: DeferralKind, Payload: payload}, program
}

func TestResumePlannerValidationAndResolutionForms(t *testing.T) {
	planner := resumePlanner{toolName: defaultTool, limits: Limits{}.withDefaults()}
	suspension, _ := validPlannerSuspension(t)
	assistant := agentic.NewTextMessage(agentic.RoleAssistant, "bad")
	for name, request := range map[string]harnessruntime.ResumeRequest{
		"wrong id":       {SuspensionID: "wrong"},
		"bad prompt":     {SuspensionID: "s", Prompt: &assistant},
		"missing":        {SuspensionID: "s"},
		"unknown":        {SuspensionID: "s", Resolutions: []harnessruntime.ToolResolution{{CallID: "other", Action: harnessruntime.ResolutionApprove}}},
		"duplicate":      {SuspensionID: "s", Resolutions: []harnessruntime.ToolResolution{{CallID: "one", Action: harnessruntime.ResolutionApprove}, {CallID: "one", Action: harnessruntime.ResolutionApprove}}},
		"invalid action": {SuspensionID: "s", Resolutions: []harnessruntime.ToolResolution{{CallID: "one", Action: harnessruntime.ResolutionInvalid}}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := planner.PlanResume(suspension, request); err == nil {
				t.Fatal("invalid resume succeeded")
			}
		})
	}
	for _, resolution := range []harnessruntime.ToolResolution{
		{CallID: "one", Action: harnessruntime.ResolutionApprove, OverrideArgs: map[string]any{"value": "new"}},
		{CallID: "one", Action: harnessruntime.ResolutionDeny},
		{CallID: "one", Action: harnessruntime.ResolutionExternalResult, Result: map[string]any{"external": true}},
	} {
		decisions, err := planner.PlanResume(suspension, harnessruntime.ResumeRequest{SuspensionID: "s", Resolutions: []harnessruntime.ToolResolution{resolution}})
		if err != nil || len(decisions) != 1 || len(decisions[0].Payload) == 0 {
			t.Fatalf("resolution=%#v decisions=%#v err=%v", resolution, decisions, err)
		}
	}
}

func TestCapabilityRejectsSelectionHazards(t *testing.T) {
	dummy := ExecutorFunc{StartFunc: func(context.Context, Request) (Step, error) { return Step{Done: true}, nil }, ResumeFunc: func(context.Context, Checkpoint, []CallResult) (Step, error) { return Step{Done: true}, nil }}
	if (*Capability)(nil).ID() != defaultID || !reflectOrdering((*Capability)(nil).Ordering(), capability.Ordering{}) {
		t.Fatal("nil capability defaults changed")
	}
	makeToolCapability := func(name string, handler agentic.ToolHandler, delegation bool) capability.Capability {
		return capability.Func{Name: "tools", Apply: func(registry *capability.Registry) error {
			tool := agentic.Tool{Function: agentic.Function{Name: name}}
			if err := registry.AddToolset(agentic.NewToolset().Add(tool, handler)); err != nil {
				return err
			}
			if delegation {
				return registry.MarkDelegationTool(name)
			}
			return nil
		}}
	}
	for name, selected := range map[string]struct {
		handler    agentic.ToolHandler
		delegation bool
		toolName   string
	}{
		"invalid selected": {handler: plainHandler{name: "bad-name"}, toolName: "bad-name"},
		"delegation":       {handler: plainHandler{name: "selected"}, delegation: true, toolName: "selected"},
		"suspendable":      {handler: suspendableHandler{plainHandler{name: "selected"}}, toolName: "selected"},
	} {
		t.Run(name, func(t *testing.T) {
			toolCapability := makeToolCapability(selected.toolName, selected.handler, selected.delegation)
			mode := New(Config{Executor: dummy, SelectedTools: []string{selected.toolName}, Order: capability.Ordering{After: []string{"tools"}}})
			if _, err := capability.Compile(toolCapability, mode); err == nil {
				t.Fatal("hazardous selection succeeded")
			}
		})
	}
}

func TestCapabilityRegistrationCollisionFrontiers(t *testing.T) {
	dummy := ExecutorFunc{
		StartFunc:  func(context.Context, Request) (Step, error) { return Step{Done: true}, nil },
		ResumeFunc: func(context.Context, Checkpoint, []CallResult) (Step, error) { return Step{Done: true}, nil },
	}
	toolCapability := func(name string, handler agentic.ToolHandler) capability.Capability {
		return capability.Func{Name: "tools", Apply: func(registry *capability.Registry) error {
			return registry.AddToolset(agentic.NewToolset().Add(agentic.Tool{Function: agentic.Function{Name: name}}, handler))
		}}
	}
	for name, capabilities := range map[string][]capability.Capability{
		"outer tool collision": {
			toolCapability(defaultTool, plainHandler{name: defaultTool}),
			New(Config{Executor: dummy, SelectedTools: []string{defaultTool}, Order: capability.Ordering{After: []string{"tools"}}}),
		},
		"unknown selected tool": {
			New(Config{Executor: dummy, SelectedTools: []string{"missing"}}),
		},
		"nil selected handler": {
			toolCapability("selected", nil),
			New(Config{Executor: dummy, SelectedTools: []string{"selected"}, Order: capability.Ordering{After: []string{"tools"}}}),
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := capability.Compile(capabilities...); err == nil {
				t.Fatal("invalid registration succeeded")
			}
		})
	}
	selected := toolCapability("selected", plainHandler{name: "selected"})
	plannerConflict := capability.Func{Name: "planner", Order: capability.Ordering{After: []string{"tools"}}, Apply: func(registry *capability.Registry) error {
		return registry.AddResumePlanner(DeferralKind, harnessruntime.ResumePlannerFunc(func(agentic.Suspension, harnessruntime.ResumeRequest) ([]agentic.ToolResumeDecision, error) {
			return nil, nil
		}))
	}}
	mode := New(Config{Executor: dummy, SelectedTools: []string{"selected"}, Order: capability.Ordering{After: []string{"planner"}}})
	if _, err := capability.Compile(selected, plannerConflict, mode); err == nil {
		t.Fatal("duplicate resume planner succeeded")
	}
}

func TestDriveAndBatchFailureFrontiers(t *testing.T) {
	valid := Step{Checkpoint: []byte("cp"), Calls: []Call{{ID: "one", Name: "selected", Input: map[string]any{}}}}
	outer := agentic.ToolCallContext{ID: "outer", Name: defaultTool}

	doneFailure := testHandler(t, ExecutorFunc{})
	doneOps := &stagedOperations{failAt: 1}
	doneContext := runtimeContext(doneOps, allowGate())
	if _, err := doneFailure.drive(doneContext, nil, mustRuntime(doneContext), outer, 0, "", Step{Done: true}); err == nil {
		t.Fatal("completion record failure succeeded")
	}

	combinedStdout := testHandler(t, ExecutorFunc{})
	combinedStdout.limits.MaxStdoutBytes = 3
	stdoutContext := runtimeContext(&fakeOperations{}, allowGate())
	if _, err := combinedStdout.drive(stdoutContext, nil, mustRuntime(stdoutContext), outer, 0, "ab", Step{Done: true, Stdout: "cd"}); err == nil {
		t.Fatal("combined stdout overflow succeeded")
	}

	invalidGate := agentic.ToolGateFunc(func(context.Context, []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
		return agentic.ToolBatchDecision{Calls: []agentic.ToolDisposition{{Kind: agentic.ToolDispositionInvalid}}}, nil
	})
	gateContext := runtimeContext(&fakeOperations{}, invalidGate)
	if _, err := testHandler(t, ExecutorFunc{}).drive(gateContext, nil, mustRuntime(gateContext), outer, 0, "", valid); err == nil {
		t.Fatal("invalid gate decision succeeded")
	}

	suspendGate := agentic.ToolGateFunc(func(context.Context, []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
		return agentic.ToolBatchDecision{
			Calls:    []agentic.ToolDisposition{{Kind: agentic.ToolDispositionSuspend}},
			Deferral: &agentic.ToolDeferral{Kind: "approval"},
		}, nil
	})
	suspendOps := &stagedOperations{failAt: 2}
	suspendContext := runtimeContext(suspendOps, suspendGate)
	if _, err := testHandler(t, ExecutorFunc{}).drive(suspendContext, nil, mustRuntime(suspendContext), outer, 0, "", valid); err == nil {
		t.Fatal("suspension record failure succeeded")
	}

	resumeFailure := testHandler(t, ExecutorFunc{ResumeFunc: func(context.Context, Checkpoint, []CallResult) (Step, error) {
		return Step{}, errors.New("resume failed")
	}})
	resumeContext := runtimeContext(&fakeOperations{}, allowGate())
	if _, err := resumeFailure.drive(resumeContext, nil, mustRuntime(resumeContext), outer, 0, "", valid); err == nil {
		t.Fatal("executor resume failure succeeded")
	}

	projectContext := runtimeContext(errorOperations{projectErr: errors.New("project failed")}, allowGate())
	if _, err := testHandler(t, ExecutorFunc{}).drive(projectContext, nil, mustRuntime(projectContext), outer, 0, "", valid); err == nil {
		t.Fatal("batch projection failure succeeded")
	}

	program := deferredProgram{
		Version: wireVersion, OuterCallID: "outer", Step: 0, Checkpoint: []byte("cp"),
		Calls:        []Call{{ID: "one", Name: "selected"}},
		Dispositions: []wireDisposition{{Kind: agentic.ToolDispositionExecute}},
	}
	startOps := &stagedOperations{failAt: 1}
	startContext := runtimeContext(startOps, allowGate())
	if _, err := testHandler(t, ExecutorFunc{}).completeBatch(startContext, nil, mustRuntime(startContext), program, nil); err == nil {
		t.Fatal("nested start record failure succeeded")
	}
	resultOps := &stagedOperations{failAt: 2}
	resultContext := runtimeContext(resultOps, allowGate())
	if _, err := testHandler(t, ExecutorFunc{}).completeBatch(resultContext, nil, mustRuntime(resultContext), program, nil); err == nil {
		t.Fatal("nested result record failure succeeded")
	}

	program.Dispositions[0] = wireDisposition{Kind: agentic.ToolDispositionReturn}
	if _, err := testHandler(t, ExecutorFunc{}).completeBatch(context.Background(), nil, mustRuntime(runtimeContext(&fakeOperations{}, allowGate())), program, nil); err == nil {
		t.Fatal("invalid returned wire result succeeded")
	}

	program.Dispositions[0] = wireDisposition{Kind: agentic.ToolDispositionSuspend}
	suspendStartOps := &stagedOperations{failAt: 1}
	suspendStartContext := runtimeContext(suspendStartOps, allowGate())
	if _, err := testHandler(t, ExecutorFunc{}).completeBatch(suspendStartContext, nil, mustRuntime(suspendStartContext), program, []resumeResolution{{CallID: "one", Action: agentic.ToolResumeExecute}}); err == nil {
		t.Fatal("resumed nested start record failure succeeded")
	}

	two := deferredProgram{
		Version: wireVersion, OuterCallID: "outer", Step: 0, Checkpoint: []byte("cp"),
		Calls:        []Call{{ID: "execute", Name: "selected"}, {ID: "suspend", Name: "selected"}},
		Dispositions: []wireDisposition{{Kind: agentic.ToolDispositionExecute}, {Kind: agentic.ToolDispositionSuspend}},
	}
	extra := []resumeResolution{
		{CallID: "execute", Action: agentic.ToolResumeExecute},
		{CallID: "suspend", Action: agentic.ToolResumeReturn, Result: &wireResult{ID: "suspend", Name: "selected"}},
	}
	if _, err := testHandler(t, ExecutorFunc{}).completeBatch(context.Background(), nil, mustRuntime(runtimeContext(&fakeOperations{}, allowGate())), two, extra); err == nil {
		t.Fatal("extra resolution succeeded")
	}

	registryFailure := testHandler(t, ExecutorFunc{})
	registryFailure.registry = failingToolRegistry{}
	result := registryFailure.executeSelected(context.Background(), nil, agentic.ToolUse{ID: "one", Name: "selected"})
	if !result.IsError || !strings.Contains(result.Content.(string), "registry failed") {
		t.Fatalf("registry error result = %#v", result)
	}

	badProgram := deferredProgram{OuterCallID: "outer", Calls: []Call{{ID: "one", Name: "selected", Input: map[string]any{"bad": make(chan int)}}}}
	if err := testHandler(t, ExecutorFunc{}).recordProgram(context.Background(), mustRuntime(runtimeContext(&fakeOperations{}, allowGate())), "outer", "planned", badProgram); err == nil {
		t.Fatal("unencodable program record succeeded")
	}
	if err := testHandler(t, ExecutorFunc{}).recordCall(context.Background(), mustRuntime(runtimeContext(&fakeOperations{}, allowGate())), badProgram, badProgram.Calls[0], "started", nil); err == nil {
		t.Fatal("unencodable call record succeeded")
	}
	returnDecision := agentic.ToolBatchDecision{Calls: []agentic.ToolDisposition{{
		Kind:   agentic.ToolDispositionReturn,
		Result: &agentic.ToolExecutionResult{ToolUseID: "one", ToolName: "selected", Content: make(chan int)},
	}}}
	if _, _, err := testHandler(t, ExecutorFunc{}).validateDecision([]agentic.ToolUse{{ID: "one", Name: "selected"}}, returnDecision); err == nil {
		t.Fatal("unencodable gate result succeeded")
	}
}

func TestResumeExecutionFailureFrontiers(t *testing.T) {
	program := deferredProgram{
		Version: wireVersion, OuterCallID: "outer", Step: 0, Checkpoint: []byte("cp"),
		Calls:        []Call{{ID: "one", Name: "selected", Input: map[string]any{}}},
		Dispositions: []wireDisposition{{Kind: agentic.ToolDispositionSuspend}},
	}
	encoded, _ := json.Marshal(program)
	runtime := mustRuntime(runtimeContext(&fakeOperations{}, allowGate()))
	outer := agentic.ToolCallContext{ID: "outer", Name: defaultTool}
	missing, _ := json.Marshal(resumePayload{Version: wireVersion, OuterCallID: "outer"})
	handler := testHandler(t, ExecutorFunc{})
	if _, err := handler.resume(context.Background(), nil, runtime, outer, agentic.ToolResumeContext{
		Deferral: agentic.ToolDeferral{Kind: DeferralKind, Payload: encoded}, Payload: missing,
	}); err == nil {
		t.Fatal("missing resolution succeeded")
	}
	decision, _ := json.Marshal(resumePayload{Version: wireVersion, OuterCallID: "outer", Resolutions: []resumeResolution{{CallID: "one", Action: agentic.ToolResumeExecute}}})
	failing := testHandler(t, ExecutorFunc{ResumeFunc: func(context.Context, Checkpoint, []CallResult) (Step, error) {
		return Step{}, errors.New("executor resume failed")
	}})
	if _, err := failing.resume(context.Background(), nil, runtime, outer, agentic.ToolResumeContext{
		Deferral: agentic.ToolDeferral{Kind: DeferralKind, Payload: encoded}, Payload: decision,
	}); err == nil || !strings.Contains(err.Error(), "executor resume") {
		t.Fatalf("resume executor error = %v", err)
	}
	program.Version = 2
	badProgram, _ := json.Marshal(program)
	if _, err := handler.resume(context.Background(), nil, runtime, outer, agentic.ToolResumeContext{
		Deferral: agentic.ToolDeferral{Kind: DeferralKind, Payload: badProgram}, Payload: decision,
	}); err == nil {
		t.Fatal("invalid deferred program succeeded")
	}
}

func plannerSuspension(program deferredProgram, handler bool, calls []agentic.ToolUse, executable []string) agentic.Suspension {
	deferral, _ := json.Marshal(program)
	payload, _ := json.Marshal(struct {
		Version           int
		SuspensionID      string
		HandlerSuspension bool
		Calls             []agentic.ToolUse
		ExecutableCallIDs []string
		Deferral          agentic.ToolDeferral
	}{1, "s", handler, calls, executable, agentic.ToolDeferral{Kind: DeferralKind, Payload: deferral}})
	return agentic.Suspension{ID: "s", Kind: DeferralKind, Payload: payload}
}

func TestResumePlannerMalformedFrontiers(t *testing.T) {
	planner := resumePlanner{toolName: defaultTool, limits: Limits{}.withDefaults()}
	request := harnessruntime.ResumeRequest{SuspensionID: "s", Resolutions: []harnessruntime.ToolResolution{{CallID: "one", Action: harnessruntime.ResolutionApprove}}}
	if _, err := planner.PlanResume(agentic.Suspension{ID: "s", Kind: DeferralKind, Payload: []byte("{")}, request); err == nil {
		t.Fatal("malformed suspension succeeded")
	}
	program := deferredProgram{
		Version: wireVersion, OuterCallID: "outer", Step: 0, Checkpoint: []byte("cp"),
		Calls:        []Call{{ID: "one", Name: "selected"}},
		Dispositions: []wireDisposition{{Kind: agentic.ToolDispositionSuspend}},
	}
	for name, suspension := range map[string]agentic.Suspension{
		"not handler":    plannerSuspension(program, false, []agentic.ToolUse{{ID: "outer", Name: defaultTool}}, []string{"outer"}),
		"two executable": plannerSuspension(program, true, []agentic.ToolUse{{ID: "outer", Name: defaultTool}}, []string{"outer", "other"}),
		"wrong tool":     plannerSuspension(program, true, []agentic.ToolUse{{ID: "outer", Name: "other"}}, []string{"outer"}),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := planner.PlanResume(suspension, request); err == nil {
				t.Fatal("invalid frontier succeeded")
			}
		})
	}
	malformed := plannerSuspension(program, true, []agentic.ToolUse{{ID: "outer", Name: defaultTool}}, []string{"outer"})
	var root map[string]any
	_ = json.Unmarshal(malformed.Payload, &root)
	root["Deferral"] = map[string]any{"Kind": DeferralKind, "Payload": "ew=="}
	malformed.Payload, _ = json.Marshal(root)
	if _, err := planner.PlanResume(malformed, request); err == nil {
		t.Fatal("malformed codemode deferral succeeded")
	}

	frontier := func(current deferredProgram) agentic.Suspension {
		return plannerSuspension(current, true, []agentic.ToolUse{{ID: "outer", Name: defaultTool}}, []string{"outer"})
	}
	invalidPrograms := map[string]deferredProgram{
		"wrong version":   program,
		"duplicate call":  program,
		"execute result":  program,
		"bad return":      program,
		"bad disposition": program,
		"no suspension":   program,
	}
	value := invalidPrograms["wrong version"]
	value.Version = 2
	invalidPrograms["wrong version"] = value
	value = invalidPrograms["duplicate call"]
	value.Calls = []Call{{ID: "one", Name: "selected"}, {ID: "one", Name: "selected"}}
	value.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionSuspend}, {Kind: agentic.ToolDispositionSuspend}}
	invalidPrograms["duplicate call"] = value
	value = invalidPrograms["execute result"]
	value.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionExecute, Result: &wireResult{ID: "one", Name: "selected"}}}
	invalidPrograms["execute result"] = value
	value = invalidPrograms["bad return"]
	value.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionReturn, Result: &wireResult{ID: "other", Name: "selected"}}}
	invalidPrograms["bad return"] = value
	value = invalidPrograms["bad disposition"]
	value.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionInvalid}}
	invalidPrograms["bad disposition"] = value
	value = invalidPrograms["no suspension"]
	value.Dispositions = []wireDisposition{{Kind: agentic.ToolDispositionExecute}}
	invalidPrograms["no suspension"] = value
	for name, current := range invalidPrograms {
		t.Run(name, func(t *testing.T) {
			if _, err := planner.PlanResume(frontier(current), request); err == nil {
				t.Fatal("malformed program succeeded")
			}
		})
	}

	oversizedOverride := request
	oversizedOverride.Resolutions = []harnessruntime.ToolResolution{{CallID: "one", Action: harnessruntime.ResolutionApprove, OverrideArgs: map[string]any{"value": strings.Repeat("x", planner.limits.MaxValueBytes)}}}
	if _, err := planner.PlanResume(frontier(program), oversizedOverride); err == nil {
		t.Fatal("oversized override succeeded")
	}
	oversizedExternal := request
	oversizedExternal.Resolutions = []harnessruntime.ToolResolution{{CallID: "one", Action: harnessruntime.ResolutionExternalResult, Result: strings.Repeat("x", planner.limits.MaxValueBytes)}}
	if _, err := planner.PlanResume(frontier(program), oversizedExternal); err == nil {
		t.Fatal("oversized external result succeeded")
	}
}

func reflectOrdering(left, right capability.Ordering) bool {
	return len(left.Before) == len(right.Before) && len(left.After) == len(right.After)
}
