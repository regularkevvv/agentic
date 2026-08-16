package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	providertest "github.com/regularkevvv/agentic/provider/test"
)

type suspensionTestHandler struct {
	name     string
	suspend  bool
	calls    atomic.Int32
	execute  func(context.Context, int) (any, error)
	mu       sync.Mutex
	resumes  []ToolResumeContext
	ordinary []bool
}

func (h *suspensionTestHandler) Name() string { return h.name }

func (h *suspensionTestHandler) MaySuspendToolExecution() bool { return h.suspend }

func (h *suspensionTestHandler) Execute(ctx context.Context, _ map[string]any, _ any) (any, error) {
	invocation := int(h.calls.Add(1))
	resume, ok := CurrentToolResume(ctx)
	h.mu.Lock()
	h.ordinary = append(h.ordinary, !ok)
	if ok {
		h.resumes = append(h.resumes, resume)
	}
	h.mu.Unlock()
	return h.execute(ctx, invocation)
}

func (h *suspensionTestHandler) snapshots() ([]ToolResumeContext, []bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	resumes := make([]ToolResumeContext, len(h.resumes))
	for index, resume := range h.resumes {
		resumes[index] = resume
		resumes[index].Deferral.Payload = append(json.RawMessage(nil), resume.Deferral.Payload...)
		resumes[index].Payload = append(json.RawMessage(nil), resume.Payload...)
	}
	return resumes, append([]bool(nil), h.ordinary...)
}

type undeclaredSuspensionHandler struct {
	name  string
	calls atomic.Int32
}

func (h *undeclaredSuspensionHandler) Name() string { return h.name }

func (h *undeclaredSuspensionHandler) Execute(context.Context, map[string]any, any) (any, error) {
	h.calls.Add(1)
	return ToolHandlerSuspension{Deferral: ToolDeferral{Kind: "undeclared"}}, nil
}

func TestSuspendableHandlerResumesExactlyAndCanSuspendAgain(t *testing.T) {
	definition := MustNewToolFromStruct("run_code", "run code", struct {
		Code string `json:"code"`
	}{})
	handler := &suspensionTestHandler{name: "run_code", suspend: true}
	handler.execute = func(ctx context.Context, invocation int) (any, error) {
		switch invocation {
		case 1:
			if _, ok := CurrentToolResume(ctx); ok {
				return nil, errors.New("ordinary execution unexpectedly had resume metadata")
			}
			return ToolHandlerSuspension{Deferral: ToolDeferral{
				Kind:    "test.nested.one",
				Payload: json.RawMessage(`{"checkpoint":1}`),
			}}, nil
		case 2:
			resume, ok := CurrentToolResume(ctx)
			if !ok || resume.Deferral.Kind != "test.nested.one" ||
				string(resume.Deferral.Payload) != `{"checkpoint":1}` ||
				string(resume.Payload) != `{"resolution":1}` {
				return nil, fmt.Errorf("first resume metadata = %#v, %t", resume, ok)
			}
			resume.Deferral.Payload[0] = 'x'
			resume.Payload[0] = 'x'
			again, ok := CurrentToolResume(ctx)
			if !ok || string(again.Deferral.Payload) != `{"checkpoint":1}` || string(again.Payload) != `{"resolution":1}` {
				return nil, fmt.Errorf("resume metadata was not defensively copied: %#v", again)
			}
			return &ToolHandlerSuspension{Deferral: ToolDeferral{
				Kind:    "test.nested.two",
				Payload: json.RawMessage(`{"checkpoint":2}`),
			}}, nil
		case 3:
			resume, ok := CurrentToolResume(ctx)
			if !ok || resume.Deferral.Kind != "test.nested.two" ||
				string(resume.Deferral.Payload) != `{"checkpoint":2}` ||
				string(resume.Payload) != `{"resolution":2}` {
				return nil, fmt.Errorf("second resume metadata = %#v, %t", resume, ok)
			}
			return "program complete", nil
		default:
			return nil, fmt.Errorf("unexpected invocation %d", invocation)
		}
	}
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{{
			ID: "code-1", Name: "run_code", Input: map[string]any{"code": "tool()"},
		}}},
		providertest.ModelResponse{Text: "done"},
	)
	recorder := &recordingInstrumentation{}
	driver := NewAgent("system", model, WithInstrumentation(recorder)).AddTool(definition, handler)
	prompt := NewTextMessage(RoleUser, "execute")

	first, err := driver.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt})
	if err != nil || first.Status != ExecutionSuspended || first.Suspension == nil ||
		first.Suspension.Kind != "test.nested.one" || len(first.Result.ToolResults) != 0 {
		t.Fatalf("first suspension = %#v, %v", first, err)
	}
	firstID := first.Suspension.ID
	second, err := driver.Resume(context.Background(), ResumeInput{
		History:    first.Result.Messages,
		Suspension: *first.Suspension,
		Decisions: []ToolResumeDecision{{
			CallID: "code-1", Action: ToolResumeExecute, Payload: json.RawMessage(`{"resolution":1}`),
		}},
	})
	if err != nil || second.Status != ExecutionSuspended || second.Suspension == nil ||
		second.Suspension.Kind != "test.nested.two" || second.Suspension.ID == firstID ||
		model.CallCount() != 1 {
		t.Fatalf("second suspension = %#v, model calls %d, error %v", second, model.CallCount(), err)
	}

	completed, err := driver.Resume(context.Background(), ResumeInput{
		History:    second.Result.Messages,
		Suspension: *second.Suspension,
		Decisions: []ToolResumeDecision{{
			CallID: "code-1", Action: ToolResumeExecute, Payload: json.RawMessage(`{"resolution":2}`),
		}},
	})
	if err != nil || completed.Status != ExecutionCompleted || completed.Result.Output != "done" ||
		handler.calls.Load() != 3 || model.CallCount() != 2 {
		t.Fatalf("completed execution = %#v, handler/model calls %d/%d, error %v", completed, handler.calls.Load(), model.CallCount(), err)
	}
	last := model.Calls()[1].Messages
	if len(last) == 0 || last[len(last)-1].Role != RoleTool ||
		last[len(last)-1].GetToolResults()[0].Content != "program complete" {
		t.Fatalf("completed request frontier = %#v", last)
	}
	resumes, ordinary := handler.snapshots()
	if len(resumes) != 2 || len(ordinary) != 3 || !ordinary[0] || ordinary[1] || ordinary[2] {
		t.Fatalf("resume snapshots/ordinary flags = %#v / %#v", resumes, ordinary)
	}
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	if len(recorder.tools) != 3 || recorder.tools[0].HandlerResumed ||
		!recorder.tools[1].HandlerResumed || !recorder.tools[2].HandlerResumed {
		t.Fatalf("instrumented handler resume lifecycle = %#v", recorder.tools)
	}
	if len(recorder.toolResults) != 3 || !recorder.toolResults[0].Suspended ||
		!recorder.toolResults[1].Suspended || recorder.toolResults[2].Suspended {
		t.Fatalf("instrumented handler outcomes = %#v", recorder.toolResults)
	}
}

func TestSuspendableHandlerIsIsolatedBeforeAnyHandlerStarts(t *testing.T) {
	suspendableDefinition := MustNewToolFromStruct("run_code", "run code", struct{}{})
	suspendable := &suspensionTestHandler{name: "run_code", suspend: true}
	suspendable.execute = func(context.Context, int) (any, error) { return "unexpected", nil }
	ordinaryDefinition, ordinaryHandler := MustToolPlain("write", "write", func(struct{}) (string, error) {
		return "unexpected", nil
	})
	var starts atomic.Int32
	wrapped := &countingToolHandler{ToolHandler: ordinaryHandler, calls: &starts}
	model := providertest.NewTestModel(providertest.ModelResponse{ToolCalls: []ToolUse{
		{ID: "code-1", Name: "run_code", Input: map[string]any{}},
		{ID: "write-1", Name: "write", Input: map[string]any{}},
	}})
	prompt := NewTextMessage(RoleUser, "execute")
	execution, err := NewAgent("system", model).
		AddTool(suspendableDefinition, suspendable).
		AddTool(ordinaryDefinition, wrapped).
		Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt})
	if !errors.Is(err, ErrToolHandlerSuspension) || execution == nil || execution.Status != ExecutionFailed {
		t.Fatalf("mixed suspension execution = %#v, %v", execution, err)
	}
	if suspendable.calls.Load() != 0 || starts.Load() != 0 {
		t.Fatalf("mixed batch started handlers: suspendable=%d ordinary=%d", suspendable.calls.Load(), starts.Load())
	}
}

type countingToolHandler struct {
	ToolHandler
	calls *atomic.Int32
}

func (h *countingToolHandler) Execute(ctx context.Context, input map[string]any, deps any) (any, error) {
	h.calls.Add(1)
	return h.ToolHandler.Execute(ctx, input, deps)
}

func TestUndeclaredAndInvalidHandlerSuspensionsFailClosed(t *testing.T) {
	definition := MustNewToolFromStruct("tool", "tool", struct{}{})
	tests := []struct {
		name    string
		handler ToolHandler
	}{
		{
			name: "undeclared",
			handler: func() ToolHandler {
				h := &undeclaredSuspensionHandler{name: "tool"}
				return h
			}(),
		},
		{
			name: "empty deferral kind",
			handler: func() ToolHandler {
				h := &suspensionTestHandler{name: "tool", suspend: true}
				h.execute = func(context.Context, int) (any, error) { return ToolHandlerSuspension{}, nil }
				return h
			}(),
		},
		{
			name: "nil suspension",
			handler: func() ToolHandler {
				h := &suspensionTestHandler{name: "tool", suspend: true}
				h.execute = func(context.Context, int) (any, error) {
					var suspension *ToolHandlerSuspension
					return suspension, nil
				}
				return h
			}(),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := providertest.NewTestModel(providertest.ModelResponse{ToolCalls: []ToolUse{{
				ID: "tool-1", Name: "tool", Input: map[string]any{},
			}}})
			prompt := NewTextMessage(RoleUser, "execute")
			execution, err := NewAgent("system", model).AddTool(definition, test.handler).
				Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt})
			if !errors.Is(err, ErrToolHandlerSuspension) || execution == nil ||
				execution.Status != ExecutionFailed || execution.Suspension != nil {
				t.Fatalf("invalid handler suspension = %#v, %v", execution, err)
			}
		})
	}
}

func TestHandlerSuspensionEncodingFailureReturnsPairedFailedExecution(t *testing.T) {
	definition := MustNewToolFromStruct("run_code", "run code", struct{}{})
	handler := &suspensionTestHandler{name: "run_code", suspend: true}
	handler.execute = func(context.Context, int) (any, error) {
		return ToolHandlerSuspension{Deferral: ToolDeferral{
			Kind:    "nested",
			Payload: json.RawMessage(`{`),
		}}, nil
	}
	model := providertest.NewTestModel(providertest.ModelResponse{ToolCalls: []ToolUse{{
		ID: "code-1", Name: "run_code", Input: map[string]any{},
	}}})
	prompt := NewTextMessage(RoleUser, "execute")
	execution, err := NewAgent("system", model).AddTool(definition, handler).
		Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt})
	if execution == nil || execution.Status != ExecutionFailed || execution.Suspension != nil || err == nil ||
		!strings.Contains(err.Error(), "marshal tool suspension") {
		t.Fatalf("encoding failure = %#v, %v", execution, err)
	}
	if handler.calls.Load() != 1 || len(execution.Result.ToolResults) != 1 ||
		!execution.Result.ToolResults[0].IsError || len(execution.Result.Messages) < 2 ||
		execution.Result.Messages[len(execution.Result.Messages)-1].Role != RoleTool {
		t.Fatalf("encoding failure pairing = %#v", execution.Result)
	}
}

func TestGateSuspensionCannotInjectHandlerResumePayload(t *testing.T) {
	definition := MustNewToolFromStruct("tool", "tool", struct{}{})
	handler := &suspensionTestHandler{name: "tool", suspend: false}
	handler.execute = func(ctx context.Context, _ int) (any, error) {
		if resume, ok := CurrentToolResume(ctx); ok {
			return nil, fmt.Errorf("gate resume leaked handler metadata: %#v", resume)
		}
		return "ordinary result", nil
	}
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []ToolUse{{ID: "tool-1", Name: "tool", Input: map[string]any{}}}},
		providertest.ModelResponse{Text: "done"},
	)
	prompt := NewTextMessage(RoleUser, "execute")
	driver := NewAgent("system", model).AddTool(definition, handler)
	suspended, err := driver.Drive(
		context.Background(),
		DriveInput{Mode: DriveStart, Prompt: &prompt},
		WithRunToolGate(ToolGateFunc(func(context.Context, []ToolUse) (ToolBatchDecision, error) {
			return ToolBatchDecision{
				Calls:    []ToolDisposition{{Kind: ToolDispositionSuspend}},
				Deferral: &ToolDeferral{Kind: "approval", Payload: json.RawMessage(`{"request":1}`)},
			}, nil
		})),
	)
	if err != nil || suspended.Status != ExecutionSuspended {
		t.Fatalf("gate suspension = %#v, %v", suspended, err)
	}
	result := &ToolExecutionResult{ToolUseID: "tool-1", ToolName: "tool", Content: "external"}
	invalid := []struct {
		name     string
		decision ToolResumeDecision
	}{
		{
			name: "handler payload",
			decision: ToolResumeDecision{
				CallID: "tool-1", Action: ToolResumeExecute, Payload: json.RawMessage(`{"not":"handler data"}`),
			},
		},
		{
			name:     "execute with result",
			decision: ToolResumeDecision{CallID: "tool-1", Action: ToolResumeExecute, Result: result},
		},
		{
			name: "return with payload",
			decision: ToolResumeDecision{
				CallID: "tool-1", Action: ToolResumeReturn, Result: result, Payload: json.RawMessage(`{}`),
			},
		},
		{
			name: "return with input",
			decision: ToolResumeDecision{
				CallID: "tool-1", Action: ToolResumeReturn, Result: result, Input: map[string]any{},
			},
		},
	}
	for _, test := range invalid {
		t.Run(test.name, func(t *testing.T) {
			execution, resumeErr := driver.Resume(context.Background(), ResumeInput{
				History: suspended.Result.Messages, Suspension: *suspended.Suspension,
				Decisions: []ToolResumeDecision{test.decision},
			})
			if execution == nil || !errors.Is(resumeErr, ErrResumeDecision) || handler.calls.Load() != 0 {
				t.Fatalf("invalid gate resume = %#v, calls %d, error %v", execution, handler.calls.Load(), resumeErr)
			}
		})
	}
	completed, err := driver.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: *suspended.Suspension,
		Decisions: []ToolResumeDecision{{CallID: "tool-1", Action: ToolResumeExecute}},
	})
	if err != nil || completed.Status != ExecutionCompleted || handler.calls.Load() != 1 {
		t.Fatalf("ordinary gate resume = %#v, calls %d, error %v", completed, handler.calls.Load(), err)
	}
}

func TestHandlerSuspensionRejectsAReplacementThatDoesNotDeclareSuspension(t *testing.T) {
	definition := MustNewToolFromStruct("run_code", "run code", struct{}{})
	suspendable := &suspensionTestHandler{name: "run_code", suspend: true}
	suspendable.execute = func(context.Context, int) (any, error) {
		return ToolHandlerSuspension{Deferral: ToolDeferral{Kind: "nested"}}, nil
	}
	replacement, replacementHandler := MustToolPlain("run_code", "run code", func(struct{}) (string, error) {
		return "must not execute", nil
	})
	model := providertest.NewTestModel(providertest.ModelResponse{ToolCalls: []ToolUse{{
		ID: "code-1", Name: "run_code", Input: map[string]any{},
	}}})
	driver := NewAgent("system", model)
	prompt := NewTextMessage(RoleUser, "execute")
	suspended, err := driver.Drive(
		context.Background(),
		DriveInput{Mode: DriveStart, Prompt: &prompt},
		WithRunToolsets(NewToolset().Add(definition, suspendable)),
	)
	if err != nil || suspended.Status != ExecutionSuspended {
		t.Fatalf("handler suspension = %#v, %v", suspended, err)
	}
	var replacementCalls atomic.Int32
	wrapped := &countingToolHandler{ToolHandler: replacementHandler, calls: &replacementCalls}
	execution, err := driver.Resume(
		context.Background(),
		ResumeInput{
			History: suspended.Result.Messages, Suspension: *suspended.Suspension,
			Decisions: []ToolResumeDecision{{CallID: "code-1", Action: ToolResumeExecute}},
		},
		WithRunToolsets(NewToolset().Add(replacement, wrapped)),
	)
	if execution == nil || !errors.Is(err, ErrSuspensionMismatch) || replacementCalls.Load() != 0 {
		t.Fatalf("replacement resume = %#v, calls %d, error %v", execution, replacementCalls.Load(), err)
	}
}
