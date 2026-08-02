package codemode

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	agentic "github.com/regularkevvv/agentic"

	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type runCodeHandler struct {
	name     string
	executor Executor
	limits   Limits
	registry agentic.ToolRegistry
	catalog  []Tool
}

func (h *runCodeHandler) Name() string { return h.name }

func (h *runCodeHandler) MaySuspendToolExecution() bool { return true }

func (h *runCodeHandler) Execute(ctx context.Context, input map[string]any, deps any) (any, error) {
	toolRuntime, ok := harnessruntime.FromContext(ctx)
	if !ok || toolRuntime.Capture == nil || toolRuntime.Operations == nil {
		return nil, errors.New("codemode requires harness tool runtime services")
	}
	outer, ok := agentic.CurrentToolCall(ctx)
	if !ok || outer.ID == "" || outer.Name != h.name {
		return nil, errors.New("codemode requires the current outer tool call")
	}
	if resumed, ok := agentic.CurrentToolResume(ctx); ok {
		return h.resume(ctx, deps, toolRuntime, outer, resumed)
	}
	code, ok := input["code"].(string)
	if !ok || code == "" {
		return nil, errors.New("codemode code must be a non-empty string")
	}
	if len(code) > h.limits.MaxCodeBytes {
		return nil, fmt.Errorf("codemode code exceeds %d bytes", h.limits.MaxCodeBytes)
	}
	step, err := h.executor.Start(ctx, Request{Code: code, Tools: cloneCatalog(h.catalog)})
	if err != nil {
		return nil, fmt.Errorf("start codemode executor: %w", err)
	}
	return h.drive(ctx, deps, toolRuntime, outer, 0, "", step)
}

func (h *runCodeHandler) resume(
	ctx context.Context,
	deps any,
	toolRuntime harnessruntime.ToolRuntime,
	outer agentic.ToolCallContext,
	resumed agentic.ToolResumeContext,
) (any, error) {
	if resumed.Deferral.Kind != DeferralKind {
		return nil, fmt.Errorf("unexpected codemode deferral kind %q", resumed.Deferral.Kind)
	}
	var program deferredProgram
	if err := json.Unmarshal(resumed.Deferral.Payload, &program); err != nil {
		return nil, fmt.Errorf("decode codemode deferral: %w", err)
	}
	if err := h.validateProgram(program, outer.ID); err != nil {
		return nil, err
	}
	var decisions resumePayload
	if err := json.Unmarshal(resumed.Payload, &decisions); err != nil {
		return nil, fmt.Errorf("decode codemode resume payload: %w", err)
	}
	if decisions.Version != wireVersion || decisions.OuterCallID != outer.ID {
		return nil, errors.New("codemode resume payload differs from the deferred outer call")
	}
	results, err := h.completeBatch(ctx, deps, toolRuntime, program, decisions.Resolutions)
	if err != nil {
		return nil, err
	}
	step, err := h.executor.Resume(ctx, append(Checkpoint(nil), program.Checkpoint...), results)
	if err != nil {
		return nil, fmt.Errorf("resume codemode executor: %w", err)
	}
	return h.drive(ctx, deps, toolRuntime, outer, program.Step+1, program.Stdout, step)
}

func (h *runCodeHandler) drive(
	ctx context.Context,
	deps any,
	toolRuntime harnessruntime.ToolRuntime,
	outer agentic.ToolCallContext,
	stepNumber int,
	stdout string,
	step Step,
) (any, error) {
	for {
		validated, err := h.validateStep(step)
		if err != nil {
			return nil, err
		}
		if len(stdout)+len(validated.Stdout) > h.limits.MaxStdoutBytes {
			return nil, fmt.Errorf("codemode stdout exceeds %d bytes", h.limits.MaxStdoutBytes)
		}
		stdout += validated.Stdout
		if validated.Done {
			payload, err := encodeBounded(operationPayload{Version: wireVersion, Step: stepNumber}, h.programLimit(), "codemode completion")
			if err != nil {
				return nil, err
			}
			if err := toolRuntime.Operations.RecordOperation(ctx, harnessruntime.Operation{
				ID:           outer.ID + "/" + strconv.Itoa(stepNumber),
				Kind:         "codemode.execution",
				Phase:        "completed",
				ParentCallID: outer.ID,
				Payload:      payload,
			}); err != nil {
				return nil, err
			}
			return Result{Output: validated.Output, Stdout: stdout}, nil
		}
		if stepNumber >= h.limits.MaxSteps {
			return nil, fmt.Errorf("codemode exceeded %d steps", h.limits.MaxSteps)
		}
		program := deferredProgram{
			Version:     wireVersion,
			OuterCallID: outer.ID,
			Step:        stepNumber,
			Checkpoint:  append(Checkpoint(nil), validated.Checkpoint...),
			Calls:       cloneCalls(validated.Calls),
			Stdout:      stdout,
		}
		if err := h.recordProgram(ctx, toolRuntime, outer.ID, "planned", program); err != nil {
			return nil, err
		}
		calls := toToolUses(program.Calls)
		decision, err := toolRuntime.Capture.ToolGate().EvaluateBatch(ctx, calls)
		if err != nil {
			return nil, fmt.Errorf("evaluate codemode tool batch: %w", err)
		}
		program.Dispositions, program.GateDeferral, err = h.validateDecision(calls, decision)
		if err != nil {
			return nil, err
		}
		if hasSuspension(program.Dispositions) {
			if err := h.recordProgram(ctx, toolRuntime, outer.ID, "suspended", program); err != nil {
				return nil, err
			}
			encoded, err := encodeBounded(program, h.programLimit(), "codemode deferral")
			if err != nil {
				return nil, err
			}
			return agentic.ToolHandlerSuspension{Deferral: agentic.ToolDeferral{
				Kind:    DeferralKind,
				Payload: encoded,
			}}, nil
		}
		results, err := h.completeBatch(ctx, deps, toolRuntime, program, nil)
		if err != nil {
			return nil, err
		}
		step, err = h.executor.Resume(ctx, append(Checkpoint(nil), program.Checkpoint...), results)
		if err != nil {
			return nil, fmt.Errorf("resume codemode executor: %w", err)
		}
		stepNumber++
	}
}

func (h *runCodeHandler) completeBatch(
	ctx context.Context,
	deps any,
	toolRuntime harnessruntime.ToolRuntime,
	program deferredProgram,
	resolutions []resumeResolution,
) ([]CallResult, error) {
	resolved := make(map[string]resumeResolution, len(resolutions))
	for _, resolution := range resolutions {
		if resolution.CallID == "" || resolved[resolution.CallID].CallID != "" {
			return nil, fmt.Errorf("duplicate codemode resolution %q", resolution.CallID)
		}
		resolved[resolution.CallID] = resolution
	}
	results := make([]CallResult, len(program.Calls))
	for index, proposed := range program.Calls {
		call := proposed
		disposition := program.Dispositions[index]
		var result agentic.ToolExecutionResult
		switch disposition.Kind {
		case agentic.ToolDispositionExecute:
			if err := h.recordCall(ctx, toolRuntime, program, call, "started", nil); err != nil {
				return nil, err
			}
			result = h.executeSelected(ctx, deps, hostToolUse(program, call))
		case agentic.ToolDispositionReturn:
			converted, err := fromWireResult(disposition.Result)
			if err != nil {
				return nil, err
			}
			result, err = reidentifyResult(call, hostToolUse(program, call), converted)
			if err != nil {
				return nil, err
			}
		case agentic.ToolDispositionSuspend:
			resolution, ok := resolved[call.ID]
			if !ok {
				return nil, fmt.Errorf("missing codemode resolution %q", call.ID)
			}
			delete(resolved, call.ID)
			switch resolution.Action {
			case agentic.ToolResumeExecute:
				if resolution.OverrideArgs != nil {
					call.Input = cloneMap(resolution.OverrideArgs)
				}
				if err := h.recordCall(ctx, toolRuntime, program, call, "started", nil); err != nil {
					return nil, err
				}
				result = h.executeSelected(ctx, deps, hostToolUse(program, call))
			case agentic.ToolResumeReturn:
				converted, err := fromWireResult(resolution.Result)
				if err != nil {
					return nil, err
				}
				result, err = reidentifyResult(call, hostToolUse(program, call), converted)
				if err != nil {
					return nil, err
				}
			default:
				return nil, fmt.Errorf("invalid codemode resolution action for %q", call.ID)
			}
		default:
			return nil, fmt.Errorf("invalid codemode disposition for %q", call.ID)
		}
		hostCall := hostToolUse(program, call)
		projected, err := toolRuntime.Operations.ProjectToolResult(ctx, hostCall, result)
		if err != nil {
			return nil, err
		}
		if projected.ToolUseID != hostCall.ID || projected.ToolName != hostCall.Name {
			return nil, fmt.Errorf("codemode projected result changed nested call identity %q", call.ID)
		}
		projected = restoreInlineContent(result, projected)
		wire, err := toWireResult(projected, h.limits.MaxValueBytes)
		if err != nil {
			return nil, err
		}
		// Host IDs are session-safe identities for effects and artifacts. The
		// executor continues to see its own stable call ID.
		wire.ID = call.ID
		if err := h.recordCall(ctx, toolRuntime, program, call, "result", wire); err != nil {
			return nil, err
		}
		results[index] = CallResult{
			ID: call.ID, Name: call.Name, Content: wire.Content, IsError: wire.IsError,
		}
	}
	if len(resolved) != 0 {
		return nil, errors.New("codemode resume payload contains unknown resolutions")
	}
	return results, nil
}

// The default result processor canonicalizes even a small structured value to
// the exact JSON string that the model would see. A codemode executor is a
// typed execution host, so retain the original JSON shape when projection did
// not actually change those bytes. Spill previews, redactions, and every other
// transformation differ from the canonical value and remain authoritative.
func restoreInlineContent(original, projected agentic.ToolExecutionResult) agentic.ToolExecutionResult {
	formatted, ok := projected.Content.(string)
	if ok && formatted == agentic.FormatToolResult(original.Content) {
		projected.Content = original.Content
	}
	return projected
}

func (h *runCodeHandler) executeSelected(ctx context.Context, deps any, call agentic.ToolUse) agentic.ToolExecutionResult {
	callCtx := agentic.WithToolCallContext(ctx, agentic.ToolCallContext{ID: call.ID, Name: call.Name, Attempt: 1})
	result, err := h.registry.Execute(callCtx, call, deps)
	if err != nil {
		return agentic.ToolExecutionResult{
			ToolUseID: call.ID, ToolName: call.Name, Content: err.Error(), IsError: true, Error: err,
		}
	}
	return result
}

func hostToolUse(program deferredProgram, call Call) agentic.ToolUse {
	return agentic.ToolUse{
		ID:    nestedCallID(program.OuterCallID, program.Step, call.ID),
		Name:  call.Name,
		Input: cloneMap(call.Input),
	}
}

// nestedCallID is length-prefixed so opaque provider IDs containing separators
// cannot alias a call from another outer invocation or executor step.
func nestedCallID(outerID string, step int, executorID string) string {
	return "codemode:" + strconv.Itoa(len(outerID)) + ":" + outerID + ":" +
		strconv.Itoa(step) + ":" + strconv.Itoa(len(executorID)) + ":" + executorID
}

func reidentifyResult(call Call, hostCall agentic.ToolUse, result agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error) {
	if result.ToolUseID != call.ID || result.ToolName != call.Name {
		return agentic.ToolExecutionResult{}, fmt.Errorf("codemode result identity differs for call %q", call.ID)
	}
	result.ToolUseID = hostCall.ID
	return result, nil
}

func (h *runCodeHandler) recordProgram(
	ctx context.Context,
	runtime harnessruntime.ToolRuntime,
	outerID string,
	phase string,
	program deferredProgram,
) error {
	payload, err := encodeBounded(operationPayload{Version: wireVersion, Step: program.Step, Program: &program}, h.programLimit(), "codemode operation")
	if err != nil {
		return err
	}
	return runtime.Operations.RecordOperation(ctx, harnessruntime.Operation{
		ID: outerID + "/" + strconv.Itoa(program.Step), Kind: "codemode.execution",
		Phase: phase, ParentCallID: outerID, Payload: payload,
	})
}

func (h *runCodeHandler) recordCall(
	ctx context.Context,
	runtime harnessruntime.ToolRuntime,
	program deferredProgram,
	call Call,
	phase string,
	result *wireResult,
) error {
	payload, err := encodeBounded(operationPayload{
		Version: wireVersion, Step: program.Step, Call: &call, Result: result,
	}, h.programLimit(), "codemode call operation")
	if err != nil {
		return err
	}
	return runtime.Operations.RecordOperation(ctx, harnessruntime.Operation{
		ID:   program.OuterCallID + "/" + strconv.Itoa(program.Step) + "/" + call.ID,
		Kind: "codemode.call", Phase: phase, ParentCallID: program.OuterCallID, Payload: payload,
	})
}

func (h *runCodeHandler) validateStep(step Step) (Step, error) {
	if step.Done == (len(step.Calls) > 0) {
		return Step{}, errors.New("codemode step must be done or contain calls, exclusively")
	}
	if len(step.Stdout) > h.limits.MaxStdoutBytes {
		return Step{}, fmt.Errorf("codemode step stdout exceeds %d bytes", h.limits.MaxStdoutBytes)
	}
	if step.Done {
		if len(step.Checkpoint) != 0 {
			return Step{}, errors.New("completed codemode step must not contain a checkpoint")
		}
		output, err := cloneJSON(step.Output, h.limits.MaxValueBytes, "codemode output")
		if err != nil {
			return Step{}, err
		}
		return Step{Done: true, Output: output, Stdout: step.Stdout}, nil
	}
	if step.Output != nil || len(step.Checkpoint) == 0 || len(step.Checkpoint) > h.limits.MaxCheckpointBytes {
		return Step{}, errors.New("nonterminal codemode step requires a bounded checkpoint and no output")
	}
	if len(step.Calls) > h.limits.MaxCallsPerStep {
		return Step{}, fmt.Errorf("codemode step exceeds %d calls", h.limits.MaxCallsPerStep)
	}
	result := Step{Checkpoint: append(Checkpoint(nil), step.Checkpoint...), Stdout: step.Stdout}
	result.Calls = make([]Call, len(step.Calls))
	seen := make(map[string]bool, len(step.Calls))
	for index, call := range step.Calls {
		if strings.TrimSpace(call.ID) == "" || seen[call.ID] || !identifierPattern.MatchString(call.Name) || !h.registry.Has(call.Name) {
			return Step{}, fmt.Errorf("invalid or duplicate codemode call %q to %q", call.ID, call.Name)
		}
		seen[call.ID] = true
		encoded, err := encodeBounded(call.Input, h.limits.MaxValueBytes, "codemode call input")
		if err != nil {
			return Step{}, err
		}
		var input map[string]any
		if err := json.Unmarshal(encoded, &input); err != nil {
			return Step{}, err
		}
		result.Calls[index] = Call{ID: call.ID, Name: call.Name, Input: input}
	}
	return result, nil
}

func (h *runCodeHandler) validateDecision(
	calls []agentic.ToolUse,
	decision agentic.ToolBatchDecision,
) ([]wireDisposition, agentic.ToolDeferral, error) {
	if len(decision.Calls) != len(calls) {
		return nil, agentic.ToolDeferral{}, fmt.Errorf("codemode gate returned %d dispositions for %d calls", len(decision.Calls), len(calls))
	}
	result := make([]wireDisposition, len(calls))
	hasSuspend := false
	for index, disposition := range decision.Calls {
		switch disposition.Kind {
		case agentic.ToolDispositionExecute:
			if disposition.Result != nil {
				return nil, agentic.ToolDeferral{}, fmt.Errorf("codemode executable call %q has a result", calls[index].ID)
			}
		case agentic.ToolDispositionReturn:
			if disposition.Result == nil || disposition.Result.ToolUseID != calls[index].ID || disposition.Result.ToolName != calls[index].Name {
				return nil, agentic.ToolDeferral{}, fmt.Errorf("codemode returned call %q has invalid identity", calls[index].ID)
			}
			wire, err := toWireResult(*disposition.Result, h.limits.MaxValueBytes)
			if err != nil {
				return nil, agentic.ToolDeferral{}, err
			}
			result[index].Result = wire
		case agentic.ToolDispositionSuspend:
			if disposition.Result != nil || disposition.Continue {
				return nil, agentic.ToolDeferral{}, fmt.Errorf("codemode suspended call %q is invalid", calls[index].ID)
			}
			hasSuspend = true
		default:
			return nil, agentic.ToolDeferral{}, fmt.Errorf("codemode call %q has an invalid gate disposition", calls[index].ID)
		}
		result[index].Kind = disposition.Kind
		result[index].Continue = disposition.Continue
	}
	if hasSuspend != (decision.Deferral != nil) || (hasSuspend && decision.Deferral.Kind == "") {
		return nil, agentic.ToolDeferral{}, errors.New("codemode gate suspension has an invalid deferral")
	}
	var deferral agentic.ToolDeferral
	if decision.Deferral != nil {
		deferral = *decision.Deferral
		deferral.Payload = append([]byte(nil), decision.Deferral.Payload...)
	}
	return result, deferral, nil
}

func (h *runCodeHandler) validateProgram(program deferredProgram, outerID string) error {
	if program.Version != wireVersion || program.OuterCallID != outerID || program.Step < 0 ||
		program.Step >= h.limits.MaxSteps || len(program.Calls) == 0 ||
		len(program.Calls) != len(program.Dispositions) || len(program.Checkpoint) == 0 ||
		len(program.Checkpoint) > h.limits.MaxCheckpointBytes || len(program.Stdout) > h.limits.MaxStdoutBytes {
		return errors.New("invalid codemode deferred program")
	}
	validated, err := h.validateStep(Step{Checkpoint: program.Checkpoint, Calls: program.Calls})
	if err != nil {
		return err
	}
	program.Calls = validated.Calls
	if !hasSuspension(program.Dispositions) {
		return errors.New("codemode deferred program has no suspended calls")
	}
	return nil
}

func (h *runCodeHandler) programLimit() int {
	return h.limits.MaxCheckpointBytes + h.limits.MaxStdoutBytes +
		h.limits.MaxCallsPerStep*h.limits.MaxValueBytes + (64 << 10)
}

func hasSuspension(dispositions []wireDisposition) bool {
	for _, disposition := range dispositions {
		if disposition.Kind == agentic.ToolDispositionSuspend {
			return true
		}
	}
	return false
}

func toToolUses(calls []Call) []agentic.ToolUse {
	result := make([]agentic.ToolUse, len(calls))
	for index, call := range calls {
		result[index] = agentic.ToolUse{ID: call.ID, Name: call.Name, Input: cloneMap(call.Input)}
	}
	return result
}

func cloneCatalog(catalog []Tool) []Tool {
	result := make([]Tool, len(catalog))
	for index, tool := range catalog {
		result[index] = tool
		result[index].Parameters = cloneMap(tool.Parameters)
	}
	return result
}
