package codemode

import (
	"encoding/json"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"

	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type resumePlanner struct {
	toolName string
	limits   Limits
}

func (p resumePlanner) PlanResume(
	suspension agentic.Suspension,
	request harnessruntime.ResumeRequest,
) ([]agentic.ToolResumeDecision, error) {
	if request.SuspensionID == "" || request.SuspensionID != suspension.ID {
		return nil, fmt.Errorf("%w: suspension ID differs", harnessruntime.ErrInvalidResumeRequest)
	}
	if request.Prompt != nil && request.Prompt.Role != agentic.RoleUser {
		return nil, fmt.Errorf("%w: resume prompt must be a user message", harnessruntime.ErrInvalidResumeRequest)
	}
	frontier, err := harnessruntime.InspectToolSuspension(suspension, DeferralKind)
	if err != nil {
		return nil, err
	}
	if !frontier.HandlerSuspension || len(frontier.ExecutableCallIDs) != 1 {
		return nil, fmt.Errorf("%w: codemode requires one handler suspension", harnessruntime.ErrInvalidResumeRequest)
	}
	outerID := frontier.ExecutableCallIDs[0]
	outerName := ""
	for _, call := range frontier.Calls {
		if call.ID == outerID {
			outerName = call.Name
			break
		}
	}
	if outerName != p.toolName {
		return nil, fmt.Errorf("%w: suspended call is not %s", harnessruntime.ErrInvalidResumeRequest, p.toolName)
	}
	var program deferredProgram
	if err := json.Unmarshal(frontier.Deferral.Payload, &program); err != nil {
		return nil, fmt.Errorf("%w: decode codemode deferral: %v", harnessruntime.ErrInvalidResumeRequest, err)
	}
	if program.Version != wireVersion || program.OuterCallID != outerID || program.Step < 0 ||
		program.Step >= p.limits.MaxSteps || len(program.Calls) == 0 ||
		len(program.Calls) != len(program.Dispositions) || len(program.Checkpoint) == 0 ||
		len(program.Checkpoint) > p.limits.MaxCheckpointBytes || len(program.Stdout) > p.limits.MaxStdoutBytes {
		return nil, fmt.Errorf("%w: malformed codemode deferral", harnessruntime.ErrInvalidResumeRequest)
	}
	required := make(map[string]Call)
	seenCalls := make(map[string]bool, len(program.Calls))
	for index, call := range program.Calls {
		if call.ID == "" || call.Name == "" || seenCalls[call.ID] {
			return nil, fmt.Errorf("%w: invalid deferred call %q", harnessruntime.ErrInvalidResumeRequest, call.ID)
		}
		seenCalls[call.ID] = true
		switch disposition := program.Dispositions[index]; disposition.Kind {
		case agentic.ToolDispositionExecute:
			if disposition.Result != nil {
				return nil, fmt.Errorf("%w: executable call has a result", harnessruntime.ErrInvalidResumeRequest)
			}
		case agentic.ToolDispositionReturn:
			if disposition.Result == nil || disposition.Result.ID != call.ID || disposition.Result.Name != call.Name {
				return nil, fmt.Errorf("%w: invalid returned call %q", harnessruntime.ErrInvalidResumeRequest, call.ID)
			}
		case agentic.ToolDispositionSuspend:
			required[call.ID] = call
		default:
			return nil, fmt.Errorf("%w: invalid deferred disposition", harnessruntime.ErrInvalidResumeRequest)
		}
	}
	if len(required) == 0 {
		return nil, fmt.Errorf("%w: codemode deferral has no required resolutions", harnessruntime.ErrInvalidResumeRequest)
	}
	resolved := make(map[string]bool, len(request.Resolutions))
	resume := resumePayload{Version: wireVersion, OuterCallID: outerID}
	for _, resolution := range request.Resolutions {
		call, ok := required[resolution.CallID]
		if !ok {
			return nil, fmt.Errorf("%w: unknown resolution %q", harnessruntime.ErrInvalidResumeRequest, resolution.CallID)
		}
		if resolved[resolution.CallID] {
			return nil, fmt.Errorf("%w: duplicate resolution %q", harnessruntime.ErrInvalidResumeRequest, resolution.CallID)
		}
		resolved[resolution.CallID] = true
		planned := resumeResolution{CallID: call.ID}
		switch resolution.Action {
		case harnessruntime.ResolutionApprove:
			planned.Action = agentic.ToolResumeExecute
			if resolution.OverrideArgs != nil {
				encoded, err := encodeBounded(resolution.OverrideArgs, p.limits.MaxValueBytes, "codemode override arguments")
				if err != nil {
					return nil, fmt.Errorf("%w: %v", harnessruntime.ErrInvalidResumeRequest, err)
				}
				if err := json.Unmarshal(encoded, &planned.OverrideArgs); err != nil {
					return nil, fmt.Errorf("%w: invalid override arguments", harnessruntime.ErrInvalidResumeRequest)
				}
			}
		case harnessruntime.ResolutionDeny:
			reason := resolution.Reason
			if reason == "" {
				reason = "denied by operator"
			}
			message := "Tool call denied: " + reason
			planned.Action = agentic.ToolResumeReturn
			planned.Result = &wireResult{ID: call.ID, Name: call.Name, Content: message, IsError: true, Error: message}
		case harnessruntime.ResolutionExternalResult:
			content, err := cloneJSON(resolution.Result, p.limits.MaxValueBytes, "codemode external result")
			if err != nil {
				return nil, fmt.Errorf("%w: %v", harnessruntime.ErrInvalidResumeRequest, err)
			}
			planned.Action = agentic.ToolResumeReturn
			planned.Result = &wireResult{ID: call.ID, Name: call.Name, Content: content}
		default:
			return nil, fmt.Errorf("%w: invalid action for %q", harnessruntime.ErrInvalidResumeRequest, call.ID)
		}
		resume.Resolutions = append(resume.Resolutions, planned)
	}
	if len(resolved) != len(required) {
		return nil, fmt.Errorf("%w: got %d resolutions for %d required calls", harnessruntime.ErrInvalidResumeRequest, len(resolved), len(required))
	}
	if len(resume.Resolutions) == 0 {
		return nil, errors.New("codemode resume unexpectedly has no resolutions")
	}
	maximum := p.limits.MaxCheckpointBytes + p.limits.MaxStdoutBytes +
		p.limits.MaxCallsPerStep*p.limits.MaxValueBytes + (64 << 10)
	payload, err := encodeBounded(resume, maximum, "codemode resume payload")
	if err != nil {
		return nil, fmt.Errorf("%w: %v", harnessruntime.ErrInvalidResumeRequest, err)
	}
	return []agentic.ToolResumeDecision{{
		CallID:  outerID,
		Action:  agentic.ToolResumeExecute,
		Payload: payload,
	}}, nil
}
