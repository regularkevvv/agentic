package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/repair"
)

// Resume resolves one durable suspension. The complete request is validated
// and persisted before the Driver can re-enter any handler.
func (s *Session[O]) Resume(ctx context.Context, request ResumeRequest) (*agentic.Execution[O], error) {
	s.mu.Lock()
	if err := s.resumeErrorLocked(request); err != nil {
		s.mu.Unlock()
		return nil, err
	}
	suspension := cloneSuspension(s.suspension)
	if suspension.Kind == "harness.recovery.indeterminate" {
		s.mu.Unlock()
		return s.resumeIndeterminate(ctx, suspension, request)
	}
	history := append(cloneMessages(s.run.history), cloneMessages(s.run.expected)...)
	limits := cloneLimitsPointer(s.run.limits)
	s.mu.Unlock()

	decisions, err := s.resume.PlanResume(*suspension, request)
	if err != nil {
		return nil, err
	}
	payload := resolutionAcceptedPayload{SuspensionID: suspension.ID, Request: cloneResumeRequest(request)}
	entry, err := pending(s.codec, kindResolutionAccepted, payload)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	if err := s.resumeErrorLocked(request); err != nil {
		s.mu.Unlock()
		return nil, err
	}
	if s.suspension.ID != suspension.ID {
		s.mu.Unlock()
		return nil, fmt.Errorf("%w: suspension changed", ErrInvalidResumeRequest)
	}
	commit, appendErr := s.journal.Append(ctx, s.cursor, entry)
	if appendErr != nil {
		s.mu.Unlock()
		return nil, appendErr
	}
	runCtx, cancel := context.WithCancel(ctx)
	s.runCancel = cancel
	s.run.resumeInProgress = true
	s.run.resumeEventSeen = false
	s.run.publicNewStart = len(s.run.expected)
	s.cursor = commit.Cursor
	s.transitionLocked(Running)
	s.mu.Unlock()
	s.publishOwn(commit.Entries, agentic.EventAuthoritative)

	runCtx = s.withToolRuntime(runCtx)
	var prompt *agentic.Message
	if request.Prompt != nil {
		copy := cloneMessages([]agentic.Message{*request.Prompt})[0]
		prompt = &copy
	}
	execution, runErr := s.driver.Resume(runCtx, agentic.ResumeInput{
		History:    history,
		Suspension: *suspension,
		Decisions:  decisions,
		Prompt:     prompt,
	}, s.runOptions(limits)...)
	return s.finishExecution(execution, runErr)
}

func (s *Session[O]) resumeErrorLocked(request ResumeRequest) error {
	switch s.state {
	case Faulted:
		return &FaultError{SessionID: s.id, Cause: s.fault}
	case Closed:
		return ErrSessionClosed
	case Suspended:
	default:
		return ErrNotRunning
	}
	if s.suspension == nil || s.run == nil {
		return fmt.Errorf("%w: suspended state has no durable frontier", ErrInvalidResumeRequest)
	}
	if request.SuspensionID == "" || request.SuspensionID != s.suspension.ID {
		return fmt.Errorf("%w: suspension ID differs", ErrInvalidResumeRequest)
	}
	if request.Prompt != nil && request.Prompt.Role != agentic.RoleUser {
		return ErrInvalidMessage
	}
	return nil
}

func (s *Session[O]) resumeIndeterminate(
	ctx context.Context,
	suspension *agentic.Suspension,
	request ResumeRequest,
) (*agentic.Execution[O], error) {
	var payload recoverySuspensionPayload
	if err := json.Unmarshal(suspension.Payload, &payload); err != nil || payload.Version != 1 {
		if err == nil {
			err = errors.New("unsupported recovery suspension version")
		}
		return nil, fmt.Errorf("%w: %v", ErrInvalidResumeRequest, err)
	}
	required := make(map[string]repair.PendingCall, len(payload.Calls))
	for _, call := range payload.Calls {
		if call.ID == "" || call.State != repair.PendingIndeterminate || required[call.ID].ID != "" {
			return nil, fmt.Errorf("%w: malformed indeterminate call set", ErrInvalidResumeRequest)
		}
		required[call.ID] = call
	}
	resolutions := make(map[string]ToolResolution, len(request.Resolutions))
	for _, resolution := range request.Resolutions {
		if required[resolution.CallID].ID == "" {
			return nil, fmt.Errorf("%w: unknown resolution %q", ErrInvalidResumeRequest, resolution.CallID)
		}
		if _, exists := resolutions[resolution.CallID]; exists {
			return nil, fmt.Errorf("%w: duplicate resolution %q", ErrInvalidResumeRequest, resolution.CallID)
		}
		if resolution.Action == ResolutionApprove {
			return nil, fmt.Errorf("%w: %s", ErrIndeterminateTool, resolution.CallID)
		}
		if resolution.Action != ResolutionDeny && resolution.Action != ResolutionExternalResult {
			return nil, fmt.Errorf("%w: invalid action for %q", ErrInvalidResumeRequest, resolution.CallID)
		}
		resolutions[resolution.CallID] = resolution
	}
	if len(resolutions) != len(required) {
		return nil, fmt.Errorf("%w: got %d resolutions for %d indeterminate calls", ErrInvalidResumeRequest, len(resolutions), len(required))
	}

	s.mu.Lock()
	if err := s.resumeErrorLocked(request); err != nil {
		s.mu.Unlock()
		return nil, err
	}
	if s.suspension.ID != suspension.ID {
		s.mu.Unlock()
		return nil, fmt.Errorf("%w: suspension changed", ErrInvalidResumeRequest)
	}
	open, err := repair.InspectFrontier(s.messages)
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}
	pendingCalls := repair.PendingCalls{Calls: make([]repair.PendingCall, len(open))}
	for index, call := range open {
		state := repair.PendingPlanned
		if required[call.ID].ID != "" {
			state = repair.PendingIndeterminate
		}
		pendingCalls.Calls[index] = repair.PendingCall{ID: call.ID, Name: call.Name, State: state}
	}
	repaired, err := repair.Process(s.messages, repair.CloseInterruptedFrontier, pendingCalls)
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}
	added := cloneMessages(repaired[len(s.messages):])
	for index, message := range added {
		results := message.GetToolResults()
		if len(results) != 1 {
			s.mu.Unlock()
			return nil, fmt.Errorf("%w: recovery repair produced an invalid result", ErrCommitProjectionMismatch)
		}
		resolution, ok := resolutions[results[0].ToolUseID]
		if !ok {
			continue
		}
		call := required[results[0].ToolUseID]
		switch resolution.Action {
		case ResolutionDeny:
			reason := resolution.Reason
			if reason == "" {
				reason = "denied by operator after indeterminate execution"
			}
			added[index] = agentic.NewToolResultMessageFor(call.ID, call.Name, reason, true)
		case ResolutionExternalResult:
			added[index] = agentic.NewToolResultMessageFor(
				call.ID,
				call.Name,
				agentic.FormatToolResult(resolution.Result),
				false,
			)
		}
	}
	var limits *agentic.UsageLimits
	if s.budget != nil {
		remaining, remainingErr := remainingLimits(*s.budget, s.usage)
		if remainingErr != nil {
			s.mu.Unlock()
			return nil, remainingErr
		}
		limits = &remaining
	}
	runID, idErr := s.ids.New("run")
	if idErr != nil {
		s.mu.Unlock()
		return nil, idErr
	}
	batch := newEntryBatch(s.codec, len(added)+4)
	batch.Add(kindResolutionAccepted, resolutionAcceptedPayload{
		SuspensionID: suspension.ID,
		Request:      cloneResumeRequest(request),
	})
	batch.Add(kindRunClosed, runClosedPayload{
		ID:     s.run.id,
		Status: agentic.ExecutionInterrupted,
		Error:  "indeterminate tool frontier resolved by operator",
	})
	for _, message := range added {
		batch.Add(kindMessage, messagePayload{Message: message, Source: "recovery_resolution"})
	}
	if request.Prompt != nil {
		batch.Add(kindMessage, messagePayload{Message: *request.Prompt, Source: "resume_prompt"})
	}
	batch.Add(kindRunOpened, runOpenedPayload{
		ID:       runID,
		Mode:     "continue",
		Recovery: true,
		Limits:   cloneLimitsPointer(limits),
	})
	pendingEntries, encodeErr := batch.Result()
	if encodeErr != nil {
		s.mu.Unlock()
		return nil, encodeErr
	}
	commit, appendErr := s.journal.Append(ctx, s.cursor, pendingEntries...)
	if appendErr != nil {
		s.mu.Unlock()
		return nil, appendErr
	}
	s.messages = append(s.messages, added...)
	if request.Prompt != nil {
		s.messages = append(s.messages, cloneMessages([]agentic.Message{*request.Prompt})[0])
	}
	history := providerHistory(s.messages, s.contextMarkers)
	runCtx, cancel := context.WithCancel(ctx)
	s.runCancel = cancel
	s.run = &activeRun{
		id:                 runID,
		mode:               agentic.DriveContinue,
		history:            history,
		contextMarkerCount: len(s.contextMarkers),
		limits:             cloneLimitsPointer(limits),
	}
	s.suspension = nil
	s.cursor = commit.Cursor
	s.transitionLocked(Running)
	s.mu.Unlock()
	s.publishOwnByKind(commit.Entries)

	runCtx = s.withToolRuntime(runCtx)
	execution, runErr := s.driver.Drive(runCtx, agentic.DriveInput{
		Mode:    agentic.DriveContinue,
		History: history,
	}, s.runOptions(limits)...)
	return s.finishExecution(execution, runErr)
}

func cloneResumeRequest(request ResumeRequest) ResumeRequest {
	result := request
	result.Resolutions = make([]ToolResolution, len(request.Resolutions))
	for index, resolution := range request.Resolutions {
		result.Resolutions[index] = resolution
		result.Resolutions[index].OverrideArgs = cloneAnyMap(resolution.OverrideArgs)
	}
	if request.Prompt != nil {
		copy := cloneMessages([]agentic.Message{*request.Prompt})[0]
		result.Prompt = &copy
	}
	return result
}

func cloneAnyMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	result := make(map[string]any, len(value))
	for key, item := range value {
		result[key] = cloneAnyValue(item)
	}
	return result
}

func cloneAnyValue(value any) any {
	switch current := value.(type) {
	case map[string]any:
		return cloneAnyMap(current)
	case []any:
		result := make([]any, len(current))
		for index, item := range current {
			result[index] = cloneAnyValue(item)
		}
		return result
	case []string:
		return append([]string(nil), current...)
	case []byte:
		return append([]byte(nil), current...)
	default:
		return current
	}
}
