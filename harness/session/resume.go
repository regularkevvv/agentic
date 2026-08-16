package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/repair"
	"github.com/regularkevvv/agentic/harness/store"
)

// Resume resolves one durable suspension. The complete request is validated
// and persisted before the Driver can re-enter any handler.
func (s *Session[O]) Resume(ctx context.Context, request ResumeRequest) (*agentic.Execution[O], error) {
	accepted, err := s.prepareResume(ctx, request, ctx)
	if err != nil {
		return nil, err
	}
	return s.driveResumed(accepted)
}

// acceptedResume is the single-use product of prepareResume: one durably
// accepted, not-yet-driven resolution. indeterminate selects the recovery
// continuation drive instead of Driver.Resume.
type acceptedResume[O any] struct {
	runID         string
	runCtx        context.Context
	history       []agentic.Message
	suspension    *agentic.Suspension
	decisions     []agentic.ToolResumeDecision
	prompt        *agentic.Message
	commit        store.Commit
	limits        *agentic.UsageLimits
	indeterminate bool
	consumed      atomic.Bool
}

func (a *acceptedResume[O]) consume() error {
	if !a.consumed.CompareAndSwap(false, true) {
		return errors.New("accepted resume was already driven")
	}
	return nil
}

// prepareResume is the durable acceptance half of Resume: it validates the
// request, persists resolution.accepted, transitions to Running, and
// publishes the committed acceptance. acceptCtx governs the acceptance
// journal append; runParent is the parent of the run context created inside
// the same locked acceptance window (the legacy Resume passes its caller
// context as both). It dispatches to prepareResumeIndeterminate through the
// same suspension-kind branch Resume has always used.
func (s *Session[O]) prepareResume(
	acceptCtx context.Context,
	request ResumeRequest,
	runParent context.Context,
) (*acceptedResume[O], error) {
	return s.prepareResumeWithCommand(acceptCtx, request, runParent, nil)
}

func (s *Session[O]) prepareResumeWithCommand(
	acceptCtx context.Context,
	request ResumeRequest,
	runParent context.Context,
	command *loopCommandAcceptedPayload,
) (*acceptedResume[O], error) {
	s.mu.Lock()
	if err := s.resumeErrorLocked(request); err != nil {
		s.mu.Unlock()
		return nil, err
	}
	suspension := cloneSuspension(s.suspension)
	if suspension.Kind == "harness.recovery.indeterminate" {
		s.mu.Unlock()
		return s.prepareResumeIndeterminateWithCommand(acceptCtx, suspension, request, runParent, command)
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
	pendingEntries := []store.PendingEntry{entry}
	if command != nil {
		accepted := *command
		accepted.RunID = s.currentRunID()
		commandEntry, encodeErr := pending(s.codec, kindCommandAccepted, accepted)
		if encodeErr != nil {
			return nil, encodeErr
		}
		pendingEntries = append(pendingEntries, commandEntry)
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
	commit, appendErr := s.journal.Append(acceptCtx, s.cursor, pendingEntries...)
	if appendErr != nil {
		s.mu.Unlock()
		return nil, appendErr
	}
	runID := s.run.id
	runCtx, cancel := context.WithCancel(runParent)
	s.runCancel = cancel
	s.run.resumeInProgress = true
	s.run.resumeEventSeen = false
	s.run.publicNewStart = len(s.run.expected)
	s.cursor = commit.Cursor
	s.transitionLocked(Running)
	s.mu.Unlock()
	s.publishOwn(commit.Entries, agentic.EventAuthoritative)

	var prompt *agentic.Message
	if request.Prompt != nil {
		copy := cloneMessages([]agentic.Message{*request.Prompt})[0]
		prompt = &copy
	}
	return &acceptedResume[O]{
		runID:      runID,
		runCtx:     runCtx,
		history:    history,
		suspension: suspension,
		decisions:  decisions,
		prompt:     prompt,
		commit:     commit,
		limits:     limits,
	}, nil
}

// driveResumed is the execution half of Resume. It re-enters the driver
// exactly once and settles through the existing finish path. Like
// driveAccepted, it has no Closed pre-check: single-use consumption plus the
// view's close-time goroutine join make a post-Close drive unreachable.
func (s *Session[O]) driveResumed(accepted *acceptedResume[O]) (*agentic.Execution[O], error) {
	if accepted.indeterminate {
		return s.driveResumedIndeterminate(accepted)
	}
	if err := accepted.consume(); err != nil {
		return nil, err
	}
	runCtx := s.withToolRuntime(accepted.runCtx)
	execution, runErr := s.driver.Resume(runCtx, agentic.ResumeInput{
		History:    accepted.history,
		Suspension: *accepted.suspension,
		Decisions:  accepted.decisions,
		Prompt:     accepted.prompt,
	}, s.runOptions(accepted.runID, accepted.limits)...)
	return s.finishExecution(execution, runErr)
}

// driveResumedIndeterminate continues the recovery-opened run through
// Driver.Drive in DriveContinue mode and settles through the finish path.
func (s *Session[O]) driveResumedIndeterminate(accepted *acceptedResume[O]) (*agentic.Execution[O], error) {
	if err := accepted.consume(); err != nil {
		return nil, err
	}
	runCtx := s.withToolRuntime(accepted.runCtx)
	execution, runErr := s.driver.Drive(runCtx, agentic.DriveInput{
		Mode:    agentic.DriveContinue,
		History: accepted.history,
	}, s.runOptions(accepted.runID, accepted.limits)...)
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

// resumeIndeterminate preserves the legacy fused acceptance+drive
// composition of the indeterminate-recovery branch: acceptance and the run
// context both derive from the caller context, exactly as before the split.
func (s *Session[O]) resumeIndeterminate(
	ctx context.Context,
	suspension *agentic.Suspension,
	request ResumeRequest,
) (*agentic.Execution[O], error) {
	accepted, err := s.prepareResumeIndeterminate(ctx, suspension, request, ctx)
	if err != nil {
		return nil, err
	}
	return s.driveResumedIndeterminate(accepted)
}

// prepareResumeIndeterminate is the durable acceptance half of the
// indeterminate-recovery resume: it validates the operator resolutions,
// closes the old run, opens the continuation run in one batch, and publishes
// the committed acceptance. The returned acceptedResume drives through
// Driver.Drive in DriveContinue mode.
func (s *Session[O]) prepareResumeIndeterminate(
	acceptCtx context.Context,
	suspension *agentic.Suspension,
	request ResumeRequest,
	runParent context.Context,
) (*acceptedResume[O], error) {
	return s.prepareResumeIndeterminateWithCommand(acceptCtx, suspension, request, runParent, nil)
}

func (s *Session[O]) prepareResumeIndeterminateWithCommand(
	acceptCtx context.Context,
	suspension *agentic.Suspension,
	request ResumeRequest,
	runParent context.Context,
	command *loopCommandAcceptedPayload,
) (*acceptedResume[O], error) {
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
	instructions := s.run.instructions
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
		ID:           runID,
		Mode:         "continue",
		Recovery:     true,
		Limits:       cloneLimitsPointer(limits),
		Instructions: instructions,
	})
	if command != nil {
		accepted := *command
		accepted.RunID = runID
		batch.Add(kindCommandAccepted, accepted)
	}
	pendingEntries, encodeErr := batch.Result()
	if encodeErr != nil {
		s.mu.Unlock()
		return nil, encodeErr
	}
	commit, appendErr := s.journal.Append(acceptCtx, s.cursor, pendingEntries...)
	if appendErr != nil {
		s.mu.Unlock()
		return nil, appendErr
	}
	s.messages = append(s.messages, added...)
	if request.Prompt != nil {
		s.messages = append(s.messages, cloneMessages([]agentic.Message{*request.Prompt})[0])
	}
	history := providerHistory(s.messages, s.contextMarkers)
	runCtx, cancel := context.WithCancel(runParent)
	s.runCancel = cancel
	s.run = &activeRun{
		id:                 runID,
		mode:               agentic.DriveContinue,
		history:            history,
		contextMarkerCount: len(s.contextMarkers),
		limits:             cloneLimitsPointer(limits),
		instructions:       instructions,
	}
	s.suspension = nil
	s.cursor = commit.Cursor
	s.transitionLocked(Running)
	s.mu.Unlock()
	s.publishOwnByKind(commit.Entries)
	return &acceptedResume[O]{
		runID:         runID,
		runCtx:        runCtx,
		history:       history,
		commit:        commit,
		limits:        limits,
		indeterminate: true,
	}, nil
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
