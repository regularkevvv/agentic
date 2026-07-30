package session

import (
	"context"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/repair"
	"github.com/regularkevvv/agentic/harness/store"
)

func (s *Session[O]) finishExecution(execution *agentic.Execution[O], runErr error) (*agentic.Execution[O], error) {
	s.mu.Lock()
	if s.runCancel != nil {
		s.runCancel()
		s.runCancel = nil
	}
	if s.state == Faulted {
		fault := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return execution, fault
	}
	interrupted := s.state == Interrupting || (execution != nil && execution.Status == agentic.ExecutionInterrupted)
	s.mu.Unlock()
	if interrupted {
		if err := s.finishInterrupt(execution); err != nil {
			return execution, err
		}
		if runErr == nil {
			runErr = context.Canceled
		}
		return execution, runErr
	}

	s.mu.Lock()
	if s.run == nil {
		s.mu.Unlock()
		return execution, errors.New("driver returned without an active session run")
	}
	if s.run.resumeInProgress && !s.run.resumeEventSeen && isResumeValidationError(runErr) {
		s.run.resumeInProgress = false
		s.transitionLocked(Suspended)
		s.mu.Unlock()
		return execution, runErr
	}
	status := agentic.ExecutionFailed
	if execution != nil {
		status = execution.Status
	}
	if status == agentic.ExecutionSuspended {
		if execution == nil || execution.Result == nil || execution.Suspension == nil {
			fault := s.persistFaultLocked(fmt.Errorf("%w: suspended execution is incomplete", ErrCommitProjectionMismatch))
			s.mu.Unlock()
			return execution, fault
		}
		system, projectionErr := s.validateProjectionLocked(execution)
		if projectionErr != nil {
			fault := s.persistFaultLocked(projectionErr)
			s.mu.Unlock()
			return execution, fault
		}
		batch := newEntryBatch(s.codec, 2)
		delta, deltaErr := usageDelta(execution.Result.Usage, s.run.lastUsage)
		if deltaErr != nil {
			fault := s.persistFaultLocked(deltaErr)
			s.mu.Unlock()
			return execution, fault
		}
		nextUsage := s.usage
		usageChanged := !usageEmpty(delta)
		if usageChanged {
			nextUsage = addUsage(s.usage, delta)
			batch.Add(kindUsageCommitted, usagePayload{Run: execution.Result.Usage, Session: nextUsage})
		}
		if system != nil {
			batch.Add(kindSystemMessage, messagePayload{Message: *system, Source: "driver_system"})
		}
		pendingEntries, encodeErr := batch.Result()
		if encodeErr != nil {
			fault := s.persistFaultLocked(encodeErr)
			s.mu.Unlock()
			return execution, fault
		}
		var committed []store.Entry
		if len(pendingEntries) > 0 {
			commit, appendErr := s.journal.Append(context.Background(), s.cursor, pendingEntries...)
			if appendErr != nil {
				cancel := s.faultLocked(appendErr)
				s.mu.Unlock()
				if cancel != nil {
					cancel()
				}
				return execution, &FaultError{SessionID: s.id, Cause: appendErr}
			}
			s.cursor = commit.Cursor
			committed = commit.Entries
		}
		if usageChanged {
			s.usage = nextUsage
		}
		s.run.lastUsage = cloneUsage(execution.Result.Usage)
		if system != nil {
			shiftContextMarkers(s.contextMarkers, 1)
			s.messages = append([]agentic.Message{*system}, s.messages...)
			s.run.history = append([]agentic.Message{*system}, s.run.history...)
		}
		s.suspension = cloneSuspension(execution.Suspension)
		s.run.resumeInProgress = false
		s.transitionLocked(Suspended)
		s.runCancel = nil
		s.mu.Unlock()
		s.publishOwnByKind(committed)
		return execution, runErr
	}
	if s.state == Running {
		s.transitionLocked(Closing)
	}

	system, projectionErr := s.validateProjectionLocked(execution)
	if projectionErr != nil {
		fault := s.persistFaultLocked(projectionErr)
		s.mu.Unlock()
		return execution, fault
	}
	batch := newEntryBatch(s.codec, 4+len(s.queue))
	var nextUsage agentic.Usage
	usageChanged := false
	if execution != nil && execution.Result != nil {
		delta, err := usageDelta(execution.Result.Usage, s.run.lastUsage)
		if err != nil {
			fault := s.persistFaultLocked(err)
			s.mu.Unlock()
			return execution, fault
		}
		if !usageEmpty(delta) {
			nextUsage = addUsage(s.usage, delta)
			batch.Add(kindUsageCommitted, usagePayload{Run: execution.Result.Usage, Session: nextUsage})
			usageChanged = true
		}
	}
	if system != nil {
		batch.Add(kindSystemMessage, messagePayload{Message: *system, Source: "driver_system"})
	}
	toCancel := []QueueEntry(nil)
	if status != agentic.ExecutionCompleted || runErr != nil {
		toCancel = s.queueByKindsLocked(QueueSteer, QueueFollowUp)
		for _, entry := range toCancel {
			batch.Add(kindQueueCancelled, queueMutationPayload{ID: entry.ID, Reason: "run ended"})
		}
	}
	if s.budget != nil && agentic.IsUsageLimitExceeded(runErr) {
		runErr = &BudgetError{Cause: runErr}
	}
	errorText := ""
	if runErr != nil {
		errorText = runErr.Error()
	}
	batch.Add(kindRunClosed, runClosedPayload{ID: s.run.id, Status: status, Error: errorText})
	pendingEntries, encodeErr := batch.Result()
	if encodeErr != nil {
		cancel := s.faultLocked(encodeErr)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return execution, &FaultError{SessionID: s.id, Cause: encodeErr}
	}
	commit, err := s.journal.Append(context.Background(), s.cursor, pendingEntries...)
	if err != nil {
		cancel := s.faultLocked(err)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return execution, &FaultError{SessionID: s.id, Cause: err}
	}
	if usageChanged {
		s.usage = nextUsage
	}
	if system != nil {
		shiftContextMarkers(s.contextMarkers, 1)
		s.messages = append([]agentic.Message{*system}, s.messages...)
	}
	for _, entry := range toCancel {
		_, _ = s.removeQueueLocked(entry.ID)
	}
	s.cursor = commit.Cursor
	s.run = nil
	s.suspension = nil
	s.transitionLocked(Idle)
	s.mu.Unlock()
	s.publishOwnByKind(commit.Entries)
	return execution, runErr
}

func isResumeValidationError(err error) bool {
	return errors.Is(err, agentic.ErrDriveInput) ||
		errors.Is(err, agentic.ErrTranscriptInvalid) ||
		errors.Is(err, agentic.ErrSuspensionVersion) ||
		errors.Is(err, agentic.ErrSuspensionMismatch) ||
		errors.Is(err, agentic.ErrResumeDecision)
}

func (s *Session[O]) validateProjectionLocked(execution *agentic.Execution[O]) (*agentic.Message, error) {
	if execution == nil || execution.Result == nil {
		return nil, nil
	}
	full := cloneMessages(execution.Result.Messages)
	index := 0
	var system *agentic.Message
	if len(full) > 0 && full[0].Role == agentic.RoleSystem &&
		(len(s.run.history) == 0 || !messagesEqual(full[:1], s.run.history[:1])) {
		copy := full[0]
		system = &copy
		index++
	}
	if len(full)-index < len(s.run.history) || !messagesEqual(full[index:index+len(s.run.history)], s.run.history) {
		return nil, fmt.Errorf("%w: result history prefix differs from the durable input", ErrCommitProjectionMismatch)
	}
	index += len(s.run.history)
	actual := full[index:]
	if !messagesEqual(actual, s.run.expected) {
		return nil, fmt.Errorf("%w: result contains %d new messages but %d were synchronously committed", ErrCommitProjectionMismatch, len(actual), len(s.run.expected))
	}
	// v0.4.0 can include a driver-inserted system/history prefix in
	// NewMessages. The full-prefix comparison above remains authoritative; this
	// suffix check still exercises the public API required by the contract.
	publicNew := execution.Result.NewMessages()
	publicExpected := s.run.expected[s.run.publicNewStart:]
	if len(publicNew) < len(publicExpected) ||
		!messagesEqual(publicNew[len(publicNew)-len(publicExpected):], publicExpected) {
		return nil, fmt.Errorf("%w: Result.NewMessages is not the committed suffix", ErrCommitProjectionMismatch)
	}
	return system, nil
}

func (s *Session[O]) persistFaultLocked(cause error) error {
	entry, encodeErr := pending(s.codec, kindFault, struct{ Error string }{cause.Error()})
	if encodeErr == nil {
		commit, appendErr := s.journal.Append(context.Background(), s.cursor, entry)
		if appendErr == nil {
			s.cursor = commit.Cursor
		}
	}
	cancel := s.faultLocked(cause)
	if cancel != nil {
		cancel()
	}
	return &FaultError{SessionID: s.id, Cause: cause}
}

func (s *Session[O]) publishOwnByKind(entries []store.Entry) {
	for _, entry := range entries {
		nature := agentic.EventAuthoritative
		switch entry.Kind {
		case kindRunClosed, kindInterruptMarker, kindFault, kindRecovered:
			nature = agentic.EventLifecycle
		}
		s.publishOwn([]store.Entry{entry}, nature)
	}
}

func (s *Session[O]) finishInterrupt(execution *agentic.Execution[O]) error {
	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return err
	}
	if s.run == nil {
		s.mu.Unlock()
		return errors.New("interrupt has no active run")
	}
	var system *agentic.Message
	if execution != nil && execution.Result != nil {
		projectedSystem, err := s.validateProjectionLocked(execution)
		if err != nil {
			fault := s.persistFaultLocked(err)
			s.mu.Unlock()
			return fault
		}
		system = projectedSystem
	}
	pendingCalls := s.pendingCallsLocked()
	repaired, err := repair.Process(s.messages, repair.CloseInterruptedFrontier, pendingCalls)
	if err != nil {
		fault := s.persistFaultLocked(err)
		s.mu.Unlock()
		return fault
	}
	if len(repaired) < len(s.messages) || !messagesEqual(repaired[:len(s.messages)], s.messages) {
		err := fmt.Errorf("%w: interrupt repair rewrote committed messages", ErrCommitProjectionMismatch)
		fault := s.persistFaultLocked(err)
		s.mu.Unlock()
		return fault
	}
	added := repaired[len(s.messages):]
	batch := newEntryBatch(s.codec, len(added)+len(s.queue)+3)
	for _, message := range added {
		batch.Add(kindRepair, messagePayload{Message: message, Source: "interrupt_repair"})
	}
	var nextUsage agentic.Usage
	usageChanged := false
	if execution != nil && execution.Result != nil {
		delta, deltaErr := usageDelta(execution.Result.Usage, s.run.lastUsage)
		if deltaErr != nil {
			fault := s.persistFaultLocked(deltaErr)
			s.mu.Unlock()
			return fault
		}
		if !usageEmpty(delta) {
			nextUsage = addUsage(s.usage, delta)
			batch.Add(kindUsageCommitted, usagePayload{Run: execution.Result.Usage, Session: nextUsage})
			usageChanged = true
		}
	}
	if system != nil {
		batch.Add(kindSystemMessage, messagePayload{Message: *system, Source: "driver_system"})
	}
	toCancel := s.queueByKindsLocked(QueueSteer, QueueFollowUp)
	for _, entry := range toCancel {
		batch.Add(kindQueueCancelled, queueMutationPayload{ID: entry.ID, Reason: "interrupted"})
	}
	marker := interruptMarkerPayload{Message: "The run was deliberately interrupted; detached work may still be running and side effects may have partially occurred."}
	batch.Add(kindInterruptMarker, marker)
	batch.Add(kindRunClosed, runClosedPayload{ID: s.run.id, Status: agentic.ExecutionInterrupted, Error: context.Canceled.Error()})
	pendingEntries, encodeErr := batch.Result()
	if encodeErr != nil {
		cancel := s.faultLocked(encodeErr)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return &FaultError{SessionID: s.id, Cause: encodeErr}
	}
	commit, appendErr := s.journal.Append(context.Background(), s.cursor, pendingEntries...)
	if appendErr != nil {
		cancel := s.faultLocked(appendErr)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return &FaultError{SessionID: s.id, Cause: appendErr}
	}
	s.messages = repaired
	if system != nil {
		shiftContextMarkers(s.contextMarkers, 1)
		s.messages = append([]agentic.Message{*system}, s.messages...)
	}
	s.contextMarkers = append(s.contextMarkers, contextMarker{
		after:   len(s.messages),
		message: interruptionContextMessage(marker.Message),
	})
	if usageChanged {
		s.usage = nextUsage
	}
	for _, entry := range toCancel {
		_, _ = s.removeQueueLocked(entry.ID)
	}
	s.cursor = commit.Cursor
	s.run = nil
	s.runCancel = nil
	s.suspension = nil
	s.transitionLocked(Idle)
	s.mu.Unlock()
	s.publishOwnByKind(commit.Entries)
	return nil
}

func (s *Session[O]) pendingCallsLocked() repair.PendingCalls {
	open, _ := repair.InspectFrontier(s.messages)
	pending := repair.PendingCalls{Calls: make([]repair.PendingCall, len(open))}
	for i, call := range open {
		state := repair.PendingPlanned
		if s.run.started[call.ID] && !s.run.results[call.ID] {
			state = repair.PendingIndeterminate
		}
		pending.Calls[i] = repair.PendingCall{ID: call.ID, Name: call.Name, State: state}
	}
	return pending
}
