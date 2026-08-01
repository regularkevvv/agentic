package session

import (
	"context"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"

	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

// RecordOperation implements runtime.OperationRuntime with synchronous,
// write-ahead durability. Unknown operation kinds remain opaque journal facts
// and are replayed through the ordinary durable event cursor.
func (s *Session[O]) RecordOperation(ctx context.Context, operation harnessruntime.Operation) error {
	if operation.ID == "" || operation.Kind == "" || operation.Phase == "" {
		return errors.New("runtime operation ID, kind, and phase are required")
	}
	operation.Payload = append([]byte(nil), operation.Payload...)
	entry, err := pending(s.codec, kindRuntimeOperation, operation)
	if err != nil {
		return s.faultOperation(err)
	}
	s.mu.Lock()
	if s.state == Faulted {
		fault := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return fault
	}
	if s.state == Closed {
		s.mu.Unlock()
		return ErrSessionClosed
	}
	if s.state != Running {
		state := s.state
		s.mu.Unlock()
		return fmt.Errorf("runtime operation requires a running session, got %s", state)
	}
	commit, appendErr := s.journal.Append(context.WithoutCancel(ctx), s.cursor, entry)
	if appendErr != nil {
		cancel := s.faultLocked(appendErr)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return &FaultError{SessionID: s.id, Cause: appendErr}
	}
	s.cursor = commit.Cursor
	s.mu.Unlock()
	s.publishOwn(commit.Entries, agentic.EventAuthoritative)
	return nil
}

// ProjectToolResult applies the session-scoped artifact/result processor to a
// nested composite-tool result without exposing its concrete storage.
func (s *Session[O]) ProjectToolResult(
	ctx context.Context,
	call agentic.ToolUse,
	result agentic.ToolExecutionResult,
) (agentic.ToolExecutionResult, error) {
	if call.ID == "" || call.Name == "" || result.ToolUseID != call.ID || result.ToolName != call.Name {
		err := fmt.Errorf("%w: nested tool result identity differs", ErrCommitProjectionMismatch)
		return agentic.ToolExecutionResult{}, s.faultOperation(err)
	}
	projected, err := s.processor.Process(ctx, call, result)
	if err != nil {
		if ctx.Err() != nil {
			return agentic.ToolExecutionResult{}, ctx.Err()
		}
		return agentic.ToolExecutionResult{}, s.faultOperation(err)
	}
	if projected.ToolUseID != call.ID || projected.ToolName != call.Name ||
		(!result.IsError && projected.IsError) || (result.IsError && !projected.IsError) {
		err = fmt.Errorf("%w: nested result processor changed identity or error disposition", ErrCommitProjectionMismatch)
		return agentic.ToolExecutionResult{}, s.faultOperation(err)
	}
	return projected, nil
}

func (s *Session[O]) faultOperation(cause error) error {
	s.mu.Lock()
	if s.state == Faulted {
		fault := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return fault
	}
	if s.state == Closed {
		s.mu.Unlock()
		return ErrSessionClosed
	}
	cancel := s.faultLocked(cause)
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	return &FaultError{SessionID: s.id, Cause: cause}
}
