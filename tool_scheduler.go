package agentic

import (
	"context"
	"errors"
	"fmt"
	"time"
)

type scheduledResult struct {
	index  int
	result ToolExecutionResult
	err    error
}

// executeAdmittedTools runs one already-gated batch concurrently while keeping
// result order deterministic. Cancellation stops waiting after the configured
// grace period; late handler results are intentionally ignored.
func (c *agentCore) executeAdmittedTools(
	ls *loopState,
	calls []ToolUse,
	resumes map[string]ToolResumeContext,
) ([]ToolExecutionResult, *ToolDeferral, error) {
	if len(calls) == 0 {
		return nil, nil, nil
	}
	for index, call := range calls {
		attempt := ls.retryCounts[call.Name] + 1
		if err := ls.emit(&ToolStartedEvent{
			eventBase: newEventBase(EventAuthoritative, EventTypeToolStarted, ls.iteration),
			Call:      call,
			Attempt:   attempt,
		}); err != nil {
			return nil, nil, err
		}
		_ = index
	}

	results := make([]ToolExecutionResult, len(calls))
	received := make([]bool, len(calls))
	resultCh := make(chan scheduledResult, len(calls))
	baseCtx := addToolExecutionState(ls.ctx, ls)
	for index, call := range calls {
		index, call := index, call
		attempt := ls.retryCounts[call.Name] + 1
		go func() {
			callCtx := WithToolCallContext(baseCtx, ToolCallContext{ID: call.ID, Name: call.Name, Attempt: attempt})
			if resume, ok := resumes[call.ID]; ok {
				callCtx = withToolResumeContext(callCtx, resume)
			}
			result, err := ls.registry.Execute(callCtx, call, ls.deps)
			resultCh <- scheduledResult{index: index, result: result, err: err}
		}()
	}

	remaining := len(calls)
	var firstErr error
	var cancellation <-chan struct{}
	if ls.ctx.Done() != nil {
		cancellation = ls.ctx.Done()
	}
	for remaining > 0 {
		select {
		case completion := <-resultCh:
			if received[completion.index] {
				continue
			}
			received[completion.index] = true
			remaining--
			if completion.err != nil {
				if firstErr == nil {
					firstErr = fmt.Errorf("execute tool %q: %w", calls[completion.index].Name, completion.err)
				}
				results[completion.index] = makeToolResultError(calls[completion.index], completion.err.Error(), completion.err)
				continue
			}
			results[completion.index] = completion.result
		case <-cancellation:
			grace := time.Second
			if ls.options.toolCancellationGrace != nil {
				grace = *ls.options.toolCancellationGrace
			}
			if grace < 0 {
				grace = 0
			}
			timer := time.NewTimer(grace)
			for remaining > 0 {
				select {
				case completion := <-resultCh:
					if received[completion.index] {
						continue
					}
					received[completion.index] = true
					remaining--
					if completion.err != nil {
						if firstErr == nil {
							firstErr = fmt.Errorf("execute tool %q: %w", calls[completion.index].Name, completion.err)
						}
						results[completion.index] = makeToolResultError(calls[completion.index], completion.err.Error(), completion.err)
					} else {
						results[completion.index] = completion.result
					}
				case <-timer.C:
					for index, call := range calls {
						if received[index] {
							continue
						}
						received[index] = true
						remaining--
						results[index] = makeToolResultError(
							call,
							"Tool execution did not stop before cancellation grace; side effects may still be running.",
							ErrExecutionInterrupted,
						)
					}
				}
			}
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			cancellation = nil
		}
	}
	for index, result := range results {
		deferral, suspended, err := handlerDeferral(ls.registry, calls[index], result)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			results[index] = makeToolResultError(calls[index], err.Error(), err)
			continue
		}
		if suspended {
			return nil, deferral, firstErr
		}
	}
	return results, nil, firstErr
}

func handlerDeferral(
	registry *executionToolRegistry,
	call ToolUse,
	result ToolExecutionResult,
) (*ToolDeferral, bool, error) {
	var suspension ToolHandlerSuspension
	switch value := result.Content.(type) {
	case ToolHandlerSuspension:
		suspension = value
	case *ToolHandlerSuspension:
		if value == nil {
			return nil, false, fmt.Errorf("%w: handler %q returned a nil suspension", ErrToolHandlerSuspension, call.Name)
		}
		suspension = *value
	default:
		return nil, false, nil
	}
	handler, ok := registry.Get(call.Name)
	marker, marked := handler.(SuspendableToolHandler)
	if !ok || !marked || !marker.MaySuspendToolExecution() {
		return nil, false, fmt.Errorf("%w: handler %q did not declare suspension", ErrToolHandlerSuspension, call.Name)
	}
	if err := validateToolResultIdentity(call, result); err != nil {
		return nil, false, fmt.Errorf("%w: handler %q returned mismatched suspension identity: %v", ErrToolHandlerSuspension, call.Name, err)
	}
	if result.IsError || result.Error != nil || suspension.Deferral.Kind == "" {
		return nil, false, fmt.Errorf("%w: handler %q returned an invalid deferral", ErrToolHandlerSuspension, call.Name)
	}
	deferral := suspension.Deferral
	deferral.Payload = append([]byte(nil), deferral.Payload...)
	return &deferral, true, nil
}

func validateToolBatchDecision(calls []ToolUse, decision ToolBatchDecision) (bool, error) {
	if len(decision.Calls) != len(calls) {
		return false, fmt.Errorf("tool gate returned %d dispositions for %d calls", len(decision.Calls), len(calls))
	}
	hasSuspend := false
	for index, disposition := range decision.Calls {
		call := calls[index]
		switch disposition.Kind {
		case ToolDispositionExecute:
			if disposition.Result != nil {
				return false, fmt.Errorf("tool gate supplied a result for executable call %q", call.ID)
			}
		case ToolDispositionReturn:
			if disposition.Result == nil {
				return false, fmt.Errorf("tool gate returned no result for call %q", call.ID)
			}
			if err := validateToolResultIdentity(call, *disposition.Result); err != nil {
				return false, err
			}
		case ToolDispositionSuspend:
			hasSuspend = true
			if disposition.Result != nil || disposition.Continue {
				return false, fmt.Errorf("tool gate supplied result or continuation for suspended call %q", call.ID)
			}
		default:
			return false, fmt.Errorf("tool gate returned an invalid disposition for call %q", call.ID)
		}
	}
	if hasSuspend {
		if decision.Deferral == nil || decision.Deferral.Kind == "" {
			return false, errors.New("tool gate suspension requires one non-empty deferral")
		}
		return true, nil
	}
	if decision.Deferral != nil {
		return false, errors.New("tool gate returned a deferral without a suspended call")
	}
	return false, nil
}

func validateToolResultIdentity(call ToolUse, result ToolExecutionResult) error {
	if result.ToolUseID != call.ID {
		return fmt.Errorf("tool result ID %q does not match call %q", result.ToolUseID, call.ID)
	}
	if result.ToolName != call.Name {
		return fmt.Errorf("tool result name %q does not match call %q", result.ToolName, call.Name)
	}
	return nil
}

func (c *agentCore) projectToolResult(ctx context.Context, ls *loopState, call ToolUse, result ToolExecutionResult) (ToolExecutionResult, error) {
	if err := validateToolResultIdentity(call, result); err != nil {
		return makeToolResultError(call, "Tool execution returned an invalid result identity.", err), err
	}
	if ls.options.toolResultProcessor == nil {
		return result, nil
	}
	projected, err := ls.options.toolResultProcessor.Process(ctx, call, result)
	if err != nil {
		return makeToolResultError(
			call,
			"Tool handler may have completed, but result processing failed: "+err.Error(),
			err,
		), err
	}
	if err := validateToolResultIdentity(call, projected); err != nil {
		return makeToolResultError(call, "Tool result processing changed call identity.", err), err
	}
	if result.IsError && !projected.IsError {
		err := errors.New("tool result processor changed an error into success")
		return makeToolResultError(call, "Tool result processing changed an error into success.", err), err
	}
	if result.Error != nil && projected.Error == nil {
		err := errors.New("tool result processor removed an execution error")
		return makeToolResultError(call, "Tool result processing removed execution error truth.", err), err
	}
	return projected, nil
}
