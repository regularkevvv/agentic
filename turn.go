package agentic

import (
	"context"
	"errors"
	"fmt"
)

type completionEvaluator[O any] func(context.Context, dependencyEnvelope, Message) (O, error)

type validatedCandidate[O any] struct {
	output O
	meta   CompletionCandidate
}

type turnOutcome[O any] struct {
	assistant    Message
	results      []ToolExecutionResult
	candidate    *validatedCandidate[O]
	continuation bool
	suspension   *Suspension
	fatal        error
}

type regularTurnState struct {
	results      map[string]ToolExecutionResult
	continuation bool
	suspension   *Suspension
	fatal        error
}

func (c *agentCore) checkOutputTool(toolUses []ToolUse) int {
	for index, toolUse := range toolUses {
		if c.outputToolNames[toolUse.Name] {
			return index
		}
	}
	return -1
}

func textCompletionEvaluator(c *agentCore) completionEvaluator[string] {
	return func(ctx context.Context, deps dependencyEnvelope, message Message) (string, error) {
		output := message.GetTextContent()
		if err := c.validateOutput(ctx, deps, output); err != nil {
			return "", err
		}
		return output, nil
	}
}

func completeAssistantTurn[O any](
	c *agentCore,
	ls *loopState,
	assistant Message,
	evaluator completionEvaluator[O],
	recordedCalls bool,
	resumed *regularTurnState,
) turnOutcome[O] {
	outcome := turnOutcome[O]{assistant: assistant}
	calls := assistant.GetToolUses()
	if !recordedCalls {
		ls.allToolCalls = append(ls.allToolCalls, cloneToolUses(calls)...)
	}
	outputIndex := c.checkOutputTool(calls)
	hasCandidate := len(calls) == 0 || outputIndex >= 0

	regular := resumed
	if regular == nil && (outputIndex < 0 || ls.endStrategy != EndStrategyEarly) {
		regular = c.executeRegularTurn(ls, calls)
	}
	if regular != nil && regular.suspension != nil {
		outcome.suspension = regular.suspension
		return outcome
	}
	if regular == nil {
		regular = &regularTurnState{results: make(map[string]ToolExecutionResult)}
	}
	if regular.results == nil {
		regular.results = make(map[string]ToolExecutionResult)
	}

	results := make([]ToolExecutionResult, len(calls))
	for index, call := range calls {
		if c.outputToolNames[call.Name] {
			if index != outputIndex {
				results[index] = makeToolResultError(call, "Skipped because the first output tool is the completion candidate.", nil)
			}
			continue
		}
		if outputIndex >= 0 && ls.endStrategy == EndStrategyEarly {
			results[index] = makeToolResultError(call, "Skipped because early output ended the tool batch.", nil)
			continue
		}
		result, ok := regular.results[call.ID]
		if !ok {
			result = makeToolResultError(call, "Tool execution did not produce a paired result.", regular.fatal)
		}
		results[index] = result
	}

	if regular.fatal != nil {
		outcome.fatal = regular.fatal
		outcome.continuation = true
		if outputIndex >= 0 {
			results[outputIndex] = makeToolResultError(calls[outputIndex], "Output discarded because tool execution failed.", regular.fatal)
		}
	} else if hasCandidate && !regular.continuation {
		output, err := evaluator(ls.ctx, ls.deps, assistant)
		if err != nil {
			outcome.continuation = true
			ls.validationRetries++
			if outputIndex >= 0 {
				results[outputIndex] = makeToolResultError(calls[outputIndex], "Output validation failed: "+err.Error(), err)
			} else {
				feedback := NewTextMessage(RoleUser, fmt.Sprintf("Output validation error: %s\nPlease try again.", err))
				ls.messages = append(ls.messages, feedback)
				if emitErr := ls.emit(&TurnMessagesInjectedEvent{
					eventBase: newEventBase(EventAuthoritative, EventTypeTurnMessagesInjected, ls.iteration),
					Messages:  cloneMessages([]Message{feedback}),
				}); emitErr != nil {
					outcome.fatal = emitErr
				}
			}
			if ls.validationRetries > c.config.maxValidationRetries {
				outcome.fatal = fmt.Errorf("output validation failed after %d retries: %w", c.config.maxValidationRetries, err)
			}
		} else {
			source := CompletionText
			toolCallID := ""
			if outputIndex >= 0 {
				source = CompletionOutputTool
				toolCallID = calls[outputIndex].ID
				results[outputIndex] = ToolExecutionResult{
					ToolUseID: calls[outputIndex].ID,
					ToolName:  calls[outputIndex].Name,
					Content:   "Output accepted.",
				}
			}
			candidate := &validatedCandidate[O]{output: output, meta: CompletionCandidate{Source: source, ToolCallID: toolCallID}}
			outcome.candidate = candidate
			if emitErr := ls.emit(&OutputValidatedEvent{
				eventBase: newEventBase(EventAuthoritative, EventTypeOutputValidated, ls.iteration),
				Candidate: candidate.meta,
			}); emitErr != nil {
				outcome.fatal = emitErr
			}
		}
	} else if outputIndex >= 0 {
		results[outputIndex] = makeToolResultError(calls[outputIndex], "Output discarded because another tool requires continuation.", nil)
		outcome.continuation = true
	}

	if regular.continuation {
		outcome.continuation = true
	}
	if len(results) > 0 {
		if err := ls.commitToolResults(results); err != nil && outcome.fatal == nil {
			outcome.fatal = err
		}
	}
	outcome.results = results
	return outcome
}

func (c *agentCore) executeRegularTurn(ls *loopState, calls []ToolUse) *regularTurnState {
	state := &regularTurnState{results: make(map[string]ToolExecutionResult)}
	regular := make([]ToolUse, 0, len(calls))
	for _, call := range calls {
		if !c.outputToolNames[call.Name] {
			regular = append(regular, call)
		}
	}
	if len(regular) == 0 {
		return state
	}
	if err := ls.emit(&ToolBatchPlannedEvent{
		eventBase: newEventBase(EventAuthoritative, EventTypeToolBatchPlanned, ls.iteration),
		Calls:     cloneToolUses(regular),
	}); err != nil {
		state.fatal = err
		for _, call := range regular {
			state.results[call.ID] = makeToolResultError(call, "Tool batch planning failed.", err)
		}
		return state
	}

	gate := ls.options.toolGate
	if gate == nil {
		gate = allowAllToolGate{}
	}
	decision, err := gate.EvaluateBatch(ls.ctx, cloneToolUses(regular))
	if err != nil {
		state.fatal = fmt.Errorf("tool gate: %w", err)
		for _, call := range regular {
			state.results[call.ID] = makeToolResultError(call, "Tool batch denied because the tool gate failed: "+err.Error(), err)
		}
		return state
	}
	suspended, err := validateToolBatchDecision(regular, decision)
	if err != nil {
		state.fatal = fmt.Errorf("tool gate: %w", err)
		for _, call := range regular {
			state.results[call.ID] = makeToolResultError(call, "Tool batch denied because the tool gate returned an invalid decision.", err)
		}
		return state
	}
	if suspended {
		suspension, err := c.newToolSuspension(ls, calls, regular, *decision.Deferral)
		if err != nil {
			state.fatal = err
			for _, call := range regular {
				state.results[call.ID] = makeToolResultError(call, "Unable to persist tool suspension.", err)
			}
			return state
		}
		state.suspension = suspension
		return state
	}

	executable := make([]ToolUse, 0, len(regular))
	for index, disposition := range decision.Calls {
		call := regular[index]
		state.continuation = state.continuation || disposition.Continue
		switch disposition.Kind {
		case ToolDispositionReturn:
			state.results[call.ID] = *disposition.Result
		case ToolDispositionExecute:
			executable = append(executable, call)
		}
	}
	if len(executable) == 0 {
		return state
	}
	if ls.registry.Count() == 0 && !ls.registry.hasExecutor {
		err := errors.New("model requested tool calls but no tools are registered")
		state.fatal = err
		for _, call := range executable {
			state.results[call.ID] = makeToolResultError(call, err.Error(), err)
		}
		return state
	}
	if err := ls.checkToolCallLimits(len(executable)); err != nil {
		state.fatal = err
		for _, call := range executable {
			state.results[call.ID] = makeToolResultError(call, err.Error(), err)
		}
		return state
	}

	results, executionErr := c.executeAdmittedTools(ls, executable)
	for index, result := range results {
		call := executable[index]
		projected, err := c.projectToolResult(ls.ctx, ls, call, result)
		state.results[call.ID] = projected
		if err != nil && state.fatal == nil {
			state.fatal = fmt.Errorf("tool result processor: %w", err)
		}
		if projected.Error != nil {
			var retry *ModelRetry
			if errors.As(projected.Error, &retry) {
				maxRetries := c.config.retryConfig.MaxRetries
				if handler, ok := ls.registry.Get(call.Name); ok {
					maxRetries = c.resolveMaxRetries(handler, maxRetries)
				}
				if ls.retryCounts[call.Name] < maxRetries {
					if ls.retryCounts == nil {
						ls.retryCounts = make(map[string]int)
					}
					ls.retryCounts[call.Name]++
					ls.totalRetries++
					state.continuation = true
				}
			}
		}
		ls.totalUsage.ToolCalls++
	}
	if executionErr != nil && state.fatal == nil {
		state.fatal = executionErr
	}
	return state
}

func (ls *loopState) commitToolResults(results []ToolExecutionResult) error {
	for _, result := range results {
		ls.allToolResults = append(ls.allToolResults, result)
		ls.messages = append(ls.messages, NewToolResultMessageFor(
			result.ToolUseID,
			result.ToolName,
			FormatToolResult(result.Content),
			result.IsError,
		))
	}
	for _, result := range results {
		if err := ls.emit(&ToolResultCommittedEvent{
			eventBase: newEventBase(EventAuthoritative, EventTypeToolResultCommitted, ls.iteration),
			Result:    result,
		}); err != nil {
			return err
		}
	}
	return nil
}
