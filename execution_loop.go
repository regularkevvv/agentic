package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

func validateDriveInput(input DriveInput, history []Message) error {
	frontier, err := inspectTranscript(history)
	if err != nil {
		return err
	}
	switch input.Mode {
	case DriveStart:
		if input.Prompt == nil || input.Prompt.Role != RoleUser {
			return fmt.Errorf("%w: DriveStart requires one RoleUser prompt", ErrDriveInput)
		}
		if len(frontier) != 0 {
			return fmt.Errorf("%w: DriveStart history has an open tool frontier", ErrTranscriptInvalid)
		}
	case DriveContinue:
		if input.Prompt != nil {
			return fmt.Errorf("%w: DriveContinue forbids a prompt", ErrDriveInput)
		}
		if len(history) == 0 {
			return fmt.Errorf("%w: DriveContinue requires non-empty history", ErrDriveInput)
		}
		if len(frontier) != 0 {
			return fmt.Errorf("%w: DriveContinue history has an open tool frontier", ErrTranscriptInvalid)
		}
		last := history[len(history)-1]
		if last.Role != RoleUser && last.Role != RoleTool {
			return fmt.Errorf("%w: DriveContinue history must end in a user or tool-result message", ErrDriveInput)
		}
	default:
		return fmt.Errorf("%w: unknown drive mode %d", ErrDriveInput, input.Mode)
	}
	return nil
}

func (ls *loopState) emit(event Event) error {
	if ls.options.eventSink == nil {
		return nil
	}
	return ls.options.eventSink.Emit(ls.ctx, event)
}

func driveWithEvaluator[O any](
	c *agentCore,
	ctx context.Context,
	input DriveInput,
	deps dependencyEnvelope,
	evaluator completionEvaluator[O],
	opts ...RunOption,
) (*Execution[O], error) {
	if err := c.preflight(ctx, deps); err != nil {
		return nil, err
	}
	ls, err := c.prepareLoopForDrive(ctx, input, deps, opts...)
	if err != nil {
		return nil, err
	}
	mode := AgentInvocationStart
	if input.Mode == DriveContinue {
		mode = AgentInvocationContinue
	}
	return observeAgentExecution(c, ls, mode, func() (*Execution[O], error) {
		return driveLoop(c, ls, evaluator, nil, nil)
	})
}

func resumeWithEvaluator[O any](
	c *agentCore,
	ctx context.Context,
	input ResumeInput,
	deps dependencyEnvelope,
	evaluator completionEvaluator[O],
	opts ...RunOption,
) (*Execution[O], error) {
	if err := c.preflight(ctx, deps); err != nil {
		return nil, err
	}
	if input.Prompt != nil && input.Prompt.Role != RoleUser {
		return nil, fmt.Errorf("%w: Resume prompt must be RoleUser", ErrDriveInput)
	}
	ls, err := c.prepareLoopForResume(ctx, input.History, deps, opts...)
	if err != nil {
		return nil, err
	}
	return observeAgentExecution(c, ls, AgentInvocationResume, func() (*Execution[O], error) {
		return resumePreparedWithEvaluator(c, ls, input, evaluator)
	})
}

func resumePreparedWithEvaluator[O any](
	c *agentCore,
	ls *loopState,
	input ResumeInput,
	evaluator completionEvaluator[O],
) (*Execution[O], error) {
	frontier, err := inspectTranscript(ls.messages)
	if err != nil {
		return failedExecution[O](c, ls, err, false)
	}
	if len(frontier) == 0 {
		return failedExecution[O](c, ls, fmt.Errorf("%w: Resume requires an open tool frontier", ErrSuspensionMismatch), false)
	}
	if input.Suspension.ID == "" || input.Suspension.FrontierHash == "" {
		return failedExecution[O](c, ls, fmt.Errorf("%w: missing suspension ID or frontier hash", ErrSuspensionMismatch), false)
	}
	if got := frontierHash(ls.messages, frontier); got != input.Suspension.FrontierHash {
		return failedExecution[O](c, ls, fmt.Errorf("%w: frontier hash differs", ErrSuspensionMismatch), false)
	}
	payload, err := decodeToolSuspension(input.Suspension)
	if err != nil {
		return failedExecution[O](c, ls, err, false)
	}
	if payload.SuspensionID != input.Suspension.ID || payload.Deferral.Kind != input.Suspension.Kind {
		return failedExecution[O](c, ls, fmt.Errorf("%w: suspension identity differs", ErrSuspensionMismatch), false)
	}
	if payload.ConfigFingerprint != ls.configFingerprint || payload.EndStrategy != ls.endStrategy {
		return failedExecution[O](c, ls, fmt.Errorf("%w: model, output, toolset, limit, retry, or end-strategy configuration changed", ErrSuspensionMismatch), false)
	}
	if !sameToolCalls(frontier, payload.Calls) {
		return failedExecution[O](c, ls, fmt.Errorf("%w: persisted open calls differ from transcript frontier", ErrSuspensionMismatch), false)
	}

	ls.totalUsage = payload.Usage
	ls.retryCounts = copyRetryCounts(payload.RetryCounts)
	ls.totalRetries = payload.TotalRetries
	ls.validationRetries = payload.ValidationRetries
	ls.iteration = payload.Iteration
	ls.lastFinishReason = payload.LastFinishReason

	regular, err := c.resolveSuspendedTools(ls, payload, input.Decisions)
	if err != nil {
		return failedExecution[O](c, ls, err, false)
	}
	assistant := ls.messages[len(ls.messages)-1]
	prompt := (*Message)(nil)
	if input.Prompt != nil {
		copy := cloneMessages([]Message{*input.Prompt})[0]
		prompt = &copy
	}
	return driveLoop(c, ls, evaluator, &resumeTurn[O]{assistant: assistant, regular: regular}, prompt)
}

func observeAgentExecution[O any](
	c *agentCore,
	ls *loopState,
	mode AgentInvocationMode,
	run func() (*Execution[O], error),
) (execution *Execution[O], err error) {
	ctx, span := safeStartAgent(ls.ctx, ls.instrumentation, AgentOperation{
		Agent:     ls.agentIdentity,
		Model:     ls.modelMetadata,
		ModelName: c.model.Name(),
		Request:   c.instrumentationRequest(ls),
		Run:       ls.runMetadata,
		Mode:      mode,
		Input:     ls.messages,
	})
	ctx = withInheritedInstrumentation(ctx, ls.instrumentation, ls.runMetadata)
	ls.ctx = ctx
	defer func() {
		result := AgentOperationResult{
			Status:   ExecutionFailed,
			Messages: ls.messages,
			Usage:    ls.totalUsage,
			Error:    err,
		}
		if execution != nil {
			result.Status = execution.Status
			if execution.Result != nil {
				result.Messages = execution.Result.Messages
				result.Usage = execution.Result.Usage
				result.FinishReason = execution.Result.FinishReason
			}
		}
		safeEndAgent(span, result)
	}()
	return run()
}

type resumeTurn[O any] struct {
	assistant Message
	regular   *regularTurnState
}

func sameToolCalls(left, right []ToolUse) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		leftBytes, leftErr := json.Marshal(left[index])
		rightBytes, rightErr := json.Marshal(right[index])
		if leftErr != nil || rightErr != nil || string(leftBytes) != string(rightBytes) {
			return false
		}
	}
	return true
}

func (c *agentCore) resolveSuspendedTools(ls *loopState, payload toolSuspensionPayload, decisions []ToolResumeDecision) (*regularTurnState, error) {
	byID := make(map[string]ToolUse, len(payload.Calls))
	for _, call := range payload.Calls {
		byID[call.ID] = call
	}
	executable := make([]ToolUse, len(payload.ExecutableCallIDs))
	for index, id := range payload.ExecutableCallIDs {
		call, ok := byID[id]
		if !ok || c.outputToolNames[call.Name] {
			return nil, fmt.Errorf("%w: suspension contains invalid executable call %q", ErrSuspensionMismatch, id)
		}
		executable[index] = call
	}
	if len(decisions) != len(executable) {
		return nil, fmt.Errorf("%w: got %d decisions for %d suspended calls", ErrResumeDecision, len(decisions), len(executable))
	}

	decisionByID := make(map[string]ToolResumeDecision, len(decisions))
	for _, decision := range decisions {
		if decision.CallID == "" {
			return nil, fmt.Errorf("%w: decision has an empty call ID", ErrResumeDecision)
		}
		if _, exists := decisionByID[decision.CallID]; exists {
			return nil, fmt.Errorf("%w: duplicate decision for %q", ErrResumeDecision, decision.CallID)
		}
		decisionByID[decision.CallID] = decision
	}

	state := &regularTurnState{results: make(map[string]ToolExecutionResult)}
	toExecute := make([]ToolUse, 0, len(executable))
	resumeContexts := make(map[string]ToolResumeContext, len(executable))
	for _, call := range executable {
		decision, ok := decisionByID[call.ID]
		if !ok {
			return nil, fmt.Errorf("%w: missing decision for %q", ErrResumeDecision, call.ID)
		}
		switch decision.Action {
		case ToolResumeExecute:
			if decision.Result != nil {
				return nil, fmt.Errorf("%w: execute decision for %q supplied a result", ErrResumeDecision, call.ID)
			}
			definition, ok := ls.registry.Tool(call.Name)
			if !ok {
				return nil, fmt.Errorf("%w: tool %q is no longer registered", ErrResumeDecision, call.Name)
			}
			if decision.Input != nil {
				if err := validateResumeInput(ls.ctx, call, definition, decision.Input); err != nil {
					return nil, err
				}
				call.Input = decision.Input
			}
			toExecute = append(toExecute, call)
			if payload.HandlerSuspension {
				handler, exists := ls.registry.Get(call.Name)
				marker, marked := handler.(SuspendableToolHandler)
				if !exists || !marked || !marker.MaySuspendToolExecution() {
					return nil, fmt.Errorf(
						"%w: suspended handler %q no longer declares suspension",
						ErrSuspensionMismatch,
						call.Name,
					)
				}
				resumeContexts[call.ID] = ToolResumeContext{
					SuspensionID: payload.SuspensionID,
					Deferral: ToolDeferral{
						Kind:    payload.Deferral.Kind,
						Payload: append(json.RawMessage(nil), payload.Deferral.Payload...),
					},
					Payload: append(json.RawMessage(nil), decision.Payload...),
				}
			} else if len(decision.Payload) != 0 {
				return nil, fmt.Errorf(
					"%w: execute decision for %q supplied handler payload for a gate suspension",
					ErrResumeDecision,
					call.ID,
				)
			}
		case ToolResumeReturn:
			if len(decision.Payload) != 0 || decision.Input != nil {
				return nil, fmt.Errorf("%w: return decision for %q supplied execute-only data", ErrResumeDecision, call.ID)
			}
			if decision.Result == nil {
				return nil, fmt.Errorf("%w: return decision for %q has no result", ErrResumeDecision, call.ID)
			}
			if err := validateToolResultIdentity(call, *decision.Result); err != nil {
				return nil, fmt.Errorf("%w: %v", ErrResumeDecision, err)
			}
			state.results[call.ID] = *decision.Result
		default:
			return nil, fmt.Errorf("%w: invalid action for %q", ErrResumeDecision, call.ID)
		}
	}
	if len(toExecute) == 0 {
		return state, nil
	}
	if err := ls.checkToolCallLimits(len(toExecute)); err != nil {
		return nil, err
	}
	if err := validateSuspendableBatch(ls.registry, executable, toExecute); err != nil {
		return nil, err
	}
	results, handlerDeferral, executionErr := c.executeAdmittedTools(ls, toExecute, resumeContexts)
	if handlerDeferral != nil {
		suspension, err := c.newToolSuspension(ls, payload.Calls, toExecute, *handlerDeferral, true)
		if err != nil {
			return nil, err
		}
		state.suspension = suspension
		return state, nil
	}
	for index, result := range results {
		call := toExecute[index]
		projected, err := c.projectToolResult(ls.ctx, ls, call, result)
		state.results[call.ID] = projected
		if err != nil && state.fatal == nil {
			state.fatal = fmt.Errorf("tool result processor: %w", err)
		}
		c.applyRetryState(ls, call, projected, state)
		ls.totalUsage.ToolCalls++
	}
	if executionErr != nil && state.fatal == nil {
		state.fatal = executionErr
	}
	return state, nil
}

func (c *agentCore) applyRetryState(ls *loopState, call ToolUse, result ToolExecutionResult, state *regularTurnState) {
	if result.Error == nil {
		return
	}
	var retry *ModelRetry
	if !errors.As(result.Error, &retry) {
		return
	}
	maxRetries := c.config.retryConfig.MaxRetries
	if handler, ok := ls.registry.Get(call.Name); ok {
		maxRetries = c.resolveMaxRetries(handler, maxRetries)
	}
	if ls.retryCounts[call.Name] >= maxRetries {
		return
	}
	if ls.retryCounts == nil {
		ls.retryCounts = make(map[string]int)
	}
	ls.retryCounts[call.Name]++
	ls.totalRetries++
	state.continuation = true
}

func driveLoop[O any](c *agentCore, ls *loopState, evaluator completionEvaluator[O], resumed *resumeTurn[O], resumePrompt *Message) (*Execution[O], error) {
	if err := ls.emit(&RunStartedEvent{eventBase: newEventBase(EventLifecycle, EventTypeRunStarted, ls.iteration)}); err != nil {
		return failedExecution[O](c, ls, err, true)
	}
	if resumed != nil {
		outcome := completeAssistantTurn(c, ls, resumed.assistant, evaluator, true, resumed.regular)
		if outcome.suspension != nil {
			return suspendedExecution[O](c, ls, *outcome.suspension)
		}
		if resumePrompt != nil && outcome.fatal == nil && outcome.suspension == nil {
			ls.messages = append(ls.messages, *resumePrompt)
			outcome.continuation = true
			if err := ls.emit(&TurnMessagesInjectedEvent{
				eventBase: newEventBase(EventAuthoritative, EventTypeTurnMessagesInjected, ls.iteration),
				Messages:  cloneMessages([]Message{*resumePrompt}),
			}); err != nil {
				outcome.fatal = err
			}
		}
		execution, continueRun, err := finishTurn(c, ls, outcome)
		if err != nil || execution != nil {
			return execution, err
		}
		if continueRun {
			ls.iteration++
		}
	}

	for {
		if err := ls.ctx.Err(); err != nil {
			return interruptedExecution[O](c, ls, err, false)
		}
		if ls.iteration >= ls.maxIterations {
			return failedExecution[O](c, ls, &MaxIterationsError{MaxIterations: ls.maxIterations}, false)
		}
		if err := ls.checkPreRequestLimits(); err != nil {
			return failedExecution[O](c, ls, err, false)
		}
		if err := ls.emit(&TurnStartedEvent{eventBase: newEventBase(EventLifecycle, EventTypeTurnStarted, ls.iteration)}); err != nil {
			return failedExecution[O](c, ls, err, true)
		}
		response, err := c.requestTurn(ls)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return interruptedExecution[O](c, ls, err, false)
			}
			return failedExecution[O](c, ls, err, false)
		}
		ls.totalUsage.Add(response.Usage)
		if err := ls.checkPostResponseLimits(); err != nil {
			return failedExecution[O](c, ls, err, false)
		}
		ls.lastFinishReason = response.FinishReason
		if response.FinishReason == FinishReasonError {
			return failedExecution[O](c, ls, &ProviderError{Reason: response.RawFinishReason}, false)
		}
		if response.FinishReason == FinishReasonLength {
			return failedExecution[O](c, ls, &ProviderError{Reason: "response reached its output limit before completing a turn"}, false)
		}
		if len(response.Message.Content) == 0 {
			return failedExecution[O](c, ls, &ProviderError{Reason: "empty response from provider"}, false)
		}

		assistant := response.Message
		ls.messages = append(ls.messages, assistant)
		if err := ls.emit(&AssistantCommittedEvent{
			eventBase: newEventBase(EventAuthoritative, EventTypeAssistantCommitted, ls.iteration),
			Message:   cloneMessages([]Message{assistant})[0],
		}); err != nil {
			return failedExecution[O](c, ls, err, true)
		}

		outcome := completeAssistantTurn(c, ls, assistant, evaluator, false, nil)
		if outcome.suspension != nil {
			return suspendedExecution[O](c, ls, *outcome.suspension)
		}
		execution, continueRun, err := finishTurn(c, ls, outcome)
		if err != nil || execution != nil {
			return execution, err
		}
		if continueRun {
			ls.iteration++
		}
	}
}

func finishTurn[O any](c *agentCore, ls *loopState, outcome turnOutcome[O]) (*Execution[O], bool, error) {
	candidate := CompletionCandidate{}
	if outcome.candidate != nil {
		candidate = outcome.candidate.meta
	}
	if err := ls.emit(&TurnEndedEvent{
		eventBase: newEventBase(EventLifecycle, EventTypeTurnEnded, ls.iteration),
		Candidate: candidate,
		Usage:     ls.totalUsage,
	}); err != nil {
		execution, failure := failedExecution[O](c, ls, err, true)
		return execution, false, failure
	}
	if outcome.fatal != nil {
		execution, failure := failedExecution[O](c, ls, outcome.fatal, false)
		return execution, false, failure
	}

	decision := TurnDecision{Action: TurnDefault}
	if ls.options.turnHook != nil {
		hookTurn := Turn{
			Index:     ls.iteration,
			Messages:  cloneMessages(ls.messages),
			Assistant: cloneMessages([]Message{outcome.assistant})[0],
			Results:   cloneToolResults(outcome.results),
			Usage:     ls.totalUsage,
			Candidate: candidate,
		}
		var err error
		decision, err = ls.options.turnHook(ls.ctx, hookTurn)
		if err != nil {
			execution, failure := failedExecution[O](c, ls, err, false)
			return execution, false, failure
		}
	}
	if err := validateTurnDecision(decision); err != nil {
		execution, failure := failedExecution[O](c, ls, err, false)
		return execution, false, failure
	}

	switch decision.Action {
	case TurnContinue:
		ls.messages = append(ls.messages, cloneMessages(decision.Inject)...)
		if err := ls.emit(&TurnMessagesInjectedEvent{
			eventBase: newEventBase(EventAuthoritative, EventTypeTurnMessagesInjected, ls.iteration),
			Messages:  cloneMessages(decision.Inject),
		}); err != nil {
			execution, failure := failedExecution[O](c, ls, err, true)
			return execution, false, failure
		}
		return nil, true, nil
	case TurnStop:
		if outcome.candidate != nil {
			return completedExecution(c, ls, outcome.candidate.output)
		}
		return stoppedExecution[O](c, ls)
	case TurnDefault:
		if outcome.candidate != nil && !outcome.continuation {
			return completedExecution(c, ls, outcome.candidate.output)
		}
		return nil, true, nil
	default:
		execution, failure := failedExecution[O](c, ls, fmt.Errorf("%w: unknown action %d", ErrTurnDecision, decision.Action), false)
		return execution, false, failure
	}
}

func validateTurnDecision(decision TurnDecision) error {
	if decision.Action != TurnDefault && decision.Action != TurnContinue && decision.Action != TurnStop {
		return fmt.Errorf("%w: unknown action %d", ErrTurnDecision, decision.Action)
	}
	if decision.Action != TurnContinue && len(decision.Inject) > 0 {
		return fmt.Errorf("%w: injected messages require TurnContinue", ErrTurnDecision)
	}
	if decision.Action == TurnContinue && len(decision.Inject) == 0 {
		return fmt.Errorf("%w: TurnContinue requires at least one user message", ErrTurnDecision)
	}
	for _, message := range decision.Inject {
		if message.Role != RoleUser {
			return fmt.Errorf("%w: turn hook may inject only user messages", ErrTurnDecision)
		}
	}
	return nil
}

func (c *agentCore) requestTurn(ls *loopState) (*ChatResponse, error) {
	streaming := ls.options.modelStreaming != nil && *ls.options.modelStreaming
	if streaming {
		if model, ok := c.model.(StreamModel); ok {
			request, err := c.buildRequest(ls, true)
			if err != nil {
				return nil, err
			}
			modelCtx, span := safeStartModel(ls.ctx, ls.instrumentation, ModelOperation{
				Agent: ls.agentIdentity, Model: ls.modelMetadata, Run: ls.runMetadata,
				Request: *request, Iteration: ls.iteration,
			})
			stream, err := model.RequestStream(modelCtx, request)
			if err != nil {
				wrapped := fmt.Errorf("model request: %w", err)
				safeEndModel(span, ModelOperationResult{Error: wrapped})
				return nil, wrapped
			}
			message, usage, finishReason, err := c.consumeStream(ls, stream, span)
			if err != nil {
				safeEndModel(span, ModelOperationResult{Error: err})
				return nil, err
			}
			response := &ChatResponse{Message: message, Usage: usage, FinishReason: finishReason, RawFinishReason: string(finishReason)}
			safeEndModel(span, ModelOperationResult{Response: response})
			return response, nil
		}
	}
	request, err := c.buildRequest(ls, false)
	if err != nil {
		return nil, err
	}
	modelCtx, span := safeStartModel(ls.ctx, ls.instrumentation, ModelOperation{
		Agent: ls.agentIdentity, Model: ls.modelMetadata, Run: ls.runMetadata,
		Request: *request, Iteration: ls.iteration,
	})
	response, err := c.model.Request(modelCtx, request)
	if err != nil {
		wrapped := fmt.Errorf("model request: %w", err)
		safeEndModel(span, ModelOperationResult{Response: response, Error: wrapped})
		return nil, wrapped
	}
	safeEndModel(span, ModelOperationResult{Response: response})
	return response, nil
}

func (c *agentCore) consumeStream(ls *loopState, stream *StreamResult, span ...ModelOperationSpan) (Message, Usage, FinishReason, error) {
	var textContent string
	var thinkingContent string
	var thinkingSignature, thinkingProvider, thinkingID string
	var usage Usage
	var finishReason FinishReason
	type accumulator struct {
		id   string
		name string
		args string
	}
	calls := make(map[string]*accumulator)
	var order []string
	for event := range stream.Events {
		if len(span) > 0 {
			safeObserveStreamEvent(span[0], event)
		}
		switch event.Type {
		case StreamEventTextDelta:
			textContent += event.Delta
			if err := ls.emit(&TextPreviewEvent{
				eventBase: newEventBase(EventPreview, EventTypeTextPreview, ls.iteration),
				Delta:     event.Delta,
			}); err != nil {
				return Message{}, Usage{}, FinishReasonError, err
			}
		case StreamEventThinkingDelta:
			thinkingContent += event.Delta
			if event.Signature != "" {
				thinkingSignature = event.Signature
			}
			if event.ProviderName != "" {
				thinkingProvider = event.ProviderName
			}
			if event.ThinkingID != "" {
				thinkingID = event.ThinkingID
			}
			if err := ls.emit(&ThinkingPreviewEvent{
				eventBase:    newEventBase(EventPreview, EventTypeThinkingPreview, ls.iteration),
				Delta:        event.Delta,
				Signature:    event.Signature,
				ProviderName: event.ProviderName,
				ThinkingID:   event.ThinkingID,
			}); err != nil {
				return Message{}, Usage{}, FinishReasonError, err
			}
		case StreamEventToolCallStart:
			if event.ToolUse == nil {
				continue
			}
			if _, exists := calls[event.ToolUse.ID]; exists {
				return Message{}, Usage{}, FinishReasonError, fmt.Errorf("stream contained duplicate tool call ID %q", event.ToolUse.ID)
			}
			calls[event.ToolUse.ID] = &accumulator{id: event.ToolUse.ID, name: event.ToolUse.Name}
			order = append(order, event.ToolUse.ID)
			if err := ls.emit(&ToolCallPreviewEvent{
				eventBase: newEventBase(EventPreview, EventTypeToolCallPreview, ls.iteration),
				Call:      *event.ToolUse,
			}); err != nil {
				return Message{}, Usage{}, FinishReasonError, err
			}
		case StreamEventToolCallDelta:
			if call, ok := calls[event.ToolCallID]; ok {
				call.args += event.Delta
			}
			if err := ls.emit(&ToolArgumentPreviewEvent{
				eventBase:  newEventBase(EventPreview, EventTypeToolArgumentPreview, ls.iteration),
				ToolCallID: event.ToolCallID,
				Delta:      event.Delta,
			}); err != nil {
				return Message{}, Usage{}, FinishReasonError, err
			}
		case StreamEventDone:
			if event.Usage != nil {
				usage = *event.Usage
			}
			if event.FinishReason != "" {
				finishReason = event.FinishReason
			}
		case StreamEventError:
			return Message{}, Usage{}, FinishReasonError, event.Error
		}
	}

	message := Message{Role: RoleAssistant}
	if thinkingContent != "" {
		message.Content = append(message.Content, Part{Type: ContentThinking, Thinking: &ThinkingBlock{
			Text:         thinkingContent,
			ID:           thinkingID,
			Signature:    thinkingSignature,
			ProviderName: thinkingProvider,
		}})
	}
	if textContent != "" {
		message.Content = append(message.Content, Part{Type: ContentText, Text: textContent})
	}
	for _, id := range order {
		call := calls[id]
		input := map[string]any{}
		if call.args != "" {
			if err := json.Unmarshal([]byte(call.args), &input); err != nil {
				return Message{}, Usage{}, FinishReasonError, fmt.Errorf("decode streamed tool arguments for %q: %w", call.id, err)
			}
		}
		message.Content = append(message.Content, Part{Type: ContentToolUse, ToolUse: &ToolUse{ID: call.id, Name: call.name, Input: input}})
	}
	return message, usage, finishReason, nil
}

func completedExecution[O any](c *agentCore, ls *loopState, output O) (*Execution[O], bool, error) {
	execution := &Execution[O]{Status: ExecutionCompleted, Result: makeResult(ls, output)}
	if err := ls.emit(&RunCompletedEvent{
		eventBase:    newEventBase(EventLifecycle, EventTypeRunCompleted, ls.iteration),
		Usage:        ls.totalUsage,
		FinishReason: ls.lastFinishReason,
	}); err != nil {
		execution, failure := failedExecution[O](c, ls, err, true)
		return execution, false, failure
	}
	if err := ls.emit(&RunEndedEvent{
		eventBase: newEventBase(EventLifecycle, EventTypeRunEnded, ls.iteration),
		Status:    ExecutionCompleted,
	}); err != nil {
		execution, failure := failedExecution[O](c, ls, err, true)
		return execution, false, failure
	}
	return execution, false, nil
}

func suspendedExecution[O any](c *agentCore, ls *loopState, suspension Suspension) (*Execution[O], error) {
	execution := &Execution[O]{
		Status:     ExecutionSuspended,
		Result:     makeResult[O](ls, *new(O)),
		Suspension: &suspension,
	}
	if err := ls.emit(&RunSuspendedEvent{
		eventBase:  newEventBase(EventLifecycle, EventTypeRunSuspended, ls.iteration),
		Suspension: suspension,
	}); err != nil {
		return failedExecution[O](c, ls, err, true)
	}
	return execution, nil
}

func stoppedExecution[O any](c *agentCore, ls *loopState) (*Execution[O], bool, error) {
	execution := &Execution[O]{Status: ExecutionStopped, Result: makeResult[O](ls, *new(O))}
	if err := ls.emit(&RunEndedEvent{
		eventBase: newEventBase(EventLifecycle, EventTypeRunEnded, ls.iteration),
		Status:    ExecutionStopped,
	}); err != nil {
		execution, failure := failedExecution[O](c, ls, err, true)
		return execution, false, failure
	}
	return execution, false, nil
}

func interruptedExecution[O any](_ *agentCore, ls *loopState, cause error, sinkFailure bool) (*Execution[O], error) {
	if !sinkFailure {
		_ = ls.emit(&RunInterruptedEvent{eventBase: newEventBase(EventLifecycle, EventTypeRunInterrupted, ls.iteration)})
		_ = ls.emit(&RunEndedEvent{
			eventBase: newEventBase(EventLifecycle, EventTypeRunEnded, ls.iteration),
			Status:    ExecutionInterrupted,
		})
	}
	return &Execution[O]{Status: ExecutionInterrupted, Result: makeResult[O](ls, *new(O))}, cause
}

func failedExecution[O any](_ *agentCore, ls *loopState, cause error, sinkFailure bool) (*Execution[O], error) {
	if !sinkFailure {
		_ = ls.emit(&RunErrorEvent{
			eventBase: newEventBase(EventLifecycle, EventTypeRunError, ls.iteration),
			Error:     cause,
		})
		_ = ls.emit(&RunEndedEvent{
			eventBase: newEventBase(EventLifecycle, EventTypeRunEnded, ls.iteration),
			Status:    ExecutionFailed,
		})
	}
	return &Execution[O]{Status: ExecutionFailed, Result: makeResult[O](ls, *new(O))}, cause
}

func makeResult[O any](ls *loopState, output O) *Result[O] {
	return &Result[O]{
		Output:       output,
		Messages:     cloneMessages(ls.messages),
		ToolCalls:    cloneToolUses(ls.allToolCalls),
		ToolResults:  cloneToolResults(ls.allToolResults),
		FinishReason: ls.lastFinishReason,
		Usage:        ls.totalUsage,
		Retries:      ls.totalRetries,
		historyLen:   ls.historyLen,
	}
}

func executionSnapshot[O any](execution *Execution[O]) ExecutionSnapshot {
	if execution == nil || execution.Result == nil {
		return ExecutionSnapshot{}
	}
	snapshot := ExecutionSnapshot{
		Status:      execution.Status,
		Messages:    cloneMessages(execution.Result.Messages),
		ToolCalls:   cloneToolUses(execution.Result.ToolCalls),
		ToolResults: cloneToolResults(execution.Result.ToolResults),
		Usage:       execution.Result.Usage,
	}
	if execution.Suspension != nil {
		suspension := *execution.Suspension
		suspension.Payload = append([]byte(nil), suspension.Payload...)
		snapshot.Suspension = &suspension
	}
	return snapshot
}
