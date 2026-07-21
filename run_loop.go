package agentic

import (
	"context"
	"errors"
	"fmt"

	"github.com/regularkevvv/agentic/internal/core"
)

// loopState contains all mutable data for one run. It is deliberately
// non-generic; exact dependency types are restored only inside typed adapters.
type loopState struct {
	ctx              context.Context
	deps             dependencyEnvelope
	messages         []Message
	historyLen       int
	maxIterations    int
	usageLimits      *UsageLimits
	endStrategy      EndStrategy
	historyProcessor HistoryProcessor
	options          *runOptions

	allToolCalls      []ToolUse
	allToolResults    []ToolExecutionResult
	totalUsage        Usage
	retryCounts       map[string]int
	totalRetries      int
	validationRetries int
}

func (c *agentCore) run(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*Result[string], error) {
	if err := c.preflight(ctx, deps); err != nil {
		return nil, err
	}
	return c.runAfterPreflight(ctx, prompt, deps, opts...)
}

func (c *agentCore) runAfterPreflight(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*Result[string], error) {
	ls, err := c.prepareLoopAfterPreflight(ctx, prompt, deps, opts...)
	if err != nil {
		return nil, err
	}

	for iteration := 0; iteration < ls.maxIterations; iteration++ {
		if err := ls.checkPreRequestLimits(); err != nil {
			return ls.makeResult("", FinishReasonStop), err
		}
		request, err := c.buildRequest(ls, false)
		if err != nil {
			return nil, err
		}
		response, err := c.model.Request(ctx, request)
		if err != nil {
			return nil, fmt.Errorf("model request: %w", err)
		}
		ls.totalUsage.Add(response.Usage)
		if err := ls.checkPostResponseLimits(); err != nil {
			return ls.makeResult("", FinishReasonStop), err
		}
		// A provider that reported an in-band failure has not produced a usable
		// turn, even though the transport succeeded. Surface it rather than
		// letting partial content read as a complete answer.
		if response.FinishReason == FinishReasonError {
			return ls.makeResult("", FinishReasonError), &ProviderError{Reason: response.RawFinishReason}
		}
		if len(response.Message.Content) == 0 {
			return ls.makeResult("", FinishReasonError), &ProviderError{Reason: "empty response from provider"}
		}

		assistantMsg := response.Message
		finishReason := response.FinishReason
		ls.messages = append(ls.messages, assistantMsg)
		toolUses := assistantMsg.GetToolUses()
		if len(toolUses) == 0 {
			output := assistantMsg.GetTextContent()
			if validationErr := c.validateOutput(ctx, deps, output); validationErr != nil {
				ls.validationRetries++
				if ls.validationRetries > c.config.maxValidationRetries {
					return nil, fmt.Errorf("output validation failed after %d retries: %w", c.config.maxValidationRetries, validationErr)
				}
				ls.messages = append(ls.messages, NewTextMessage(RoleUser, fmt.Sprintf("Output validation error: %s\nPlease try again.", validationErr)))
				continue
			}
			return ls.makeResult(output, finishReason), nil
		}

		outcome, err := c.processToolUses(ls, toolUses)
		if err != nil {
			return ls.makeResult("", FinishReasonStop), err
		}
		if outcome.hasOutput && !outcome.retryRequested {
			return ls.makeResult(assistantMsg.GetTextContent(), finishReason), nil
		}
	}

	return ls.makeResult("Maximum iterations reached", FinishReasonLength), &MaxIterationsError{MaxIterations: ls.maxIterations}
}

func (c *agentCore) prepareLoop(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*loopState, error) {
	if err := c.preflight(ctx, deps); err != nil {
		return nil, err
	}
	return c.prepareLoopAfterPreflight(ctx, prompt, deps, opts...)
}

func (c *agentCore) prepareLoopAfterPreflight(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*loopState, error) {
	options := applyRunOptions(opts)
	systemPrompt, err := c.resolveSystemPrompt(ctx, deps)
	if err != nil {
		return nil, fmt.Errorf("system prompt: %w", err)
	}

	messages := append([]Message(nil), options.messages...)
	if c.systemPromptSuffix != "" {
		if systemPrompt != "" {
			systemPrompt += "\n\n" + c.systemPromptSuffix
		} else {
			systemPrompt = c.systemPromptSuffix
		}
	}
	hasSystemPrompt := false
	for _, message := range messages {
		if message.Role == RoleSystem {
			hasSystemPrompt = true
			break
		}
	}
	if !hasSystemPrompt && systemPrompt != "" {
		messages = append([]Message{NewTextMessage(RoleSystem, systemPrompt)}, messages...)
	}
	messages = append(messages, NewTextMessage(RoleUser, prompt))

	maxIterations := c.config.maxIterations
	if options.maxIterations != nil {
		maxIterations = *options.maxIterations
	}
	usageLimits := c.config.usageLimits
	if options.usageLimits != nil {
		usageLimits = options.usageLimits
	}
	endStrategy := c.config.endStrategy
	if options.endStrategy != nil {
		endStrategy = *options.endStrategy
	}
	historyProcessor := c.config.historyProcessor
	if options.historyProcessor != nil {
		historyProcessor = options.historyProcessor
	}

	return &loopState{
		ctx:              ctx,
		deps:             deps,
		messages:         messages,
		historyLen:       len(options.messages),
		maxIterations:    maxIterations,
		usageLimits:      usageLimits,
		endStrategy:      endStrategy,
		historyProcessor: historyProcessor,
		options:          options,
	}, nil
}

func (c *agentCore) buildRequest(ls *loopState, stream bool) (*ChatRequest, error) {
	requestMessages := ls.messages
	if ls.historyProcessor != nil {
		var err error
		requestMessages, err = ls.historyProcessor.Process(ls.ctx, ls.messages)
		if err != nil {
			return nil, fmt.Errorf("history processor: %w", err)
		}
	}
	request := &ChatRequest{
		Model:       c.model.Name(),
		Messages:    requestMessages,
		Temperature: firstNonNil(ls.options.temperature, c.config.temperature),
		MaxTokens:   firstNonNil(ls.options.maxTokens, c.config.maxTokens),
		TopP:        firstNonNil(ls.options.topP, c.config.topP),
		Stream:      stream,
	}
	if c.registry != nil && c.registry.Count() > 0 {
		availableTools := c.registry.Tools()
		if c.config.toolPrepareFunc != nil {
			var err error
			availableTools, err = c.config.toolPrepareFunc(ls.ctx, ls.deps, availableTools)
			if err != nil {
				return nil, fmt.Errorf("tool prepare: %w", err)
			}
		}
		request.Tools = availableTools
		request.ToolChoice = firstNonNil(ls.options.toolChoice, c.config.toolChoice)
	}
	request.ResponseFormat = c.responseFormat
	request.Thinking = c.config.thinking
	return request, nil
}

func (ls *loopState) checkPreRequestLimits() error {
	if ls.usageLimits == nil {
		return nil
	}
	return ls.usageLimits.checkBeforeRequest(ls.totalUsage)
}

func (ls *loopState) checkPostResponseLimits() error {
	if ls.usageLimits == nil {
		return nil
	}
	return ls.usageLimits.checkAfterResponse(ls.totalUsage)
}

func (ls *loopState) checkToolCallLimits(pendingCalls int) error {
	if ls.usageLimits == nil {
		return nil
	}
	return ls.usageLimits.checkBeforeToolCalls(ls.totalUsage.ToolCalls, pendingCalls)
}

func (c *agentCore) checkOutputTool(toolUses []ToolUse) int {
	for i, toolUse := range toolUses {
		if c.outputToolNames[toolUse.Name] {
			return i
		}
	}
	return -1
}

type toolUseOutcome struct {
	hasOutput      bool
	retryRequested bool
	results        []ToolExecutionResult
}

func (c *agentCore) processToolUses(ls *loopState, toolUses []ToolUse) (toolUseOutcome, error) {
	outputIndex := c.checkOutputTool(toolUses)
	if outputIndex >= 0 && ls.endStrategy == EndStrategyEarly {
		ls.allToolCalls = append(ls.allToolCalls, toolUses[:outputIndex+1]...)
		return toolUseOutcome{hasOutput: true}, nil
	}

	ls.allToolCalls = append(ls.allToolCalls, toolUses...)
	executable := make([]ToolUse, 0, len(toolUses))
	for _, toolUse := range toolUses {
		if !c.outputToolNames[toolUse.Name] {
			executable = append(executable, toolUse)
		}
	}
	if len(executable) == 0 {
		return toolUseOutcome{hasOutput: outputIndex >= 0}, nil
	}
	if c.registry == nil {
		return toolUseOutcome{}, fmt.Errorf("model requested tool calls but no tools are registered")
	}
	if err := ls.checkToolCallLimits(len(executable)); err != nil {
		return toolUseOutcome{}, err
	}

	parentMessages := ls.messages
	if len(parentMessages) > 0 {
		parentMessages = parentMessages[:len(parentMessages)-1]
	}
	var retryCounts map[string]int
	if len(ls.retryCounts) > 0 {
		retryCounts = make(map[string]int, len(ls.retryCounts))
		for name, count := range ls.retryCounts {
			retryCounts[name] = count
		}
	}
	executionCtx := core.WithToolExecutionState(ls.ctx, core.ToolExecutionState{
		Messages:    append([]Message(nil), parentMessages...),
		RetryCounts: retryCounts,
	})
	results, err := c.registry.ExecuteBatch(executionCtx, executable, ls.deps)
	if err != nil {
		return toolUseOutcome{}, err
	}
	if len(results) != len(executable) {
		return toolUseOutcome{}, fmt.Errorf("execute tool batch: expected %d results, got %d", len(executable), len(results))
	}

	outcome := toolUseOutcome{hasOutput: outputIndex >= 0, results: results}
	for i, result := range results {
		toolCall := executable[i]
		if result.Error != nil {
			var retry *ModelRetry
			if errors.As(result.Error, &retry) {
				maxRetries := c.config.retryConfig.MaxRetries
				if handler, ok := c.registry.Get(toolCall.Name); ok {
					maxRetries = c.resolveMaxRetries(handler, maxRetries)
				}
				if ls.retryCounts[toolCall.Name] < maxRetries {
					if ls.retryCounts == nil {
						ls.retryCounts = make(map[string]int)
					}
					ls.retryCounts[toolCall.Name]++
					ls.totalRetries++
					ls.messages = append(ls.messages, NewToolResultMessageFor(toolCall.ID, toolCall.Name, retry.Message, true))
					outcome.retryRequested = true
					continue
				}
			}
		}
		ls.totalUsage.ToolCalls++
		ls.allToolResults = append(ls.allToolResults, result)
		ls.messages = append(ls.messages, NewToolResultMessageFor(result.ToolUseID, result.ToolName, FormatToolResult(result.Content), result.IsError))
	}

	if outcome.hasOutput && outcome.retryRequested {
		for _, toolUse := range toolUses {
			if c.outputToolNames[toolUse.Name] {
				ls.messages = append(ls.messages, NewToolResultMessageFor(toolUse.ID, toolUse.Name, "Output discarded because another tool requested a retry.", true))
			}
		}
	}
	return outcome, nil
}

func (ls *loopState) makeResult(output string, finishReason FinishReason) *Result[string] {
	return &Result[string]{
		Output:       output,
		Messages:     ls.messages,
		ToolCalls:    ls.allToolCalls,
		ToolResults:  ls.allToolResults,
		FinishReason: finishReason,
		Usage:        ls.totalUsage,
		Retries:      ls.totalRetries,
		historyLen:   ls.historyLen,
	}
}
