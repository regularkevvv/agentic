package agentic

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/regularkevvv/agentic/internal/core"
)

func (a *Agent) RunStream(ctx context.Context, prompt string, opts ...RunOption) (*StreamResult, error) {
	return a.core.runStream(ctx, prompt, dependencyEnvelope{}, opts...)
}

func (a *AgentWithDeps[D]) RunStream(ctx context.Context, prompt string, deps D, opts ...RunOption) (*StreamResult, error) {
	return a.core.runStream(ctx, prompt, core.NewDependencyEnvelope(deps), opts...)
}

func (c *agentCore) runStream(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*StreamResult, error) {
	if err := c.preflight(ctx, deps); err != nil {
		return nil, err
	}
	streamModel, ok := c.model.(StreamModel)
	if !ok {
		return c.runStreamFallback(ctx, prompt, deps, opts...)
	}
	return c.runStreamTrue(ctx, prompt, deps, streamModel, opts...)
}

func (c *agentCore) runStreamTrue(ctx context.Context, prompt string, deps dependencyEnvelope, model StreamModel, opts ...RunOption) (*StreamResult, error) {
	ls, err := c.prepareLoopAfterPreflight(ctx, prompt, deps, opts...)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamEvent, 64)
	result := NewStreamResult(ch)

	go func() {
		defer close(ch)
		for iteration := 0; iteration < ls.maxIterations; iteration++ {
			if err := ls.checkPreRequestLimits(); err != nil {
				ch <- StreamEvent{Type: StreamEventError, Error: err}
				return
			}
			request, err := c.buildRequest(ls, true)
			if err != nil {
				ch <- StreamEvent{Type: StreamEventError, Error: err}
				return
			}
			stream, err := model.RequestStream(ctx, request)
			if err != nil {
				ch <- StreamEvent{Type: StreamEventError, Error: fmt.Errorf("model request: %w", err)}
				return
			}
			message, usage, err := c.consumeAndForward(stream, ch)
			if err != nil {
				ch <- StreamEvent{Type: StreamEventError, Error: err}
				return
			}
			ls.totalUsage.Add(usage)
			if err := ls.checkPostResponseLimits(); err != nil {
				ch <- StreamEvent{Type: StreamEventError, Error: err}
				return
			}
			ls.messages = append(ls.messages, message)
			toolUses := message.GetToolUses()
			if len(toolUses) == 0 {
				output := message.GetTextContent()
				if validationErr := c.validateOutput(ctx, deps, output); validationErr != nil {
					ls.validationRetries++
					if ls.validationRetries > c.config.maxValidationRetries {
						ch <- StreamEvent{Type: StreamEventError, Error: fmt.Errorf("output validation failed after %d retries: %w", c.config.maxValidationRetries, validationErr)}
						return
					}
					ls.messages = append(ls.messages, NewTextMessage(RoleUser, fmt.Sprintf("Output validation error: %s\nPlease try again.", validationErr)))
					continue
				}
				ch <- StreamEvent{Type: StreamEventDone, Usage: &ls.totalUsage}
				return
			}
			outcome, err := c.processToolUses(ls, toolUses)
			if err != nil {
				ch <- StreamEvent{Type: StreamEventError, Error: err}
				return
			}
			for _, toolResult := range outcome.results {
				ch <- StreamEvent{Type: StreamEventToolResult, Delta: FormatToolResult(toolResult.Content), ToolCallID: toolResult.ToolUseID}
			}
			if outcome.hasOutput && !outcome.retryRequested {
				ch <- StreamEvent{Type: StreamEventDone, Usage: &ls.totalUsage}
				return
			}
		}
		ch <- StreamEvent{Type: StreamEventError, Error: &MaxIterationsError{MaxIterations: ls.maxIterations}}
	}()
	return result, nil
}

func (c *agentCore) consumeAndForward(stream *StreamResult, out chan<- StreamEvent) (Message, Usage, error) {
	var textContent string
	var thinkingContent string
	var usage Usage
	type toolCallAccumulator struct {
		id   string
		name string
		args string
	}
	toolCalls := make(map[string]*toolCallAccumulator)
	var toolCallOrder []string
	for event := range stream.Events {
		switch event.Type {
		case StreamEventTextDelta:
			textContent += event.Delta
			out <- event
		case StreamEventThinkingDelta:
			thinkingContent += event.Delta
			out <- event
		case StreamEventToolCallStart:
			if event.ToolUse != nil {
				toolCalls[event.ToolUse.ID] = &toolCallAccumulator{id: event.ToolUse.ID, name: event.ToolUse.Name}
				toolCallOrder = append(toolCallOrder, event.ToolUse.ID)
			}
			out <- event
		case StreamEventToolCallDelta:
			if toolCall, ok := toolCalls[event.ToolCallID]; ok {
				toolCall.args += event.Delta
			}
			out <- event
		case StreamEventDone:
			if event.Usage != nil {
				usage = *event.Usage
			}
		case StreamEventError:
			return Message{}, Usage{}, event.Error
		default:
			out <- event
		}
	}

	message := Message{Role: RoleAssistant, Content: make([]Part, 0)}
	if thinkingContent != "" {
		message.Content = append(message.Content, Part{Type: ContentThinking, Thinking: &ThinkingBlock{Text: thinkingContent}})
	}
	if textContent != "" {
		message.Content = append(message.Content, Part{Type: ContentText, Text: textContent})
	}
	for _, id := range toolCallOrder {
		toolCall := toolCalls[id]
		var input map[string]interface{}
		if toolCall.args != "" {
			_ = json.Unmarshal([]byte(toolCall.args), &input)
		}
		message.Content = append(message.Content, Part{Type: ContentToolUse, ToolUse: &ToolUse{ID: toolCall.id, Name: toolCall.name, Input: input}})
	}
	return message, usage, nil
}

func (c *agentCore) runStreamFallback(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*StreamResult, error) {
	ch := make(chan StreamEvent, 32)
	result := NewStreamResult(ch)
	go func() {
		defer close(ch)
		runResult, err := c.runAfterPreflight(ctx, prompt, deps, opts...)
		if err != nil {
			ch <- StreamEvent{Type: StreamEventError, Error: err}
			return
		}
		for _, call := range runResult.ToolCalls {
			call := call
			ch <- StreamEvent{Type: StreamEventToolCallStart, ToolUse: &call}
		}
		if runResult.Output != "" {
			ch <- StreamEvent{Type: StreamEventTextDelta, Delta: runResult.Output}
		}
		usage := runResult.Usage
		ch <- StreamEvent{Type: StreamEventDone, Usage: &usage}
	}()
	return result, nil
}
