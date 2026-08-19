package agentic

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/regularkevvv/agentic/internal/core"
)

func (a *Agent) RunStream(ctx context.Context, prompt string, opts ...RunOption) (*StreamResult, error) {
	return a.core.runStream(ctx, prompt, dependencyEnvelope{}, opts...)
}

func (a *AgentWithDeps[D]) RunStream(ctx context.Context, prompt string, deps D, opts ...RunOption) (*StreamResult, error) {
	return a.core.runStream(ctx, prompt, core.NewDependencyEnvelope(deps), opts...)
}

func (c *agentCore) runStream(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*StreamResult, error) {
	return runStreamWithEvaluator(c, ctx, prompt, deps, textCompletionEvaluator(c), opts...)
}

// runStreamWithEvaluator is the backwards-compatible StreamResult projection
// of the canonical Driver fold. The channel remains intentionally lossless and
// backpressured; harness subscribers use EventSink instead.
func runStreamWithEvaluator[O any](
	c *agentCore,
	ctx context.Context,
	prompt string,
	deps dependencyEnvelope,
	evaluator completionEvaluator[O],
	opts ...RunOption,
) (*StreamResult, error) {
	if err := c.preflight(ctx, deps); err != nil {
		return nil, err
	}
	message := NewTextMessage(RoleUser, prompt)
	ch := make(chan StreamEvent, 64)
	stream := NewStreamResult(ch)
	legacy := newLegacyStreamSink(ch)
	existing := applyRunOptions(opts).eventSink
	combined := fanoutEventSink{first: existing, second: legacy}
	preparedOpts := append(append([]RunOption(nil), opts...), WithRunModelStreaming(true), WithRunEventSink(combined))
	ls, err := c.prepareLoopForDrive(ctx, DriveInput{Mode: DriveStart, Prompt: &message}, deps, preparedOpts...)
	if err != nil {
		return nil, err
	}

	go func() {
		defer close(ch)
		execution, runErr := observeAgentExecution(c, ls, AgentInvocationStart, func() (*Execution[O], error) {
			return driveLoop(c, ls, evaluator, nil, nil)
		})
		if execution != nil {
			stream.SetSnapshot(executionSnapshot(execution))
		}
		if runErr != nil {
			legacy.sendErrorIfNeeded(runErr)
			return
		}
		if execution != nil {
			switch execution.Status {
			case ExecutionSuspended:
				legacy.sendErrorIfNeeded(ErrExecutionSuspended)
			case ExecutionStopped:
				legacy.sendErrorIfNeeded(ErrExecutionStopped)
			case ExecutionInterrupted:
				legacy.sendErrorIfNeeded(ErrExecutionInterrupted)
			case ExecutionFailed:
				legacy.sendErrorIfNeeded(ErrExecutionFailed)
			}
		}
	}()
	return stream, nil
}

type fanoutEventSink struct {
	first  EventSink
	second EventSink
}

func (s fanoutEventSink) Emit(ctx context.Context, event Event) error {
	if s.first != nil {
		if err := s.first.Emit(ctx, event); err != nil {
			return err
		}
	}
	if s.second != nil {
		return s.second.Emit(ctx, event)
	}
	return nil
}

type legacyStreamSink struct {
	ch chan<- StreamEvent

	mu           sync.Mutex
	textSeen     map[int]bool
	callSeen     map[string]bool
	terminalSent bool
}

func newLegacyStreamSink(ch chan<- StreamEvent) *legacyStreamSink {
	return &legacyStreamSink{
		ch:       ch,
		textSeen: make(map[int]bool),
		callSeen: make(map[string]bool),
	}
}

func (s *legacyStreamSink) Emit(_ context.Context, event Event) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch value := event.(type) {
	case *TextPreviewEvent:
		s.textSeen[value.TurnIndex()] = true
		s.ch <- StreamEvent{Type: StreamEventTextDelta, Delta: value.Delta}
	case *ThinkingPreviewEvent:
		s.ch <- StreamEvent{
			Type:         StreamEventThinkingDelta,
			Delta:        value.Delta,
			Signature:    value.Signature,
			ProviderName: value.ProviderName,
			ThinkingID:   value.ThinkingID,
		}
	case *ToolCallPreviewEvent:
		s.sendToolCallStart(value.Call)
	case *ToolArgumentPreviewEvent:
		s.ch <- StreamEvent{Type: StreamEventToolCallDelta, ToolCallID: value.ToolCallID, Delta: value.Delta}
	case *AssistantCommittedEvent:
		if !s.textSeen[value.TurnIndex()] {
			if text := value.Message.GetTextContent(); text != "" {
				s.ch <- StreamEvent{Type: StreamEventTextDelta, Delta: text}
			}
		}
		for _, call := range value.Message.GetToolUses() {
			s.sendToolCallStart(call)
		}
	case *ToolStartedEvent:
		s.sendToolCallStart(value.Call)
	case *ToolResultCommittedEvent:
		s.ch <- StreamEvent{
			Type:       StreamEventToolResult,
			ToolCallID: value.Result.ToolUseID,
			Delta:      FormatToolResult(value.Result.Content),
		}
	case *RunCompletedEvent:
		if !s.terminalSent {
			usage := value.Usage
			s.ch <- StreamEvent{Type: StreamEventDone, Usage: &usage, FinishReason: value.FinishReason}
			s.terminalSent = true
		}
	case *RunErrorEvent:
		s.sendErrorLocked(value.Error)
	case *RunInterruptedEvent:
		s.sendErrorLocked(ErrExecutionInterrupted)
	case *RunSuspendedEvent:
		s.sendErrorLocked(ErrExecutionSuspended)
	case *RunEndedEvent:
		if value.Status == ExecutionStopped {
			s.sendErrorLocked(ErrExecutionStopped)
		}
	}
	return nil
}

func (s *legacyStreamSink) sendToolCallStart(call ToolUse) {
	if s.callSeen[call.ID] {
		return
	}
	s.callSeen[call.ID] = true
	copy := call
	s.ch <- StreamEvent{Type: StreamEventToolCallStart, ToolUse: &copy}
}

func (s *legacyStreamSink) sendErrorIfNeeded(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sendErrorLocked(err)
}

func (s *legacyStreamSink) sendErrorLocked(err error) {
	if s.terminalSent {
		return
	}
	s.ch <- StreamEvent{Type: StreamEventError, Error: err}
	s.terminalSent = true
}

// consumeAndForward remains a private compatibility helper for provider and
// regression tests. Agent runs no longer use it: previews now flow through the
// canonical EventSink before being projected by legacyStreamSink.
func (c *agentCore) consumeAndForward(stream *StreamResult, out chan<- StreamEvent) (Message, Usage, FinishReason, error) {
	var textContent string
	var thinkingContent string
	var thinkingSignature, thinkingProvider, thinkingID string
	var usage Usage
	var finishReason FinishReason
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
			if event.Signature != "" {
				thinkingSignature = event.Signature
			}
			if event.ProviderName != "" {
				thinkingProvider = event.ProviderName
			}
			if event.ThinkingID != "" {
				thinkingID = event.ThinkingID
			}
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
			if event.FinishReason != "" {
				finishReason = event.FinishReason
			}
		case StreamEventError:
			return Message{}, Usage{}, FinishReasonError, event.Error
		default:
			out <- event
		}
	}

	message := Message{Role: RoleAssistant, Content: make([]Part, 0)}
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
	for _, id := range toolCallOrder {
		toolCall := toolCalls[id]
		var input map[string]any
		if toolCall.args != "" {
			_ = json.Unmarshal([]byte(toolCall.args), &input)
		}
		message.Content = append(message.Content, Part{Type: ContentToolUse, ToolUse: &ToolUse{ID: toolCall.id, Name: toolCall.name, Input: input}})
	}
	return message, usage, finishReason, nil
}
