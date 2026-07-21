package agentic

import (
	"context"
	"fmt"
	"strings"
)

// HistoryProcessor transforms the message history before it's sent to the model.
// The processor affects only what's sent to the model — the agent's internal
// message list remains complete for RunResult.AllMessages().
type HistoryProcessor interface {
	Process(ctx context.Context, messages []Message) ([]Message, error)
}

// HistoryProcessorFunc is a function adapter for HistoryProcessor.
type HistoryProcessorFunc func(ctx context.Context, messages []Message) ([]Message, error)

func (f HistoryProcessorFunc) Process(ctx context.Context, messages []Message) ([]Message, error) {
	return f(ctx, messages)
}

// TruncateHistory keeps the system prompt (if any) plus the last maxMessages messages.
func TruncateHistory(maxMessages int) HistoryProcessor {
	return HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
		if len(messages) <= maxMessages {
			return messages, nil
		}

		// Preserve system prompt if it's the first message
		var prefix []Message
		rest := messages
		if len(messages) > 0 && messages[0].Role == RoleSystem {
			prefix = messages[:1]
			rest = messages[1:]
		}

		if len(rest) <= maxMessages {
			return messages, nil
		}

		// Keep last maxMessages from rest
		truncated := rest[len(rest)-maxMessages:]
		result := make([]Message, 0, len(prefix)+len(truncated))
		result = append(result, prefix...)
		result = append(result, truncated...)
		return result, nil
	})
}

// SlidingWindowHistory keeps a sliding window of messages within a token budget.
// The system prompt is always preserved. tokenCounter estimates the token count
// for a single message.
func SlidingWindowHistory(maxTokens int, tokenCounter func(Message) int) HistoryProcessor {
	return HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
		if len(messages) == 0 {
			return messages, nil
		}

		// Preserve system prompt
		var systemMsg []Message
		rest := messages
		budget := maxTokens
		if messages[0].Role == RoleSystem {
			systemMsg = messages[:1]
			rest = messages[1:]
			budget -= tokenCounter(messages[0])
		}

		if budget <= 0 {
			return systemMsg, nil
		}

		// Walk backwards, accumulating messages that fit
		var window []Message
		for i := len(rest) - 1; i >= 0; i-- {
			cost := tokenCounter(rest[i])
			if budget-cost < 0 {
				break
			}
			budget -= cost
			window = append([]Message{rest[i]}, window...)
		}

		result := make([]Message, 0, len(systemMsg)+len(window))
		result = append(result, systemMsg...)
		result = append(result, window...)
		return result, nil
	})
}

// SummarizeHistory uses an LLM to summarize older messages when the history
// exceeds maxMessages. The summary replaces older messages as a single user message.
// The most recent maxMessages are always preserved verbatim.
func SummarizeHistory(model Model, maxMessages int) HistoryProcessor {
	return HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
		if len(messages) <= maxMessages {
			return messages, nil
		}

		// Preserve system prompt
		var systemMsg []Message
		rest := messages
		if len(messages) > 0 && messages[0].Role == RoleSystem {
			systemMsg = messages[:1]
			rest = messages[1:]
		}

		if len(rest) <= maxMessages {
			return messages, nil
		}

		// Split into old (to summarize) and recent (to keep)
		cutoff := len(rest) - maxMessages
		old := rest[:cutoff]
		recent := rest[cutoff:]

		// Build summary prompt from old messages
		var sb strings.Builder
		sb.WriteString("Summarize the following conversation concisely, preserving key facts and decisions:\n\n")
		for _, msg := range old {
			sb.WriteString(fmt.Sprintf("[%s]: %s\n", msg.Role, msg.GetTextContent()))
		}

		summaryReq := &ChatRequest{
			Model: model.Name(),
			Messages: []Message{
				NewTextMessage(RoleUser, sb.String()),
			},
		}

		resp, err := model.Request(ctx, summaryReq)
		if err != nil {
			return nil, fmt.Errorf("summarize history: %w", err)
		}

		// Refuse to replace real history with an empty summary. The messages
		// being summarized are dropped from what the model sees, so accepting
		// an empty summary here discards them permanently.
		summaryText := resp.Message.GetTextContent()
		if summaryText == "" {
			return nil, fmt.Errorf("summarize history: %w", &ProviderError{Reason: "summary response contained no text"})
		}
		summaryMsg := NewTextMessage(RoleUser, fmt.Sprintf("[Conversation summary]: %s", summaryText))

		result := make([]Message, 0, len(systemMsg)+1+len(recent))
		result = append(result, systemMsg...)
		result = append(result, summaryMsg)
		result = append(result, recent...)
		return result, nil
	})
}

// ChainProcessors applies multiple processors in sequence.
// Each processor receives the output of the previous one.
func ChainProcessors(processors ...HistoryProcessor) HistoryProcessor {
	return HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
		var err error
		current := messages
		for _, p := range processors {
			current, err = p.Process(ctx, current)
			if err != nil {
				return nil, err
			}
		}
		return current, nil
	})
}
