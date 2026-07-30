package contextpolicy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	agentic "github.com/regularkevvv/agentic"
)

const (
	DefaultStructuredSummaryBytes = 8 * 1024
	DefaultStructuredEntryBytes   = 512
)

// StructuredConfig bounds deterministic extraction.
type StructuredConfig struct {
	MaxSummaryBytes int
	MaxEntryBytes   int
}

// StructuredCompactor extracts a deterministic, bounded description of an old
// protocol-valid prefix.
type StructuredCompactor struct {
	config StructuredConfig
}

func NewStructuredCompactor(config StructuredConfig) (*StructuredCompactor, error) {
	if config.MaxSummaryBytes == 0 {
		config.MaxSummaryBytes = DefaultStructuredSummaryBytes
	}
	if config.MaxEntryBytes == 0 {
		config.MaxEntryBytes = DefaultStructuredEntryBytes
	}
	if config.MaxSummaryBytes < 256 || config.MaxEntryBytes < 32 ||
		config.MaxEntryBytes > config.MaxSummaryBytes {
		return nil, errors.New("invalid structured compactor limits")
	}
	return &StructuredCompactor{config: config}, nil
}

func (c *StructuredCompactor) Summarize(ctx context.Context, messages []agentic.Message) (agentic.Message, error) {
	if err := ctx.Err(); err != nil {
		return agentic.Message{}, err
	}
	const prefix = "<harness_compaction version=\"1\" kind=\"structured\">\n"
	const suffix = "\n</harness_compaction>"
	var builder strings.Builder
	builder.WriteString(prefix)
	for index, message := range messages {
		line := structuredLine(index, message, c.config.MaxEntryBytes)
		if builder.Len()+len(line)+len(suffix) > c.config.MaxSummaryBytes {
			remaining := len(messages) - index
			marker := fmt.Sprintf("... %d older messages omitted", remaining)
			if builder.Len()+len(marker)+1+len(suffix) <= c.config.MaxSummaryBytes {
				if builder.Len() > len(prefix) {
					builder.WriteByte('\n')
				}
				builder.WriteString(marker)
			}
			break
		}
		if builder.Len() > len(prefix) {
			builder.WriteByte('\n')
		}
		builder.WriteString(line)
	}
	builder.WriteString(suffix)
	return agentic.NewTextMessage(agentic.RoleUser, builder.String()), nil
}

func structuredLine(index int, message agentic.Message, limit int) string {
	var details []string
	text := strings.TrimSpace(message.GetTextContent())
	if text != "" {
		details = append(details, "text="+quotedBounded(text, limit/2))
	}
	uses := message.GetToolUses()
	if len(uses) > 0 {
		names := make([]string, len(uses))
		for i, call := range uses {
			names[i] = call.Name + "#" + call.ID
		}
		details = append(details, "calls="+strings.Join(names, ","))
	}
	results := message.GetToolResults()
	if len(results) > 0 {
		values := make([]string, len(results))
		for i, result := range results {
			status := "ok"
			if result.IsError {
				status = "error"
			}
			values[i] = result.Name + "#" + result.ToolUseID + ":" + status
		}
		details = append(details, "results="+strings.Join(values, ","))
	}
	line := fmt.Sprintf("%d role=%s", index, message.Role)
	if len(details) > 0 {
		line += " " + strings.Join(details, " ")
	}
	return truncateUTF8(line, limit)
}

func quotedBounded(value string, limit int) string {
	value = strings.ReplaceAll(value, "\n", "\\n")
	return fmt.Sprintf("%q", truncateUTF8(value, limit))
}

// Summarizer is the model-independent port used by LLMSummaryCompactor.
type Summarizer interface {
	Summarize(context.Context, []agentic.Message) (string, error)
}

// SummarizerFunc adapts a function to Summarizer.
type SummarizerFunc func(context.Context, []agentic.Message) (string, error)

func (f SummarizerFunc) Summarize(ctx context.Context, messages []agentic.Message) (string, error) {
	return f(ctx, messages)
}

// ModelSummarizer is the standard Agentic Model adapter for the LLM
// compactor. It uses a separate request and never mutates the caller's
// transcript.
type ModelSummarizer struct {
	model       agentic.Model
	instruction string
}

func NewModelSummarizer(model agentic.Model, instruction string) (*ModelSummarizer, error) {
	if model == nil {
		return nil, errors.New("summary model is required")
	}
	if strings.TrimSpace(instruction) == "" {
		instruction = "Summarize the conversation while preserving facts, decisions, unresolved work, and tool outcomes."
	}
	return &ModelSummarizer{model: model, instruction: instruction}, nil
}

func (s *ModelSummarizer) Summarize(ctx context.Context, messages []agentic.Message) (string, error) {
	encoded, err := json.Marshal(cloneMessages(messages))
	if err != nil {
		return "", fmt.Errorf("encode summary input: %w", err)
	}
	response, err := s.model.Request(ctx, &agentic.ChatRequest{
		Model: s.model.Name(),
		Messages: []agentic.Message{agentic.NewTextMessage(
			agentic.RoleUser,
			s.instruction+"\n\nCanonical conversation JSON:\n"+string(encoded),
		)},
	})
	if err != nil {
		return "", err
	}
	if response == nil {
		return "", errors.New("summary model returned no response")
	}
	return response.Message.GetTextContent(), nil
}

// LLMSummaryCompactor delegates extraction while retaining deterministic
// framing and a hard output bound. Persisted Compaction recipes prevent the
// summary from being recomputed until a new prefix must be compacted.
type LLMSummaryCompactor struct {
	summarizer Summarizer
	maxBytes   int
}

func NewLLMSummaryCompactor(summarizer Summarizer, maxBytes int) (*LLMSummaryCompactor, error) {
	if summarizer == nil {
		return nil, errors.New("context summarizer is required")
	}
	if maxBytes == 0 {
		maxBytes = DefaultStructuredSummaryBytes
	}
	if maxBytes < 256 {
		return nil, errors.New("LLM summary limit is too small")
	}
	return &LLMSummaryCompactor{summarizer: summarizer, maxBytes: maxBytes}, nil
}

func (c *LLMSummaryCompactor) Summarize(ctx context.Context, messages []agentic.Message) (agentic.Message, error) {
	summary, err := c.summarizer.Summarize(ctx, cloneMessages(messages))
	if err != nil {
		return agentic.Message{}, fmt.Errorf("summarize context: %w", err)
	}
	summary = strings.TrimSpace(summary)
	if summary == "" {
		return agentic.Message{}, errors.New("summarize context: empty summary")
	}
	const prefix = "<harness_compaction version=\"1\" kind=\"llm\">\n"
	const suffix = "\n</harness_compaction>"
	available := c.maxBytes - len(prefix) - len(suffix)
	if available < 1 {
		return agentic.Message{}, errors.New("LLM summary framing exceeds limit")
	}
	summary = truncateUTF8(summary, available)
	return agentic.NewTextMessage(agentic.RoleUser, prefix+summary+suffix), nil
}

func truncateUTF8(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	data := []byte(value)
	if len(data) <= limit {
		return value
	}
	end := limit
	for end > 0 && !utf8.Valid(data[:end]) {
		end--
	}
	return string(data[:end])
}
