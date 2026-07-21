package agentic

import (
	"context"
	"errors"
	"testing"
	"time"
)

type historyStubModel struct {
	requests []*ChatRequest
	resp     *ChatResponse
	err      error
}

func (m *historyStubModel) Request(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	m.requests = append(m.requests, req)
	if m.err != nil {
		return nil, m.err
	}
	if m.resp != nil {
		return m.resp, nil
	}
	return &ChatResponse{
		Model:   "history-stub",
		Created: time.Unix(0, 0),
		Message: NewTextMessage(RoleAssistant, "summary"),
	}, nil
}

func (m *historyStubModel) Name() string {
	return "history-stub"
}

func TestHistoryProcessorFunc(t *testing.T) {
	processor := HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
		return append(messages, NewTextMessage(RoleAssistant, "processed")), nil
	})

	got, err := processor.Process(context.Background(), []Message{NewTextMessage(RoleUser, "hello")})
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(got))
	}
	if got[1].GetTextContent() != "processed" {
		t.Fatalf("expected appended message, got %q", got[1].GetTextContent())
	}
}

func TestTruncateHistory(t *testing.T) {
	t.Run("returns original when total length fits", func(t *testing.T) {
		messages := []Message{
			NewTextMessage(RoleUser, "one"),
			NewTextMessage(RoleAssistant, "two"),
		}

		got, err := TruncateHistory(2).Process(context.Background(), messages)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if len(got) != len(messages) {
			t.Fatalf("expected %d messages, got %d", len(messages), len(got))
		}
	})

	t.Run("returns original when only system plus max tail remains", func(t *testing.T) {
		messages := []Message{
			NewTextMessage(RoleSystem, "system"),
			NewTextMessage(RoleUser, "one"),
			NewTextMessage(RoleAssistant, "two"),
		}

		got, err := TruncateHistory(2).Process(context.Background(), messages)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if len(got) != len(messages) {
			t.Fatalf("expected %d messages, got %d", len(messages), len(got))
		}
	})

	t.Run("preserves system and last messages", func(t *testing.T) {
		messages := []Message{
			NewTextMessage(RoleSystem, "system"),
			NewTextMessage(RoleUser, "one"),
			NewTextMessage(RoleAssistant, "two"),
			NewTextMessage(RoleUser, "three"),
			NewTextMessage(RoleAssistant, "four"),
		}

		got, err := TruncateHistory(2).Process(context.Background(), messages)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if len(got) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(got))
		}
		if got[0].Role != RoleSystem {
			t.Fatalf("expected first message to remain system, got %q", got[0].Role)
		}
		if got[1].GetTextContent() != "three" || got[2].GetTextContent() != "four" {
			t.Fatalf("unexpected tail: %#v", got)
		}
	})
}

func TestSlidingWindowHistory(t *testing.T) {
	t.Run("returns empty history unchanged", func(t *testing.T) {
		got, err := SlidingWindowHistory(10, func(Message) int { return 1 }).Process(context.Background(), nil)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("expected empty history, got %d messages", len(got))
		}
	})

	t.Run("returns only system when budget is exhausted by system prompt", func(t *testing.T) {
		messages := []Message{
			NewTextMessage(RoleSystem, "system"),
			NewTextMessage(RoleUser, "one"),
		}

		got, err := SlidingWindowHistory(1, func(Message) int { return 2 }).Process(context.Background(), messages)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if len(got) != 1 || got[0].Role != RoleSystem {
			t.Fatalf("expected only system message, got %#v", got)
		}
	})

	t.Run("keeps most recent messages that fit", func(t *testing.T) {
		messages := []Message{
			NewTextMessage(RoleSystem, "system"),
			NewTextMessage(RoleUser, "one"),
			NewTextMessage(RoleAssistant, "two"),
			NewTextMessage(RoleUser, "three"),
			NewTextMessage(RoleAssistant, "four"),
		}

		costs := map[string]int{
			"system": 1,
			"one":    2,
			"two":    2,
			"three":  3,
			"four":   2,
		}

		got, err := SlidingWindowHistory(6, func(msg Message) int {
			return costs[msg.GetTextContent()]
		}).Process(context.Background(), messages)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if len(got) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(got))
		}
		if got[0].Role != RoleSystem || got[1].GetTextContent() != "three" || got[2].GetTextContent() != "four" {
			t.Fatalf("unexpected window: %#v", got)
		}
	})
}

func TestSummarizeHistory(t *testing.T) {
	t.Run("returns original when history fits", func(t *testing.T) {
		model := &historyStubModel{}
		messages := []Message{
			NewTextMessage(RoleUser, "one"),
			NewTextMessage(RoleAssistant, "two"),
		}

		got, err := SummarizeHistory(model, 2).Process(context.Background(), messages)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if len(got) != len(messages) {
			t.Fatalf("expected %d messages, got %d", len(messages), len(got))
		}
		if len(model.requests) != 0 {
			t.Fatalf("expected summarizer not to be called, got %d requests", len(model.requests))
		}
	})

	t.Run("summarizes older messages and keeps recent tail", func(t *testing.T) {
		model := &historyStubModel{
			resp: &ChatResponse{
				Model:   "history-stub",
				Created: time.Unix(0, 0),
				Message: NewTextMessage(RoleAssistant, "compact summary"),
			},
		}
		messages := []Message{
			NewTextMessage(RoleSystem, "system"),
			NewTextMessage(RoleUser, "first"),
			NewTextMessage(RoleAssistant, "second"),
			NewTextMessage(RoleUser, "third"),
			NewTextMessage(RoleAssistant, "fourth"),
		}

		got, err := SummarizeHistory(model, 2).Process(context.Background(), messages)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if len(model.requests) != 1 {
			t.Fatalf("expected 1 summarizer request, got %d", len(model.requests))
		}
		if model.requests[0].Model != model.Name() {
			t.Fatalf("expected model name %q, got %q", model.Name(), model.requests[0].Model)
		}
		if got[0].Role != RoleSystem {
			t.Fatalf("expected system prompt to be preserved, got %q", got[0].Role)
		}
		if got[1].GetTextContent() != "[Conversation summary]: compact summary" {
			t.Fatalf("unexpected summary message: %#v", got[1])
		}
		if got[2].GetTextContent() != "third" || got[3].GetTextContent() != "fourth" {
			t.Fatalf("unexpected recent messages: %#v", got)
		}
	})

	t.Run("wraps model error", func(t *testing.T) {
		model := &historyStubModel{err: errors.New("boom")}
		messages := []Message{
			NewTextMessage(RoleUser, "first"),
			NewTextMessage(RoleAssistant, "second"),
			NewTextMessage(RoleUser, "third"),
		}

		_, err := SummarizeHistory(model, 1).Process(context.Background(), messages)
		if err == nil || err.Error() != "summarize history: boom" {
			t.Fatalf("expected wrapped error, got %v", err)
		}
	})

	// A summary response with no content cannot stand in for the messages it
	// replaces: accepting it would drop the summarized history permanently and
	// leave an empty "[Conversation summary]: " in its place. The exact error
	// text is not pinned here — only that the empty response is rejected.
	t.Run("errors when summary response has no content", func(t *testing.T) {
		model := &historyStubModel{
			resp: &ChatResponse{
				Model:   "history-stub",
				Created: time.Unix(0, 0),
				Message: Message{Role: RoleAssistant},
			},
		}
		messages := []Message{
			NewTextMessage(RoleUser, "first"),
			NewTextMessage(RoleAssistant, "second"),
			NewTextMessage(RoleUser, "third"),
		}

		_, err := SummarizeHistory(model, 1).Process(context.Background(), messages)
		if err == nil {
			t.Fatal("expected an error for a summary response with no content")
		}
	})
}

func TestChainProcessors(t *testing.T) {
	t.Run("applies processors in sequence", func(t *testing.T) {
		p1 := HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
			return append(messages, NewTextMessage(RoleAssistant, "one")), nil
		})
		p2 := HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
			return append(messages, NewTextMessage(RoleAssistant, "two")), nil
		})

		got, err := ChainProcessors(p1, p2).Process(context.Background(), []Message{NewTextMessage(RoleUser, "start")})
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if len(got) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(got))
		}
		if got[1].GetTextContent() != "one" || got[2].GetTextContent() != "two" {
			t.Fatalf("unexpected chained output: %#v", got)
		}
	})

	t.Run("returns first processor error", func(t *testing.T) {
		expected := errors.New("stop")
		p1 := HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
			return nil, expected
		})
		p2 := HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
			t.Fatal("second processor should not be called")
			return messages, nil
		})

		_, err := ChainProcessors(p1, p2).Process(context.Background(), []Message{NewTextMessage(RoleUser, "start")})
		if !errors.Is(err, expected) {
			t.Fatalf("expected %v, got %v", expected, err)
		}
	})
}
