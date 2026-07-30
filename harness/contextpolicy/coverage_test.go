package contextpolicy

import (
	"context"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

func TestProjectorAdapterAndRemainingProjectionFailures(t *testing.T) {
	called := false
	projector := ProjectorFunc(func(_ context.Context, request ProjectionRequest) (Projection, error) {
		called = true
		return Projection{Messages: request.Messages}, nil
	})
	if projection, err := projector.Project(context.Background(), ProjectionRequest{}); err != nil ||
		projection.Messages != nil || !called {
		t.Fatalf("projector adapter = %#v, %v, called=%v", projection, err, called)
	}

	for _, invalid := range []agentic.Message{
		agentic.NewTextMessage(agentic.RoleAssistant, "assistant"),
		{Role: agentic.RoleUser},
		agentic.NewToolUseMessage(agentic.ToolUse{ID: "call", Name: "tool"}),
	} {
		policy, err := New(Config{}, []Transform{TransformFunc(func(
			_ context.Context,
			value *TransformContext,
		) error {
			value.Durable.Messages = append(value.Durable.Messages, invalid)
			return nil
		})}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := policy.Project(context.Background(), ProjectionRequest{}); !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("invalid durable addition %#v error = %v", invalid, err)
		}
	}

	policy, err := New(Config{}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := policy.Project(context.Background(), ProjectionRequest{
		Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "one")},
		Compaction: &Compaction{
			Version: 1,
			Start:   0,
			Cut:     1,
			Summary: agentic.NewTextMessage(agentic.RoleUser, "summary"),
		},
	}); !errors.Is(err, ErrCompactionInvalid) {
		t.Fatalf("stale persisted compaction = %v", err)
	}

	counterErr := errors.New("counter")
	policy, err = New(Config{
		ContextWindowTokens: 100,
		Counter: TokenCounterFunc(func(context.Context, []byte) (int, error) {
			return 0, counterErr
		}),
	}, nil, CompactorFunc(func(context.Context, []agentic.Message) (agentic.Message, error) {
		return agentic.NewTextMessage(agentic.RoleUser, "summary"), nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := policy.Project(context.Background(), ProjectionRequest{
		Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "one")},
	}); !errors.Is(err, counterErr) {
		t.Fatalf("initial estimate error = %v", err)
	}
}

func TestProjectionSecondEstimateAndTargetFailures(t *testing.T) {
	messages := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "one"),
		agentic.NewTextMessage(agentic.RoleAssistant, "two"),
		agentic.NewTextMessage(agentic.RoleUser, "three"),
	}
	counterErr := errors.New("second estimate")
	calls := 0
	policy, err := New(Config{
		ContextWindowTokens: 20,
		TriggerPercent:      70,
		TargetPercent:       50,
		RecentMessages:      1,
		MessageOverhead:     1,
		PartOverhead:        1,
		ToolOverhead:        1,
		Counter: TokenCounterFunc(func(context.Context, []byte) (int, error) {
			calls++
			if calls > len(messages) {
				return 0, counterErr
			}
			return 10, nil
		}),
	}, nil, CompactorFunc(func(context.Context, []agentic.Message) (agentic.Message, error) {
		return agentic.NewTextMessage(agentic.RoleUser, "summary"), nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := policy.Project(context.Background(), ProjectionRequest{Messages: messages}); !errors.Is(err, counterErr) {
		t.Fatalf("second estimate error = %v", err)
	}

	policy, err = New(Config{
		ContextWindowTokens: 20,
		TriggerPercent:      70,
		TargetPercent:       50,
		RecentMessages:      1,
		MessageOverhead:     1,
		PartOverhead:        1,
		ToolOverhead:        1,
		Counter: TokenCounterFunc(func(context.Context, []byte) (int, error) {
			return 10, nil
		}),
	}, nil, CompactorFunc(func(context.Context, []agentic.Message) (agentic.Message, error) {
		return agentic.NewTextMessage(agentic.RoleUser, "summary"), nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := policy.Project(context.Background(), ProjectionRequest{Messages: messages}); !errors.Is(err, ErrContextWindowExceeded) {
		t.Fatalf("target overflow = %v", err)
	}
}

func TestEstimateToolFailuresAndInternalEdgeHelpers(t *testing.T) {
	policy, err := New(Config{}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	policy.config.Tools = []agentic.Tool{{
		Function: agentic.Function{Parameters: map[string]any{"bad": make(chan int)}},
	}}
	if _, err := policy.Estimate(context.Background(), nil); err == nil {
		t.Fatal("unencodable tool schema was estimated")
	}

	counterErr := errors.New("tool counter")
	policy.config.Tools = []agentic.Tool{{}}
	policy.config.Counter = TokenCounterFunc(func(context.Context, []byte) (int, error) {
		return 0, counterErr
	})
	if _, err := policy.Estimate(context.Background(), nil); !errors.Is(err, counterErr) {
		t.Fatalf("tool counter error = %v", err)
	}

	badMessage := agentic.NewToolUseMessage(agentic.ToolUse{
		ID: "bad", Name: "bad", Input: map[string]any{"channel": make(chan int)},
	})
	if cloned := cloneMessages([]agentic.Message{badMessage}); len(cloned) != 1 {
		t.Fatalf("fallback clone = %#v", cloned)
	}
	if messagesEqual(nil, []agentic.Message{{}}) {
		t.Fatal("different message lengths compared equal")
	}
	if messagesEqual([]agentic.Message{badMessage}, []agentic.Message{badMessage}) {
		t.Fatal("unencodable messages compared equal")
	}

	if _, _, ok, err := compactionCut([]agentic.Message{
		agentic.NewToolUseMessage(agentic.ToolUse{ID: "open", Name: "tool"}),
	}, 0); err != nil || ok {
		t.Fatalf("frontier-only cut = ok=%v err=%v", ok, err)
	}
	start, cut, ok, err := compactionCut([]agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "stable"),
		agentic.NewToolUseMessage(agentic.ToolUse{ID: "open", Name: "tool"}),
	}, 0)
	if err != nil || !ok || start != 0 || cut != 1 {
		t.Fatalf("stable-before-frontier cut = %d,%d,%v,%v", start, cut, ok, err)
	}

	if err := validateContextTail([]agentic.Message{{Role: agentic.RoleUser}}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("empty context tail = %v", err)
	}
	if err := validateContextTail([]agentic.Message{agentic.NewToolResultMessageFor(
		"call", "tool", "result", false,
	)}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("non-text context tail = %v", err)
	}
	if _, _, ok, err := compactionCut([]agentic.Message{
		agentic.NewTextMessage(agentic.RoleSystem, "system"),
		agentic.NewTextMessage(agentic.RoleUser, "recent"),
	}, 1); err != nil || ok {
		t.Fatalf("no eligible post-system cut = ok=%v err=%v", ok, err)
	}
}

func TestLLMCompactorDefaultAndDefensiveFramingBound(t *testing.T) {
	summarizer := SummarizerFunc(func(context.Context, []agentic.Message) (string, error) {
		return "summary", nil
	})
	compactor, err := NewLLMSummaryCompactor(summarizer, 0)
	if err != nil || compactor.maxBytes != DefaultStructuredSummaryBytes {
		t.Fatalf("default LLM compactor = %#v, %v", compactor, err)
	}
	compactor.maxBytes = 1
	if _, err := compactor.Summarize(context.Background(), nil); err == nil {
		t.Fatal("undersized defensive framing succeeded")
	}
}

func TestProjectRejectsExplicitNonTextDurableAddition(t *testing.T) {
	policy, err := New(Config{}, []Transform{TransformFunc(func(
		_ context.Context,
		value *TransformContext,
	) error {
		value.Durable.Messages = append(value.Durable.Messages, agentic.Message{
			Role: agentic.RoleUser,
			Content: []agentic.Part{{
				Type: agentic.ContentImageURL,
			}},
		})
		return nil
	})}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := policy.Project(context.Background(), ProjectionRequest{}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("non-text durable addition = %v", err)
	}
}
