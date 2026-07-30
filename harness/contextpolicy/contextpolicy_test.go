package contextpolicy

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

func TestPolicySeparatesDurableAndEphemeralContext(t *testing.T) {
	t.Parallel()
	input := []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "prompt")}
	policy, err := New(Config{}, []Transform{
		TransformFunc(func(_ context.Context, value *TransformContext) error {
			value.Durable.Messages = append(value.Durable.Messages, agentic.NewTextMessage(agentic.RoleUser, "durable"))
			*value.Ephemeral = append(*value.Ephemeral, agentic.NewTextMessage(agentic.RoleUser, "ephemeral"))
			return nil
		}),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	projected, err := policy.Project(context.Background(), ProjectionRequest{Messages: input})
	if err != nil {
		t.Fatal(err)
	}
	if len(projected.Messages) != 3 || len(projected.DurableAdditions) != 1 ||
		projected.Messages[1].GetTextContent() != "durable" ||
		projected.Messages[2].GetTextContent() != "ephemeral" {
		t.Fatalf("projection = %#v", projected)
	}
	if len(input) != 1 || input[0].GetTextContent() != "prompt" {
		t.Fatalf("input mutated: %#v", input)
	}
}

func TestPolicyRejectsDurableRewriteAndInvalidTail(t *testing.T) {
	t.Parallel()
	tests := []Transform{
		TransformFunc(func(_ context.Context, value *TransformContext) error {
			value.Durable.Messages[0] = agentic.NewTextMessage(agentic.RoleUser, "rewritten")
			return nil
		}),
		TransformFunc(func(_ context.Context, value *TransformContext) error {
			*value.Ephemeral = append(*value.Ephemeral, agentic.NewTextMessage(agentic.RoleAssistant, "bad"))
			return nil
		}),
		TransformFunc(func(_ context.Context, value *TransformContext) error {
			value.Durable = nil
			return nil
		}),
	}
	for index, transform := range tests {
		policy, err := New(Config{}, []Transform{transform}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := policy.Project(context.Background(), ProjectionRequest{
			Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "original")},
		}); !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("case %d error = %v", index, err)
		}
	}
}

func TestPolicyEnforcesAppendOnlyAcrossTransforms(t *testing.T) {
	t.Parallel()
	policy, err := New(Config{}, []Transform{
		TransformFunc(func(_ context.Context, value *TransformContext) error {
			value.Durable.Messages = append(
				value.Durable.Messages,
				agentic.NewTextMessage(agentic.RoleUser, "first"),
			)
			return nil
		}),
		TransformFunc(func(_ context.Context, value *TransformContext) error {
			value.Durable.Messages[len(value.Durable.Messages)-1] =
				agentic.NewTextMessage(agentic.RoleUser, "rewritten")
			return nil
		}),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := policy.Project(context.Background(), ProjectionRequest{}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("later transform rewrote an earlier addition: %v", err)
	}
}

func TestPolicyUsesReplacementEphemeralPartition(t *testing.T) {
	t.Parallel()
	policy, err := New(Config{}, []Transform{
		TransformFunc(func(_ context.Context, value *TransformContext) error {
			replacement := []agentic.Message{
				agentic.NewTextMessage(agentic.RoleUser, "replacement"),
			}
			value.Ephemeral = &replacement
			return nil
		}),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	projected, err := policy.Project(context.Background(), ProjectionRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(projected.Messages) != 1 || projected.Messages[0].GetTextContent() != "replacement" {
		t.Fatalf("projection = %#v", projected.Messages)
	}
}

func TestStructuredCompactionIsStablePairSafeAndReusable(t *testing.T) {
	t.Parallel()
	compactor, err := NewStructuredCompactor(StructuredConfig{MaxSummaryBytes: 512, MaxEntryBytes: 96})
	if err != nil {
		t.Fatal(err)
	}
	policy, err := New(Config{
		ContextWindowTokens: 1000,
		TriggerPercent:      70,
		TargetPercent:       60,
		RecentMessages:      2,
		MessageOverhead:     1,
		PartOverhead:        1,
		ToolOverhead:        1,
	}, nil, compactor)
	if err != nil {
		t.Fatal(err)
	}
	call := agentic.ToolUse{ID: "call-1", Name: "lookup", Input: map[string]any{"q": "x"}}
	messages := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleSystem, "system"),
		agentic.NewTextMessage(agentic.RoleUser, strings.Repeat("old-user-", 25)),
		agentic.NewToolUseMessage(call),
		agentic.NewToolResultMessageFor(call.ID, call.Name, strings.Repeat("result-", 20), false),
		agentic.NewTextMessage(agentic.RoleAssistant, strings.Repeat("old-answer-", 18)),
		agentic.NewTextMessage(agentic.RoleUser, "recent question"),
		agentic.NewTextMessage(agentic.RoleAssistant, "recent answer"),
	}
	first, err := policy.Project(context.Background(), ProjectionRequest{Messages: messages})
	if err != nil {
		t.Fatal(err)
	}
	if first.Compaction == nil || !first.CompactionChanged {
		t.Fatalf("missing compaction: %#v", first)
	}
	if first.Compaction.Start != 1 || first.Compaction.Cut < 4 {
		t.Fatalf("pair was split by cut %#v", first.Compaction)
	}
	if first.Messages[0].Role != agentic.RoleSystem ||
		!strings.Contains(first.Messages[1].GetTextContent(), "harness_compaction") {
		t.Fatalf("compacted view = %#v", first.Messages)
	}
	second, err := policy.Project(context.Background(), ProjectionRequest{
		Messages:   messages,
		Compaction: first.Compaction,
	})
	if err != nil {
		t.Fatal(err)
	}
	if second.CompactionChanged || !reflect.DeepEqual(first.Messages, second.Messages) {
		t.Fatalf("reapplied projection changed: first=%#v second=%#v", first, second)
	}
	repeated, err := compactor.Summarize(context.Background(), messages[1:first.Compaction.Cut])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(repeated, first.Compaction.Summary) {
		t.Fatalf("structured summary is not byte-stable")
	}
}

func TestLLMCompactionReusesPersistedSummary(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	compactor, err := NewLLMSummaryCompactor(SummarizerFunc(func(_ context.Context, _ []agentic.Message) (string, error) {
		calls.Add(1)
		return "stable facts", nil
	}), 300)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := New(Config{
		ContextWindowTokens: 420,
		TriggerPercent:      80,
		TargetPercent:       70,
		RecentMessages:      1,
		MessageOverhead:     1,
		PartOverhead:        1,
	}, nil, compactor)
	if err != nil {
		t.Fatal(err)
	}
	messages := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, strings.Repeat("a", 180)),
		agentic.NewTextMessage(agentic.RoleAssistant, strings.Repeat("b", 180)),
		agentic.NewTextMessage(agentic.RoleUser, "recent"),
	}
	first, err := policy.Project(context.Background(), ProjectionRequest{Messages: messages})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := policy.Project(context.Background(), ProjectionRequest{Messages: messages, Compaction: first.Compaction}); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("summarizer calls = %d", calls.Load())
	}
}

func TestEstimatorIncludesToolSchemasAndFraming(t *testing.T) {
	t.Parallel()
	tool := agentic.MustNewToolFromStruct("large", strings.Repeat("description", 8), struct {
		Value string `json:"value"`
	}{})
	without, err := New(Config{}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	with, err := New(Config{Tools: []agentic.Tool{tool}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	messages := []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "hello")}
	base, err := without.Estimate(context.Background(), messages)
	if err != nil {
		t.Fatal(err)
	}
	total, err := with.Estimate(context.Background(), messages)
	if err != nil {
		t.Fatal(err)
	}
	if total <= base {
		t.Fatalf("tool schema was not counted: base=%d total=%d", base, total)
	}
}

func TestCompactionRejectsChangedPrefixAndImpossibleWindow(t *testing.T) {
	t.Parallel()
	messages := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "old"),
		agentic.NewTextMessage(agentic.RoleAssistant, "new"),
	}
	compaction := Compaction{
		Version:    1,
		Start:      0,
		Cut:        1,
		PrefixHash: prefixHash(messages[:1]),
		Summary:    agentic.NewTextMessage(agentic.RoleUser, "summary"),
	}
	changed := cloneMessages(messages)
	changed[0] = agentic.NewTextMessage(agentic.RoleUser, "changed")
	if _, err := Apply(changed, compaction); !errors.Is(err, ErrCompactionInvalid) {
		t.Fatalf("changed prefix error = %v", err)
	}

	compactor, _ := NewStructuredCompactor(StructuredConfig{MaxSummaryBytes: 256, MaxEntryBytes: 64})
	policy, err := New(Config{
		ContextWindowTokens: 100,
		TriggerPercent:      50,
		TargetPercent:       40,
		RecentMessages:      2,
	}, nil, compactor)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := policy.Project(context.Background(), ProjectionRequest{
		Messages: []agentic.Message{
			agentic.NewTextMessage(agentic.RoleUser, strings.Repeat("x", 100)),
			agentic.NewTextMessage(agentic.RoleAssistant, strings.Repeat("y", 100)),
		},
	}); !errors.Is(err, ErrContextWindowExceeded) {
		t.Fatalf("impossible window error = %v", err)
	}
}

type summaryModel struct {
	request *agentic.ChatRequest
	err     error
	nilResp bool
}

func (m *summaryModel) Name() string { return "test:summary" }

func (m *summaryModel) Request(_ context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	m.request = request
	if m.err != nil {
		return nil, m.err
	}
	if m.nilResp {
		return nil, nil
	}
	return &agentic.ChatResponse{
		Model:   m.Name(),
		Message: agentic.NewTextMessage(agentic.RoleAssistant, "model summary"),
	}, nil
}

func TestModelSummarizerAdapter(t *testing.T) {
	t.Parallel()
	model := &summaryModel{}
	summarizer, err := NewModelSummarizer(model, "")
	if err != nil {
		t.Fatal(err)
	}
	summary, err := summarizer.Summarize(context.Background(), []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "fact"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if summary != "model summary" || model.request == nil ||
		!strings.Contains(model.request.Messages[0].GetTextContent(), `"fact"`) {
		t.Fatalf("summary=%q request=%#v", summary, model.request)
	}
	if _, err := NewModelSummarizer(nil, ""); err == nil {
		t.Fatal("nil model succeeded")
	}
}

func TestConfigTransformAndEstimatorErrorPaths(t *testing.T) {
	t.Parallel()
	invalid := []Config{
		{ContextWindowTokens: -1},
		{TriggerPercent: 101},
		{TriggerPercent: 50, TargetPercent: 50},
		{RecentMessages: -1},
		{MessageOverhead: -1},
		{PartOverhead: -1},
		{ToolOverhead: -1},
	}
	for index, config := range invalid {
		if _, err := New(config, nil, nil); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("invalid config %d = %v", index, err)
		}
	}
	compactor, _ := NewStructuredCompactor(StructuredConfig{})
	if _, err := New(Config{}, nil, compactor); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("compactor without window = %v", err)
	}
	badTool := agentic.Tool{Function: agentic.Function{
		Name:       "bad",
		Parameters: map[string]any{"channel": make(chan int)},
	}}
	if _, err := New(Config{Tools: []agentic.Tool{badTool}}, nil, nil); err == nil {
		t.Fatal("uncloneable tool schema succeeded")
	}
	if _, err := New(Config{}, []Transform{nil}, nil); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("nil transform = %v", err)
	}

	wantErr := errors.New("transform failed")
	policy, err := New(Config{}, []Transform{TransformFunc(func(context.Context, *TransformContext) error {
		return wantErr
	})}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := policy.Project(context.Background(), ProjectionRequest{}); !errors.Is(err, wantErr) {
		t.Fatalf("transform error = %v", err)
	}
	policy, _ = New(Config{}, []Transform{TransformFunc(func(_ context.Context, value *TransformContext) error {
		value.Ephemeral = nil
		return nil
	})}, nil)
	if _, err := policy.Project(context.Background(), ProjectionRequest{}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("cleared ephemeral = %v", err)
	}
	policy, _ = New(Config{}, []Transform{TransformFunc(func(_ context.Context, value *TransformContext) error {
		value.Durable.Messages = nil
		return nil
	})}, nil)
	if _, err := policy.Project(context.Background(), ProjectionRequest{
		Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "base")},
	}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("truncated durable = %v", err)
	}
	for name, message := range map[string]agentic.Message{
		"empty":     {Role: agentic.RoleUser},
		"assistant": agentic.NewTextMessage(agentic.RoleAssistant, "bad"),
		"tool":      agentic.NewToolUseMessage(agentic.ToolUse{ID: "call", Name: "tool"}),
	} {
		message := message
		t.Run(name, func(t *testing.T) {
			policy, _ := New(Config{}, []Transform{TransformFunc(func(_ context.Context, value *TransformContext) error {
				value.Durable.Messages = append(value.Durable.Messages, message)
				return nil
			})}, nil)
			if _, err := policy.Project(context.Background(), ProjectionRequest{}); !errors.Is(err, ErrInvalidTransform) {
				t.Fatalf("invalid durable tail = %v", err)
			}
		})
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Passthrough().Project(ctx, ProjectionRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled projection = %v", err)
	}

	counterErr := errors.New("counter")
	policy, _ = New(Config{Counter: TokenCounterFunc(func(context.Context, []byte) (int, error) {
		return 0, counterErr
	})}, nil, nil)
	if _, err := policy.Estimate(context.Background(), []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "message"),
	}); !errors.Is(err, counterErr) {
		t.Fatalf("message counter error = %v", err)
	}
	policy, _ = New(Config{
		Counter: TokenCounterFunc(func(context.Context, []byte) (int, error) { return 0, counterErr }),
		Tools:   []agentic.Tool{{Function: agentic.Function{Name: "tool"}}},
	}, nil, nil)
	if _, err := policy.Estimate(context.Background(), nil); !errors.Is(err, counterErr) {
		t.Fatalf("tool counter error = %v", err)
	}
	badMessage := agentic.NewToolUseMessage(agentic.ToolUse{
		ID: "bad", Name: "bad", Input: map[string]any{"channel": make(chan int)},
	})
	policy, _ = New(Config{}, nil, nil)
	if _, err := policy.Estimate(context.Background(), []agentic.Message{badMessage}); err == nil {
		t.Fatal("unencodable message was counted")
	}
}

func TestCompactionValidationFrontiersAndCompactorFailures(t *testing.T) {
	t.Parallel()
	base := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "old"),
		agentic.NewTextMessage(agentic.RoleAssistant, "answer"),
	}
	valid := Compaction{
		Version:    1,
		Start:      0,
		Cut:        1,
		PrefixHash: prefixHash(base[:1]),
		Summary:    agentic.NewTextMessage(agentic.RoleUser, "summary"),
	}
	invalid := []Compaction{
		{},
		{Version: 2, Start: 0, Cut: 1, Summary: valid.Summary},
		{Version: 1, Start: -1, Cut: 1, Summary: valid.Summary},
		{Version: 1, Start: 1, Cut: 1, Summary: valid.Summary},
		{Version: 1, Start: 0, Cut: 3, Summary: valid.Summary},
		{Version: 1, Start: 0, Cut: 1, Summary: agentic.NewTextMessage(agentic.RoleAssistant, "bad")},
	}
	for index, value := range invalid {
		if _, err := Apply(base, value); !errors.Is(err, ErrCompactionInvalid) {
			t.Fatalf("invalid compaction %d = %v", index, err)
		}
	}
	applied, err := Apply(base, valid)
	if err != nil || len(applied) != 2 || applied[0].GetTextContent() != "summary" {
		t.Fatalf("valid apply = %#v, %v", applied, err)
	}

	frontiers := []struct {
		name     string
		messages []agentic.Message
	}{
		{"orphan", []agentic.Message{agentic.NewToolResultMessageFor("missing", "tool", "x", false), agentic.NewTextMessage(agentic.RoleUser, "recent")}},
		{"duplicate", []agentic.Message{agentic.NewToolUseMessage(
			agentic.ToolUse{ID: "same", Name: "a"},
			agentic.ToolUse{ID: "same", Name: "b"},
		), agentic.NewTextMessage(agentic.RoleUser, "recent")}},
		{"overlap", []agentic.Message{
			agentic.NewToolUseMessage(agentic.ToolUse{ID: "one", Name: "a"}),
			agentic.NewToolUseMessage(agentic.ToolUse{ID: "two", Name: "b"}),
			agentic.NewTextMessage(agentic.RoleUser, "recent"),
		}},
	}
	compactor, _ := NewStructuredCompactor(StructuredConfig{})
	for _, test := range frontiers {
		test := test
		t.Run(test.name, func(t *testing.T) {
			policy, err := New(Config{
				ContextWindowTokens: 10,
				TriggerPercent:      50,
				TargetPercent:       40,
				RecentMessages:      1,
				Counter: TokenCounterFunc(func(context.Context, []byte) (int, error) {
					return 100, nil
				}),
			}, nil, compactor)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := policy.Project(context.Background(), ProjectionRequest{Messages: test.messages}); !errors.Is(err, ErrCompactionInvalid) {
				t.Fatalf("frontier error = %v", err)
			}
		})
	}

	summaryErr := errors.New("summary")
	policy, _ := New(Config{
		ContextWindowTokens: 100,
		TriggerPercent:      50,
		TargetPercent:       40,
		RecentMessages:      1,
		Counter: TokenCounterFunc(func(context.Context, []byte) (int, error) {
			return 100, nil
		}),
	}, nil, CompactorFunc(func(context.Context, []agentic.Message) (agentic.Message, error) {
		return agentic.Message{}, summaryErr
	}))
	if _, err := policy.Project(context.Background(), ProjectionRequest{Messages: base}); !errors.Is(err, summaryErr) {
		t.Fatalf("compactor error = %v", err)
	}
	policy, _ = New(Config{
		ContextWindowTokens: 100,
		TriggerPercent:      50,
		TargetPercent:       40,
		RecentMessages:      1,
		Counter: TokenCounterFunc(func(context.Context, []byte) (int, error) {
			return 100, nil
		}),
	}, nil, CompactorFunc(func(context.Context, []agentic.Message) (agentic.Message, error) {
		return agentic.NewTextMessage(agentic.RoleAssistant, "invalid"), nil
	}))
	if _, err := policy.Project(context.Background(), ProjectionRequest{Messages: base}); !errors.Is(err, ErrCompactionInvalid) {
		t.Fatalf("invalid summary = %v", err)
	}
}

func TestStructuredAndLLMCompactorBoundsAndErrors(t *testing.T) {
	t.Parallel()
	for index, config := range []StructuredConfig{
		{MaxSummaryBytes: 255},
		{MaxSummaryBytes: 256, MaxEntryBytes: 31},
		{MaxSummaryBytes: 256, MaxEntryBytes: 257},
	} {
		if _, err := NewStructuredCompactor(config); err == nil {
			t.Fatalf("invalid structured config %d succeeded", index)
		}
	}
	compactor, err := NewStructuredCompactor(StructuredConfig{MaxSummaryBytes: 300, MaxEntryBytes: 64})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := compactor.Summarize(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled structured summary = %v", err)
	}
	call := agentic.ToolUse{ID: "call", Name: "lookup"}
	messages := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, strings.Repeat("é", 100)),
		agentic.NewToolUseMessage(call),
		agentic.NewToolResultMessageFor(call.ID, call.Name, "failed", true),
	}
	for range 12 {
		messages = append(messages, agentic.NewTextMessage(agentic.RoleAssistant, strings.Repeat("tail", 100)))
	}
	summary, err := compactor.Summarize(context.Background(), messages)
	if err != nil {
		t.Fatal(err)
	}
	text := summary.GetTextContent()
	if len(text) > 300 || !strings.Contains(text, "older messages omitted") || !strings.Contains(text, "calls=lookup#call") {
		t.Fatalf("bounded structured summary = %q", text)
	}
	if !strings.Contains(structuredLine(0, messages[2], 64), "error") {
		t.Fatal("structured result status missing")
	}
	if got := truncateUTF8("éé", 3); got != "é" {
		t.Fatalf("UTF-8 truncation = %q", got)
	}
	if got := truncateUTF8("x", 0); got != "" {
		t.Fatalf("zero truncation = %q", got)
	}

	if _, err := NewLLMSummaryCompactor(nil, 0); err == nil {
		t.Fatal("nil summarizer succeeded")
	}
	if _, err := NewLLMSummaryCompactor(SummarizerFunc(func(context.Context, []agentic.Message) (string, error) {
		return "x", nil
	}), 255); err == nil {
		t.Fatal("small LLM limit succeeded")
	}
	wantErr := errors.New("summarizer")
	llm, _ := NewLLMSummaryCompactor(SummarizerFunc(func(context.Context, []agentic.Message) (string, error) {
		return "", wantErr
	}), 256)
	if _, err := llm.Summarize(context.Background(), messages); !errors.Is(err, wantErr) {
		t.Fatalf("summarizer error = %v", err)
	}
	llm, _ = NewLLMSummaryCompactor(SummarizerFunc(func(context.Context, []agentic.Message) (string, error) {
		return "   ", nil
	}), 256)
	if _, err := llm.Summarize(context.Background(), messages); err == nil {
		t.Fatal("empty summary succeeded")
	}
	llm, _ = NewLLMSummaryCompactor(SummarizerFunc(func(context.Context, []agentic.Message) (string, error) {
		return strings.Repeat("é", 300), nil
	}), 256)
	result, err := llm.Summarize(context.Background(), messages)
	if err != nil || len(result.GetTextContent()) > 256 {
		t.Fatalf("bounded LLM summary bytes=%d err=%v", len(result.GetTextContent()), err)
	}
}

func TestModelSummarizerErrorsAndCustomInstruction(t *testing.T) {
	t.Parallel()
	wantErr := errors.New("model")
	model := &summaryModel{err: wantErr}
	summarizer, err := NewModelSummarizer(model, "custom instruction")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := summarizer.Summarize(context.Background(), nil); !errors.Is(err, wantErr) {
		t.Fatalf("model error = %v", err)
	}
	if model.request == nil || !strings.HasPrefix(model.request.Messages[0].GetTextContent(), "custom instruction") {
		t.Fatalf("custom request = %#v", model.request)
	}
	model = &summaryModel{nilResp: true}
	summarizer, _ = NewModelSummarizer(model, "")
	if _, err := summarizer.Summarize(context.Background(), nil); err == nil {
		t.Fatal("nil model response succeeded")
	}
	bad := agentic.NewToolUseMessage(agentic.ToolUse{
		ID: "bad", Name: "bad", Input: map[string]any{"channel": make(chan int)},
	})
	model = &summaryModel{}
	summarizer, _ = NewModelSummarizer(model, "")
	if _, err := summarizer.Summarize(context.Background(), []agentic.Message{bad}); err == nil {
		t.Fatal("unencodable summary input succeeded")
	}
}
