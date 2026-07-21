package agentic

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/testutil"
	testprovider "github.com/regularkevvv/agentic/provider/test"
)

func TestReadyHandoffRunsThroughParent(t *testing.T) {
	childModel := testprovider.NewTestModel(testprovider.ModelResponse{Text: "child answer"})
	child := NewAgent("child", childModel)
	handoff := NewHandoff("delegate", "delegate work", child)
	parentModel := testprovider.NewTestModel(
		testprovider.ModelResponse{ToolCalls: []ToolUse{{ID: "handoff_1", Name: "delegate", Input: map[string]interface{}{"task": "help"}}}},
		testprovider.ModelResponse{Text: "parent done"},
	)
	parent := NewAgent("parent", parentModel).AddHandoff(handoff)

	result, err := parent.Run(context.Background(), "delegate this")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Output != "parent done" || childModel.CallCount() != 1 {
		t.Fatalf("unexpected result %#v or child calls %d", result, childModel.CallCount())
	}
	if result.ToolResults[0].ToolUseID != "handoff_1" || result.ToolResults[0].Content != "child answer" {
		t.Fatalf("handoff did not retain tool id/output: %#v", result.ToolResults[0])
	}
}

func TestReadyHandoffWithProviderChild(t *testing.T) {
	type deps struct{ Marker string }
	childModel := testprovider.NewTestModel(testprovider.ModelResponse{Text: "child answer"})
	child := NewAgentWithDepsDynamic[*deps](func(ctx RunContext[*deps]) (string, error) {
		return "marker=" + ctx.Deps.Marker, nil
	}, childModel)
	var providerCalls atomic.Int32
	readyChild := child.BindProvider(func(context.Context) (*deps, error) {
		providerCalls.Add(1)
		return &deps{Marker: "provider"}, nil
	})
	parentModel := testprovider.NewTestModel(
		testprovider.ModelResponse{ToolCalls: []ToolUse{{ID: "provider_1", Name: "delegate", Input: map[string]interface{}{"task": "help"}}}},
		testprovider.ModelResponse{Text: "parent done"},
	)
	result, err := NewAgent("parent", parentModel).
		AddHandoff(NewHandoff("delegate", "delegate work", readyChild)).
		Run(context.Background(), "delegate this")
	if err != nil {
		t.Fatal(err)
	}
	if providerCalls.Load() != 1 || childModel.CallCount() != 1 {
		t.Fatalf("provider=%d child=%d", providerCalls.Load(), childModel.CallCount())
	}
	if got := childModel.Calls()[0].Messages[0].GetTextContent(); got != "marker=provider" {
		t.Fatalf("provider deps not observed: %q", got)
	}
	if result.ToolResults[0].ToolUseID != "provider_1" {
		t.Fatalf("tool id not retained: %#v", result.ToolResults[0])
	}
}

func TestHandoffHistoryFiltersAndOptions(t *testing.T) {
	baseHistory := []Message{
		NewTextMessage(RoleSystem, "secret system"),
		NewTextMessage(RoleUser, "older user context"),
	}
	t.Run("full history", func(t *testing.T) {
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"})
		handoff := NewHandoff("delegate", "delegate", NewAgent("child", model),
			WithHandoffFilter(HandoffFullHistory),
			WithHandoffSystemPrompt("extra"),
			WithHandoffMaxTokens(77),
		)
		ctx := core.WithToolExecutionState(context.Background(), core.ToolExecutionState{Messages: baseHistory})
		out, err := (&handoffHandler{handoff: handoff}).Execute(ctx, map[string]interface{}{"task": "work"}, nil)
		if err != nil || out != "ok" {
			t.Fatalf("Execute: out=%v err=%v", out, err)
		}
		request := model.Calls()[0]
		if request.MaxTokens == nil || *request.MaxTokens != 77 {
			t.Fatalf("max tokens not forwarded: %#v", request.MaxTokens)
		}
		joined := messagesText(request.Messages)
		if !strings.Contains(joined, "older user context") || !strings.Contains(joined, "extra") || strings.Contains(joined, "secret system") {
			t.Fatalf("unexpected child messages: %s", joined)
		}
	})

	t.Run("custom summary", func(t *testing.T) {
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"})
		handoff := NewHandoff("delegate", "delegate", NewAgent("child", model),
			WithHandoffFilter(HandoffSummary),
			WithHandoffSummaryFunc(func(context.Context, []Message) (string, error) { return "summary text", nil }),
		)
		ctx := core.WithToolExecutionState(context.Background(), core.ToolExecutionState{Messages: baseHistory})
		if _, err := (&handoffHandler{handoff: handoff}).Execute(ctx, map[string]interface{}{"task": "work"}, nil); err != nil {
			t.Fatal(err)
		}
		if joined := messagesText(model.Calls()[0].Messages); !strings.Contains(joined, "summary text") {
			t.Fatalf("summary missing: %s", joined)
		}
	})

	t.Run("summary error", func(t *testing.T) {
		handoff := NewHandoff("delegate", "delegate", NewAgent("child", testprovider.NewTestModel()),
			WithHandoffFilter(HandoffSummary),
			WithHandoffSummaryFunc(func(context.Context, []Message) (string, error) { return "", errors.New("boom") }),
		)
		ctx := core.WithToolExecutionState(context.Background(), core.ToolExecutionState{Messages: baseHistory})
		_, err := (&handoffHandler{handoff: handoff}).Execute(ctx, map[string]interface{}{"task": "work"}, nil)
		if err == nil || !strings.Contains(err.Error(), "summarize handoff history") {
			t.Fatalf("expected summary error, got %v", err)
		}
	})
}

func TestMappedHandoffs(t *testing.T) {
	type parentDeps struct{ User string }
	type childDeps struct{ User string }

	t.Run("different dependency types", func(t *testing.T) {
		childModel := testprovider.NewTestModel(testprovider.ModelResponse{Text: "child"})
		child := NewAgentWithDeps[*childDeps]("", childModel).SetDynamicPrompt(func(ctx RunContext[*childDeps]) (string, error) {
			return "user=" + ctx.Deps.User, nil
		})
		handoff := NewTextHandoffWithDeps("delegate", "delegate", child,
			func(ctx RunContext[*parentDeps]) (*childDeps, error) { return &childDeps{User: ctx.Deps.User}, nil },
		)
		parentModel := testprovider.NewTestModel(
			testprovider.ModelResponse{ToolCalls: []ToolUse{{ID: "map_1", Name: "delegate", Input: map[string]interface{}{"task": "work"}}}},
			testprovider.ModelResponse{Text: "done"},
		)
		parent := NewAgentWithDeps[*parentDeps]("parent", parentModel).AddHandoffWithDeps(handoff)
		if _, err := parent.Run(context.Background(), "go", &parentDeps{User: "alice"}); err != nil {
			t.Fatal(err)
		}
		if got := childModel.Calls()[0].Messages[0].GetTextContent(); got != "user=alice" {
			t.Fatalf("mapped deps not observed: %q", got)
		}
	})

	t.Run("identity", func(t *testing.T) {
		child := NewAgentWithDeps[*parentDeps]("child", testprovider.NewTestModel())
		handoff := NewIdentityTextHandoff("delegate", "delegate", child)
		if handoff == nil {
			t.Fatal("expected identity handoff")
		}
	})

	t.Run("mapper fails before child model", func(t *testing.T) {
		childModel := testprovider.NewTestModel()
		child := NewAgentWithDeps[*childDeps]("child", childModel)
		handoff := NewTextHandoffWithDeps("delegate", "delegate", child,
			func(RunContext[*parentDeps]) (*childDeps, error) { return nil, errors.New("denied") },
		)
		handler := &handoffWithDepsHandler[*parentDeps]{handoff: handoff}
		_, err := handler.Execute(context.Background(), map[string]interface{}{"task": "work"}, core.NewDependencyEnvelope(&parentDeps{}))
		if err == nil || !strings.Contains(err.Error(), "map handoff dependencies") || childModel.CallCount() != 0 {
			t.Fatalf("err=%v child calls=%d", err, childModel.CallCount())
		}
	})

	t.Run("mapper failure retains parent tool id", func(t *testing.T) {
		childModel := testprovider.NewTestModel()
		child := NewAgentWithDeps[*childDeps]("child", childModel)
		handoff := NewTextHandoffWithDeps("delegate", "delegate", child,
			func(RunContext[*parentDeps]) (*childDeps, error) { return nil, errors.New("denied") },
		)
		parentModel := testprovider.NewTestModel(
			testprovider.ModelResponse{ToolCalls: []ToolUse{{ID: "mapper_failure_1", Name: "delegate", Input: map[string]interface{}{"task": "work"}}}},
			testprovider.ModelResponse{Text: "handled"},
		)
		result, err := NewAgentWithDeps[*parentDeps]("parent", parentModel).
			AddHandoffWithDeps(handoff).
			Run(context.Background(), "go", &parentDeps{})
		if err != nil {
			t.Fatal(err)
		}
		if childModel.CallCount() != 0 || len(result.ToolResults) != 1 || result.ToolResults[0].ToolUseID != "mapper_failure_1" || !result.ToolResults[0].IsError {
			t.Fatalf("child=%d results=%#v", childModel.CallCount(), result.ToolResults)
		}
	})

	t.Run("invalid envelope", func(t *testing.T) {
		child := NewAgentWithDeps[*childDeps]("child", testprovider.NewTestModel())
		handoff := NewTextHandoffWithDeps("delegate", "delegate", child,
			func(RunContext[*parentDeps]) (*childDeps, error) { return &childDeps{}, nil },
		)
		_, err := (&handoffWithDepsHandler[*parentDeps]{handoff: handoff}).Execute(context.Background(), map[string]interface{}{"task": "work"}, "bad")
		if err == nil || !strings.Contains(err.Error(), "envelope") {
			t.Fatalf("expected envelope error, got %v", err)
		}
	})
}

type concurrentHandoffModel struct {
	child bool
	mu    sync.Mutex
	seen  map[string]int
}

func (m *concurrentHandoffModel) Name() string { return "handoff:concurrent" }

func (m *concurrentHandoffModel) Request(_ context.Context, request *ChatRequest) (*ChatResponse, error) {
	if m.child {
		marker := request.Messages[0].GetTextContent()
		m.mu.Lock()
		m.seen[marker]++
		m.mu.Unlock()
		return &ChatResponse{Message: NewTextMessage(RoleAssistant, marker), FinishReason: FinishReasonStop}, nil
	}
	for _, message := range request.Messages {
		if message.Role == RoleTool {
			return &ChatResponse{Message: NewTextMessage(RoleAssistant, "done"), FinishReason: FinishReasonStop}, nil
		}
	}
	return &ChatResponse{Message: NewToolUseMessage(ToolUse{
		ID: "concurrent_handoff", Name: "delegate", Input: map[string]interface{}{"task": "work"},
	}), FinishReason: FinishReasonToolCalls}, nil
}

func TestConcurrentMappedHandoffParentsKeepDependenciesIsolated(t *testing.T) {
	type parentDeps struct{ Marker string }
	type childDeps struct{ Marker string }
	childModel := &concurrentHandoffModel{child: true, seen: make(map[string]int)}
	child := NewAgentWithDepsDynamic[*childDeps](func(ctx RunContext[*childDeps]) (string, error) {
		return ctx.Deps.Marker, nil
	}, childModel)
	handoff := NewTextHandoffWithDeps("delegate", "delegate", child, func(ctx RunContext[*parentDeps]) (*childDeps, error) {
		return &childDeps{Marker: ctx.Deps.Marker}, nil
	})
	parent := NewAgentWithDeps[*parentDeps]("parent", &concurrentHandoffModel{}).AddHandoffWithDeps(handoff)

	const runs = 24
	var wg sync.WaitGroup
	errs := make(chan error, runs*2)
	for _, marker := range []string{"alice", "bob"} {
		for range runs {
			wg.Add(1)
			go func(marker string) {
				defer wg.Done()
				_, err := parent.Run(context.Background(), "go", &parentDeps{Marker: marker})
				errs <- err
			}(marker)
		}
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	childModel.mu.Lock()
	defer childModel.mu.Unlock()
	if childModel.seen["alice"] != runs || childModel.seen["bob"] != runs {
		t.Fatalf("handoff dependencies crossed: %#v", childModel.seen)
	}
}

func TestTypedMappedHandoffConstructors(t *testing.T) {
	type deps struct{ ID string }
	type output struct {
		Value string `json:"value"`
	}
	child := NewTypedAgentWithDeps[output, *deps]("child", testprovider.NewTestModel(), "output")
	mapped := NewHandoffWithDeps("typed", "typed", child, func(ctx RunContext[*deps]) (*deps, error) { return ctx.Deps, nil })
	identity := NewIdentityHandoff("identity", "identity", child)
	if mapped == nil || identity == nil {
		t.Fatal("expected typed handoffs")
	}
}

func TestHandoffHandlerErrorsAndHelpers(t *testing.T) {
	t.Run("marshal input", func(t *testing.T) {
		h := NewHandoff("delegate", "delegate", NewAgent("child", testprovider.NewTestModel()))
		_, err := (&handoffHandler{handoff: h}).Execute(context.Background(), map[string]interface{}{"task": func() {}}, nil)
		if err == nil || !strings.Contains(err.Error(), "marshal input") {
			t.Fatalf("expected marshal error, got %v", err)
		}
	})
	t.Run("unmarshal input", func(t *testing.T) {
		h := NewHandoff("delegate", "delegate", NewAgent("child", testprovider.NewTestModel()))
		_, err := (&handoffHandler{handoff: h}).Execute(context.Background(), map[string]interface{}{"task": 12}, nil)
		if err == nil || !strings.Contains(err.Error(), "unmarshal handoff input") {
			t.Fatalf("expected unmarshal error, got %v", err)
		}
	})
	t.Run("child error", func(t *testing.T) {
		h := NewHandoff("delegate", "delegate", NewAgent("child", &testutil.StubModel{Err: errors.New("boom")}))
		_, err := (&handoffHandler{handoff: h}).Execute(context.Background(), map[string]interface{}{"task": "work"}, nil)
		if err == nil || !strings.Contains(err.Error(), `handoff to "delegate"`) {
			t.Fatalf("expected child error, got %v", err)
		}
	})

	messages := make([]Message, 10)
	for i := range messages {
		messages[i] = NewTextMessage(RoleUser, strings.Repeat("x", 600))
	}
	summary := compactHandoffHistory(messages)
	if strings.Count(summary, "user:") != 8 || !strings.Contains(summary, "…") {
		t.Fatalf("unexpected compact summary: %s", summary)
	}
	if clipHandoffText("short", 10) != "short" {
		t.Fatal("short text should be unchanged")
	}
}

func messagesText(messages []Message) string {
	var parts []string
	for _, message := range messages {
		parts = append(parts, message.GetTextContent())
	}
	return strings.Join(parts, "\n")
}
