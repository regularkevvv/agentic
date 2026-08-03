package session

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/permission"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func requestSystem(request agentic.ChatRequest) string {
	var parts []string
	for _, message := range request.Messages {
		if message.Role == agentic.RoleSystem {
			parts = append(parts, message.GetTextContent())
		}
	}
	return strings.Join(parts, "\n\n")
}

func TestExchangeInstructionsResolveOnceAndRefreshNextPrompt(t *testing.T) {
	repository := storememory.New()
	model := &scriptedModel{steps: []modelStep{textStep("first"), textStep("second")}}
	agent := agentic.NewAgent("base", model)
	value := "exchange-v1"
	var resolutions atomic.Int32
	config := sessionConfig(t, agent, repository, artifactmemory.New(), spill.Config{})
	config.Instructions = harnessruntime.ExchangeInstructionProviderFunc(func(_ context.Context, exchange harnessruntime.ExchangeContext) (string, error) {
		resolutions.Add(1)
		if exchange.SessionID != config.ID || exchange.RunID == "" || exchange.Scope.SessionID != config.ID {
			t.Fatalf("exchange context = %+v", exchange)
		}
		return value, nil
	})
	current, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := current.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "one")); err != nil {
		t.Fatal(err)
	}
	value = "exchange-v2"
	if _, err := current.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "two")); err != nil {
		t.Fatal(err)
	}
	calls := model.Calls()
	if len(calls) != 2 || resolutions.Load() != 2 {
		t.Fatalf("model calls=%d resolutions=%d", len(calls), resolutions.Load())
	}
	if first := requestSystem(calls[0]); !strings.Contains(first, "base") || !strings.Contains(first, "exchange-v1") {
		t.Fatalf("first system = %q", first)
	}
	if second := requestSystem(calls[1]); !strings.Contains(second, "base") || !strings.Contains(second, "exchange-v2") || strings.Contains(second, "exchange-v1") {
		t.Fatalf("second system = %q", second)
	}
}

func TestExchangeInstructionsCreateSystemMessageWhenRunnerHasNone(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{textStep("done")}}
	config := sessionConfig(t, agentic.NewAgent("", model), storememory.New(), artifactmemory.New(), spill.Config{})
	config.Instructions = harnessruntime.ExchangeInstructionProviderFunc(func(context.Context, harnessruntime.ExchangeContext) (string, error) {
		return "trusted-only", nil
	})
	current, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := current.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "hello")); err != nil {
		t.Fatal(err)
	}
	calls := model.Calls()
	if len(calls) != 1 || requestSystem(calls[0]) != "trusted-only" {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestExchangeInstructionFailureLeavesNoStartedRun(t *testing.T) {
	repository := storememory.New()
	driver := &countingDriver{}
	config := sessionConfig(t, driver, repository, artifactmemory.New(), spill.Config{})
	want := errors.New("instruction backend unavailable")
	config.Instructions = harnessruntime.ExchangeInstructionProviderFunc(func(context.Context, harnessruntime.ExchangeContext) (string, error) {
		return "", want
	})
	current, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := current.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "blocked")); !errors.Is(err, want) {
		t.Fatalf("prompt error = %v, want %v", err, want)
	}
	if current.State() != Idle || driver.Count() != 0 || current.run != nil {
		t.Fatalf("state=%s drives=%d run=%+v", current.State(), driver.Count(), current.run)
	}
	loaded, err := current.journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range loaded.Entries {
		if entry.Kind == kindRunOpened {
			t.Fatalf("instruction failure persisted run.opened: %+v", entry)
		}
	}
}

func TestZeroToolBudgetAllowsToolFreeExchange(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{textStep("done")}}
	current, err := New(context.Background(), sessionConfig(
		t, agentic.NewAgent("base", model), storememory.New(), artifactmemory.New(), spill.Config{},
	), WithBudget(agentic.UsageLimits{MaxToolCalls: agentic.IntPtr(0)}))
	if err != nil {
		t.Fatal(err)
	}
	execution, err := current.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "direct"))
	if err != nil || execution.Status != agentic.ExecutionCompleted {
		t.Fatalf("execution=%+v err=%v", execution, err)
	}
}

func TestExchangeInstructionsStayStableThroughSuspensionResume(t *testing.T) {
	repository := storememory.New()
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "danger-1", Name: "danger", Input: map[string]any{"value": "x"}})},
		textStep("finished"),
	}}
	agent := agentic.NewAgent("base", model)
	agentic.AddTool(agent,
		func(context.Context, resumeToolInput) (string, error) { return "ok", nil },
		agentic.AutoToolName("danger"),
		agentic.AutoToolDescription("perform a gated action"),
	)
	policy, err := permission.New(permission.DecisionDeny, permission.Rule{Pattern: "tool/danger/**", Decision: permission.DecisionAsk})
	if err != nil {
		t.Fatal(err)
	}
	permissionCapability, err := permission.NewCapability(policy)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(permissionCapability)
	if err != nil {
		t.Fatal(err)
	}
	value := "stable-v1"
	var resolutions atomic.Int32
	config := sessionConfig(t, agent, repository, artifactmemory.New(), spill.Config{})
	config.ToolGate = plan.ToolGate()
	config.Context = plan.ContextPolicy()
	config.Instructions = harnessruntime.ExchangeInstructionProviderFunc(func(context.Context, harnessruntime.ExchangeContext) (string, error) {
		resolutions.Add(1)
		return value, nil
	})
	current, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := current.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "start"))
	if err != nil || execution.Status != agentic.ExecutionSuspended {
		t.Fatalf("execution=%+v err=%v", execution, err)
	}
	value = "stable-v2"
	if _, err := current.Resume(context.Background(), ResumeRequest{
		SuspensionID: execution.Suspension.ID,
		Resolutions:  []ToolResolution{{CallID: "danger-1", Action: ResolutionApprove}},
	}); err != nil {
		t.Fatal(err)
	}
	calls := model.Calls()
	if len(calls) != 2 || resolutions.Load() != 1 {
		t.Fatalf("model calls=%d resolutions=%d", len(calls), resolutions.Load())
	}
	if second := requestSystem(calls[1]); !strings.Contains(second, "stable-v1") || strings.Contains(second, "stable-v2") {
		t.Fatalf("resumed system = %q", second)
	}
}
