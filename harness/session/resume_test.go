package session

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/permission"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type resumeToolInput struct {
	Value string `json:"value"`
}

func TestDeferredResumeValidatesPersistsAndRecoversBeforeEffects(t *testing.T) {
	repository := storememory.New()
	model := &scriptedModel{steps: []modelStep{
		{
			message: agentic.NewToolUseMessage(agentic.ToolUse{
				ID:    "danger-1",
				Name:  "danger",
				Input: map[string]any{"value": "original"},
			}),
			usage: agentic.Usage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6, Requests: 1},
		},
		textStep("finished"),
	}}
	agent := agentic.NewAgent("system", model)
	var handlerCalls atomic.Int32
	var observedValue atomic.Value
	var current *Session[string]
	agentic.AddTool(agent, func(_ context.Context, input resumeToolInput) (string, error) {
		handlerCalls.Add(1)
		observedValue.Store(input.Value)
		loaded, err := current.journal.Load(context.Background())
		if err != nil {
			return "", err
		}
		for _, entry := range loaded.Entries {
			if entry.Kind == kindResolutionAccepted {
				return "executed", nil
			}
		}
		return "", errors.New("handler ran before durable resolution acceptance")
	}, agentic.AutoToolName("danger"), agentic.AutoToolDescription("Perform a dangerous operation"))

	policy, err := permission.New(permission.DecisionDeny,
		permission.Rule{Pattern: "tool/danger/**", Decision: permission.DecisionAsk},
	)
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
	config := sessionConfig(t, agent, repository, artifactmemory.New(), spill.Config{})
	config.ToolGate = plan.ToolGate()
	config.Context = plan.ContextPolicy()
	current, err = New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := current.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "start"))
	if err != nil {
		t.Fatal(err)
	}
	if execution.Status != agentic.ExecutionSuspended || current.State() != Suspended ||
		execution.Suspension == nil || handlerCalls.Load() != 0 {
		t.Fatalf("execution=%#v state=%s calls=%d", execution, current.State(), handlerCalls.Load())
	}

	invalid := []ResumeRequest{
		{SuspensionID: execution.Suspension.ID},
		{
			SuspensionID: execution.Suspension.ID,
			Resolutions:  []ToolResolution{{CallID: "unknown", Action: ResolutionApprove}},
		},
		{
			SuspensionID: execution.Suspension.ID,
			Resolutions: []ToolResolution{
				{CallID: "danger-1", Action: ResolutionApprove},
				{CallID: "danger-1", Action: ResolutionDeny},
			},
		},
	}
	for index, request := range invalid {
		if _, err := current.Resume(context.Background(), request); !errors.Is(err, ErrInvalidResumeRequest) {
			t.Fatalf("invalid request %d error = %v", index, err)
		}
		if handlerCalls.Load() != 0 || current.State() != Suspended {
			t.Fatalf("invalid request %d state=%s calls=%d", index, current.State(), handlerCalls.Load())
		}
	}
	loaded, err := current.journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range loaded.Entries {
		if entry.Kind == kindResolutionAccepted {
			t.Fatal("invalid resume request was persisted")
		}
	}

	sessionID := current.ID()
	if err := current.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	current, err = Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if current.ID() != sessionID || current.State() != Suspended {
		t.Fatalf("recovered id=%s state=%s", current.ID(), current.State())
	}
	prompt := agentic.NewTextMessage(agentic.RoleUser, "after approval")
	execution, err = current.Resume(context.Background(), ResumeRequest{
		SuspensionID: execution.Suspension.ID,
		Resolutions: []ToolResolution{{
			CallID:       "danger-1",
			Action:       ResolutionApprove,
			OverrideArgs: map[string]any{"value": "approved"},
		}},
		Prompt: &prompt,
	})
	if err != nil {
		t.Fatal(err)
	}
	if execution.Status != agentic.ExecutionCompleted || current.State() != Idle ||
		handlerCalls.Load() != 1 || observedValue.Load() != "approved" {
		t.Fatalf("execution=%#v state=%s calls=%d value=%v", execution, current.State(), handlerCalls.Load(), observedValue.Load())
	}
	calls := model.Calls()
	if len(calls) != 2 {
		t.Fatalf("model calls = %d", len(calls))
	}
	last := calls[1].Messages
	if len(last) < 5 || last[len(last)-1].Role != agentic.RoleUser ||
		last[len(last)-1].GetTextContent() != "after approval" ||
		last[len(last)-2].Role != agentic.RoleTool {
		t.Fatalf("resume message order = %#v", last)
	}
}

func TestIndeterminateRecoveryRejectsApprovalAndAcceptsExternalResult(t *testing.T) {
	call := agentic.ToolUse{ID: "indeterminate-1", Name: "charge"}
	config, driver, _ := crashedConfig(t, []agentic.ToolUse{call}, "started")
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := recovered.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != Suspended || snapshot.Suspension == nil {
		t.Fatalf("snapshot = %#v", snapshot)
	}
	if _, err := recovered.Resume(context.Background(), ResumeRequest{
		SuspensionID: snapshot.Suspension.ID,
		Resolutions:  []ToolResolution{{CallID: call.ID, Action: ResolutionApprove}},
	}); !errors.Is(err, ErrIndeterminateTool) {
		t.Fatalf("approval error = %v", err)
	}
	if driver.Count() != 0 {
		t.Fatalf("indeterminate handler path repeated: drives=%d", driver.Count())
	}
	execution, err := recovered.Resume(context.Background(), ResumeRequest{
		SuspensionID: snapshot.Suspension.ID,
		Resolutions: []ToolResolution{{
			CallID: call.ID,
			Action: ResolutionExternalResult,
			Result: "already charged",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if execution.Status != agentic.ExecutionCompleted || driver.Count() != 1 {
		t.Fatalf("execution=%#v drives=%d", execution, driver.Count())
	}
	last := driver.Last()
	if last.Mode != agentic.DriveContinue ||
		last.History[len(last.History)-1].Role != agentic.RoleTool ||
		last.History[len(last.History)-1].GetToolResults()[0].Content != "already charged" {
		t.Fatalf("continued history = %#v", last.History)
	}
}

func TestSuspendedUsageAndBudgetRestoreAcrossResume(t *testing.T) {
	repository := storememory.New()
	model := &scriptedModel{steps: []modelStep{
		{
			message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "budget-1", Name: "budget_tool"}),
			usage:   agentic.Usage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6, Requests: 1},
		},
		{
			message: agentic.NewTextMessage(agentic.RoleAssistant, "done"),
			usage:   agentic.Usage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8, Requests: 1},
		},
	}}
	agent := agentic.NewAgent("", model)
	agentic.AddTool(agent, func(context.Context, struct{}) (string, error) {
		return "ok", nil
	}, agentic.AutoToolName("budget_tool"), agentic.AutoToolDescription("Budget test tool"))
	policy, _ := permission.New(permission.DecisionDeny,
		permission.Rule{Pattern: "tool/budget_tool/**", Decision: permission.DecisionAsk},
	)
	permissionCapability, _ := permission.NewCapability(policy)
	plan, err := capability.Compile(permissionCapability)
	if err != nil {
		t.Fatal(err)
	}
	config := sessionConfig(t, agent, repository, artifactmemory.New(), spill.Config{})
	config.ToolGate = plan.ToolGate()
	config.Context = plan.ContextPolicy()
	maxTotal, maxRequests := 20, 2
	first, err := New(context.Background(), config, WithBudget(agentic.UsageLimits{
		MaxTotalTokens: &maxTotal,
		MaxRequests:    &maxRequests,
	}))
	if err != nil {
		t.Fatal(err)
	}
	execution, err := first.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "start"))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := first.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Usage.TotalTokens != 6 || snapshot.Usage.Requests != 1 {
		t.Fatalf("suspended usage = %#v", snapshot.Usage)
	}
	if err := first.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ = recovered.Snapshot(context.Background())
	if snapshot.Usage.TotalTokens != 6 || snapshot.Usage.Requests != 1 {
		t.Fatalf("recovered usage = %#v", snapshot.Usage)
	}
	execution, err = recovered.Resume(context.Background(), ResumeRequest{
		SuspensionID: execution.Suspension.ID,
		Resolutions:  []ToolResolution{{CallID: "budget-1", Action: ResolutionApprove}},
	})
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ = recovered.Snapshot(context.Background())
	if execution.Status != agentic.ExecutionCompleted ||
		snapshot.Usage.TotalTokens != 14 || snapshot.Usage.Requests != 2 {
		t.Fatalf("completed execution=%#v usage=%#v", execution, snapshot.Usage)
	}
}
