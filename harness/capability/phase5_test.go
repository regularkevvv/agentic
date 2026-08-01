package capability

import (
	"context"
	"errors"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type namedHandler string

func (h namedHandler) Name() string { return string(h) }
func (h namedHandler) Execute(context.Context, map[string]any, any) (any, error) {
	return string(h), nil
}

func TestTakeToolsetPreservesOrderHidesDefinitionsAndReservesNames(t *testing.T) {
	registry := newRegistry()
	tools := []agentic.Tool{
		{Function: agentic.Function{Name: "one"}},
		{Function: agentic.Function{Name: "two"}},
		{Function: agentic.Function{Name: "three"}},
	}
	if err := registry.AddToolset(testToolset{tools: tools[:2], handlers: []agentic.ToolHandler{namedHandler("one"), namedHandler("two")}}); err != nil {
		t.Fatal(err)
	}
	if err := registry.AddToolset(testToolset{tools: tools[2:], handlers: []agentic.ToolHandler{namedHandler("three")}}); err != nil {
		t.Fatal(err)
	}
	if err := registry.MarkDelegationTool("three"); err != nil || !registry.IsDelegationTool("three") || !registry.HasTool("two") {
		t.Fatalf("classification err=%v delegation=%v has=%v", err, registry.IsDelegationTool("three"), registry.HasTool("two"))
	}
	selected, err := registry.TakeToolset("three", "one")
	if err != nil {
		t.Fatal(err)
	}
	selectedTools, selectedHandlers := selected.ToolsAndHandlers()
	if got := []string{selectedTools[0].Function.Name, selectedTools[1].Function.Name}; !reflect.DeepEqual(got, []string{"three", "one"}) ||
		selectedHandlers[0].Name() != "three" {
		t.Fatalf("selected tools=%v handlers=%v", got, selectedHandlers)
	}
	if len(registry.tools) != 1 || registry.tools[0].Function.Name != "two" || len(registry.toolsets) != 1 || !registry.HasTool("one") {
		t.Fatalf("remaining tools=%#v sets=%d reserved=%v", registry.tools, len(registry.toolsets), registry.HasTool("one"))
	}
	if _, err := registry.TakeToolset(); err == nil {
		t.Fatal("empty selection succeeded")
	}
	if _, err := registry.TakeToolset("missing"); !errors.Is(err, ErrUnknownTool) {
		t.Fatalf("unknown selection = %v", err)
	}
	if _, err := registry.TakeToolset("two", "two"); !errors.Is(err, ErrDuplicateSelection) {
		t.Fatalf("duplicate selection = %v", err)
	}
	registry.frozen = true
	if _, err := registry.TakeToolset("two"); !errors.Is(err, ErrRegistryFrozen) {
		t.Fatalf("frozen selection = %v", err)
	}
}

func TestResumePlannerRegistrationAndPlanRouting(t *testing.T) {
	planner := harnessruntime.ResumePlannerFunc(func(suspension agentic.Suspension, request harnessruntime.ResumeRequest) ([]agentic.ToolResumeDecision, error) {
		return []agentic.ToolResumeDecision{{CallID: request.SuspensionID, Action: agentic.ToolResumeReturn}}, nil
	})
	plan, err := Compile(Func{Name: "planner", Apply: func(registry *Registry) error {
		return registry.AddResumePlanner("test.resume", planner)
	}})
	if err != nil {
		t.Fatal(err)
	}
	decisions, err := plan.ResumePlanner().PlanResume(agentic.Suspension{ID: "s", Kind: "test.resume"}, harnessruntime.ResumeRequest{SuspensionID: "s"})
	if err != nil || len(decisions) != 1 || decisions[0].CallID != "s" {
		t.Fatalf("decisions=%#v err=%v", decisions, err)
	}
	registry := newRegistry()
	if err := registry.AddResumePlanner("", planner); err == nil {
		t.Fatal("empty kind succeeded")
	}
	if err := registry.AddResumePlanner("test", nil); err == nil {
		t.Fatal("nil planner succeeded")
	}
	if err := registry.AddResumePlanner(harnessruntime.PermissionDeferralKind, planner); !errors.Is(err, ErrDuplicatePlanner) {
		t.Fatalf("permission planner = %v", err)
	}
	if err := registry.AddResumePlanner("test", planner); err != nil {
		t.Fatal(err)
	}
	if err := registry.AddResumePlanner("test", planner); !errors.Is(err, ErrDuplicatePlanner) {
		t.Fatalf("duplicate planner = %v", err)
	}
	registry.frozen = true
	if err := registry.AddResumePlanner("other", planner); !errors.Is(err, ErrRegistryFrozen) {
		t.Fatalf("frozen planner = %v", err)
	}
}
