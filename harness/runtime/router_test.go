package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

func TestResumeRouterDispatchesAndDeepCopies(t *testing.T) {
	called := false
	planner := ResumePlannerFunc(func(suspension agentic.Suspension, request ResumeRequest) ([]agentic.ToolResumeDecision, error) {
		called = true
		suspension.Payload[0] = 'X'
		request.Resolutions[0].OverrideArgs["value"] = "changed"
		request.Resolutions[0].Result.(map[string]any)["result"] = "changed"
		return []agentic.ToolResumeDecision{{CallID: "call", Action: agentic.ToolResumeExecute}}, nil
	})
	router, err := NewResumeRouter(map[string]ResumePlanner{"custom": planner})
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("payload")
	request := ResumeRequest{Resolutions: []ToolResolution{{OverrideArgs: map[string]any{"value": "original"}, Result: map[string]any{"result": "original"}}}}
	decisions, err := router.PlanResume(agentic.Suspension{Kind: "custom", Payload: payload}, request)
	if err != nil || !called || decisions[0].CallID != "call" {
		t.Fatalf("decisions=%#v called=%v err=%v", decisions, called, err)
	}
	if string(payload) != "payload" || request.Resolutions[0].OverrideArgs["value"] != "original" || request.Resolutions[0].Result.(map[string]any)["result"] != "original" {
		t.Fatal("router exposed caller-owned values")
	}
	if _, err := router.PlanResume(agentic.Suspension{Kind: "unknown"}, ResumeRequest{}); !errors.Is(err, ErrUnsupportedDeferral) {
		t.Fatalf("unknown kind = %v", err)
	}
	var nilRouter *ResumeRouter
	if _, err := nilRouter.PlanResume(agentic.Suspension{Kind: "custom"}, ResumeRequest{}); !errors.Is(err, ErrUnsupportedDeferral) {
		t.Fatalf("nil router = %v", err)
	}
}

func TestResumeRouterValidationAndDefaultPermissionPlanner(t *testing.T) {
	if _, err := NewResumeRouter(map[string]ResumePlanner{"": ResumePlannerFunc(nil)}); err == nil {
		t.Fatal("invalid planner succeeded")
	}
	if _, err := NewResumeRouter(map[string]ResumePlanner{PermissionDeferralKind: DefaultResumePlanner()}); !errors.Is(err, ErrDuplicatePlanner) {
		t.Fatalf("duplicate permission planner = %v", err)
	}
	call := agentic.ToolUse{ID: "one", Name: "tool"}
	suspension := testPermissionSuspension(t, []agentic.ToolUse{call}, []string{"one"}, nil)
	router, err := NewResumeRouter(nil)
	if err != nil {
		t.Fatal(err)
	}
	decisions, err := router.PlanResume(suspension, ResumeRequest{
		SuspensionID: suspension.ID,
		Resolutions:  []ToolResolution{{CallID: "one", Action: ResolutionApprove}},
	})
	if err != nil || len(decisions) != 1 || decisions[0].Action != agentic.ToolResumeExecute {
		t.Fatalf("permission decision=%#v err=%v", decisions, err)
	}
	if _, err := DefaultResumePlanner().PlanResume(suspension, ResumeRequest{SuspensionID: "wrong"}); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("default planner error = %v", err)
	}
}

func TestInspectToolSuspensionRejectsRootEnvelopeViolations(t *testing.T) {
	makeValue := func(root rootToolSuspensionPayload) agentic.Suspension {
		payload, _ := json.Marshal(root)
		return agentic.Suspension{ID: "s", Kind: "kind", Payload: payload}
	}
	valid := rootToolSuspensionPayload{
		Version: 1, SuspensionID: "s", HandlerSuspension: true,
		Calls: []agentic.ToolUse{{ID: "one", Name: "tool"}}, ExecutableCallIDs: []string{"one"},
		Deferral: agentic.ToolDeferral{Kind: "kind", Payload: []byte(`{}`)},
	}
	frontier, err := InspectToolSuspension(makeValue(valid), "kind")
	if err != nil || !frontier.HandlerSuspension || frontier.Calls[0].ID != "one" {
		t.Fatalf("frontier=%#v err=%v", frontier, err)
	}
	for name, mutate := range map[string]func(*rootToolSuspensionPayload){
		"duplicate call":       func(value *rootToolSuspensionPayload) { value.Calls = append(value.Calls, value.Calls[0]) },
		"empty name":           func(value *rootToolSuspensionPayload) { value.Calls[0].Name = "" },
		"unknown executable":   func(value *rootToolSuspensionPayload) { value.ExecutableCallIDs[0] = "other" },
		"duplicate executable": func(value *rootToolSuspensionPayload) { value.ExecutableCallIDs = []string{"one", "one"} },
	} {
		t.Run(name, func(t *testing.T) {
			value := valid
			value.Calls = cloneRuntimeCalls(valid.Calls)
			value.ExecutableCallIDs = append([]string(nil), valid.ExecutableCallIDs...)
			mutate(&value)
			if _, err := InspectToolSuspension(makeValue(value), "kind"); !errors.Is(err, ErrInvalidResumeRequest) {
				t.Fatalf("error = %v", err)
			}
		})
	}
	if _, err := InspectToolSuspension(agentic.Suspension{ID: "s", Kind: "other"}, "kind"); !errors.Is(err, ErrUnsupportedDeferral) {
		t.Fatalf("kind mismatch = %v", err)
	}
	if _, err := InspectToolSuspension(agentic.Suspension{ID: "s", Kind: "kind", Payload: []byte("{")}, "kind"); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("decode error = %v", err)
	}
}

func TestResumePlannerFuncAndClonePrompt(t *testing.T) {
	prompt := agentic.NewTextMessage(agentic.RoleUser, "original")
	called := false
	planner := ResumePlannerFunc(func(_ agentic.Suspension, request ResumeRequest) ([]agentic.ToolResumeDecision, error) {
		called = request.Prompt.GetTextContent() == "original"
		request.Prompt.Content[0].Text = "changed"
		return nil, nil
	})
	router, _ := NewResumeRouter(map[string]ResumePlanner{"custom": planner})
	_, _ = router.PlanResume(agentic.Suspension{Kind: "custom"}, ResumeRequest{Prompt: &prompt})
	if !called || prompt.GetTextContent() != "original" {
		t.Fatal("prompt was not defensively copied")
	}
	if _, err := Resume[string](context.Background(), nil, nil, agentic.Suspension{}, ResumeRequest{}); err == nil {
		t.Fatal("nil driver succeeded")
	}
}
