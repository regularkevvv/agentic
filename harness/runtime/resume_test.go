package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

type resumeDriver struct {
	calls  int
	input  agentic.ResumeInput
	result *agentic.Execution[string]
}

func (d *resumeDriver) Run(context.Context, string, ...agentic.RunOption) (*agentic.Result[string], error) {
	return nil, errors.New("unexpected Run")
}

func (d *resumeDriver) Drive(context.Context, agentic.DriveInput, ...agentic.RunOption) (*agentic.Execution[string], error) {
	return nil, errors.New("unexpected Drive")
}

func (d *resumeDriver) Resume(_ context.Context, input agentic.ResumeInput, _ ...agentic.RunOption) (*agentic.Execution[string], error) {
	d.calls++
	d.input = input
	return d.result, nil
}

func TestPlanResumeValidatesWholeResolutionSetBeforeEffects(t *testing.T) {
	t.Parallel()
	calls := []agentic.ToolUse{
		{ID: "allowed", Name: "read"},
		{ID: "asked", Name: "shell"},
		{ID: "external", Name: "lookup"},
	}
	suspension := testPermissionSuspension(t, calls, []string{"asked", "external"}, []string{"allowed"})
	valid := ResumeRequest{
		SuspensionID: suspension.ID,
		Resolutions: []ToolResolution{
			{CallID: "asked", Action: ResolutionDeny, Reason: "not now"},
			{CallID: "external", Action: ResolutionExternalResult, Result: map[string]any{"value": 7}},
		},
	}
	decisions, err := PlanResume(suspension, valid)
	if err != nil {
		t.Fatal(err)
	}
	if len(decisions) != 3 ||
		decisions[0].CallID != "allowed" || decisions[0].Action != agentic.ToolResumeExecute ||
		decisions[1].CallID != "asked" || decisions[1].Action != agentic.ToolResumeReturn ||
		decisions[1].Result == nil || !decisions[1].Result.IsError ||
		decisions[2].CallID != "external" || decisions[2].Result == nil ||
		!reflect.DeepEqual(decisions[2].Result.Content, map[string]any{"value": 7}) {
		t.Fatalf("decisions = %#v", decisions)
	}

	tests := map[string]ResumeRequest{
		"wrong suspension": {
			SuspensionID: "wrong",
		},
		"missing": {
			SuspensionID: suspension.ID,
			Resolutions:  []ToolResolution{{CallID: "asked", Action: ResolutionApprove}},
		},
		"unknown": {
			SuspensionID: suspension.ID,
			Resolutions: []ToolResolution{
				{CallID: "asked", Action: ResolutionApprove},
				{CallID: "unknown", Action: ResolutionApprove},
			},
		},
		"duplicate": {
			SuspensionID: suspension.ID,
			Resolutions: []ToolResolution{
				{CallID: "asked", Action: ResolutionApprove},
				{CallID: "asked", Action: ResolutionDeny},
			},
		},
		"invalid action": {
			SuspensionID: suspension.ID,
			Resolutions: []ToolResolution{
				{CallID: "asked", Action: ResolutionInvalid},
				{CallID: "external", Action: ResolutionApprove},
			},
		},
	}
	for name, request := range tests {
		name, request := name, request
		t.Run(name, func(t *testing.T) {
			driver := &resumeDriver{}
			if _, err := Resume(context.Background(), driver, nil, suspension, request); !errors.Is(err, ErrInvalidResumeRequest) {
				t.Fatalf("error = %v", err)
			}
			if driver.calls != 0 {
				t.Fatalf("driver called %d times", driver.calls)
			}
		})
	}
}

func TestResumeDelegatesCompleteOrderedDecisions(t *testing.T) {
	t.Parallel()
	calls := []agentic.ToolUse{{ID: "one", Name: "write", Input: map[string]any{"value": 1}}}
	suspension := testPermissionSuspension(t, calls, []string{"one"}, nil)
	want := &agentic.Execution[string]{Status: agentic.ExecutionCompleted}
	driver := &resumeDriver{result: want}
	prompt := agentic.NewTextMessage(agentic.RoleUser, "continue")
	history := []agentic.Message{agentic.NewToolUseMessage(calls...)}
	execution, err := Resume(context.Background(), driver, history, suspension, ResumeRequest{
		SuspensionID: suspension.ID,
		Resolutions: []ToolResolution{{
			CallID:       "one",
			Action:       ResolutionApprove,
			OverrideArgs: map[string]any{"value": 2},
		}},
		Prompt: &prompt,
	})
	if err != nil {
		t.Fatal(err)
	}
	if execution != want || driver.calls != 1 || len(driver.input.Decisions) != 1 ||
		driver.input.Decisions[0].Input["value"] != 2 || driver.input.Prompt == nil {
		t.Fatalf("execution=%#v input=%#v calls=%d", execution, driver.input, driver.calls)
	}
	history[0] = agentic.NewTextMessage(agentic.RoleUser, "mutated")
	prompt = agentic.NewTextMessage(agentic.RoleUser, "mutated")
	if driver.input.History[0].Role != agentic.RoleAssistant ||
		driver.input.Prompt.GetTextContent() != "continue" {
		t.Fatal("Resume exposed caller-owned messages")
	}
}

func TestInspectDeferredRejectsMalformedPayload(t *testing.T) {
	t.Parallel()
	if _, err := InspectDeferred(agentic.Suspension{ID: "s", Kind: "other"}); !errors.Is(err, ErrUnsupportedDeferral) {
		t.Fatalf("kind error = %v", err)
	}
	calls := []agentic.ToolUse{{ID: "one", Name: "tool"}}
	value := testPermissionSuspension(t, calls, []string{"one"}, nil)
	value.Payload = []byte("{")
	if _, err := InspectDeferred(value); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("payload error = %v", err)
	}
}

func TestInspectDeferredValidatesEveryEnvelopeInvariant(t *testing.T) {
	t.Parallel()
	call := agentic.ToolUse{ID: "one", Name: "tool"}
	inner := func(batch DeferredBatch) agentic.ToolDeferral {
		payload, err := json.Marshal(batch)
		if err != nil {
			t.Fatal(err)
		}
		return agentic.ToolDeferral{Kind: PermissionDeferralKind, Payload: payload}
	}
	makeSuspension := func(root rootToolSuspensionPayload) agentic.Suspension {
		payload, err := json.Marshal(root)
		if err != nil {
			t.Fatal(err)
		}
		return agentic.Suspension{ID: "s", Kind: PermissionDeferralKind, Payload: payload}
	}
	valid := rootToolSuspensionPayload{
		Version:           1,
		SuspensionID:      "s",
		Calls:             []agentic.ToolUse{call},
		ExecutableCallIDs: []string{"one"},
		Deferral: inner(DeferredBatch{
			Version:               DeferredBatchVersion,
			RequiredResolutionIDs: []string{"one"},
		}),
	}
	malformed := map[string]rootToolSuspensionPayload{
		"version":       func() rootToolSuspensionPayload { value := valid; value.Version = 2; return value }(),
		"id":            func() rootToolSuspensionPayload { value := valid; value.SuspensionID = "other"; return value }(),
		"kind":          func() rootToolSuspensionPayload { value := valid; value.Deferral.Kind = "other"; return value }(),
		"no calls":      func() rootToolSuspensionPayload { value := valid; value.Calls = nil; return value }(),
		"no executable": func() rootToolSuspensionPayload { value := valid; value.ExecutableCallIDs = nil; return value }(),
		"bad inner JSON": func() rootToolSuspensionPayload {
			value := valid
			value.Deferral.Payload = []byte(`"{"`)
			return value
		}(),
		"bad inner version": func() rootToolSuspensionPayload {
			value := valid
			value.Deferral = inner(DeferredBatch{Version: 2, RequiredResolutionIDs: []string{"one"}})
			return value
		}(),
		"no required": func() rootToolSuspensionPayload {
			value := valid
			value.Deferral = inner(DeferredBatch{Version: DeferredBatchVersion})
			return value
		}(),
		"empty executable": func() rootToolSuspensionPayload {
			value := valid
			value.ExecutableCallIDs = []string{""}
			return value
		}(),
		"duplicate executable": func() rootToolSuspensionPayload {
			value := valid
			value.ExecutableCallIDs = []string{"one", "one"}
			return value
		}(),
		"unknown required": func() rootToolSuspensionPayload {
			value := valid
			value.Deferral = inner(DeferredBatch{Version: DeferredBatchVersion, RequiredResolutionIDs: []string{"unknown"}})
			return value
		}(),
		"duplicate required": func() rootToolSuspensionPayload {
			value := valid
			value.Deferral = inner(DeferredBatch{Version: DeferredBatchVersion, RequiredResolutionIDs: []string{"one", "one"}})
			return value
		}(),
		"required and preallowed": func() rootToolSuspensionPayload {
			value := valid
			value.Deferral = inner(DeferredBatch{
				Version:               DeferredBatchVersion,
				RequiredResolutionIDs: []string{"one"},
				PreAllowedCallIDs:     []string{"one"},
			})
			return value
		}(),
		"duplicate preallowed": func() rootToolSuspensionPayload {
			value := valid
			value.ExecutableCallIDs = []string{"one", "two"}
			value.Deferral = inner(DeferredBatch{
				Version:               DeferredBatchVersion,
				RequiredResolutionIDs: []string{"one"},
				PreAllowedCallIDs:     []string{"two", "two"},
			})
			return value
		}(),
		"unclassified executable": func() rootToolSuspensionPayload {
			value := valid
			value.ExecutableCallIDs = []string{"one", "two"}
			return value
		}(),
	}
	for name, root := range malformed {
		root := root
		t.Run(name, func(t *testing.T) {
			if _, err := InspectDeferred(makeSuspension(root)); !errors.Is(err, ErrInvalidResumeRequest) {
				t.Fatalf("malformed error = %v", err)
			}
		})
	}

	frontier, err := InspectDeferred(makeSuspension(valid))
	if err != nil || len(frontier.Calls) != 1 || frontier.Calls[0].ID != "one" {
		t.Fatalf("valid frontier = %#v, %v", frontier, err)
	}
	frontier.Calls[0].Input = map[string]any{"mutated": true}
	if call.Input != nil {
		t.Fatal("frontier mutated source call")
	}
}

func TestResumeValidationDefaultsAndDeepCopies(t *testing.T) {
	t.Parallel()
	call := agentic.ToolUse{ID: "one", Name: "tool"}
	suspension := testPermissionSuspension(t, []agentic.ToolUse{call}, []string{"one"}, nil)
	if _, err := Resume[string](context.Background(), nil, nil, suspension, ResumeRequest{}); err == nil {
		t.Fatal("nil driver succeeded")
	}
	assistant := agentic.NewTextMessage(agentic.RoleAssistant, "bad")
	if _, err := PlanResume(suspension, ResumeRequest{
		SuspensionID: suspension.ID,
		Resolutions:  []ToolResolution{{CallID: "one", Action: ResolutionApprove}},
		Prompt:       &assistant,
	}); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("assistant prompt = %v", err)
	}

	decisions, err := PlanResume(suspension, ResumeRequest{
		SuspensionID: suspension.ID,
		Resolutions:  []ToolResolution{{CallID: "one", Action: ResolutionDeny}},
	})
	if err != nil || decisions[0].Result == nil ||
		decisions[0].Result.Content != "Tool call denied: denied by operator" {
		t.Fatalf("default denial = %#v, %v", decisions, err)
	}

	override := map[string]any{
		"map":   map[string]any{"value": "original"},
		"slice": []any{map[string]any{"value": "original"}},
		"text":  []string{"original"},
		"bytes": []byte("original"),
	}
	decisions, err = PlanResume(suspension, ResumeRequest{
		SuspensionID: suspension.ID,
		Resolutions:  []ToolResolution{{CallID: "one", Action: ResolutionApprove, OverrideArgs: override}},
	})
	if err != nil {
		t.Fatal(err)
	}
	override["map"].(map[string]any)["value"] = "changed"
	override["slice"].([]any)[0].(map[string]any)["value"] = "changed"
	override["text"].([]string)[0] = "changed"
	override["bytes"].([]byte)[0] = 'X'
	got := decisions[0].Input
	if got["map"].(map[string]any)["value"] != "original" ||
		got["slice"].([]any)[0].(map[string]any)["value"] != "original" ||
		got["text"].([]string)[0] != "original" ||
		string(got["bytes"].([]byte)) != "original" {
		t.Fatalf("decision aliases caller input: %#v", got)
	}

	missingCall := testPermissionSuspension(t, []agentic.ToolUse{{ID: "other", Name: "tool"}}, []string{"one"}, nil)
	if _, err := PlanResume(missingCall, ResumeRequest{
		SuspensionID: missingCall.ID,
		Resolutions:  []ToolResolution{{CallID: "one", Action: ResolutionApprove}},
	}); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("missing frontier call = %v", err)
	}
	duplicateCall := testPermissionSuspension(
		t,
		[]agentic.ToolUse{{ID: "one", Name: "tool"}, {ID: "one", Name: "tool"}},
		[]string{"one"},
		nil,
	)
	if _, err := PlanResume(duplicateCall, ResumeRequest{
		SuspensionID: duplicateCall.ID,
		Resolutions:  []ToolResolution{{CallID: "one", Action: ResolutionApprove}},
	}); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("duplicate frontier call = %v", err)
	}
}

func TestPlanResumeRejectsFrontierCallMapInconsistencies(t *testing.T) {
	makeSuspension := func(calls []agentic.ToolUse, executable string) agentic.Suspension {
		t.Helper()
		inner, err := json.Marshal(DeferredBatch{
			Version:               DeferredBatchVersion,
			RequiredResolutionIDs: []string{executable},
		})
		if err != nil {
			t.Fatal(err)
		}
		id := "suspension-inconsistent"
		payload, err := json.Marshal(rootToolSuspensionPayload{
			Version:           1,
			SuspensionID:      id,
			Calls:             calls,
			ExecutableCallIDs: []string{executable},
			Deferral: agentic.ToolDeferral{
				Kind:    PermissionDeferralKind,
				Payload: inner,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return agentic.Suspension{ID: id, Kind: PermissionDeferralKind, Payload: payload}
	}
	request := func(suspension agentic.Suspension, callID string) ResumeRequest {
		return ResumeRequest{
			SuspensionID: suspension.ID,
			Resolutions:  []ToolResolution{{CallID: callID, Action: ResolutionApprove}},
		}
	}

	duplicate := makeSuspension([]agentic.ToolUse{
		{ID: "one", Name: "tool"},
		{ID: "one", Name: "tool"},
	}, "one")
	if _, err := PlanResume(duplicate, request(duplicate, "one")); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("duplicate call map = %v", err)
	}

	missing := makeSuspension([]agentic.ToolUse{{ID: "other", Name: "tool"}}, "one")
	if _, err := PlanResume(missing, request(missing, "one")); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("missing executable call = %v", err)
	}

	if cloneRuntimeMessages(nil) != nil || cloneRuntimeCalls(nil) != nil {
		t.Fatal("empty runtime clones were non-nil")
	}
}

func testPermissionSuspension(
	t *testing.T,
	calls []agentic.ToolUse,
	required []string,
	preAllowed []string,
) agentic.Suspension {
	t.Helper()
	inner, err := json.Marshal(DeferredBatch{
		Version:               DeferredBatchVersion,
		RequiredResolutionIDs: required,
		PreAllowedCallIDs:     preAllowed,
	})
	if err != nil {
		t.Fatal(err)
	}
	id := "suspension-test"
	known := make(map[string]bool, len(required)+len(preAllowed))
	for _, callID := range required {
		known[callID] = true
	}
	for _, callID := range preAllowed {
		known[callID] = true
	}
	executable := make([]string, 0, len(known))
	for _, call := range calls {
		if known[call.ID] {
			executable = append(executable, call.ID)
		}
	}
	payload, err := json.Marshal(rootToolSuspensionPayload{
		Version:           1,
		SuspensionID:      id,
		Calls:             calls,
		ExecutableCallIDs: executable,
		Deferral: agentic.ToolDeferral{
			Kind:    PermissionDeferralKind,
			Payload: inner,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return agentic.Suspension{
		ID:           id,
		Kind:         PermissionDeferralKind,
		FrontierHash: "v1:test",
		Payload:      payload,
	}
}
