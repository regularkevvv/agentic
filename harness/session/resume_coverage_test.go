package session

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/repair"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
)

type hookCodec struct {
	base   jsoncodec.Codec
	before func(any)
}

func (c hookCodec) Encode(value any) ([]byte, error) {
	if c.before != nil {
		c.before(value)
	}
	return c.base.Encode(value)
}

func (c hookCodec) Decode(payload []byte, value any) error {
	return c.base.Decode(payload, value)
}

func permissionSuspension(t *testing.T, id string, calls ...agentic.ToolUse) agentic.Suspension {
	t.Helper()
	required := make([]string, len(calls))
	for index, call := range calls {
		required[index] = call.ID
	}
	deferred, err := json.Marshal(harnessruntime.DeferredBatch{
		Version:               harnessruntime.DeferredBatchVersion,
		RequiredResolutionIDs: required,
	})
	if err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(struct {
		Version           int
		SuspensionID      string
		Calls             []agentic.ToolUse
		ExecutableCallIDs []string
		Deferral          agentic.ToolDeferral
	}{
		Version:           1,
		SuspensionID:      id,
		Calls:             calls,
		ExecutableCallIDs: required,
		Deferral: agentic.ToolDeferral{
			Kind:    harnessruntime.PermissionDeferralKind,
			Payload: deferred,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return agentic.Suspension{
		ID:      id,
		Kind:    harnessruntime.PermissionDeferralKind,
		Payload: payload,
	}
}

func suspendedPermissionSession(t *testing.T) (*Session[string], ResumeRequest) {
	t.Helper()
	session := newRunningSession(t)
	call := agentic.ToolUse{ID: "call", Name: "tool"}
	suspension := permissionSuspension(t, "suspension", call)
	session.mu.Lock()
	session.suspension = &suspension
	session.transitionLocked(Suspended)
	session.mu.Unlock()
	return session, ResumeRequest{
		SuspensionID: suspension.ID,
		Resolutions: []ToolResolution{{
			CallID: call.ID,
			Action: ResolutionDeny,
			Reason: "no",
		}},
	}
}

func TestResumeStatePreflightAndPlannerErrors(t *testing.T) {
	boom := errors.New("boom")
	session, request := suspendedPermissionSession(t)
	for _, test := range []struct {
		name   string
		mutate func(*Session[string], *ResumeRequest)
		want   error
	}{
		{"faulted", func(s *Session[string], _ *ResumeRequest) {
			s.state = Faulted
			s.fault = boom
		}, ErrSessionFaulted},
		{"closed", func(s *Session[string], _ *ResumeRequest) { s.state = Closed }, ErrSessionClosed},
		{"running", func(s *Session[string], _ *ResumeRequest) { s.state = Running }, ErrNotRunning},
		{"missing frontier", func(s *Session[string], _ *ResumeRequest) { s.suspension = nil }, ErrInvalidResumeRequest},
		{"missing run", func(s *Session[string], _ *ResumeRequest) { s.run = nil }, ErrInvalidResumeRequest},
		{"wrong ID", func(_ *Session[string], r *ResumeRequest) { r.SuspensionID = "other" }, ErrInvalidResumeRequest},
		{"bad prompt", func(_ *Session[string], r *ResumeRequest) {
			message := agentic.NewTextMessage(agentic.RoleAssistant, "bad")
			r.Prompt = &message
		}, ErrInvalidMessage},
	} {
		t.Run(test.name, func(t *testing.T) {
			current, currentRequest := suspendedPermissionSession(t)
			current.mu.Lock()
			test.mutate(current, &currentRequest)
			current.mu.Unlock()
			if _, err := current.Resume(context.Background(), currentRequest); !errors.Is(err, test.want) {
				t.Fatalf("Resume error = %v", err)
			}
		})
	}

	session.suspension.Kind = "unsupported"
	if _, err := session.Resume(context.Background(), request); err == nil {
		t.Fatal("unsupported suspension reached the driver")
	}
}

func TestResumeEncodingSecondCheckSuspensionChangeAndAppendFailures(t *testing.T) {
	boom := errors.New("boom")
	session, request := suspendedPermissionSession(t)
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	if _, err := session.Resume(context.Background(), request); !errors.Is(err, boom) {
		t.Fatalf("resolution encoding error = %v", err)
	}

	session, request = suspendedPermissionSession(t)
	session.codec = hookCodec{
		base: jsoncodec.New(),
		before: func(value any) {
			if reflect.TypeOf(value) == reflect.TypeOf(resolutionAcceptedPayload{}) {
				session.mu.Lock()
				session.transitionLocked(Closed)
				session.mu.Unlock()
			}
		},
	}
	if _, err := session.Resume(context.Background(), request); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("second resume state check = %v", err)
	}

	session, request = suspendedPermissionSession(t)
	session.codec = hookCodec{
		base: jsoncodec.New(),
		before: func(value any) {
			if reflect.TypeOf(value) == reflect.TypeOf(resolutionAcceptedPayload{}) {
				session.mu.Lock()
				copy := *session.suspension
				copy.ID = "changed"
				session.suspension = &copy
				session.mu.Unlock()
			}
		},
	}
	if _, err := session.Resume(context.Background(), request); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("changed suspension = %v", err)
	}

	session, request = suspendedPermissionSession(t)
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	if _, err := session.Resume(context.Background(), request); !errors.Is(err, boom) ||
		session.State() != Suspended {
		t.Fatalf("resolution append error = %v, state=%s", err, session.State())
	}
}

func recoveryPayload(t *testing.T, calls ...repair.PendingCall) []byte {
	t.Helper()
	payload, err := json.Marshal(recoverySuspensionPayload{Version: 1, Calls: calls})
	if err != nil {
		t.Fatal(err)
	}
	return payload
}

func indeterminateSession(t *testing.T, calls ...agentic.ToolUse) (*Session[string], agentic.Suspension) {
	t.Helper()
	session := newRunningSession(t)
	pending := make([]repair.PendingCall, len(calls))
	for index, call := range calls {
		pending[index] = repair.PendingCall{ID: call.ID, Name: call.Name, State: repair.PendingIndeterminate}
		session.run.started[call.ID] = true
	}
	suspension := agentic.Suspension{
		ID:      "recovery_suspension",
		Kind:    "harness.recovery.indeterminate",
		Payload: recoveryPayload(t, pending...),
	}
	session.messages = []agentic.Message{agentic.NewToolUseMessage(calls...)}
	session.suspension = &suspension
	session.state = Suspended
	return session, suspension
}

func TestResumeIndeterminateRejectsMalformedPayloadAndResolutionSets(t *testing.T) {
	call := agentic.ToolUse{ID: "call", Name: "tool"}
	session, suspension := indeterminateSession(t, call)
	for _, payload := range [][]byte{
		[]byte("{"),
		func() []byte {
			value, _ := json.Marshal(recoverySuspensionPayload{Version: 2})
			return value
		}(),
		recoveryPayload(t, repair.PendingCall{ID: "", State: repair.PendingIndeterminate}),
		recoveryPayload(t, repair.PendingCall{ID: call.ID, State: repair.PendingPlanned}),
		recoveryPayload(t,
			repair.PendingCall{ID: call.ID, State: repair.PendingIndeterminate},
			repair.PendingCall{ID: call.ID, State: repair.PendingIndeterminate},
		),
	} {
		copy := suspension
		copy.Payload = payload
		if _, err := session.resumeIndeterminate(context.Background(), &copy, ResumeRequest{
			SuspensionID: copy.ID,
		}); !errors.Is(err, ErrInvalidResumeRequest) {
			t.Fatalf("malformed payload %s error = %v", payload, err)
		}
	}

	cases := []ResumeRequest{
		{
			SuspensionID: suspension.ID,
			Resolutions:  []ToolResolution{{CallID: "unknown", Action: ResolutionDeny}},
		},
		{
			SuspensionID: suspension.ID,
			Resolutions: []ToolResolution{
				{CallID: call.ID, Action: ResolutionDeny},
				{CallID: call.ID, Action: ResolutionDeny},
			},
		},
		{
			SuspensionID: suspension.ID,
			Resolutions:  []ToolResolution{{CallID: call.ID, Action: ResolutionApprove}},
		},
		{
			SuspensionID: suspension.ID,
			Resolutions:  []ToolResolution{{CallID: call.ID, Action: ResolutionInvalid}},
		},
		{SuspensionID: suspension.ID},
	}
	for index, request := range cases {
		if _, err := session.resumeIndeterminate(context.Background(), &suspension, request); err == nil {
			t.Fatalf("invalid resolution set %d was accepted", index)
		}
	}
}

func validIndeterminateRequest(suspension agentic.Suspension, calls ...agentic.ToolUse) ResumeRequest {
	resolutions := make([]ToolResolution, len(calls))
	for index, call := range calls {
		resolutions[index] = ToolResolution{
			CallID: call.ID,
			Action: ResolutionDeny,
		}
	}
	return ResumeRequest{SuspensionID: suspension.ID, Resolutions: resolutions}
}

func TestResumeIndeterminateStateFrontierBudgetIDEncodingAndAppendFailures(t *testing.T) {
	boom := errors.New("boom")
	call := agentic.ToolUse{ID: "call", Name: "tool"}
	session, suspension := indeterminateSession(t, call)
	request := validIndeterminateRequest(suspension, call)
	session.state = Closed
	if _, err := session.resumeIndeterminate(context.Background(), &suspension, request); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("second indeterminate state check = %v", err)
	}

	session, suspension = indeterminateSession(t, call)
	request = validIndeterminateRequest(suspension, call)
	changed := suspension
	changed.ID = "changed"
	if _, err := session.resumeIndeterminate(context.Background(), &changed, request); !errors.Is(err, ErrInvalidResumeRequest) {
		t.Fatalf("changed indeterminate suspension = %v", err)
	}

	session, suspension = indeterminateSession(t, call)
	session.messages = []agentic.Message{
		agentic.NewToolUseMessage(call),
		agentic.NewToolResultMessageFor(call.ID, call.Name, "one", false),
		agentic.NewToolResultMessageFor(call.ID, call.Name, "two", false),
	}
	request = validIndeterminateRequest(suspension, call)
	if _, err := session.resumeIndeterminate(context.Background(), &suspension, request); err == nil {
		t.Fatal("corrupt indeterminate frontier was accepted")
	}

	session, suspension = indeterminateSession(t, call)
	request = validIndeterminateRequest(suspension, call)
	zero := 0
	session.budget = &agentic.UsageLimits{MaxRequests: &zero}
	if _, err := session.resumeIndeterminate(context.Background(), &suspension, request); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("indeterminate budget error = %v", err)
	}

	session, suspension = indeterminateSession(t, call)
	request = validIndeterminateRequest(suspension, call)
	session.ids = idsFunc(func(string) (string, error) { return "", boom })
	if _, err := session.resumeIndeterminate(context.Background(), &suspension, request); !errors.Is(err, boom) {
		t.Fatalf("indeterminate run ID error = %v", err)
	}

	session, suspension = indeterminateSession(t, call)
	request = validIndeterminateRequest(suspension, call)
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	if _, err := session.resumeIndeterminate(context.Background(), &suspension, request); !errors.Is(err, boom) {
		t.Fatalf("indeterminate encoding error = %v", err)
	}

	session, suspension = indeterminateSession(t, call)
	request = validIndeterminateRequest(suspension, call)
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	if _, err := session.resumeIndeterminate(context.Background(), &suspension, request); !errors.Is(err, boom) {
		t.Fatalf("indeterminate append error = %v", err)
	}
}

func TestResumeIndeterminateDenialPromptAndPlannedCompanion(t *testing.T) {
	indeterminate := agentic.ToolUse{ID: "indeterminate", Name: "danger"}
	planned := agentic.ToolUse{ID: "planned", Name: "safe"}
	session, suspension := indeterminateSession(t, indeterminate, planned)
	maxRequests := 10
	session.budget = &agentic.UsageLimits{MaxRequests: &maxRequests}
	// Only the first call is indeterminate. The second exercises the stable
	// planned-abandonment result, which has no operator resolution.
	session.run.started[planned.ID] = false
	suspension.Payload = recoveryPayload(t, repair.PendingCall{
		ID: indeterminate.ID, Name: indeterminate.Name, State: repair.PendingIndeterminate,
	})
	session.suspension = &suspension
	prompt := agentic.NewTextMessage(agentic.RoleUser, "continue")
	execution, err := session.resumeIndeterminate(context.Background(), &suspension, ResumeRequest{
		SuspensionID: suspension.ID,
		Resolutions: []ToolResolution{{
			CallID: indeterminate.ID,
			Action: ResolutionDeny,
		}},
		Prompt: &prompt,
	})
	if err != nil || execution == nil || execution.Status != agentic.ExecutionCompleted ||
		session.State() != Idle {
		t.Fatalf("denial resume = %#v, %v, state=%s", execution, err, session.State())
	}
	last := session.driver.(*countingDriver).Last()
	if len(last.History) != 4 ||
		last.History[1].GetToolResults()[0].Content != "denied by operator after indeterminate execution" ||
		last.History[2].GetToolResults()[0].ToolUseID != planned.ID ||
		last.History[3].GetTextContent() != "continue" {
		t.Fatalf("denial/planned history = %#v", last.History)
	}
}

func TestCloneResumeRequestCopiesMutableInputs(t *testing.T) {
	prompt := agentic.NewTextMessage(agentic.RoleUser, "prompt")
	request := ResumeRequest{
		SuspensionID: "suspension",
		Resolutions: []ToolResolution{{
			CallID: "call",
			Action: ResolutionApprove,
			OverrideArgs: map[string]any{
				"value":  "original",
				"nested": map[string]any{"items": []any{"one"}},
				"words":  []string{"first"},
				"bytes":  []byte{1},
			},
		}},
		Prompt: &prompt,
	}
	cloned := cloneResumeRequest(request)
	cloned.Resolutions[0].OverrideArgs["value"] = "changed"
	cloned.Resolutions[0].OverrideArgs["nested"].(map[string]any)["items"].([]any)[0] = "changed"
	cloned.Resolutions[0].OverrideArgs["words"].([]string)[0] = "changed"
	cloned.Resolutions[0].OverrideArgs["bytes"].([]byte)[0] = 2
	cloned.Prompt.Role = agentic.RoleAssistant
	if request.Resolutions[0].OverrideArgs["value"] != "original" ||
		request.Resolutions[0].OverrideArgs["nested"].(map[string]any)["items"].([]any)[0] != "one" ||
		request.Resolutions[0].OverrideArgs["words"].([]string)[0] != "first" ||
		request.Resolutions[0].OverrideArgs["bytes"].([]byte)[0] != 1 ||
		request.Prompt.Role != agentic.RoleUser || cloneAnyMap(nil) != nil {
		t.Fatalf("resume request clone aliased input: %#v / %#v", request, cloned)
	}
}
