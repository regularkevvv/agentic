package permission

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

func TestPatternMatchingRejectsShortAndExtraValues(t *testing.T) {
	t.Parallel()
	if matchPattern([]string{"one", "two"}, []string{"one"}) {
		t.Fatal("pattern matched a shorter value")
	}
	if matchPattern([]string{"one"}, []string{"one", "two"}) {
		t.Fatal("exact pattern matched an extra value segment")
	}
}

func TestPolicyUsesMostSpecificRuleAndDefaultDeny(t *testing.T) {
	t.Parallel()
	policy, err := New(DecisionDeny,
		Rule{Pattern: "filesystem/**", Decision: DecisionAsk},
		Rule{Pattern: "filesystem/read/**", Decision: DecisionAllow},
		Rule{Pattern: "filesystem/read/file/private/**", Decision: DecisionDeny},
		Rule{Pattern: "filesystem/read/file/private/safe/**", Decision: DecisionAllow},
	)
	if err != nil {
		t.Fatal(err)
	}
	request := func(action, id string) PermissionRequest {
		return PermissionRequest{
			Capability:        "filesystem",
			Action:            action,
			CanonicalResource: env.CanonicalResource{Scheme: "file", ID: id},
		}
	}
	if got := policy.Evaluate(request("read", "/public/file")); got != DecisionAllow {
		t.Fatalf("public read = %v", got)
	}
	if got := policy.Evaluate(request("write", "/public/file")); got != DecisionAsk {
		t.Fatalf("write = %v", got)
	}
	if got := policy.Evaluate(request("read", "/private/secret")); got != DecisionDeny {
		t.Fatalf("private read = %v", got)
	}
	if got := policy.Evaluate(request("read", "/private/safe/file")); got != DecisionAllow {
		t.Fatalf("safe read = %v", got)
	}
	if got := policy.Evaluate(PermissionRequest{
		Capability:        "tool",
		Action:            "unknown",
		CanonicalResource: env.CanonicalResource{Scheme: "tool", ID: "unknown"},
	}); got != DecisionDeny {
		t.Fatalf("unknown tool = %v", got)
	}
	if _, err := New(DecisionInvalid); err == nil {
		t.Fatal("invalid fallback succeeded")
	}
	if _, err := New(DecisionDeny, Rule{Pattern: "bad/**/tail", Decision: DecisionAllow}); err == nil {
		t.Fatal("invalid pattern succeeded")
	}
}

func TestPermissionCapabilityDeniesWholeMixedBatch(t *testing.T) {
	t.Parallel()
	policy, err := New(DecisionDeny,
		Rule{Pattern: "filesystem/read/**", Decision: DecisionAllow},
	)
	if err != nil {
		t.Fatal(err)
	}
	plan := permissionPlan(t, policy, false)
	calls := []agentic.ToolUse{
		{ID: "read-1", Name: "read_file", Input: map[string]any{"path": "/allowed"}},
		{ID: "app-1", Name: "application_tool"},
	}
	decision, err := plan.ToolGate().EvaluateBatch(toolContext(t), calls)
	if err != nil {
		t.Fatal(err)
	}
	if len(decision.Calls) != 2 {
		t.Fatalf("decision = %#v", decision)
	}
	for _, disposition := range decision.Calls {
		if disposition.Kind != agentic.ToolDispositionReturn || disposition.Result == nil ||
			!disposition.Result.IsError || !disposition.Continue {
			t.Fatalf("disposition = %#v", disposition)
		}
	}
	if !reflect.DeepEqual(
		[]string{decision.Calls[0].Result.ToolUseID, decision.Calls[1].Result.ToolUseID},
		[]string{"read-1", "app-1"},
	) {
		t.Fatalf("result order = %#v", decision.Calls)
	}
}

func TestPermissionCapabilitySuspendsAskedBatchWithPreAllowedCalls(t *testing.T) {
	t.Parallel()
	policy, err := New(DecisionDeny,
		Rule{Pattern: "filesystem/read/**", Decision: DecisionAllow},
		Rule{Pattern: "shell/**", Decision: DecisionAsk},
	)
	if err != nil {
		t.Fatal(err)
	}
	plan := permissionPlan(t, policy, true)
	calls := []agentic.ToolUse{
		{ID: "read-1", Name: "read_file", Input: map[string]any{"path": "/allowed"}},
		{ID: "shell-1", Name: "run_command", Input: map[string]any{"name": "echo"}},
	}
	decision, err := plan.ToolGate().EvaluateBatch(toolContext(t), calls)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Deferral == nil || decision.Deferral.Kind != harnessruntime.PermissionDeferralKind {
		t.Fatalf("deferral = %#v", decision.Deferral)
	}
	for _, disposition := range decision.Calls {
		if disposition.Kind != agentic.ToolDispositionSuspend {
			t.Fatalf("disposition = %#v", disposition)
		}
	}
	var payload struct {
		harnessruntime.DeferredBatch
		Requests []deferredRequest `json:"requests"`
	}
	if err := json.Unmarshal(decision.Deferral.Payload, &payload); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(payload.RequiredResolutionIDs, []string{"shell-1"}) ||
		!reflect.DeepEqual(payload.PreAllowedCallIDs, []string{"read-1"}) ||
		len(payload.Requests) != 1 || payload.Requests[0].Request.Capability != "shell" {
		t.Fatalf("payload = %#v", payload)
	}
}

func TestPermissionPresets(t *testing.T) {
	t.Parallel()
	file := func(action string) PermissionRequest {
		return PermissionRequest{
			Capability:        "filesystem",
			Action:            action,
			CanonicalResource: env.CanonicalResource{Scheme: "file", ID: "/workspace/file"},
		}
	}
	shell := PermissionRequest{
		Capability:        "shell",
		Action:            "exec",
		CanonicalResource: env.CanonicalResource{Scheme: "command", ID: "echo"},
	}
	if ReadOnly().Evaluate(file("read")) != DecisionAllow ||
		ReadOnly().Evaluate(file("write")) != DecisionDeny ||
		ReadOnly().Evaluate(shell) != DecisionDeny {
		t.Fatal("ReadOnly preset has unexpected decisions")
	}
	if WorkspaceWrite().Evaluate(file("write")) != DecisionAllow ||
		WorkspaceWrite().Evaluate(shell) != DecisionAsk {
		t.Fatal("WorkspaceWrite preset has unexpected decisions")
	}
}

func permissionPlan(t *testing.T, policy *Policy, shell bool) capability.Plan {
	t.Helper()
	permissionCapability, err := NewCapability(policy)
	if err != nil {
		t.Fatal(err)
	}
	effects := capability.Func{Name: "effects", Apply: func(registry *capability.Registry) error {
		if err := registry.AddEffectResolver("read_file", capability.EffectResolverFunc(func(
			ctx context.Context,
			call agentic.ToolUse,
			environment env.Environment,
		) (capability.Effect, error) {
			path, _ := call.Input["path"].(string)
			resource, err := environment.Files().CanonicalPath(ctx, path)
			return capability.Effect{Capability: "filesystem", Action: "read", Resource: resource}, err
		})); err != nil {
			return err
		}
		if !shell {
			return nil
		}
		return registry.AddEffectResolver("run_command", capability.EffectResolverFunc(func(
			_ context.Context,
			call agentic.ToolUse,
			_ env.Environment,
		) (capability.Effect, error) {
			name, _ := call.Input["name"].(string)
			return capability.Effect{
				Capability: "shell",
				Action:     "exec",
				Resource:   env.CanonicalResource{Scheme: "command", ID: name, Display: name},
			}, nil
		}))
	}}
	plan, err := capability.Compile(effects, permissionCapability)
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func toolContext(t *testing.T) context.Context {
	t.Helper()
	environment, err := envmemory.New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := environment.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})
	return harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: environment,
		SessionID:   "session",
	})
}

func TestPermissionGateRequiresRuntime(t *testing.T) {
	t.Parallel()
	plan := permissionPlan(t, WorkspaceWrite(), false)
	if _, err := plan.ToolGate().EvaluateBatch(context.Background(), []agentic.ToolUse{{ID: "x", Name: "read_file"}}); err == nil {
		t.Fatal("gate without runtime succeeded")
	}
}

func TestCapabilityOptionsValidate(t *testing.T) {
	t.Parallel()
	if _, err := NewCapability(nil); err == nil {
		t.Fatal("nil policy succeeded")
	}
	if _, err := NewCapability(WorkspaceWrite(), nil); err == nil {
		t.Fatal("nil option succeeded")
	}
	if _, err := NewCapability(WorkspaceWrite(), WithID("")); err == nil {
		t.Fatal("empty ID succeeded")
	}
	want := errors.New("registration")
	bad := capability.Func{Name: "bad", Apply: func(*capability.Registry) error { return want }}
	if _, err := capability.Compile(bad); !errors.Is(err, want) {
		t.Fatalf("registration error = %v", err)
	}
}

func TestPolicyPatternValidationTiesAndNilDefault(t *testing.T) {
	t.Parallel()
	if (*Policy)(nil).Evaluate(PermissionRequest{}) != DecisionDeny {
		t.Fatal("nil policy did not deny")
	}
	invalid := []Rule{
		{Pattern: "", Decision: DecisionAllow},
		{Pattern: "filesystem//read", Decision: DecisionAllow},
		{Pattern: "filesystem/[", Decision: DecisionAllow},
		{Pattern: "filesystem/**/read", Decision: DecisionAllow},
		{Pattern: "filesystem/read", Decision: DecisionInvalid},
	}
	for index, rule := range invalid {
		if _, err := New(DecisionDeny, rule); err == nil {
			t.Fatalf("invalid rule %d succeeded", index)
		}
	}
	policy, err := New(DecisionDeny,
		Rule{Pattern: "tool/name/**", Decision: DecisionAsk},
		Rule{Pattern: "tool/name/**", Decision: DecisionAllow},
		Rule{Pattern: "too/long/pattern", Decision: DecisionAsk},
		Rule{Pattern: "tool/oth?r/**", Decision: DecisionAsk},
	)
	if err != nil {
		t.Fatal(err)
	}
	request := PermissionRequest{
		Capability:        "tool",
		Action:            "name",
		CanonicalResource: env.CanonicalResource{Scheme: "tool", ID: `path\to\value`},
	}
	if decision := policy.Evaluate(request); decision != DecisionAllow {
		t.Fatalf("later specificity tie = %v", decision)
	}
	if decision := policy.Evaluate(PermissionRequest{Capability: "tool", Action: "other"}); decision != DecisionAsk {
		t.Fatalf("single-segment wildcard = %v", decision)
	}
}

func TestPermissionGateAllowResolutionFailuresAndPriorDisposition(t *testing.T) {
	t.Parallel()
	allow, err := New(DecisionAllow)
	if err != nil {
		t.Fatal(err)
	}
	plan := permissionPlan(t, allow, false)
	decision, err := plan.ToolGate().EvaluateBatch(toolContext(t), []agentic.ToolUse{{
		ID: "unknown", Name: "unknown",
	}})
	if err != nil || decision.Calls[0].Kind != agentic.ToolDispositionExecute {
		t.Fatalf("allow decision = %#v, %v", decision, err)
	}

	resolutionErr := errors.New("canonicalization")
	effects := capability.Func{Name: "effects", Apply: func(registry *capability.Registry) error {
		return registry.AddEffectResolver("broken", capability.EffectResolverFunc(func(
			context.Context,
			agentic.ToolUse,
			env.Environment,
		) (capability.Effect, error) {
			return capability.Effect{}, resolutionErr
		}))
	}}
	permissions, _ := NewCapability(allow, WithID("custom"), WithOrdering(capability.Ordering{After: []string{"effects"}}))
	plan, err = capability.Compile(effects, permissions)
	if err != nil {
		t.Fatal(err)
	}
	decision, err = plan.ToolGate().EvaluateBatch(toolContext(t), []agentic.ToolUse{{ID: "broken", Name: "broken"}})
	if err != nil || decision.Calls[0].Result == nil ||
		!strings.Contains(agentic.FormatToolResult(decision.Calls[0].Result.Content), resolutionErr.Error()) {
		t.Fatalf("resolution denial = %#v, %v", decision, err)
	}

	incomplete := capability.Func{Name: "incomplete", Apply: func(registry *capability.Registry) error {
		return registry.AddEffectResolver("incomplete", capability.EffectResolverFunc(func(
			context.Context,
			agentic.ToolUse,
			env.Environment,
		) (capability.Effect, error) {
			return capability.Effect{}, nil
		}))
	}}
	permissions, _ = NewCapability(allow, WithOrdering(capability.Ordering{After: []string{"incomplete"}}))
	plan, err = capability.Compile(incomplete, permissions)
	if err != nil {
		t.Fatal(err)
	}
	decision, err = plan.ToolGate().EvaluateBatch(toolContext(t), []agentic.ToolUse{{ID: "bad", Name: "incomplete"}})
	if err != nil || decision.Calls[0].Kind != agentic.ToolDispositionReturn {
		t.Fatalf("incomplete effect = %#v, %v", decision, err)
	}

	prior := capability.Func{Name: "prior", Apply: func(registry *capability.Registry) error {
		return registry.AddToolGateMiddleware(capability.ToolGateMiddlewareFunc(func(
			_ context.Context,
			calls []agentic.ToolUse,
			current agentic.ToolBatchDecision,
		) (agentic.ToolBatchDecision, error) {
			result := agentic.ToolExecutionResult{
				ToolUseID: calls[0].ID,
				ToolName:  calls[0].Name,
				Content:   "already returned",
			}
			current.Calls[0] = agentic.ToolDisposition{Kind: agentic.ToolDispositionReturn, Result: &result}
			return current, nil
		}))
	}}
	permissions, _ = NewCapability(WorkspaceWrite(), WithOrdering(capability.Ordering{After: []string{"prior"}}))
	plan, err = capability.Compile(prior, permissions)
	if err != nil {
		t.Fatal(err)
	}
	decision, err = plan.ToolGate().EvaluateBatch(toolContext(t), []agentic.ToolUse{{ID: "prior", Name: "unknown"}})
	if err != nil || decision.Calls[0].Result.Content != "already returned" {
		t.Fatalf("prior disposition = %#v, %v", decision, err)
	}

	optionErr := errors.New("option")
	if _, err := NewCapability(allow, Option(func(*Capability) error { return optionErr })); !errors.Is(err, optionErr) {
		t.Fatalf("option error = %v", err)
	}
}
