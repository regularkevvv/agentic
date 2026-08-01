package capability

import (
	"context"
	"errors"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

func TestCompileStableOrderingAndValidation(t *testing.T) {
	t.Parallel()
	var registered []string
	capability := func(id string, order Ordering) Capability {
		return Func{Name: id, Order: order, Apply: func(*Registry) error {
			registered = append(registered, id)
			return nil
		}}
	}
	plan, err := Compile(
		capability("second", Ordering{After: []string{"first"}}),
		capability("independent", Ordering{}),
		capability("first", Ordering{}),
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"independent", "first", "second"}
	if !reflect.DeepEqual(plan.IDs(), want) || !reflect.DeepEqual(registered, want) {
		t.Fatalf("ids=%v registered=%v want=%v", plan.IDs(), registered, want)
	}
	ids := plan.IDs()
	ids[0] = "changed"
	if reflect.DeepEqual(plan.IDs(), ids) {
		t.Fatal("Plan.IDs exposed mutable storage")
	}

	for name, capabilities := range map[string][]Capability{
		"duplicate": {
			Func{Name: "same"}, Func{Name: "same"},
		},
		"missing": {
			Func{Name: "one", Order: Ordering{After: []string{"absent"}}},
		},
		"cycle": {
			Func{Name: "one", Order: Ordering{After: []string{"two"}}},
			Func{Name: "two", Order: Ordering{After: []string{"one"}}},
		},
		"empty": {
			Func{},
		},
		"surrounding whitespace": {
			Func{Name: " invalid "},
		},
	} {
		name, capabilities := name, capabilities
		t.Run(name, func(t *testing.T) {
			if _, err := Compile(capabilities...); err == nil {
				t.Fatal("Compile succeeded")
			}
		})
	}
}

func TestRegistryFreezesToolsAndRejectsDuplicateContributions(t *testing.T) {
	t.Parallel()
	tool, handler := agentic.MustToolWithContext(
		"echo",
		"echo input",
		func(_ context.Context, input struct {
			Value string `json:"value"`
		}) (string, error) {
			return input.Value, nil
		},
	)
	toolset := agentic.NewToolset().Add(tool, handler)
	var retained *Registry
	plan, err := Compile(Func{Name: "tools", Apply: func(registry *Registry) error {
		retained = registry
		return registry.AddToolset(toolset)
	}})
	if err != nil {
		t.Fatal(err)
	}
	tool.Function.Description = "mutated"
	toolset.Add(tool, handler)
	got := plan.Tools()
	if len(got) != 1 || got[0].Function.Description != "echo input" {
		t.Fatalf("frozen tools = %#v", got)
	}
	if err := retained.AddToolset(toolset); !errors.Is(err, ErrRegistryFrozen) {
		t.Fatalf("post-build mutation error = %v", err)
	}

	if _, err := Compile(Func{Name: "duplicates", Apply: func(registry *Registry) error {
		if err := registry.AddToolset(agentic.NewToolset().Add(tool, handler)); err != nil {
			return err
		}
		return registry.AddToolset(agentic.NewToolset().Add(tool, handler))
	}}); !errors.Is(err, ErrDuplicateTool) {
		t.Fatalf("duplicate tool error = %v", err)
	}
	resolver := EffectResolverFunc(func(context.Context, agentic.ToolUse, env.Environment) (Effect, error) {
		return Effect{}, nil
	})
	if _, err := Compile(Func{Name: "effects", Apply: func(registry *Registry) error {
		if err := registry.AddEffectResolver("echo", resolver); err != nil {
			return err
		}
		return registry.AddEffectResolver("echo", resolver)
	}}); !errors.Is(err, ErrDuplicateEffect) {
		t.Fatalf("duplicate effect error = %v", err)
	}
}

func TestGuardedGateMiddlewareNarrowsInOrder(t *testing.T) {
	t.Parallel()
	calls := []agentic.ToolUse{{ID: "one", Name: "a"}, {ID: "two", Name: "b"}}
	var order []string
	first := ToolGateMiddlewareFunc(func(
		_ context.Context,
		_ []agentic.ToolUse,
		current agentic.ToolBatchDecision,
	) (agentic.ToolBatchDecision, error) {
		order = append(order, "first")
		result := agentic.ToolExecutionResult{ToolUseID: "one", ToolName: "a", Content: "denied", IsError: true, Error: errors.New("denied")}
		current.Calls[0] = agentic.ToolDisposition{Kind: agentic.ToolDispositionReturn, Result: &result, Continue: true}
		return current, nil
	})
	second := ToolGateMiddlewareFunc(func(
		_ context.Context,
		_ []agentic.ToolUse,
		current agentic.ToolBatchDecision,
	) (agentic.ToolBatchDecision, error) {
		order = append(order, "second")
		return current, nil
	})
	plan, err := Compile(
		Func{Name: "first", Apply: func(registry *Registry) error { return registry.AddToolGateMiddleware(first) }},
		Func{Name: "second", Order: Ordering{After: []string{"first"}}, Apply: func(registry *Registry) error {
			return registry.AddToolGateMiddleware(second)
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	decision, err := plan.ToolGate().EvaluateBatch(context.Background(), calls)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []string{"first", "second"}) ||
		decision.Calls[0].Kind != agentic.ToolDispositionReturn ||
		decision.Calls[1].Kind != agentic.ToolDispositionExecute {
		t.Fatalf("order=%v decision=%#v", order, decision)
	}

	broadening := ToolGateMiddlewareFunc(func(
		_ context.Context,
		_ []agentic.ToolUse,
		current agentic.ToolBatchDecision,
	) (agentic.ToolBatchDecision, error) {
		current.Calls[0] = agentic.ToolDisposition{Kind: agentic.ToolDispositionExecute}
		return current, nil
	})
	plan, err = Compile(
		Func{Name: "first", Apply: func(registry *Registry) error { return registry.AddToolGateMiddleware(first) }},
		Func{Name: "bad", Apply: func(registry *Registry) error { return registry.AddToolGateMiddleware(broadening) }},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := plan.ToolGate().EvaluateBatch(context.Background(), calls); !errors.Is(err, ErrGateBroadened) {
		t.Fatalf("broadening error = %v", err)
	}

	mutatePriorResult := ToolGateMiddlewareFunc(func(
		_ context.Context,
		_ []agentic.ToolUse,
		current agentic.ToolBatchDecision,
	) (agentic.ToolBatchDecision, error) {
		current.Calls[0].Result.Content.(map[string]any)["value"] = "changed"
		return current, nil
	})
	firstWithMutableResult := ToolGateMiddlewareFunc(func(
		_ context.Context,
		_ []agentic.ToolUse,
		current agentic.ToolBatchDecision,
	) (agentic.ToolBatchDecision, error) {
		result := agentic.ToolExecutionResult{
			ToolUseID: "one",
			ToolName:  "a",
			Content:   map[string]any{"value": "original"},
		}
		current.Calls[0] = agentic.ToolDisposition{Kind: agentic.ToolDispositionReturn, Result: &result}
		return current, nil
	})
	plan, err = Compile(
		Func{Name: "first", Apply: func(registry *Registry) error {
			return registry.AddToolGateMiddleware(firstWithMutableResult)
		}},
		Func{Name: "mutator", Apply: func(registry *Registry) error {
			return registry.AddToolGateMiddleware(mutatePriorResult)
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := plan.ToolGate().EvaluateBatch(context.Background(), calls); !errors.Is(err, ErrGateBroadened) {
		t.Fatalf("mutable-result broadening error = %v", err)
	}
}

func TestPlanCarriesOrderedContextEventAndLifecycleHooks(t *testing.T) {
	t.Parallel()
	transform := contextpolicy.TransformFunc(func(_ context.Context, value *contextpolicy.TransformContext) error {
		*value.Ephemeral = append(*value.Ephemeral, agentic.NewTextMessage(agentic.RoleUser, "tail"))
		return nil
	})
	middleware := event.MiddlewareHandler(func(ctx context.Context, value agentic.Event, next agentic.EventSink) error {
		return next.Emit(ctx, value)
	})
	hook := harnessruntime.LifecycleHookFunc(func(context.Context, harnessruntime.LifecycleEvent) error { return nil })
	plan, err := Compile(Func{Name: "all", Apply: func(registry *Registry) error {
		if err := registry.AddContextTransform(transform); err != nil {
			return err
		}
		if err := registry.AddEventMiddleware(middleware); err != nil {
			return err
		}
		return registry.AddLifecycleHook(hook)
	}})
	if err != nil {
		t.Fatal(err)
	}
	projection, err := plan.ContextPolicy().Project(context.Background(), contextpolicy.ProjectionRequest{
		Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "base")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(projection.Messages) != 2 || projection.Messages[1].GetTextContent() != "tail" ||
		len(plan.EventMiddleware()) != 1 || len(plan.LifecycleHooks()) != 1 {
		t.Fatalf("projection=%#v middleware=%d hooks=%d", projection, len(plan.EventMiddleware()), len(plan.LifecycleHooks()))
	}
}

type testToolset struct {
	tools    []agentic.Tool
	handlers []agentic.ToolHandler
}

func (s testToolset) ToolsAndHandlers() ([]agentic.Tool, []agentic.ToolHandler) {
	return s.tools, s.handlers
}

func TestRegistryContributionValidationAndSnapshots(t *testing.T) {
	t.Parallel()
	if _, err := Compile(nil); err == nil {
		t.Fatal("nil capability compiled")
	}
	if _, err := Compile(Func{Name: "missing-before", Order: Ordering{Before: []string{"absent"}}}); !errors.Is(err, ErrMissingOrdering) {
		t.Fatalf("missing before = %v", err)
	}
	if _, err := Compile(Func{Name: "register-error", Apply: func(*Registry) error {
		return errors.New("register")
	}}); err == nil {
		t.Fatal("registration error was ignored")
	}

	tool := agentic.Tool{Function: agentic.Function{Name: "tool"}}
	cases := []struct {
		name string
		fn   func(*Registry) error
	}{
		{"nil toolset", func(r *Registry) error { return r.AddToolset(nil) }},
		{"mismatched toolset", func(r *Registry) error {
			return r.AddToolset(testToolset{tools: []agentic.Tool{tool}})
		}},
		{"uncloneable tool", func(r *Registry) error {
			bad := tool
			bad.Function.Parameters = map[string]any{"bad": make(chan int)}
			return r.AddToolset(testToolset{tools: []agentic.Tool{bad}, handlers: []agentic.ToolHandler{nil}})
		}},
		{"empty tool name", func(r *Registry) error {
			return r.AddToolset(testToolset{
				tools:    []agentic.Tool{{}},
				handlers: []agentic.ToolHandler{nil},
			})
		}},
		{"empty effect name", func(r *Registry) error {
			return r.AddEffectResolver("", EffectResolverFunc(func(context.Context, agentic.ToolUse, env.Environment) (Effect, error) {
				return Effect{}, nil
			}))
		}},
		{"nil effect", func(r *Registry) error { return r.AddEffectResolver("tool", nil) }},
		{"nil gate", func(r *Registry) error { return r.AddToolGateMiddleware(nil) }},
		{"nil context", func(r *Registry) error { return r.AddContextTransform(nil) }},
		{"nil event", func(r *Registry) error { return r.AddEventMiddleware(nil) }},
		{"nil lifecycle", func(r *Registry) error { return r.AddLifecycleHook(nil) }},
	}
	for _, test := range cases {
		test := test
		t.Run(test.name, func(t *testing.T) {
			if _, err := Compile(Func{Name: test.name, Apply: test.fn}); err == nil {
				t.Fatal("invalid contribution compiled")
			}
		})
	}

	var registry *Registry
	plan, err := Compile(Func{Name: "snapshots", Apply: func(r *Registry) error {
		registry = r
		resolver := EffectResolverFunc(func(context.Context, agentic.ToolUse, env.Environment) (Effect, error) {
			return Effect{Capability: "test"}, nil
		})
		if err := r.AddEffectResolver("tool", resolver); err != nil {
			return err
		}
		if _, ok := r.EffectResolver("tool"); !ok {
			return errors.New("resolver unavailable during registration")
		}
		if err := r.AddToolset(testToolset{
			tools:    []agentic.Tool{tool},
			handlers: []agentic.ToolHandler{nil},
		}); err != nil {
			return err
		}
		return r.ConfigureContext(contextpolicy.Config{}, nil)
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Toolsets()) != 1 || len(plan.Tools()) != 1 {
		t.Fatalf("plan snapshots = %#v %#v", plan.Toolsets(), plan.Tools())
	}
	sets := plan.Toolsets()
	sets[0] = nil
	if plan.Toolsets()[0] == nil {
		t.Fatal("toolset slice was mutable")
	}
	tools := plan.Tools()
	tools[0].Function.Name = "mutated"
	if plan.Tools()[0].Function.Name != "tool" {
		t.Fatal("tool schema was mutable")
	}
	if _, ok := registry.EffectResolver("absent"); ok {
		t.Fatal("unknown resolver found")
	}
	if err := registry.ConfigureContext(contextpolicy.Config{}, nil); !errors.Is(err, ErrRegistryFrozen) {
		t.Fatalf("frozen context error = %v", err)
	}
	if _, err := Compile(Func{Name: "duplicate-context", Apply: func(r *Registry) error {
		if err := r.ConfigureContext(contextpolicy.Config{}, nil); err != nil {
			return err
		}
		return r.ConfigureContext(contextpolicy.Config{}, nil)
	}}); !errors.Is(err, ErrContextConfigured) {
		t.Fatalf("duplicate context = %v", err)
	}
}

func TestGuardedGateRejectsInvalidMiddlewareResults(t *testing.T) {
	t.Parallel()
	calls := []agentic.ToolUse{{ID: "one", Name: "tool"}}
	tests := []struct {
		name string
		next func(agentic.ToolBatchDecision) agentic.ToolBatchDecision
	}{
		{"cardinality", func(agentic.ToolBatchDecision) agentic.ToolBatchDecision {
			return agentic.ToolBatchDecision{}
		}},
		{"execute result", func(current agentic.ToolBatchDecision) agentic.ToolBatchDecision {
			current.Calls[0].Result = &agentic.ToolExecutionResult{}
			return current
		}},
		{"return no result", func(current agentic.ToolBatchDecision) agentic.ToolBatchDecision {
			current.Calls[0].Kind = agentic.ToolDispositionReturn
			return current
		}},
		{"return wrong identity", func(current agentic.ToolBatchDecision) agentic.ToolBatchDecision {
			result := agentic.ToolExecutionResult{ToolUseID: "wrong", ToolName: "tool"}
			current.Calls[0] = agentic.ToolDisposition{Kind: agentic.ToolDispositionReturn, Result: &result}
			return current
		}},
		{"suspend result", func(current agentic.ToolBatchDecision) agentic.ToolBatchDecision {
			result := agentic.ToolExecutionResult{}
			current.Calls[0] = agentic.ToolDisposition{Kind: agentic.ToolDispositionSuspend, Result: &result}
			return current
		}},
		{"suspend continue", func(current agentic.ToolBatchDecision) agentic.ToolBatchDecision {
			current.Calls[0] = agentic.ToolDisposition{Kind: agentic.ToolDispositionSuspend, Continue: true}
			return current
		}},
		{"suspend no deferral", func(current agentic.ToolBatchDecision) agentic.ToolBatchDecision {
			current.Calls[0] = agentic.ToolDisposition{Kind: agentic.ToolDispositionSuspend}
			return current
		}},
		{"invalid kind", func(current agentic.ToolBatchDecision) agentic.ToolBatchDecision {
			current.Calls[0].Kind = agentic.ToolDispositionInvalid
			return current
		}},
		{"deferral no suspension", func(current agentic.ToolBatchDecision) agentic.ToolBatchDecision {
			current.Deferral = &agentic.ToolDeferral{Kind: "test"}
			return current
		}},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			middleware := ToolGateMiddlewareFunc(func(
				context.Context,
				[]agentic.ToolUse,
				agentic.ToolBatchDecision,
			) (agentic.ToolBatchDecision, error) {
				return test.next(allowAll(len(calls))), nil
			})
			plan, err := Compile(Func{Name: "invalid", Apply: func(r *Registry) error {
				return r.AddToolGateMiddleware(middleware)
			}})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := plan.ToolGate().EvaluateBatch(context.Background(), calls); err == nil {
				t.Fatal("invalid gate result was accepted")
			}
		})
	}

	wantErr := errors.New("gate failed")
	plan, err := Compile(Func{Name: "error", Apply: func(r *Registry) error {
		return r.AddToolGateMiddleware(ToolGateMiddlewareFunc(func(
			context.Context,
			[]agentic.ToolUse,
			agentic.ToolBatchDecision,
		) (agentic.ToolBatchDecision, error) {
			return agentic.ToolBatchDecision{}, wantErr
		}))
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := plan.ToolGate().EvaluateBatch(context.Background(), calls); !errors.Is(err, wantErr) {
		t.Fatalf("gate error = %v", err)
	}

	called := false
	plan, err = Compile(
		Func{Name: "suspend", Apply: func(r *Registry) error {
			return r.AddToolGateMiddleware(ToolGateMiddlewareFunc(func(
				_ context.Context,
				_ []agentic.ToolUse,
				current agentic.ToolBatchDecision,
			) (agentic.ToolBatchDecision, error) {
				current.Calls[0].Kind = agentic.ToolDispositionSuspend
				current.Deferral = &agentic.ToolDeferral{Kind: "pause"}
				return current, nil
			}))
		}},
		Func{Name: "after", Apply: func(r *Registry) error {
			return r.AddToolGateMiddleware(ToolGateMiddlewareFunc(func(
				context.Context,
				[]agentic.ToolUse,
				agentic.ToolBatchDecision,
			) (agentic.ToolBatchDecision, error) {
				called = true
				return agentic.ToolBatchDecision{}, nil
			}))
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	decision, err := plan.ToolGate().EvaluateBatch(context.Background(), calls)
	if err != nil || decision.Deferral == nil || called {
		t.Fatalf("suspension decision=%#v called=%t err=%v", decision, called, err)
	}
}
