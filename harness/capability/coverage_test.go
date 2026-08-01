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

func TestStableOrderBeforeConstraint(t *testing.T) {
	ordered, err := stableOrder([]Capability{
		Func{Name: "first", Order: Ordering{Before: []string{"second"}}},
		Func{Name: "second"},
	})
	if err != nil || len(ordered) != 2 || ordered[0].ID() != "first" || ordered[1].ID() != "second" {
		t.Fatalf("before ordering = %#v, %v", ordered, err)
	}
}

func TestSmallAdaptersFrozenRegistryAndCloneBranches(t *testing.T) {
	registry := newRegistry()
	if err := (Func{Name: "noop"}).Register(registry); err != nil {
		t.Fatal(err)
	}
	resolverCalled := false
	resolver := EffectResolverFunc(func(
		context.Context,
		agentic.ToolUse,
		env.Environment,
	) (Effect, error) {
		resolverCalled = true
		return Effect{Capability: "test"}, nil
	})
	if effect, err := resolver.ResolveEffect(context.Background(), agentic.ToolUse{}, nil); err != nil ||
		effect.Capability != "test" || !resolverCalled {
		t.Fatalf("resolver = %#v, %v, called=%v", effect, err, resolverCalled)
	}

	registry.frozen = true
	for name, call := range map[string]func() error{
		"effect": func() error { return registry.AddEffectResolver("tool", resolver) },
		"gate": func() error {
			return registry.AddToolGateMiddleware(ToolGateMiddlewareFunc(func(
				context.Context,
				[]agentic.ToolUse,
				agentic.ToolBatchDecision,
			) (agentic.ToolBatchDecision, error) {
				return agentic.ToolBatchDecision{}, nil
			}))
		},
		"context": func() error {
			return registry.AddContextTransform(contextpolicy.TransformFunc(func(
				context.Context,
				*contextpolicy.TransformContext,
			) error {
				return nil
			}))
		},
		"event": func() error {
			return registry.AddEventMiddleware(event.MiddlewareFunc(func(next agentic.EventSink) agentic.EventSink {
				return next
			}))
		},
		"lifecycle": func() error {
			return registry.AddLifecycleHook(harnessruntime.LifecycleHookFunc(func(
				context.Context,
				harnessruntime.LifecycleEvent,
			) error {
				return nil
			}))
		},
	} {
		if err := call(); !errors.Is(err, ErrRegistryFrozen) {
			t.Fatalf("%s frozen error = %v", name, err)
		}
	}

	if _, err := Compile(Func{Name: "bad-context", Apply: func(registry *Registry) error {
		return registry.ConfigureContext(contextpolicy.Config{TriggerPercent: 101}, nil)
	}}); !errors.Is(err, contextpolicy.ErrInvalidConfig) {
		t.Fatalf("invalid context compile = %v", err)
	}

	edges := []map[int]bool{{}, {}}
	indegree := []int{0, 0}
	addEdge(edges, indegree, 0, 1)
	addEdge(edges, indegree, 0, 1)
	if indegree[1] != 1 {
		t.Fatalf("duplicate edge changed indegree: %#v", indegree)
	}
	frozen := &frozenToolset{
		tools:    []agentic.Tool{{}},
		handlers: []agentic.ToolHandler{nil},
	}
	tools, handlers := frozen.ToolsAndHandlers()
	if len(tools) != 1 || len(handlers) != 1 {
		t.Fatalf("frozen toolset = %#v / %#v", tools, handlers)
	}
	if tools, err := cloneTools(nil); err != nil || tools != nil {
		t.Fatalf("empty tool clone = %#v, %v", tools, err)
	}
	if cloneCalls(nil) != nil {
		t.Fatal("empty call clone was non-nil")
	}
	original := map[string]any{
		"any":     []any{map[string]any{"value": "original"}},
		"strings": []string{"original"},
		"bytes":   []byte("original"),
	}
	cloned := cloneMutableValue(original).(map[string]any)
	cloned["any"].([]any)[0].(map[string]any)["value"] = "changed"
	cloned["strings"].([]string)[0] = "changed"
	cloned["bytes"].([]byte)[0] = 'x'
	if !reflect.DeepEqual(original, map[string]any{
		"any":     []any{map[string]any{"value": "original"}},
		"strings": []string{"original"},
		"bytes":   []byte("original"),
	}) {
		t.Fatalf("mutable clone aliased original: %#v", original)
	}
}

func TestDelegationToolClassificationIsValidatedFrozenAndCopied(t *testing.T) {
	toolA, handlerA := agentic.MustToolWithContext(
		"alpha",
		"alpha",
		func(context.Context, struct{}) (string, error) { return "alpha", nil },
	)
	toolB, handlerB := agentic.MustToolWithContext(
		"beta",
		"beta",
		func(context.Context, struct{}) (string, error) { return "beta", nil },
	)
	registry := newRegistry()
	if err := registry.MarkDelegationTool("missing"); !errors.Is(err, ErrUnknownTool) {
		t.Fatalf("unknown delegation tool = %v", err)
	}
	if err := registry.AddToolset(agentic.NewToolset().
		Add(toolB, handlerB).
		Add(toolA, handlerA)); err != nil {
		t.Fatal(err)
	}
	if err := registry.MarkDelegationTool("beta"); err != nil {
		t.Fatal(err)
	}
	if err := registry.MarkDelegationTool("alpha"); err != nil {
		t.Fatal(err)
	}
	if err := registry.MarkDelegationTool("alpha"); err != nil {
		t.Fatalf("idempotent delegation classification = %v", err)
	}
	registry.frozen = true
	if err := registry.MarkDelegationTool("alpha"); !errors.Is(err, ErrRegistryFrozen) {
		t.Fatalf("frozen delegation classification = %v", err)
	}

	plan, err := Compile(Func{Name: "delegation", Apply: func(registry *Registry) error {
		if err := registry.AddToolset(agentic.NewToolset().
			Add(toolB, handlerB).
			Add(toolA, handlerA)); err != nil {
			return err
		}
		if err := registry.MarkDelegationTool("beta"); err != nil {
			return err
		}
		return registry.MarkDelegationTool("alpha")
	}})
	if err != nil {
		t.Fatal(err)
	}
	names := plan.DelegationTools()
	if !reflect.DeepEqual(names, []string{"alpha", "beta"}) {
		t.Fatalf("delegation names = %#v", names)
	}
	names[0] = "mutated"
	if plan.DelegationTools()[0] != "alpha" {
		t.Fatal("delegation names exposed mutable plan storage")
	}
}
