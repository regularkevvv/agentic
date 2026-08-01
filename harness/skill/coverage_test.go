package skill

import (
	"context"
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

func skillHandlers(t *testing.T, source Source, resolver ScopeResolver, limits Limits) map[string]agentic.ToolHandler {
	t.Helper()
	value, err := New(Config{Source: source, Scope: resolver, Limits: limits})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	result := make(map[string]agentic.ToolHandler)
	for _, toolset := range plan.Toolsets() {
		tools, handlers := toolset.ToolsAndHandlers()
		for index, tool := range tools {
			result[tool.Function.Name] = handlers[index]
		}
	}
	return result
}

func TestToolFailureFrontiers(t *testing.T) {
	resolver := ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return "tenant", nil })
	runtimeContext := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{})
	for name, current := range map[string]struct {
		source  *fakeSource
		tool    string
		input   map[string]any
		limits  Limits
		context context.Context
	}{
		"list negative limit":     {source: &fakeSource{}, tool: ToolList, input: map[string]any{"limit": -1}, limits: Limits{MaxSkills: 2}, context: runtimeContext},
		"list excessive limit":    {source: &fakeSource{}, tool: ToolList, input: map[string]any{"limit": 3}, limits: Limits{MaxSkills: 2}, context: runtimeContext},
		"list source error":       {source: &fakeSource{err: errors.New("list failed")}, tool: ToolList, input: map[string]any{}, context: runtimeContext},
		"list invalid descriptor": {source: &fakeSource{list: []Descriptor{{Name: "bad/name", Description: "bad"}}}, tool: ToolList, input: map[string]any{}, context: runtimeContext},
		"read invalid name":       {source: &fakeSource{}, tool: ToolRead, input: map[string]any{"name": "../bad"}, context: runtimeContext},
		"read source error":       {source: &fakeSource{err: errors.New("read failed")}, tool: ToolRead, input: map[string]any{"name": "alpha"}, context: runtimeContext},
		"read invalid skill":      {source: &fakeSource{skill: Skill{Name: "alpha", Description: "ok"}}, tool: ToolRead, input: map[string]any{"name": "alpha"}, context: runtimeContext},
		"missing runtime":         {source: &fakeSource{}, tool: ToolList, input: map[string]any{}, context: context.Background()},
	} {
		t.Run(name, func(t *testing.T) {
			handlers := skillHandlers(t, current.source, resolver, current.limits)
			if _, err := handlers[current.tool].Execute(current.context, current.input, nil); err == nil {
				t.Fatal("invalid skill operation succeeded")
			}
		})
	}

	for name, badResolver := range map[string]ScopeResolver{
		"resolver error": ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) {
			return "", errors.New("scope failed")
		}),
		"invalid scope": ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return " bad ", nil }),
	} {
		t.Run(name, func(t *testing.T) {
			handlers := skillHandlers(t, &fakeSource{}, badResolver, Limits{})
			if _, err := handlers[ToolList].Execute(runtimeContext, map[string]any{}, nil); err == nil {
				t.Fatal("invalid scope succeeded")
			}
		})
	}
	readScopeFailure := skillHandlers(t, &fakeSource{}, ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) {
		return "", errors.New("read scope failed")
	}), Limits{})
	if _, err := readScopeFailure[ToolRead].Execute(runtimeContext, map[string]any{"name": "alpha"}, nil); err == nil {
		t.Fatal("read scope failure was hidden")
	}
}

func TestRegistrationEffectsAndValidationFrontiers(t *testing.T) {
	resolver := ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return "tenant", nil })
	value, _ := New(Config{ID: "custom-skills", Source: &fakeSource{}, Scope: resolver, Order: capability.Ordering{After: []string{"before"}}})
	if value.ID() != "custom-skills" || len(value.Ordering().After) != 1 {
		t.Fatalf("identity/order = %q %#v", value.ID(), value.Ordering())
	}
	conflictingTool, conflictingHandler := agentic.MustToolWithContext(
		ToolList, "conflict", func(context.Context, struct{}) (struct{}, error) { return struct{}{}, nil },
	)
	conflict := capability.Func{Name: "conflict", Apply: func(registry *capability.Registry) error {
		return registry.AddToolset(agentic.NewToolset().Add(conflictingTool, conflictingHandler))
	}}
	registerValue, _ := New(Config{Source: &fakeSource{}, Scope: resolver})
	if _, err := capability.Compile(conflict, registerValue); !errors.Is(err, capability.ErrDuplicateTool) {
		t.Fatalf("duplicate registration = %v", err)
	}
	effectConflict := capability.Func{Name: "effect", Apply: func(registry *capability.Registry) error {
		return registry.AddEffectResolver(ToolList, skillEffect("existing"))
	}}
	if _, err := capability.Compile(effectConflict, registerValue); err == nil {
		t.Fatal("duplicate effect registration succeeded")
	}
	defaultEffect, err := skillEffect("list").ResolveEffect(context.Background(), agentic.ToolUse{}, nil)
	if err != nil || defaultEffect.Resource.ID != "catalog" {
		t.Fatalf("catalog effect = %#v, %v", defaultEffect, err)
	}
	if _, err := skillEffect("read").ResolveEffect(context.Background(), agentic.ToolUse{Input: map[string]any{"name": "../bad"}}, nil); err == nil {
		t.Fatal("invalid effect name succeeded")
	}

	for _, scope := range []Scope{"", " bad", "bad\n", Scope(strings.Repeat("x", 257))} {
		if ValidateScope(scope) == nil {
			t.Fatalf("invalid scope %q succeeded", scope)
		}
	}
	if ValidateDescriptor(Descriptor{Name: "bad/name", Description: "value"}, 10) == nil ||
		ValidateDescriptor(Descriptor{Name: "valid", Description: "too long"}, 2) == nil {
		t.Fatal("invalid descriptor succeeded")
	}
	for _, value := range []Skill{
		{Name: "valid", Description: "desc", Instructions: "too long"},
		{Name: "valid", Description: "desc", Instructions: "ok", Resources: []string{"one", "two"}},
		{Name: "valid", Description: "desc", Instructions: "ok", Resources: []string{""}},
		{Name: "valid", Description: "desc", Instructions: "ok", Resources: []string{strings.Repeat("x", 257)}},
		{Name: "valid", Description: "desc", Instructions: "ok", Resources: []string{"same", "same"}},
	} {
		if ValidateSkill(value, 10, 2, 1) == nil {
			t.Fatalf("invalid skill %#v succeeded", value)
		}
	}
	if err := ValidateSkill(Skill{Name: "valid", Description: "desc", Instructions: "ok", Resources: []string{"one"}}, 10, 2, 1); err != nil {
		t.Fatalf("valid skill = %v", err)
	}
}
