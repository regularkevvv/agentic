package skill

import (
	"context"
	"errors"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type fakeSource struct {
	list  []Descriptor
	skill Skill
	err   error
	scope Scope
}

func (s *fakeSource) List(_ context.Context, scope Scope, limit int) ([]Descriptor, error) {
	s.scope = scope
	if s.err != nil {
		return nil, s.err
	}
	values := append([]Descriptor(nil), s.list...)
	if len(values) > limit {
		values = values[:limit]
	}
	return values, nil
}

func (s *fakeSource) Read(_ context.Context, scope Scope, _ string, _ int) (Skill, error) {
	s.scope = scope
	return Clone(s.skill), s.err
}

func TestCapabilityToolsUseHostScopeAndEffects(t *testing.T) {
	source := &fakeSource{
		list:  []Descriptor{{Name: "alpha", Description: "first"}},
		skill: Skill{Name: "alpha", Description: "first", Instructions: "Do the bounded thing.", Resources: []string{"example.txt"}},
	}
	resolved := harnessruntime.Scope{}
	value, err := New(Config{
		Source: source,
		Scope: ScopeResolverFunc(func(_ context.Context, scope harnessruntime.Scope) (Scope, error) {
			resolved = scope
			return "tenant", nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	registry := agentic.NewRegistry()
	for _, toolset := range plan.Toolsets() {
		if err := agentic.RegisterToolset(registry, toolset); err != nil {
			t.Fatal(err)
		}
	}
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Scope: harnessruntime.Scope{SessionID: "session", Agent: "worker", Depth: 1, ParentSessionID: "parent"},
	})
	listed, err := registry.Execute(ctx, agentic.ToolUse{ID: "list", Name: ToolList, Input: map[string]any{}}, nil)
	if err != nil || listed.IsError {
		t.Fatalf("list = %#v, %v", listed, err)
	}
	if !reflect.DeepEqual(source.scope, Scope("tenant")) || resolved.SessionID != "session" {
		t.Fatalf("resolved scope = %#v, %q", resolved, source.scope)
	}
	read, err := registry.Execute(ctx, agentic.ToolUse{ID: "read", Name: ToolRead, Input: map[string]any{"name": "alpha"}}, nil)
	if err != nil || read.IsError {
		t.Fatalf("read = %#v, %v", read, err)
	}
	effect, err := skillEffect("read").ResolveEffect(context.Background(), agentic.ToolUse{Name: ToolRead, Input: map[string]any{"name": "alpha"}}, nil)
	if err != nil || effect.Resource.ID != "alpha" || effect.Action != "read" {
		t.Fatalf("effect = %#v, %v", effect, err)
	}
}

func TestCapabilityRejectsSourceAndScopeViolations(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("missing dependencies succeeded")
	}
	if _, err := New(Config{Source: &fakeSource{}, Scope: ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) {
		return "scope", nil
	}), Limits: Limits{MaxSkills: -1}}); err == nil {
		t.Fatal("negative limits succeeded")
	}
	for name, source := range map[string]*fakeSource{
		"duplicate": {list: []Descriptor{{Name: "same", Description: "one"}, {Name: "same", Description: "two"}}},
		"different": {skill: Skill{Name: "other", Description: "desc", Instructions: "body"}},
	} {
		t.Run(name, func(t *testing.T) {
			capabilityValue, _ := New(Config{Source: source, Scope: ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) {
				return "scope", nil
			})})
			plan, err := capability.Compile(capabilityValue)
			if err != nil {
				t.Fatal(err)
			}
			registry := agentic.NewRegistry()
			_ = agentic.RegisterToolset(registry, plan.Toolsets()[0])
			ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{Scope: harnessruntime.Scope{SessionID: "session"}})
			call := agentic.ToolUse{ID: "call", Name: ToolList, Input: map[string]any{}}
			if name == "different" {
				call.Name = ToolRead
				call.Input = map[string]any{"name": "expected"}
			}
			result, _ := registry.Execute(ctx, call, nil)
			if !result.IsError {
				t.Fatalf("violation result = %#v", result)
			}
		})
	}
	source := &fakeSource{list: []Descriptor{{Name: "one", Description: "ok"}}}
	capabilityValue, _ := New(Config{Source: source, Scope: ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) {
		return "", errors.New("tenant unavailable")
	})})
	plan, _ := capability.Compile(capabilityValue)
	registry := agentic.NewRegistry()
	_ = agentic.RegisterToolset(registry, plan.Toolsets()[0])
	result, _ := registry.Execute(context.Background(), agentic.ToolUse{ID: "call", Name: ToolList, Input: map[string]any{}}, nil)
	if !result.IsError {
		t.Fatalf("missing runtime result = %#v", result)
	}
}

func TestValidationAndClone(t *testing.T) {
	if err := ValidateScope(""); err == nil {
		t.Fatal("empty scope succeeded")
	}
	for _, name := range []string{"", "bad/name", "-bad"} {
		if ValidateName(name) == nil {
			t.Fatalf("invalid name %q succeeded", name)
		}
	}
	if ValidateDescriptor(Descriptor{Name: "ok", Description: ""}, 10) == nil {
		t.Fatal("empty description succeeded")
	}
	if ValidateSkill(Skill{Name: "ok", Description: "desc", Instructions: "", Resources: nil}, 10, 10, 1) == nil {
		t.Fatal("empty instructions succeeded")
	}
	value := Skill{Name: "ok", Description: "desc", Instructions: "body", Resources: []string{"one"}}
	copy := Clone(value)
	copy.Resources[0] = "two"
	if value.Resources[0] != "one" {
		t.Fatal("clone mutated source")
	}
}
