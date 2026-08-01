package skill

import (
	"context"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

const (
	CapabilityID = "skills"
	ToolList     = "list_skills"
	ToolRead     = "read_skill"
)

type ScopeResolver interface {
	ResolveSkillScope(context.Context, harnessruntime.Scope) (Scope, error)
}

type ScopeResolverFunc func(context.Context, harnessruntime.Scope) (Scope, error)

func (f ScopeResolverFunc) ResolveSkillScope(ctx context.Context, scope harnessruntime.Scope) (Scope, error) {
	return f(ctx, scope)
}

type Limits struct {
	MaxSkills           int
	MaxDescriptionBytes int
	MaxInstructionBytes int
	MaxResources        int
}

type Config struct {
	ID     string
	Order  capability.Ordering
	Source Source
	Scope  ScopeResolver
	Limits Limits
}

type Capability struct{ config Config }

func New(config Config) (*Capability, error) {
	if config.ID == "" {
		config.ID = CapabilityID
	}
	if config.Source == nil || config.Scope == nil {
		return nil, errors.New("skill source and scope resolver are required")
	}
	if config.Limits.MaxSkills == 0 {
		config.Limits.MaxSkills = 100
	}
	if config.Limits.MaxDescriptionBytes == 0 {
		config.Limits.MaxDescriptionBytes = 4 << 10
	}
	if config.Limits.MaxInstructionBytes == 0 {
		config.Limits.MaxInstructionBytes = 128 << 10
	}
	if config.Limits.MaxResources == 0 {
		config.Limits.MaxResources = 100
	}
	if config.Limits.MaxSkills <= 0 || config.Limits.MaxDescriptionBytes <= 0 ||
		config.Limits.MaxInstructionBytes <= 0 || config.Limits.MaxResources <= 0 {
		return nil, errors.New("skill limits must be positive")
	}
	return &Capability{config: config}, nil
}

func (c *Capability) ID() string { return c.config.ID }

func (c *Capability) Ordering() capability.Ordering { return c.config.Order }

type listInput struct {
	Limit int `json:"limit,omitempty"`
}

type readInput struct {
	Name string `json:"name"`
}

func (c *Capability) Register(registry *capability.Registry) error {
	listTool, listHandler, err := agentic.ToolWithContext(
		ToolList,
		"List bounded application-provided skills available in the host-selected scope.",
		func(ctx context.Context, input listInput) ([]Descriptor, error) {
			scope, err := c.resolve(ctx)
			if err != nil {
				return nil, err
			}
			limit := input.Limit
			if limit == 0 {
				limit = c.config.Limits.MaxSkills
			}
			if limit < 1 || limit > c.config.Limits.MaxSkills {
				return nil, fmt.Errorf("skill list limit must be between 1 and %d", c.config.Limits.MaxSkills)
			}
			values, err := c.config.Source.List(ctx, scope, limit)
			if err != nil {
				return nil, err
			}
			seen := make(map[string]bool, len(values))
			for _, value := range values {
				if err := ValidateDescriptor(value, c.config.Limits.MaxDescriptionBytes); err != nil {
					return nil, err
				}
				if seen[value.Name] {
					return nil, fmt.Errorf("duplicate skill %q", value.Name)
				}
				seen[value.Name] = true
			}
			return append([]Descriptor(nil), values...), nil
		},
	)
	if err != nil {
		return err
	}
	readTool, readHandler, err := agentic.ToolWithContext(
		ToolRead,
		"Load bounded application-provided guidance. It remains below the root system prompt.",
		func(ctx context.Context, input readInput) (Skill, error) {
			scope, err := c.resolve(ctx)
			if err != nil {
				return Skill{}, err
			}
			if err := ValidateName(input.Name); err != nil {
				return Skill{}, err
			}
			value, err := c.config.Source.Read(ctx, scope, input.Name, c.config.Limits.MaxInstructionBytes)
			if err != nil {
				return Skill{}, err
			}
			if value.Name != input.Name {
				return Skill{}, errors.New("skill source returned a different name")
			}
			if err := ValidateSkill(value, c.config.Limits.MaxDescriptionBytes, c.config.Limits.MaxInstructionBytes, c.config.Limits.MaxResources); err != nil {
				return Skill{}, err
			}
			return Clone(value), nil
		},
	)
	if err != nil {
		return err
	}
	if err := registry.AddToolset(agentic.NewToolset().Add(listTool, listHandler).Add(readTool, readHandler)); err != nil {
		return err
	}
	if err := registry.AddEffectResolver(ToolList, skillEffect("list")); err != nil {
		return err
	}
	return registry.AddEffectResolver(ToolRead, skillEffect("read"))
}

func (c *Capability) resolve(ctx context.Context) (Scope, error) {
	runtime, ok := harnessruntime.FromContext(ctx)
	if !ok {
		return "", errors.New("skill capability requires harness ToolRuntime")
	}
	scope, err := c.config.Scope.ResolveSkillScope(ctx, runtime.Scope)
	if err != nil {
		return "", fmt.Errorf("resolve skill scope: %w", err)
	}
	if err := ValidateScope(scope); err != nil {
		return "", err
	}
	return scope, nil
}

func skillEffect(action string) capability.EffectResolverFunc {
	return func(_ context.Context, call agentic.ToolUse, _ env.Environment) (capability.Effect, error) {
		name := "catalog"
		if value, ok := call.Input["name"].(string); ok {
			if err := ValidateName(value); err != nil {
				return capability.Effect{}, err
			}
			name = value
		}
		return capability.Effect{
			Capability: "skill", Action: action,
			Resource: env.CanonicalResource{Scheme: "skill", ID: name, Display: name},
		}, nil
	}
}

var _ capability.Capability = (*Capability)(nil)
