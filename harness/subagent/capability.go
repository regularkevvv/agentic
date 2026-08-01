package subagent

import (
	"context"
	"errors"
	"fmt"
	"strings"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type delegationInput struct {
	Task string `json:"task" description:"The task to delegate to the child agent"`
}

// BindRunner derives an already-bound child runner from the exact dependency
// type of the parent runner. Mode is the resolved dependency capture policy,
// so applications implement share, isolate, and narrow without reflection or
// a harness-owned dependency representation.
type BindRunner[D, O any] func(agentic.RunContext[D], Mode) (agentic.Runner[O], error)

// Capability contributes one delegation tool and owns only its live addressed
// child-routing table. Child transcripts and queues remain ordinary sessions.
type Capability struct {
	config  Config
	toolset agentic.Toolset
	router  *router
}

// New constructs a child capability from an already-bound runner. Dependency
// capture is explicit in how the caller bound that runner; the harness never
// extracts or rewrites an opaque dependency value. Tool capture applies to
// parent harness capability toolsets; tools intrinsic to this child runner are
// child-native and remain the caller's explicit construction choice.
func New[O any](runner agentic.Runner[O], config Config) (*Capability, error) {
	if runner == nil {
		return nil, errors.New("subagent runner is required")
	}
	if _, err := agentic.RequireDriver(runner); err != nil {
		return nil, err
	}
	resolved, err := resolveConfig(config)
	if err != nil {
		return nil, err
	}
	result := &Capability{config: resolved, router: newRouter()}
	tool, handler, err := agentic.ToolWithContext(
		resolved.Name,
		resolved.Description,
		func(ctx context.Context, input delegationInput) (Result, error) {
			return runChild(ctx, result, input.Task, runner)
		},
	)
	if err != nil {
		return nil, fmt.Errorf("build subagent tool: %w", err)
	}
	result.toolset = agentic.NewToolset().Add(tool, handler)
	return result, nil
}

// NewWithDeps constructs a dependency-aware delegation tool. The binder runs
// inside the parent's exact RunContext and returns a bound child runner, making
// share, isolate, and narrowed dependency mapping an application choice rather
// than reflection-based harness behavior.
func NewWithDeps[D, O any](config Config, bind BindRunner[D, O]) (*Capability, error) {
	if bind == nil {
		return nil, errors.New("subagent dependency binder is required")
	}
	resolved, err := resolveConfig(config)
	if err != nil {
		return nil, err
	}
	result := &Capability{config: resolved, router: newRouter()}
	tool, handler, err := agentic.ToolWithDeps(
		resolved.Name,
		resolved.Description,
		func(ctx agentic.RunContext[D], input delegationInput) (Result, error) {
			runner, bindErr := bind(ctx, resolved.Capture.Dependencies)
			if bindErr != nil {
				return Result{}, fmt.Errorf("bind child runner: %w", bindErr)
			}
			if runner == nil {
				return Result{}, errors.New("subagent dependency binder returned nil")
			}
			return runChild(ctx.Ctx, result, input.Task, runner)
		},
	)
	if err != nil {
		return nil, fmt.Errorf("build dependency-aware subagent tool: %w", err)
	}
	result.toolset = agentic.NewToolset().Add(tool, handler)
	return result, nil
}

func (c *Capability) ID() string {
	if c == nil {
		return ""
	}
	return "subagent:" + c.config.Name
}

func (c *Capability) Ordering() capability.Ordering {
	if c == nil {
		return capability.Ordering{}
	}
	return c.config.Ordering
}

func (c *Capability) Register(registry *capability.Registry) error {
	if c == nil || c.toolset == nil {
		return errors.New("subagent capability is not initialized")
	}
	if err := registry.AddToolset(c.toolset); err != nil {
		return err
	}
	if err := registry.MarkDelegationTool(c.config.Name); err != nil {
		return err
	}
	return registry.AddEffectResolver(c.config.Name, capability.EffectResolverFunc(func(
		_ context.Context,
		call agentic.ToolUse,
		_ env.Environment,
	) (capability.Effect, error) {
		task, _ := call.Input["task"].(string)
		if strings.TrimSpace(task) == "" {
			return capability.Effect{}, errors.New("subagent task is required")
		}
		return capability.Effect{
			Capability: "subagent",
			Action:     "delegate",
			Resource: env.CanonicalResource{
				Scheme:  "agent",
				ID:      c.config.Name,
				Display: c.config.Name,
			},
		}, nil
	}))
}

func requireParentRuntime(ctx context.Context) (harnessruntime.ToolRuntime, error) {
	value, ok := harnessruntime.FromContext(ctx)
	if !ok || value.Environment == nil || value.Capture == nil || value.SessionID == "" {
		return harnessruntime.ToolRuntime{}, errors.New("subagent requires harness ToolRuntime capture services")
	}
	if value.Scope.SessionID != value.SessionID {
		return harnessruntime.ToolRuntime{}, errors.New("subagent ToolRuntime scope is inconsistent")
	}
	return value, nil
}

var _ capability.Capability = (*Capability)(nil)
