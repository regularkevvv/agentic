package permission

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

// Capability contributes one atomic permission gate through the ordinary
// public capability registry.
type Capability struct {
	name   string
	order  capability.Ordering
	policy *Policy
}

type Option func(*Capability) error

func WithID(id string) Option {
	return func(value *Capability) error {
		if id == "" {
			return errors.New("permission capability ID is required")
		}
		value.name = id
		return nil
	}
}

func WithOrdering(order capability.Ordering) Option {
	return func(value *Capability) error {
		value.order = order
		return nil
	}
}

func NewCapability(policy *Policy, options ...Option) (*Capability, error) {
	if policy == nil {
		return nil, errors.New("permission policy is required")
	}
	result := &Capability{name: "permissions", policy: policy}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("permission option must not be nil")
		}
		if err := option(result); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func (c *Capability) ID() string                    { return c.name }
func (c *Capability) Ordering() capability.Ordering { return c.order }
func (c *Capability) Register(registry *capability.Registry) error {
	return registry.AddToolGateMiddleware(&gate{policy: c.policy, registry: registry})
}

type gate struct {
	policy   *Policy
	registry *capability.Registry
}

type deferredRequest struct {
	CallID  string            `json:"call_id"`
	Request PermissionRequest `json:"request"`
}

type deferralPayload struct {
	harnessruntime.DeferredBatch
	Requests []deferredRequest `json:"requests"`
}

func (g *gate) EvaluateBatch(
	ctx context.Context,
	calls []agentic.ToolUse,
	current agentic.ToolBatchDecision,
) (agentic.ToolBatchDecision, error) {
	toolRuntime, ok := harnessruntime.FromContext(ctx)
	if !ok || toolRuntime.Environment == nil {
		return agentic.ToolBatchDecision{}, errors.New("permission gate requires harness ToolRuntime")
	}
	type evaluated struct {
		index    int
		call     agentic.ToolUse
		request  PermissionRequest
		decision Decision
		err      error
	}
	values := make([]evaluated, 0, len(calls))
	hasDeny := false
	hasAsk := false
	for index, call := range calls {
		if current.Calls[index].Kind != agentic.ToolDispositionExecute {
			continue
		}
		request, err := g.resolve(ctx, call, toolRuntime)
		decision := DecisionDeny
		if err == nil {
			decision = g.policy.Evaluate(request)
		}
		hasDeny = hasDeny || decision == DecisionDeny
		hasAsk = hasAsk || decision == DecisionAsk
		values = append(values, evaluated{index: index, call: call, request: request, decision: decision, err: err})
	}
	if hasDeny {
		for _, value := range values {
			message := "Tool call denied by permission policy."
			if value.decision != DecisionDeny {
				message = "Tool call skipped because another call in the atomic batch was denied."
			} else if value.err != nil {
				message = "Tool call denied because its canonical effect could not be resolved: " + value.err.Error()
			}
			result := agentic.ToolExecutionResult{
				ToolUseID: value.call.ID,
				ToolName:  value.call.Name,
				Content:   message,
				IsError:   true,
				Error:     errors.New(message),
			}
			current.Calls[value.index] = agentic.ToolDisposition{
				Kind:     agentic.ToolDispositionReturn,
				Result:   &result,
				Continue: true,
			}
		}
		return current, nil
	}
	if !hasAsk {
		return current, nil
	}
	required := make([]string, 0, len(values))
	preAllowed := make([]string, 0, len(values))
	requests := make([]deferredRequest, 0, len(values))
	for _, value := range values {
		if value.decision == DecisionAsk {
			required = append(required, value.call.ID)
			requests = append(requests, deferredRequest{CallID: value.call.ID, Request: value.request})
		} else {
			preAllowed = append(preAllowed, value.call.ID)
		}
		current.Calls[value.index] = agentic.ToolDisposition{Kind: agentic.ToolDispositionSuspend}
	}
	payload, err := json.Marshal(deferralPayload{
		DeferredBatch: harnessruntime.DeferredBatch{
			Version:               1,
			RequiredResolutionIDs: required,
			PreAllowedCallIDs:     preAllowed,
		},
		Requests: requests,
	})
	if err != nil {
		return agentic.ToolBatchDecision{}, fmt.Errorf("encode permission deferral: %w", err)
	}
	current.Deferral = &agentic.ToolDeferral{
		Kind:    harnessruntime.PermissionDeferralKind,
		Payload: payload,
	}
	return current, nil
}

func (g *gate) resolve(
	ctx context.Context,
	call agentic.ToolUse,
	toolRuntime harnessruntime.ToolRuntime,
) (PermissionRequest, error) {
	resolver, ok := g.registry.EffectResolver(call.Name)
	if !ok {
		return PermissionRequest{
			Capability:        "tool",
			Action:            call.Name,
			CanonicalResource: capabilityResource("tool", call.Name),
		}, nil
	}
	effect, err := resolver.ResolveEffect(ctx, call, toolRuntime.Environment)
	if err != nil {
		return PermissionRequest{}, err
	}
	if effect.Capability == "" || effect.Action == "" || !effect.Resource.Valid() {
		return PermissionRequest{}, errors.New("effect resolver returned an incomplete canonical effect")
	}
	return PermissionRequest{
		Capability:        effect.Capability,
		Action:            effect.Action,
		CanonicalResource: effect.Resource,
	}, nil
}

func capabilityResource(scheme, id string) env.CanonicalResource {
	return env.CanonicalResource{Scheme: scheme, ID: id, Display: id}
}

var _ capability.Capability = (*Capability)(nil)
