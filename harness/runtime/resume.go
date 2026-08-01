package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
)

const (
	PermissionDeferralKind = "harness.permission.v1"
	DeferredBatchVersion   = 1
)

var (
	ErrInvalidResumeRequest = errors.New("invalid harness resume request")
	ErrUnsupportedDeferral  = errors.New("unsupported harness deferral")
	ErrDuplicatePlanner     = errors.New("duplicate harness resume planner")
)

// ResumePlanner converts one capability-owned durable suspension into the
// root driver's ordinary resume decisions. Implementations must be stateless:
// every fact required to validate a request belongs in the suspension.
type ResumePlanner interface {
	PlanResume(agentic.Suspension, ResumeRequest) ([]agentic.ToolResumeDecision, error)
}

// ResumePlannerFunc adapts a function to ResumePlanner.
type ResumePlannerFunc func(agentic.Suspension, ResumeRequest) ([]agentic.ToolResumeDecision, error)

func (f ResumePlannerFunc) PlanResume(
	suspension agentic.Suspension,
	request ResumeRequest,
) ([]agentic.ToolResumeDecision, error) {
	return f(suspension, request)
}

type permissionResumePlanner struct{}

func (permissionResumePlanner) PlanResume(
	suspension agentic.Suspension,
	request ResumeRequest,
) ([]agentic.ToolResumeDecision, error) {
	return PlanResume(suspension, request)
}

// DefaultResumePlanner preserves the policy-neutral runtime's permission
// suspension behavior when no capability graph is present.
func DefaultResumePlanner() ResumePlanner { return permissionResumePlanner{} }

// ResumeRouter dispatches by the public suspension kind. The permission
// planner is always installed; capabilities may add disjoint kinds.
type ResumeRouter struct {
	planners map[string]ResumePlanner
}

func NewResumeRouter(additional map[string]ResumePlanner) (*ResumeRouter, error) {
	planners := map[string]ResumePlanner{PermissionDeferralKind: DefaultResumePlanner()}
	for kind, planner := range additional {
		if kind == "" || planner == nil {
			return nil, errors.New("resume planner kind and implementation are required")
		}
		if _, exists := planners[kind]; exists {
			return nil, fmt.Errorf("%w: %s", ErrDuplicatePlanner, kind)
		}
		planners[kind] = planner
	}
	return &ResumeRouter{planners: planners}, nil
}

func (r *ResumeRouter) PlanResume(
	suspension agentic.Suspension,
	request ResumeRequest,
) ([]agentic.ToolResumeDecision, error) {
	if r == nil {
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedDeferral, suspension.Kind)
	}
	planner, ok := r.planners[suspension.Kind]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedDeferral, suspension.Kind)
	}
	return planner.PlanResume(cloneRuntimeSuspension(suspension), cloneResumeRequest(request))
}

// ResolutionAction selects how one required deferred call is completed.
type ResolutionAction uint8

const (
	ResolutionInvalid ResolutionAction = iota
	ResolutionApprove
	ResolutionDeny
	ResolutionExternalResult
)

// ResumeRequest resolves every ID required by one durable suspension.
type ResumeRequest struct {
	SuspensionID string
	Resolutions  []ToolResolution
	Prompt       *agentic.Message
}

// ToolResolution is an operator decision for one deferred call.
type ToolResolution struct {
	CallID       string
	Action       ResolutionAction
	OverrideArgs map[string]any
	Result       any
	Reason       string
}

// DeferredBatch is the stable inner payload shared by permission gates and the
// stateless resume planner.
type DeferredBatch struct {
	Version               int      `json:"version"`
	RequiredResolutionIDs []string `json:"required_resolution_ids"`
	PreAllowedCallIDs     []string `json:"pre_allowed_call_ids,omitempty"`
}

type rootToolSuspensionPayload struct {
	Version           int
	SuspensionID      string
	HandlerSuspension bool
	Calls             []agentic.ToolUse
	ExecutableCallIDs []string
	Deferral          agentic.ToolDeferral
}

// ToolSuspensionFrontier is the capability-neutral portion of the released
// root tool-suspension envelope used by resume planners.
type ToolSuspensionFrontier struct {
	HandlerSuspension bool
	Calls             []agentic.ToolUse
	ExecutableCallIDs []string
	Deferral          agentic.ToolDeferral
}

// InspectToolSuspension validates the root envelope without interpreting the
// capability-owned deferral payload.
func InspectToolSuspension(
	suspension agentic.Suspension,
	expectedKind string,
) (ToolSuspensionFrontier, error) {
	if suspension.ID == "" || expectedKind == "" || suspension.Kind != expectedKind {
		return ToolSuspensionFrontier{}, fmt.Errorf("%w: %s", ErrUnsupportedDeferral, suspension.Kind)
	}
	var root rootToolSuspensionPayload
	if err := json.Unmarshal(suspension.Payload, &root); err != nil {
		return ToolSuspensionFrontier{}, fmt.Errorf("%w: decode root suspension: %v", ErrInvalidResumeRequest, err)
	}
	if root.Version != 1 || root.SuspensionID != suspension.ID ||
		root.Deferral.Kind != expectedKind || len(root.Calls) == 0 || len(root.ExecutableCallIDs) == 0 {
		return ToolSuspensionFrontier{}, fmt.Errorf("%w: malformed root suspension", ErrInvalidResumeRequest)
	}
	known := make(map[string]bool, len(root.Calls))
	for _, call := range root.Calls {
		if call.ID == "" || call.Name == "" || known[call.ID] {
			return ToolSuspensionFrontier{}, fmt.Errorf("%w: duplicate root call %q", ErrInvalidResumeRequest, call.ID)
		}
		known[call.ID] = true
	}
	executable := make(map[string]bool, len(root.ExecutableCallIDs))
	for _, id := range root.ExecutableCallIDs {
		if !known[id] || executable[id] {
			return ToolSuspensionFrontier{}, fmt.Errorf("%w: invalid executable call %q", ErrInvalidResumeRequest, id)
		}
		executable[id] = true
	}
	deferral := root.Deferral
	deferral.Payload = append([]byte(nil), root.Deferral.Payload...)
	return ToolSuspensionFrontier{
		HandlerSuspension: root.HandlerSuspension,
		Calls:             cloneRuntimeCalls(root.Calls),
		ExecutableCallIDs: append([]string(nil), root.ExecutableCallIDs...),
		Deferral:          deferral,
	}, nil
}

// DeferredFrontier is the validated, public harness view of a root suspension.
type DeferredFrontier struct {
	Calls                 []agentic.ToolUse
	ExecutableCallIDs     []string
	RequiredResolutionIDs []string
	PreAllowedCallIDs     []string
}

// InspectDeferred validates the root envelope and harness deferral before
// returning any operator-visible IDs.
func InspectDeferred(suspension agentic.Suspension) (DeferredFrontier, error) {
	root, err := InspectToolSuspension(suspension, PermissionDeferralKind)
	if err != nil {
		return DeferredFrontier{}, err
	}
	var batch DeferredBatch
	if err := json.Unmarshal(root.Deferral.Payload, &batch); err != nil {
		return DeferredFrontier{}, fmt.Errorf("%w: decode deferred batch: %v", ErrInvalidResumeRequest, err)
	}
	if batch.Version != DeferredBatchVersion || len(batch.RequiredResolutionIDs) == 0 {
		return DeferredFrontier{}, fmt.Errorf("%w: malformed deferred batch", ErrInvalidResumeRequest)
	}
	executable := make(map[string]bool, len(root.ExecutableCallIDs))
	for _, id := range root.ExecutableCallIDs {
		if id == "" || executable[id] {
			return DeferredFrontier{}, fmt.Errorf("%w: duplicate executable call %q", ErrInvalidResumeRequest, id)
		}
		executable[id] = true
	}
	required := make(map[string]bool, len(batch.RequiredResolutionIDs))
	for _, id := range batch.RequiredResolutionIDs {
		if !executable[id] || required[id] {
			return DeferredFrontier{}, fmt.Errorf("%w: invalid required call %q", ErrInvalidResumeRequest, id)
		}
		required[id] = true
	}
	preAllowed := make(map[string]bool, len(batch.PreAllowedCallIDs))
	for _, id := range batch.PreAllowedCallIDs {
		if !executable[id] || required[id] || preAllowed[id] {
			return DeferredFrontier{}, fmt.Errorf("%w: invalid pre-allowed call %q", ErrInvalidResumeRequest, id)
		}
		preAllowed[id] = true
	}
	for id := range executable {
		if !required[id] && !preAllowed[id] {
			return DeferredFrontier{}, fmt.Errorf("%w: executable call %q has no policy disposition", ErrInvalidResumeRequest, id)
		}
	}
	return DeferredFrontier{
		Calls:                 cloneRuntimeCalls(root.Calls),
		ExecutableCallIDs:     append([]string(nil), root.ExecutableCallIDs...),
		RequiredResolutionIDs: append([]string(nil), batch.RequiredResolutionIDs...),
		PreAllowedCallIDs:     append([]string(nil), batch.PreAllowedCallIDs...),
	}, nil
}

// PlanResume validates missing, unknown, and duplicate resolutions and then
// emits one root decision in original executable-call order.
func PlanResume(suspension agentic.Suspension, request ResumeRequest) ([]agentic.ToolResumeDecision, error) {
	if request.SuspensionID == "" || request.SuspensionID != suspension.ID {
		return nil, fmt.Errorf("%w: suspension ID differs", ErrInvalidResumeRequest)
	}
	if request.Prompt != nil && request.Prompt.Role != agentic.RoleUser {
		return nil, fmt.Errorf("%w: resume prompt must be a user message", ErrInvalidResumeRequest)
	}
	frontier, err := InspectDeferred(suspension)
	if err != nil {
		return nil, err
	}
	required := make(map[string]bool, len(frontier.RequiredResolutionIDs))
	for _, id := range frontier.RequiredResolutionIDs {
		required[id] = true
	}
	resolutions := make(map[string]ToolResolution, len(request.Resolutions))
	for _, resolution := range request.Resolutions {
		if resolution.CallID == "" || !required[resolution.CallID] {
			return nil, fmt.Errorf("%w: unknown resolution %q", ErrInvalidResumeRequest, resolution.CallID)
		}
		if _, exists := resolutions[resolution.CallID]; exists {
			return nil, fmt.Errorf("%w: duplicate resolution %q", ErrInvalidResumeRequest, resolution.CallID)
		}
		switch resolution.Action {
		case ResolutionApprove, ResolutionDeny, ResolutionExternalResult:
		default:
			return nil, fmt.Errorf("%w: invalid action for %q", ErrInvalidResumeRequest, resolution.CallID)
		}
		resolution.OverrideArgs = cloneRuntimeMap(resolution.OverrideArgs)
		resolutions[resolution.CallID] = resolution
	}
	if len(resolutions) != len(required) {
		return nil, fmt.Errorf("%w: got %d resolutions for %d required calls", ErrInvalidResumeRequest, len(resolutions), len(required))
	}
	byID := make(map[string]agentic.ToolUse, len(frontier.Calls))
	for _, call := range frontier.Calls {
		if call.ID == "" || byID[call.ID].ID != "" {
			return nil, fmt.Errorf("%w: duplicate frontier call %q", ErrInvalidResumeRequest, call.ID)
		}
		byID[call.ID] = call
	}
	decisions := make([]agentic.ToolResumeDecision, 0, len(frontier.ExecutableCallIDs))
	for _, id := range frontier.ExecutableCallIDs {
		call, ok := byID[id]
		if !ok {
			return nil, fmt.Errorf("%w: executable call %q is absent", ErrInvalidResumeRequest, id)
		}
		resolution, needsResolution := resolutions[id]
		if !needsResolution {
			decisions = append(decisions, agentic.ToolResumeDecision{
				CallID: id,
				Action: agentic.ToolResumeExecute,
			})
			continue
		}
		switch resolution.Action {
		case ResolutionApprove:
			decisions = append(decisions, agentic.ToolResumeDecision{
				CallID: id,
				Action: agentic.ToolResumeExecute,
				Input:  cloneRuntimeMap(resolution.OverrideArgs),
			})
		case ResolutionDeny:
			reason := resolution.Reason
			if reason == "" {
				reason = "denied by operator"
			}
			message := "Tool call denied: " + reason
			decisions = append(decisions, agentic.ToolResumeDecision{
				CallID: id,
				Action: agentic.ToolResumeReturn,
				Result: &agentic.ToolExecutionResult{
					ToolUseID: id,
					ToolName:  call.Name,
					Content:   message,
					IsError:   true,
					Error:     errors.New(message),
				},
			})
		case ResolutionExternalResult:
			decisions = append(decisions, agentic.ToolResumeDecision{
				CallID: id,
				Action: agentic.ToolResumeReturn,
				Result: &agentic.ToolExecutionResult{
					ToolUseID: id,
					ToolName:  call.Name,
					Content:   resolution.Result,
				},
			})
		}
	}
	return decisions, nil
}

// Resume is the low-level stateless primitive. It delegates actual tool
// re-entry to the released Agentic Driver.
func Resume[O any](
	ctx context.Context,
	driver agentic.Driver[O],
	transcript []agentic.Message,
	suspension agentic.Suspension,
	request ResumeRequest,
	options ...agentic.RunOption,
) (*agentic.Execution[O], error) {
	if driver == nil {
		return nil, errors.New("resume driver is required")
	}
	decisions, err := PlanResume(suspension, request)
	if err != nil {
		return nil, err
	}
	var prompt *agentic.Message
	if request.Prompt != nil {
		copy := cloneRuntimeMessages([]agentic.Message{*request.Prompt})[0]
		prompt = &copy
	}
	return driver.Resume(ctx, agentic.ResumeInput{
		History:    cloneRuntimeMessages(transcript),
		Suspension: cloneRuntimeSuspension(suspension),
		Decisions:  decisions,
		Prompt:     prompt,
	}, options...)
}

func cloneRuntimeSuspension(value agentic.Suspension) agentic.Suspension {
	value.Payload = append([]byte(nil), value.Payload...)
	return value
}

func cloneResumeRequest(request ResumeRequest) ResumeRequest {
	result := request
	if request.Prompt != nil {
		message := cloneRuntimeMessages([]agentic.Message{*request.Prompt})[0]
		result.Prompt = &message
	}
	result.Resolutions = make([]ToolResolution, len(request.Resolutions))
	for index, resolution := range request.Resolutions {
		result.Resolutions[index] = resolution
		result.Resolutions[index].OverrideArgs = cloneRuntimeMap(resolution.OverrideArgs)
		result.Resolutions[index].Result = cloneRuntimeValue(resolution.Result)
	}
	return result
}

func cloneRuntimeMessages(messages []agentic.Message) []agentic.Message {
	if len(messages) == 0 {
		return nil
	}
	encoded, _ := json.Marshal(messages)
	var result []agentic.Message
	_ = json.Unmarshal(encoded, &result)
	return result
}

func cloneRuntimeCalls(calls []agentic.ToolUse) []agentic.ToolUse {
	if len(calls) == 0 {
		return nil
	}
	encoded, _ := json.Marshal(calls)
	var result []agentic.ToolUse
	_ = json.Unmarshal(encoded, &result)
	return result
}

func cloneRuntimeMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	result := make(map[string]any, len(value))
	for key, item := range value {
		result[key] = cloneRuntimeValue(item)
	}
	return result
}

func cloneRuntimeValue(value any) any {
	switch current := value.(type) {
	case map[string]any:
		return cloneRuntimeMap(current)
	case []any:
		result := make([]any, len(current))
		for index, item := range current {
			result[index] = cloneRuntimeValue(item)
		}
		return result
	case []string:
		return append([]string(nil), current...)
	case []byte:
		return append([]byte(nil), current...)
	default:
		return current
	}
}
