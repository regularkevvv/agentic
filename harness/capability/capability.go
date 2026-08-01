// Package capability compiles ordered, immutable harness execution plans.
package capability

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

var (
	ErrDuplicateCapability = errors.New("duplicate capability ID")
	ErrMissingOrdering     = errors.New("capability ordering reference is missing")
	ErrCapabilityCycle     = errors.New("capability ordering contains a cycle")
	ErrRegistryFrozen      = errors.New("capability registry is frozen")
	ErrDuplicateTool       = errors.New("duplicate capability tool")
	ErrDuplicateEffect     = errors.New("duplicate tool effect resolver")
	ErrContextConfigured   = errors.New("context policy is already configured")
	ErrGateBroadened       = errors.New("tool gate middleware broadened a prior decision")
	ErrUnknownTool         = errors.New("capability tool is not registered")
	ErrDuplicateSelection  = errors.New("duplicate selected capability tool")
	ErrDuplicatePlanner    = errors.New("duplicate capability resume planner")
)

// Ordering declares capability graph edges by stable capability ID.
type Ordering struct {
	Before []string
	After  []string
}

// Capability contributes only through Registry's public extension points.
type Capability interface {
	ID() string
	Ordering() Ordering
	Register(*Registry) error
}

// RegisterFunc adapts an ordinary function to capability registration.
type RegisterFunc func(*Registry) error

// Func is a compact, exported capability implementation useful for
// application composition.
type Func struct {
	Name  string
	Order Ordering
	Apply RegisterFunc
}

func (f Func) ID() string         { return f.Name }
func (f Func) Ordering() Ordering { return f.Order }
func (f Func) Register(registry *Registry) error {
	if f.Apply == nil {
		return nil
	}
	return f.Apply(registry)
}

// Effect is a backend-neutral proposed effect.
type Effect struct {
	Capability string
	Action     string
	Resource   env.CanonicalResource
}

// EffectResolver translates one capability tool call into a canonical effect.
type EffectResolver interface {
	ResolveEffect(context.Context, agentic.ToolUse, env.Environment) (Effect, error)
}

// EffectResolverFunc adapts a function to EffectResolver.
type EffectResolverFunc func(context.Context, agentic.ToolUse, env.Environment) (Effect, error)

func (f EffectResolverFunc) ResolveEffect(ctx context.Context, call agentic.ToolUse, environment env.Environment) (Effect, error) {
	return f(ctx, call, environment)
}

// ToolGateMiddleware may only change calls whose current disposition remains
// executable. Calls already returned or suspended are immutable.
type ToolGateMiddleware interface {
	EvaluateBatch(context.Context, []agentic.ToolUse, agentic.ToolBatchDecision) (agentic.ToolBatchDecision, error)
}

// ToolGateMiddlewareFunc adapts a function to ToolGateMiddleware.
type ToolGateMiddlewareFunc func(context.Context, []agentic.ToolUse, agentic.ToolBatchDecision) (agentic.ToolBatchDecision, error)

func (f ToolGateMiddlewareFunc) EvaluateBatch(
	ctx context.Context,
	calls []agentic.ToolUse,
	current agentic.ToolBatchDecision,
) (agentic.ToolBatchDecision, error) {
	return f(ctx, calls, current)
}

// Registry is mutable only while one capability graph is compiling.
type Registry struct {
	frozen            bool
	toolsets          []agentic.Toolset
	tools             []agentic.Tool
	toolNames         map[string]bool
	delegationTools   map[string]bool
	effects           map[string]EffectResolver
	gates             []ToolGateMiddleware
	contextTransforms []contextpolicy.Transform
	contextConfigured bool
	contextConfig     contextpolicy.Config
	compactor         contextpolicy.Compactor
	eventMiddleware   []event.Middleware
	lifecycleHooks    []harnessruntime.LifecycleHook
	resumePlanners    map[string]harnessruntime.ResumePlanner
}

func newRegistry() *Registry {
	return &Registry{
		toolNames:       make(map[string]bool),
		delegationTools: make(map[string]bool),
		effects:         make(map[string]EffectResolver),
		resumePlanners:  make(map[string]harnessruntime.ResumePlanner),
	}
}

// AddToolset snapshots a toolset so caller mutation cannot change a built
// Harness.
func (r *Registry) AddToolset(toolset agentic.Toolset) error {
	if err := r.mutable(); err != nil {
		return err
	}
	if toolset == nil {
		return errors.New("capability toolset must not be nil")
	}
	tools, handlers := toolset.ToolsAndHandlers()
	if len(tools) != len(handlers) {
		return fmt.Errorf("capability toolset has %d tools and %d handlers", len(tools), len(handlers))
	}
	cloned, err := cloneTools(tools)
	if err != nil {
		return err
	}
	for _, tool := range cloned {
		name := tool.Function.Name
		if name == "" {
			return errors.New("capability tool has an empty name")
		}
		if r.toolNames[name] {
			return fmt.Errorf("%w: %s", ErrDuplicateTool, name)
		}
		r.toolNames[name] = true
	}
	frozen := &frozenToolset{
		tools:    cloned,
		handlers: append([]agentic.ToolHandler(nil), handlers...),
	}
	r.toolsets = append(r.toolsets, frozen)
	r.tools = append(r.tools, cloned...)
	return nil
}

// TakeToolset removes explicitly named tools from the model-visible registry
// and returns a frozen toolset containing them in caller order. Tool names
// remain reserved and effect resolvers remain registered so a composite host
// can route the selected handlers through the same policy graph.
func (r *Registry) TakeToolset(names ...string) (agentic.Toolset, error) {
	if err := r.mutable(); err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return nil, errors.New("selected capability tools are required")
	}
	wanted := make(map[string]bool, len(names))
	for _, name := range names {
		if name == "" || !r.toolNames[name] {
			return nil, fmt.Errorf("%w: %s", ErrUnknownTool, name)
		}
		if wanted[name] {
			return nil, fmt.Errorf("%w: %s", ErrDuplicateSelection, name)
		}
		wanted[name] = true
	}
	type pair struct {
		tool    agentic.Tool
		handler agentic.ToolHandler
	}
	selected := make(map[string]pair, len(names))
	remainingSets := make([]agentic.Toolset, 0, len(r.toolsets))
	for _, set := range r.toolsets {
		tools, handlers := set.ToolsAndHandlers()
		keptTools := make([]agentic.Tool, 0, len(tools))
		keptHandlers := make([]agentic.ToolHandler, 0, len(handlers))
		for index, tool := range tools {
			name := tool.Function.Name
			if wanted[name] {
				selected[name] = pair{tool: tool, handler: handlers[index]}
				continue
			}
			keptTools = append(keptTools, tool)
			keptHandlers = append(keptHandlers, handlers[index])
		}
		if len(keptTools) > 0 {
			remainingSets = append(remainingSets, &frozenToolset{tools: keptTools, handlers: keptHandlers})
		}
	}
	chosenTools := make([]agentic.Tool, len(names))
	chosenHandlers := make([]agentic.ToolHandler, len(names))
	for index, name := range names {
		item, ok := selected[name]
		if !ok {
			return nil, fmt.Errorf("%w: %s", ErrUnknownTool, name)
		}
		chosenTools[index] = item.tool
		chosenHandlers[index] = item.handler
	}
	remainingTools := make([]agentic.Tool, 0, len(r.tools)-len(names))
	for _, tool := range r.tools {
		if !wanted[tool.Function.Name] {
			remainingTools = append(remainingTools, tool)
		}
	}
	r.toolsets = remainingSets
	r.tools = remainingTools
	return &frozenToolset{tools: chosenTools, handlers: chosenHandlers}, nil
}

// IsDelegationTool reports whether a registered tool changes session
// topology. Composite execution capabilities use it to reject recursive or
// ownership-changing selections before extracting a toolset.
func (r *Registry) IsDelegationTool(name string) bool { return r.delegationTools[name] }

// HasTool reports whether a name is already reserved by the graph, including
// tools hidden by a prior composite capability.
func (r *Registry) HasTool(name string) bool { return r.toolNames[name] }

// MarkDelegationTool classifies a registered tool as topology-changing.
// Inherited toolsets exclude these tools unless recursion is explicitly
// enabled by a topology capability.
func (r *Registry) MarkDelegationTool(toolName string) error {
	if err := r.mutable(); err != nil {
		return err
	}
	if !r.toolNames[toolName] {
		return fmt.Errorf("%w: %s", ErrUnknownTool, toolName)
	}
	r.delegationTools[toolName] = true
	return nil
}

func (r *Registry) AddEffectResolver(toolName string, resolver EffectResolver) error {
	if err := r.mutable(); err != nil {
		return err
	}
	if toolName == "" || resolver == nil {
		return errors.New("tool effect name and resolver are required")
	}
	if _, exists := r.effects[toolName]; exists {
		return fmt.Errorf("%w: %s", ErrDuplicateEffect, toolName)
	}
	r.effects[toolName] = resolver
	return nil
}

// EffectResolver returns a resolver from the eventual immutable registry.
func (r *Registry) EffectResolver(toolName string) (EffectResolver, bool) {
	resolver, ok := r.effects[toolName]
	return resolver, ok
}

func (r *Registry) AddToolGateMiddleware(middleware ToolGateMiddleware) error {
	if err := r.mutable(); err != nil {
		return err
	}
	if middleware == nil {
		return errors.New("tool gate middleware must not be nil")
	}
	r.gates = append(r.gates, middleware)
	return nil
}

func (r *Registry) AddContextTransform(transform contextpolicy.Transform) error {
	if err := r.mutable(); err != nil {
		return err
	}
	if transform == nil {
		return errors.New("context transform must not be nil")
	}
	r.contextTransforms = append(r.contextTransforms, transform)
	return nil
}

func (r *Registry) ConfigureContext(config contextpolicy.Config, compactor contextpolicy.Compactor) error {
	if err := r.mutable(); err != nil {
		return err
	}
	if r.contextConfigured {
		return ErrContextConfigured
	}
	r.contextConfigured = true
	r.contextConfig = config
	r.compactor = compactor
	return nil
}

func (r *Registry) AddEventMiddleware(middleware event.Middleware) error {
	if err := r.mutable(); err != nil {
		return err
	}
	if middleware == nil {
		return errors.New("event middleware must not be nil")
	}
	r.eventMiddleware = append(r.eventMiddleware, middleware)
	return nil
}

func (r *Registry) AddLifecycleHook(hook harnessruntime.LifecycleHook) error {
	if err := r.mutable(); err != nil {
		return err
	}
	if hook == nil {
		return errors.New("lifecycle hook must not be nil")
	}
	r.lifecycleHooks = append(r.lifecycleHooks, hook)
	return nil
}

// AddResumePlanner registers one capability-owned suspension kind.
func (r *Registry) AddResumePlanner(kind string, planner harnessruntime.ResumePlanner) error {
	if err := r.mutable(); err != nil {
		return err
	}
	if kind == "" || planner == nil {
		return errors.New("resume planner kind and implementation are required")
	}
	if kind == harnessruntime.PermissionDeferralKind {
		return fmt.Errorf("%w: %s", ErrDuplicatePlanner, kind)
	}
	if _, exists := r.resumePlanners[kind]; exists {
		return fmt.Errorf("%w: %s", ErrDuplicatePlanner, kind)
	}
	r.resumePlanners[kind] = planner
	return nil
}

func (r *Registry) mutable() error {
	if r.frozen {
		return ErrRegistryFrozen
	}
	return nil
}

// Plan is the immutable result of graph compilation.
type Plan struct {
	ids             []string
	toolsets        []agentic.Toolset
	tools           []agentic.Tool
	gate            agentic.ToolGate
	context         contextpolicy.Projector
	eventMiddleware []event.Middleware
	lifecycleHooks  []harnessruntime.LifecycleHook
	delegationTools []string
	resumePlanner   harnessruntime.ResumePlanner
}

func (p Plan) IDs() []string {
	return append([]string(nil), p.ids...)
}

func (p Plan) Toolsets() []agentic.Toolset {
	return append([]agentic.Toolset(nil), p.toolsets...)
}

func (p Plan) Tools() []agentic.Tool {
	tools, _ := cloneTools(p.tools)
	return tools
}

func (p Plan) ToolGate() agentic.ToolGate {
	return p.gate
}

func (p Plan) ContextPolicy() contextpolicy.Projector {
	return p.context
}

func (p Plan) EventMiddleware() []event.Middleware {
	return append([]event.Middleware(nil), p.eventMiddleware...)
}

func (p Plan) LifecycleHooks() []harnessruntime.LifecycleHook {
	return append([]harnessruntime.LifecycleHook(nil), p.lifecycleHooks...)
}

func (p Plan) ResumePlanner() harnessruntime.ResumePlanner { return p.resumePlanner }

// DelegationTools returns the stable names of topology-changing tools.
func (p Plan) DelegationTools() []string {
	return append([]string(nil), p.delegationTools...)
}

// Compile validates, stably sorts, and freezes one capability graph.
func Compile(capabilities ...Capability) (Plan, error) {
	sorted, err := stableOrder(capabilities)
	if err != nil {
		return Plan{}, err
	}
	registry := newRegistry()
	ids := make([]string, len(sorted))
	for index, current := range sorted {
		ids[index] = current.ID()
		if err := current.Register(registry); err != nil {
			return Plan{}, fmt.Errorf("register capability %q: %w", current.ID(), err)
		}
	}
	registry.frozen = true
	contextConfig := registry.contextConfig
	contextConfig.Tools = append(contextConfig.Tools, registry.tools...)
	projector, err := contextpolicy.New(contextConfig, registry.contextTransforms, registry.compactor)
	if err != nil {
		return Plan{}, fmt.Errorf("build context policy: %w", err)
	}
	delegationTools := make([]string, 0, len(registry.delegationTools))
	for name := range registry.delegationTools {
		delegationTools = append(delegationTools, name)
	}
	sort.Strings(delegationTools)
	resumePlanner, err := harnessruntime.NewResumeRouter(registry.resumePlanners)
	if err != nil {
		return Plan{}, fmt.Errorf("build resume planner: %w", err)
	}
	return Plan{
		ids:             ids,
		toolsets:        append([]agentic.Toolset(nil), registry.toolsets...),
		tools:           append([]agentic.Tool(nil), registry.tools...),
		gate:            composedGate{middleware: append([]ToolGateMiddleware(nil), registry.gates...)},
		context:         projector,
		eventMiddleware: append([]event.Middleware(nil), registry.eventMiddleware...),
		lifecycleHooks:  append([]harnessruntime.LifecycleHook(nil), registry.lifecycleHooks...),
		delegationTools: delegationTools,
		resumePlanner:   resumePlanner,
	}, nil
}

func stableOrder(capabilities []Capability) ([]Capability, error) {
	byID := make(map[string]int, len(capabilities))
	for index, current := range capabilities {
		if current == nil {
			return nil, errors.New("capability must not be nil")
		}
		id := current.ID()
		if id == "" || id != strings.TrimSpace(id) {
			return nil, errors.New("capability ID is required")
		}
		if _, exists := byID[id]; exists {
			return nil, fmt.Errorf("%w: %s", ErrDuplicateCapability, id)
		}
		byID[id] = index
	}
	edges := make([]map[int]bool, len(capabilities))
	indegree := make([]int, len(capabilities))
	for index := range edges {
		edges[index] = make(map[int]bool)
	}
	for index, current := range capabilities {
		order := current.Ordering()
		for _, before := range order.Before {
			target, ok := byID[before]
			if !ok {
				return nil, fmt.Errorf("%w: %s before %s", ErrMissingOrdering, current.ID(), before)
			}
			addEdge(edges, indegree, index, target)
		}
		for _, after := range order.After {
			source, ok := byID[after]
			if !ok {
				return nil, fmt.Errorf("%w: %s after %s", ErrMissingOrdering, current.ID(), after)
			}
			addEdge(edges, indegree, source, index)
		}
	}
	ready := make([]int, 0, len(capabilities))
	for index, degree := range indegree {
		if degree == 0 {
			ready = append(ready, index)
		}
	}
	result := make([]Capability, 0, len(capabilities))
	for len(ready) > 0 {
		sort.Ints(ready)
		index := ready[0]
		ready = ready[1:]
		result = append(result, capabilities[index])
		for target := range edges[index] {
			indegree[target]--
			if indegree[target] == 0 {
				ready = append(ready, target)
			}
		}
	}
	if len(result) != len(capabilities) {
		return nil, ErrCapabilityCycle
	}
	return result, nil
}

func addEdge(edges []map[int]bool, indegree []int, source, target int) {
	if edges[source][target] {
		return
	}
	edges[source][target] = true
	indegree[target]++
}

type composedGate struct {
	middleware []ToolGateMiddleware
}

func (g composedGate) EvaluateBatch(ctx context.Context, calls []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
	current := allowAll(len(calls))
	for _, middleware := range g.middleware {
		next, err := middleware.EvaluateBatch(ctx, cloneCalls(calls), cloneDecision(current))
		if err != nil {
			return agentic.ToolBatchDecision{}, err
		}
		if err := validateNarrowing(calls, current, next); err != nil {
			return agentic.ToolBatchDecision{}, err
		}
		current = cloneDecision(next)
		if current.Deferral != nil {
			break
		}
	}
	return cloneDecision(current), nil
}

func allowAll(count int) agentic.ToolBatchDecision {
	calls := make([]agentic.ToolDisposition, count)
	for index := range calls {
		calls[index].Kind = agentic.ToolDispositionExecute
	}
	return agentic.ToolBatchDecision{Calls: calls}
}

func validateNarrowing(calls []agentic.ToolUse, current, next agentic.ToolBatchDecision) error {
	if len(next.Calls) != len(calls) {
		return fmt.Errorf("gate middleware returned %d dispositions for %d calls", len(next.Calls), len(calls))
	}
	hasSuspend := false
	for index := range calls {
		previous := current.Calls[index]
		candidate := next.Calls[index]
		if previous.Kind != agentic.ToolDispositionExecute && !reflect.DeepEqual(previous, candidate) {
			return fmt.Errorf("%w: call %s", ErrGateBroadened, calls[index].ID)
		}
		switch candidate.Kind {
		case agentic.ToolDispositionExecute:
			if candidate.Result != nil {
				return fmt.Errorf("gate middleware supplied a result for executable call %s", calls[index].ID)
			}
		case agentic.ToolDispositionReturn:
			if candidate.Result == nil || candidate.Result.ToolUseID != calls[index].ID ||
				candidate.Result.ToolName != calls[index].Name {
				return fmt.Errorf("gate middleware returned an invalid result for call %s", calls[index].ID)
			}
		case agentic.ToolDispositionSuspend:
			if candidate.Result != nil || candidate.Continue {
				return fmt.Errorf("gate middleware returned an invalid suspension for call %s", calls[index].ID)
			}
			hasSuspend = true
		default:
			return fmt.Errorf("gate middleware returned an invalid disposition for call %s", calls[index].ID)
		}
	}
	if hasSuspend {
		if next.Deferral == nil || next.Deferral.Kind == "" {
			return errors.New("gate middleware suspension requires a deferral")
		}
	} else if next.Deferral != nil {
		return errors.New("gate middleware returned a deferral without suspension")
	}
	return nil
}

type frozenToolset struct {
	tools    []agentic.Tool
	handlers []agentic.ToolHandler
}

func (f *frozenToolset) ToolsAndHandlers() ([]agentic.Tool, []agentic.ToolHandler) {
	tools, _ := cloneTools(f.tools)
	return tools, append([]agentic.ToolHandler(nil), f.handlers...)
}

func cloneTools(tools []agentic.Tool) ([]agentic.Tool, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	encoded, err := json.Marshal(tools)
	if err != nil {
		return nil, fmt.Errorf("clone capability tools: %w", err)
	}
	var result []agentic.Tool
	if err := json.Unmarshal(encoded, &result); err != nil {
		return nil, fmt.Errorf("clone capability tools: %w", err)
	}
	return result, nil
}

func cloneCalls(calls []agentic.ToolUse) []agentic.ToolUse {
	if len(calls) == 0 {
		return nil
	}
	encoded, _ := json.Marshal(calls)
	var result []agentic.ToolUse
	_ = json.Unmarshal(encoded, &result)
	return result
}

func cloneDecision(value agentic.ToolBatchDecision) agentic.ToolBatchDecision {
	result := agentic.ToolBatchDecision{Calls: make([]agentic.ToolDisposition, len(value.Calls))}
	for index, disposition := range value.Calls {
		result.Calls[index] = disposition
		if disposition.Result != nil {
			copy := *disposition.Result
			copy.Content = cloneMutableValue(disposition.Result.Content)
			result.Calls[index].Result = &copy
		}
	}
	if value.Deferral != nil {
		copy := *value.Deferral
		copy.Payload = append([]byte(nil), value.Deferral.Payload...)
		result.Deferral = &copy
	}
	return result
}

func cloneMutableValue(value any) any {
	switch current := value.(type) {
	case map[string]any:
		result := make(map[string]any, len(current))
		for key, item := range current {
			result[key] = cloneMutableValue(item)
		}
		return result
	case []any:
		result := make([]any, len(current))
		for index, item := range current {
			result[index] = cloneMutableValue(item)
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
