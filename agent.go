package agentic

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"
)

// Runner is a ready-to-run agent capability. Dependencies, if any, have
// already been supplied.
type Runner[O any] interface {
	Run(context.Context, string, ...RunOption) (*Result[O], error)
}

// StreamRunner is the independent streaming capability.
type StreamRunner interface {
	RunStream(context.Context, string, ...RunOption) (*StreamResult, error)
}

// StreamingRunner combines typed execution and streaming.
type StreamingRunner[O any] interface {
	Runner[O]
	StreamRunner
}

// DynamicPrompt is a dependency-free prompt callback.
type DynamicPrompt func(context.Context) (string, error)

// DynamicPromptWithDeps is a prompt callback connected to an exact D.
type DynamicPromptWithDeps[D any] func(RunContext[D]) (string, error)

// DepsProvider obtains dependencies once at the beginning of each run.
type DepsProvider[D any] func(context.Context) (D, error)

// DepsValidator checks application invariants before any run side effect.
type DepsValidator[D any] func(context.Context, D) error

// ToolPrepareFuncWithDeps customizes tools with access to exact run deps.
type ToolPrepareFuncWithDeps[D any] func(RunContext[D], []Tool) ([]Tool, error)

type promptAdapter func(context.Context, dependencyEnvelope) (string, error)

type dependencyPlan struct {
	required  bool
	typ       reflect.Type
	validator func(context.Context, dependencyEnvelope) error
}

// agentCore is the one non-generic execution definition shared by every
// facade. It never stores a per-run dependency value.
type agentCore struct {
	model              Model
	systemPrompt       string
	dynamicPrompt      promptAdapter
	registry           ToolRegistry
	config             agentConfig
	outputToolNames    map[string]bool
	responseFormat     *ResponseFormat
	systemPromptSuffix string
	dependencyPlan     dependencyPlan
}

func newAgentCore(systemPrompt string, model Model, opts ...AgentOption) *agentCore {
	config := defaultAgentConfig()
	for _, opt := range opts {
		opt(&config)
	}
	return &agentCore{model: model, systemPrompt: systemPrompt, config: config}
}

func newAgentCoreWithDeps[D any](systemPrompt string, model Model, opts ...AgentOption) *agentCore {
	c := newAgentCore(systemPrompt, model, opts...)
	c.dependencyPlan = dependencyPlan{required: true, typ: reflect.TypeFor[D]()}
	return c
}

// Agent is a non-generic text agent without application dependencies.
type Agent struct {
	core *agentCore
}

// AgentWithDeps is a text agent whose run graph carries exact D.
type AgentWithDeps[D any] struct {
	core *agentCore
}

func NewAgent(systemPrompt string, model Model, opts ...AgentOption) *Agent {
	return &Agent{core: newAgentCore(systemPrompt, model, opts...)}
}

func NewAgentDynamic(promptFn DynamicPrompt, model Model, opts ...AgentOption) *Agent {
	a := NewAgent("", model, opts...)
	a.core.dynamicPrompt = func(ctx context.Context, _ dependencyEnvelope) (string, error) {
		return promptFn(ctx)
	}
	return a
}

func NewAgentWithDeps[D any](systemPrompt string, model Model, opts ...AgentOption) *AgentWithDeps[D] {
	return &AgentWithDeps[D]{core: newAgentCoreWithDeps[D](systemPrompt, model, opts...)}
}

func NewAgentWithDepsDynamic[D any](promptFn DynamicPromptWithDeps[D], model Model, opts ...AgentOption) *AgentWithDeps[D] {
	a := NewAgentWithDeps[D]("", model, opts...)
	a.SetDynamicPrompt(promptFn)
	return a
}

func (a *Agent) Run(ctx context.Context, prompt string, opts ...RunOption) (*Result[string], error) {
	return a.core.run(ctx, prompt, dependencyEnvelope{}, opts...)
}

func (a *AgentWithDeps[D]) Run(ctx context.Context, prompt string, deps D, opts ...RunOption) (*Result[string], error) {
	return a.core.run(ctx, prompt, core.NewDependencyEnvelope(deps), opts...)
}

func (a *AgentWithDeps[D]) Bind(deps D) Runner[string] {
	return newBoundRunner(a.Run, a.RunStream, func(context.Context) (D, error) { return deps, nil })
}

func (a *AgentWithDeps[D]) BindProvider(provider DepsProvider[D]) Runner[string] {
	return newBoundRunner(a.Run, a.RunStream, provider)
}

func (a *AgentWithDeps[D]) SetDynamicPrompt(fn DynamicPromptWithDeps[D]) *AgentWithDeps[D] {
	a.core.dynamicPrompt = adaptDynamicPrompt(fn)
	return a
}

func (a *AgentWithDeps[D]) SetDepsValidator(fn DepsValidator[D]) *AgentWithDeps[D] {
	a.core.dependencyPlan.validator = adaptDepsValidator(fn)
	return a
}

func (a *AgentWithDeps[D]) AddOutputValidator(v OutputValidatorWithDeps[D]) *AgentWithDeps[D] {
	a.core.config.outputValidators = append(a.core.config.outputValidators, adaptTextValidator(v))
	return a
}

func (a *AgentWithDeps[D]) SetToolPrepare(fn ToolPrepareFuncWithDeps[D]) *AgentWithDeps[D] {
	a.core.config.toolPrepareFunc = adaptToolPrepare(fn)
	return a
}

func adaptDynamicPrompt[D any](fn DynamicPromptWithDeps[D]) promptAdapter {
	return func(ctx context.Context, envelope dependencyEnvelope) (string, error) {
		deps, err := core.ExtractDependency[D](envelope)
		if err != nil {
			return "", err
		}
		return fn(RunContext[D]{Ctx: ctx, Deps: deps})
	}
}

func adaptDepsValidator[D any](fn DepsValidator[D]) func(context.Context, dependencyEnvelope) error {
	return func(ctx context.Context, envelope dependencyEnvelope) error {
		deps, err := core.ExtractDependency[D](envelope)
		if err != nil {
			return err
		}
		return fn(ctx, deps)
	}
}

func adaptTextValidator[D any](v OutputValidatorWithDeps[D]) textValidatorAdapter {
	return func(ctx context.Context, envelope dependencyEnvelope, output string) error {
		deps, err := core.ExtractDependency[D](envelope)
		if err != nil {
			return err
		}
		return v.Validate(RunContext[D]{Ctx: ctx, Deps: deps}, output)
	}
}

func adaptToolPrepare[D any](fn ToolPrepareFuncWithDeps[D]) toolPrepareAdapter {
	return func(ctx context.Context, envelope dependencyEnvelope, tools []Tool) ([]Tool, error) {
		deps, err := core.ExtractDependency[D](envelope)
		if err != nil {
			return nil, err
		}
		return fn(RunContext[D]{Ctx: ctx, Deps: deps}, tools)
	}
}

func (c *agentCore) preflight(ctx context.Context, deps dependencyEnvelope) error {
	if !c.dependencyPlan.required {
		if deps.DependencyType() != nil {
			return fmt.Errorf("dependency type mismatch: agent does not accept dependencies")
		}
		return nil
	}
	if deps.DependencyType() != c.dependencyPlan.typ {
		return fmt.Errorf("dependency type mismatch: expected %v, got %v", c.dependencyPlan.typ, deps.DependencyType())
	}
	if deps.DependencyIsNil() {
		return ErrNilDeps
	}
	if c.dependencyPlan.validator != nil {
		if err := c.dependencyPlan.validator(ctx, deps); err != nil {
			return fmt.Errorf("validate dependencies: %w", err)
		}
	}
	return nil
}

func (c *agentCore) resolveSystemPrompt(ctx context.Context, deps dependencyEnvelope) (string, error) {
	if len(c.config.systemPrompts) > 0 {
		parts := make([]string, 0, len(c.config.systemPrompts))
		for _, prompt := range c.config.systemPrompts {
			if prompt != "" {
				parts = append(parts, prompt)
			}
		}
		return strings.Join(parts, "\n\n"), nil
	}
	if c.dynamicPrompt != nil {
		return c.dynamicPrompt(ctx, deps)
	}
	return c.systemPrompt, nil
}

func (c *agentCore) validateOutput(ctx context.Context, deps dependencyEnvelope, output string) error {
	for _, validator := range c.config.outputValidators {
		if err := validator(ctx, deps, output); err != nil {
			return err
		}
	}
	return nil
}

func (c *agentCore) resolveMaxRetries(handler ToolHandler, globalDefault int) int {
	if configurable, ok := handler.(interface{ ToolConfig() *ToolConfig }); ok {
		if cfg := configurable.ToolConfig(); cfg != nil && cfg.MaxRetries != nil {
			return *cfg.MaxRetries
		}
	}
	return globalDefault
}

func (c *agentCore) addTool(tool Tool, handler ToolHandler) {
	if c.registry == nil {
		c.registry = NewRegistry()
	}
	if err := c.registry.Register(tool, handler); err != nil {
		panic(fmt.Sprintf("failed to register tool: %v", err))
	}
}

func (a *Agent) AddTool(tool Tool, handler ToolHandler) *Agent {
	a.core.addTool(tool, handler)
	return a
}

func (a *AgentWithDeps[D]) AddTool(tool Tool, handler ToolHandler) *AgentWithDeps[D] {
	a.core.addTool(tool, handler)
	return a
}

func (a *Agent) AddAutoTool(tool Tool, handler ToolHandler) *Agent {
	return a.AddTool(tool, handler)
}

func (a *AgentWithDeps[D]) AddAutoTool(tool Tool, handler ToolHandler) *AgentWithDeps[D] {
	return a.AddTool(tool, handler)
}

func (a *Agent) AddToolset(set Toolset) *Agent {
	addToolset(a.core, set)
	return a
}

func (a *AgentWithDeps[D]) AddToolset(set Toolset) *AgentWithDeps[D] {
	addToolset(a.core, set)
	return a
}

func addToolset(c *agentCore, set Toolset) {
	tools, handlers := set.ToolsAndHandlers()
	for i := range tools {
		c.addTool(tools[i], handlers[i])
	}
}

func (a *Agent) SetRegistry(registry ToolRegistry) *Agent {
	a.core.registry = registry
	return a
}

func (a *AgentWithDeps[D]) SetRegistry(registry ToolRegistry) *AgentWithDeps[D] {
	a.core.registry = registry
	return a
}

func (a *Agent) SetOutputToolNames(names map[string]bool) *Agent {
	a.core.outputToolNames = copyNameSet(names)
	return a
}

func (a *AgentWithDeps[D]) SetOutputToolNames(names map[string]bool) *AgentWithDeps[D] {
	a.core.outputToolNames = copyNameSet(names)
	return a
}

func copyNameSet(names map[string]bool) map[string]bool {
	copySet := make(map[string]bool, len(names))
	for name, enabled := range names {
		copySet[name] = enabled
	}
	return copySet
}

func (a *Agent) registerTool(tool Tool, handler ToolHandler) { a.core.addTool(tool, handler) }
func (a *AgentWithDeps[D]) registerTool(tool Tool, handler ToolHandler) {
	a.core.addTool(tool, handler)
}
func (a *AgentWithDeps[D]) dependencyType(D) {}

func firstNonNil[T any](values ...*T) *T {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

type boundRunner[O, D any] struct {
	run      func(context.Context, string, D, ...RunOption) (*Result[O], error)
	stream   func(context.Context, string, D, ...RunOption) (*StreamResult, error)
	provider DepsProvider[D]
}

func newBoundRunner[O, D any](
	run func(context.Context, string, D, ...RunOption) (*Result[O], error),
	stream func(context.Context, string, D, ...RunOption) (*StreamResult, error),
	provider DepsProvider[D],
) *boundRunner[O, D] {
	return &boundRunner[O, D]{run: run, stream: stream, provider: provider}
}

func (b *boundRunner[O, D]) Run(ctx context.Context, prompt string, opts ...RunOption) (*Result[O], error) {
	deps, err := b.provider(ctx)
	if err != nil {
		return nil, fmt.Errorf("dependency provider: %w", err)
	}
	return b.run(ctx, prompt, deps, opts...)
}

func (b *boundRunner[O, D]) RunStream(ctx context.Context, prompt string, opts ...RunOption) (*StreamResult, error) {
	deps, err := b.provider(ctx)
	if err != nil {
		return nil, fmt.Errorf("dependency provider: %w", err)
	}
	return b.stream(ctx, prompt, deps, opts...)
}

var _ StreamingRunner[string] = (*Agent)(nil)
var _ StreamingRunner[string] = (*boundRunner[string, struct{}])(nil)
