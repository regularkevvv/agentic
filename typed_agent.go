package agentic

import (
	"context"

	"github.com/regularkevvv/agentic/internal/core"
)

type typedValidatorAdapter[O any] func(context.Context, dependencyEnvelope, O) error

type typedRuntime[O any] struct {
	core       *agentCore
	output     OutputSpec[O]
	validators []typedValidatorAdapter[O]
}

// TypedAgent is a structured-output agent without application dependencies.
type TypedAgent[O any] struct {
	runtime *typedRuntime[O]
}

// TypedAgentWithDeps is a structured-output agent carrying exact O and D.
type TypedAgentWithDeps[O, D any] struct {
	runtime *typedRuntime[O]
}

func NewTypedAgent[O any](systemPrompt string, model Model, outputDescription string, opts ...AgentOption) *TypedAgent[O] {
	return NewTypedAgentWithMode(systemPrompt, model, NewToolOutput[O](outputDescription), opts...)
}

func NewTypedAgentWithDeps[O, D any](systemPrompt string, model Model, outputDescription string, opts ...AgentOption) *TypedAgentWithDeps[O, D] {
	return NewTypedAgentWithDepsMode[O, D](systemPrompt, model, NewToolOutput[O](outputDescription), opts...)
}

func NewTypedAgentWithMode[O any](systemPrompt string, model Model, output OutputSpec[O], opts ...AgentOption) *TypedAgent[O] {
	runtime := newTypedRuntime(newAgentCore(systemPrompt, model, opts...), output)
	return &TypedAgent[O]{runtime: runtime}
}

func NewTypedAgentWithDepsMode[O, D any](systemPrompt string, model Model, output OutputSpec[O], opts ...AgentOption) *TypedAgentWithDeps[O, D] {
	runtime := newTypedRuntime(newAgentCoreWithDeps[D](systemPrompt, model, opts...), output)
	return &TypedAgentWithDeps[O, D]{runtime: runtime}
}

func NewTypedAgentDynamic[O any](promptFn DynamicPrompt, model Model, outputDescription string, opts ...AgentOption) *TypedAgent[O] {
	agent := NewTypedAgent[O]("", model, outputDescription, opts...)
	agent.runtime.core.dynamicPrompt = func(ctx context.Context, _ dependencyEnvelope) (string, error) {
		return promptFn(ctx)
	}
	return agent
}

func NewTypedAgentWithDepsDynamic[O, D any](promptFn DynamicPromptWithDeps[D], model Model, outputDescription string, opts ...AgentOption) *TypedAgentWithDeps[O, D] {
	agent := NewTypedAgentWithDeps[O, D]("", model, outputDescription, opts...)
	agent.SetDynamicPrompt(promptFn)
	return agent
}

func newTypedRuntime[O any](c *agentCore, output OutputSpec[O]) *typedRuntime[O] {
	configureAgentOutput(c, output)
	return &typedRuntime[O]{core: c, output: output}
}

func configureAgentOutput[O any](c *agentCore, output OutputSpec[O]) {
	if responseSpec, ok := any(output).(ResponseFormatSpec); ok {
		c.responseFormat = responseSpec.ResponseFormat()
	}
	if appender, ok := any(output).(SystemPromptAppender); ok {
		c.systemPromptSuffix = appender.SystemPromptSuffix()
	}
	tools := output.Tools()
	if len(tools) == 0 {
		return
	}
	if c.outputToolNames == nil {
		c.outputToolNames = make(map[string]bool)
	}
	for _, definition := range tools {
		name := definition.Function.Name
		c.outputToolNames[name] = true
		c.addTool(definition, &noopToolHandler{name: name})
	}
}

func (a *TypedAgent[O]) Run(ctx context.Context, prompt string, opts ...RunOption) (*Result[O], error) {
	message := NewTextMessage(RoleUser, prompt)
	execution, err := a.Drive(ctx, DriveInput{Mode: DriveStart, Prompt: &message}, opts...)
	if execution == nil {
		return nil, err
	}
	return execution.Result, executionError(execution, err)
}

func (a *TypedAgentWithDeps[O, D]) Run(ctx context.Context, prompt string, deps D, opts ...RunOption) (*Result[O], error) {
	message := NewTextMessage(RoleUser, prompt)
	execution, err := a.Drive(ctx, DriveInput{Mode: DriveStart, Prompt: &message}, deps, opts...)
	if execution == nil {
		return nil, err
	}
	return execution.Result, executionError(execution, err)
}

func (a *TypedAgent[O]) RunStream(ctx context.Context, prompt string, opts ...RunOption) (*StreamResult, error) {
	return a.runtime.runStream(ctx, prompt, dependencyEnvelope{}, opts...)
}

func (a *TypedAgentWithDeps[O, D]) RunStream(ctx context.Context, prompt string, deps D, opts ...RunOption) (*StreamResult, error) {
	return a.runtime.runStream(ctx, prompt, core.NewDependencyEnvelope(deps), opts...)
}

// Drive starts or continues a dependency-free typed execution.
func (a *TypedAgent[O]) Drive(ctx context.Context, input DriveInput, opts ...RunOption) (*Execution[O], error) {
	return a.runtime.drive(ctx, input, dependencyEnvelope{}, opts...)
}

// Resume completes a suspended dependency-free typed execution.
func (a *TypedAgent[O]) Resume(ctx context.Context, input ResumeInput, opts ...RunOption) (*Execution[O], error) {
	return a.runtime.resume(ctx, input, dependencyEnvelope{}, opts...)
}

// Drive starts or continues a dependency-aware typed execution with its exact
// dependency value.
func (a *TypedAgentWithDeps[O, D]) Drive(ctx context.Context, input DriveInput, deps D, opts ...RunOption) (*Execution[O], error) {
	return a.runtime.drive(ctx, input, core.NewDependencyEnvelope(deps), opts...)
}

// Resume completes a suspended dependency-aware typed execution with its exact
// dependency value.
func (a *TypedAgentWithDeps[O, D]) Resume(ctx context.Context, input ResumeInput, deps D, opts ...RunOption) (*Execution[O], error) {
	return a.runtime.resume(ctx, input, core.NewDependencyEnvelope(deps), opts...)
}

func (a *TypedAgentWithDeps[O, D]) Bind(deps D) Runner[O] {
	return newBoundRunner(a.RunStream, a.Drive, a.Resume, func(context.Context) (D, error) { return deps, nil })
}

func (a *TypedAgentWithDeps[O, D]) BindProvider(provider DepsProvider[D]) Runner[O] {
	return newBoundRunner(a.RunStream, a.Drive, a.Resume, provider)
}

func (a *TypedAgent[O]) AddOutputValidator(v TypedOutputValidator[O]) *TypedAgent[O] {
	a.runtime.validators = append(a.runtime.validators, func(ctx context.Context, _ dependencyEnvelope, output O) error {
		return v.ValidateTyped(ctx, output)
	})
	return a
}

func (a *TypedAgentWithDeps[O, D]) AddOutputValidator(v TypedOutputValidator[O]) *TypedAgentWithDeps[O, D] {
	a.runtime.validators = append(a.runtime.validators, func(ctx context.Context, _ dependencyEnvelope, output O) error {
		return v.ValidateTyped(ctx, output)
	})
	return a
}

func (a *TypedAgentWithDeps[O, D]) AddOutputValidatorWithDeps(v TypedOutputValidatorWithDeps[D, O]) *TypedAgentWithDeps[O, D] {
	a.runtime.validators = append(a.runtime.validators, adaptTypedValidator(v))
	return a
}

func adaptTypedValidator[D, O any](v TypedOutputValidatorWithDeps[D, O]) typedValidatorAdapter[O] {
	return func(ctx context.Context, envelope dependencyEnvelope, output O) error {
		deps, err := core.ExtractDependency[D](envelope)
		if err != nil {
			return err
		}
		return v.ValidateTyped(RunContext[D]{Ctx: ctx, Deps: deps}, output)
	}
}

func (a *TypedAgentWithDeps[O, D]) SetDynamicPrompt(fn DynamicPromptWithDeps[D]) *TypedAgentWithDeps[O, D] {
	a.runtime.core.dynamicPrompt = adaptDynamicPrompt(fn)
	return a
}

func (a *TypedAgentWithDeps[O, D]) SetDepsValidator(fn DepsValidator[D]) *TypedAgentWithDeps[O, D] {
	a.runtime.core.dependencyPlan.validator = adaptDepsValidator(fn)
	return a
}

func (a *TypedAgentWithDeps[O, D]) AddTextOutputValidator(v OutputValidatorWithDeps[D]) *TypedAgentWithDeps[O, D] {
	a.runtime.core.config.outputValidators = append(a.runtime.core.config.outputValidators, adaptTextValidator(v))
	return a
}

func (a *TypedAgentWithDeps[O, D]) SetToolPrepare(fn ToolPrepareFuncWithDeps[D]) *TypedAgentWithDeps[O, D] {
	a.runtime.core.config.toolPrepareFunc = adaptToolPrepare(fn)
	return a
}

func (r *typedRuntime[O]) drive(ctx context.Context, input DriveInput, deps dependencyEnvelope, opts ...RunOption) (*Execution[O], error) {
	runOpts := r.driverOptions(opts)
	return driveWithEvaluator(r.core, ctx, input, deps, r.completionEvaluator(), runOpts...)
}

func (r *typedRuntime[O]) resume(ctx context.Context, input ResumeInput, deps dependencyEnvelope, opts ...RunOption) (*Execution[O], error) {
	runOpts := r.driverOptions(opts)
	return resumeWithEvaluator(r.core, ctx, input, deps, r.completionEvaluator(), runOpts...)
}

func (r *typedRuntime[O]) runStream(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*StreamResult, error) {
	runOpts := r.driverOptions(opts)
	return runStreamWithEvaluator(r.core, ctx, prompt, deps, r.completionEvaluator(), runOpts...)
}

func (r *typedRuntime[O]) driverOptions(opts []RunOption) []RunOption {
	runOpts := append([]RunOption(nil), opts...)
	if r.outputMode() == OutputModeTool {
		runOpts = append(runOpts, WithRunToolChoice(ToolChoiceRequired))
	}
	return runOpts
}

func (r *typedRuntime[O]) completionEvaluator() completionEvaluator[O] {
	return func(ctx context.Context, deps dependencyEnvelope, message Message) (O, error) {
		output, err := r.output.Parse(message)
		if err != nil {
			var zero O
			return zero, err
		}
		for _, validator := range r.validators {
			if err := validator(ctx, deps, output); err != nil {
				var zero O
				if IsValidationError(err) {
					return zero, err
				}
				return zero, NewValidationError(err.Error())
			}
		}
		return output, nil
	}
}

func (r *typedRuntime[O]) outputMode() OutputMode {
	if spec, ok := any(r.output).(interface{ Mode() OutputMode }); ok {
		return spec.Mode()
	}
	return OutputModeTool
}

func (a *TypedAgent[O]) AddTool(tool Tool, handler ToolHandler) *TypedAgent[O] {
	a.runtime.core.addTool(tool, handler)
	return a
}

func (a *TypedAgentWithDeps[O, D]) AddTool(tool Tool, handler ToolHandler) *TypedAgentWithDeps[O, D] {
	a.runtime.core.addTool(tool, handler)
	return a
}

func (a *TypedAgent[O]) AddAutoTool(tool Tool, handler ToolHandler) *TypedAgent[O] {
	return a.AddTool(tool, handler)
}

func (a *TypedAgentWithDeps[O, D]) AddAutoTool(tool Tool, handler ToolHandler) *TypedAgentWithDeps[O, D] {
	return a.AddTool(tool, handler)
}

func (a *TypedAgent[O]) AddToolset(set Toolset) *TypedAgent[O] {
	addToolset(a.runtime.core, set)
	return a
}

func (a *TypedAgentWithDeps[O, D]) AddToolset(set Toolset) *TypedAgentWithDeps[O, D] {
	addToolset(a.runtime.core, set)
	return a
}

func (a *TypedAgent[O]) SetRegistry(registry ToolRegistry) *TypedAgent[O] {
	a.runtime.core.registry = registry
	registerOutputTools(a.runtime.core, a.runtime.output)
	return a
}

func (a *TypedAgentWithDeps[O, D]) SetRegistry(registry ToolRegistry) *TypedAgentWithDeps[O, D] {
	a.runtime.core.registry = registry
	registerOutputTools(a.runtime.core, a.runtime.output)
	return a
}

func registerOutputTools[O any](c *agentCore, output OutputSpec[O]) {
	for _, definition := range output.Tools() {
		name := definition.Function.Name
		c.addTool(definition, &noopToolHandler{name: name})
	}
}

func (a *TypedAgent[O]) registerTool(tool Tool, handler ToolHandler) {
	a.runtime.core.addTool(tool, handler)
}

func (a *TypedAgentWithDeps[O, D]) registerTool(tool Tool, handler ToolHandler) {
	a.runtime.core.addTool(tool, handler)
}

func (a *TypedAgentWithDeps[O, D]) dependencyType(D) {}

var _ StreamingRunner[struct{}] = (*TypedAgent[struct{}])(nil)
var _ Driver[struct{}] = (*TypedAgent[struct{}])(nil)
