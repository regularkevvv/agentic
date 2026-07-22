package agentic

import (
	"context"
	"time"
)

// AgentOption configures static agent behavior. Options that consume a
// dependency or typed output attach through the corresponding typed facade.
type AgentOption func(*agentConfig)

type agentConfig struct {
	temperature          *float64
	maxTokens            *int
	topP                 *float64
	maxIterations        int
	toolChoice           *ToolChoice
	retryConfig          RetryConfig
	systemPrompts        []string
	outputValidators     []textValidatorAdapter
	maxValidationRetries int
	usageLimits          *UsageLimits
	thinking             *ThinkingConfig
	endStrategy          EndStrategy
	toolPrepareFunc      toolPrepareAdapter
	historyProcessor     HistoryProcessor
}

func defaultAgentConfig() agentConfig {
	return agentConfig{
		maxIterations:        10,
		retryConfig:          RetryConfig{MaxRetries: 1},
		maxValidationRetries: 3,
	}
}

func WithTemperature(temp float64) AgentOption {
	return func(c *agentConfig) { c.temperature = &temp }
}

func WithMaxTokens(maxTokens int) AgentOption {
	return func(c *agentConfig) { c.maxTokens = &maxTokens }
}

func WithTopP(topP float64) AgentOption {
	return func(c *agentConfig) { c.topP = &topP }
}

func WithMaxIterations(maxIter int) AgentOption {
	return func(c *agentConfig) { c.maxIterations = maxIter }
}

func WithToolChoice(choice ToolChoice) AgentOption {
	return func(c *agentConfig) { c.toolChoice = &choice }
}

// RetryConfig controls retries requested by tools.
type RetryConfig struct {
	MaxRetries int
}

func WithRetries(config RetryConfig) AgentOption {
	return func(c *agentConfig) { c.retryConfig = config }
}

// WithSystemPrompts replaces the constructor prompt with static prompt
// segments joined by a blank line.
func WithSystemPrompts(prompts ...string) AgentOption {
	return func(c *agentConfig) { c.systemPrompts = append([]string(nil), prompts...) }
}

// WithOutputValidator adds a dependency-free text output validator.
func WithOutputValidator(v OutputValidator) AgentOption {
	return func(c *agentConfig) {
		c.outputValidators = append(c.outputValidators, func(ctx context.Context, _ dependencyEnvelope, output string) error {
			return v.Validate(ctx, output)
		})
	}
}

func WithOutputValidatorFunc(fn func(context.Context, string) error) AgentOption {
	return WithOutputValidator(OutputValidatorFunc(fn))
}

func WithMaxValidationRetries(n int) AgentOption {
	return func(c *agentConfig) { c.maxValidationRetries = n }
}

func WithThinking(config ThinkingConfig) AgentOption {
	return func(c *agentConfig) { c.thinking = &config }
}

func WithUsageLimits(limits UsageLimits) AgentOption {
	return func(c *agentConfig) { c.usageLimits = &limits }
}

// EndStrategy controls completion when one response contains multiple tools.
type EndStrategy int

const (
	EndStrategyExhaustive EndStrategy = iota
	EndStrategyEarly
)

func WithEndStrategy(strategy EndStrategy) AgentOption {
	return func(c *agentConfig) { c.endStrategy = strategy }
}

// ToolPrepareFunc customizes the tools sent before each model request.
type ToolPrepareFunc func(context.Context, []Tool) ([]Tool, error)

type toolPrepareAdapter func(context.Context, dependencyEnvelope, []Tool) ([]Tool, error)

func WithToolPrepare(fn ToolPrepareFunc) AgentOption {
	return func(c *agentConfig) {
		c.toolPrepareFunc = func(ctx context.Context, _ dependencyEnvelope, tools []Tool) ([]Tool, error) {
			return fn(ctx, tools)
		}
	}
}

func WithHistoryProcessor(p HistoryProcessor) AgentOption {
	return func(c *agentConfig) { c.historyProcessor = p }
}

// RunOption configures one run.
type RunOption func(*runOptions)

type runOptions struct {
	messages              []Message
	temperature           *float64
	maxTokens             *int
	topP                  *float64
	maxIterations         *int
	toolChoice            *ToolChoice
	usageLimits           *UsageLimits
	endStrategy           *EndStrategy
	historyProcessor      HistoryProcessor
	turnHook              TurnHook
	eventSink             EventSink
	toolsets              []Toolset
	toolGate              ToolGate
	toolResultProcessor   ToolResultProcessor
	modelStreaming        *bool
	toolCancellationGrace *time.Duration
}

func WithMessages(messages ...Message) RunOption {
	return func(o *runOptions) { o.messages = append(o.messages, messages...) }
}

func WithRunTemperature(temp float64) RunOption {
	return func(o *runOptions) { o.temperature = &temp }
}

func WithRunMaxTokens(maxTokens int) RunOption {
	return func(o *runOptions) { o.maxTokens = &maxTokens }
}

func WithRunMaxIterations(maxIter int) RunOption {
	return func(o *runOptions) { o.maxIterations = &maxIter }
}

func WithRunToolChoice(choice ToolChoice) RunOption {
	return func(o *runOptions) { o.toolChoice = &choice }
}

func WithRunUsageLimits(limits UsageLimits) RunOption {
	return func(o *runOptions) { o.usageLimits = &limits }
}

func WithRunEndStrategy(strategy EndStrategy) RunOption {
	return func(o *runOptions) { o.endStrategy = &strategy }
}

func WithRunHistoryProcessor(p HistoryProcessor) RunOption {
	return func(o *runOptions) { o.historyProcessor = p }
}

// WithRunTurnHook installs a control hook for this execution only. It is a
// RunOption so bound runners can be controlled after dependencies are bound.
func WithRunTurnHook(hook TurnHook) RunOption {
	return func(o *runOptions) { o.turnHook = hook }
}

// WithRunEventSink installs a synchronous canonical event sink for this
// execution only.
func WithRunEventSink(sink EventSink) RunOption {
	return func(o *runOptions) { o.eventSink = sink }
}

// WithRunToolsets overlays immutable toolsets for this execution. The shared
// agent registry is never mutated.
func WithRunToolsets(toolsets ...Toolset) RunOption {
	return func(o *runOptions) { o.toolsets = append(o.toolsets, toolsets...) }
}

// WithRunToolGate preflights regular tool batches for this execution.
func WithRunToolGate(gate ToolGate) RunOption {
	return func(o *runOptions) { o.toolGate = gate }
}

// WithRunToolResultProcessor projects handler results before commit.
func WithRunToolResultProcessor(processor ToolResultProcessor) RunOption {
	return func(o *runOptions) { o.toolResultProcessor = processor }
}

// WithRunModelStreaming asks Driver to use the model streaming transport when
// it is available. The driver still returns one ordinary Execution.
func WithRunModelStreaming(enabled bool) RunOption {
	return func(o *runOptions) { o.modelStreaming = &enabled }
}

// WithRunToolCancellationGrace bounds how long the scheduler waits for
// already-admitted handlers after execution context cancellation.
func WithRunToolCancellationGrace(grace time.Duration) RunOption {
	return func(o *runOptions) { o.toolCancellationGrace = &grace }
}

func applyRunOptions(opts []RunOption) *runOptions {
	options := &runOptions{}
	for _, opt := range opts {
		opt(options)
	}
	return options
}
