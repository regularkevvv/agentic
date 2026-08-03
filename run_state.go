package agentic

import (
	"context"
	"fmt"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
	agentictool "github.com/regularkevvv/agentic/tool"
)

// loopState contains all mutable state for one execution. It deliberately
// remains non-generic; typed output is introduced only by the completion
// evaluator used by the shared fold.
type loopState struct {
	ctx              context.Context
	deps             dependencyEnvelope
	messages         []Message
	historyLen       int
	maxIterations    int
	usageLimits      *UsageLimits
	endStrategy      EndStrategy
	historyProcessor HistoryProcessor
	options          *runOptions
	registry         *executionToolRegistry

	allToolCalls      []ToolUse
	allToolResults    []ToolExecutionResult
	totalUsage        Usage
	retryCounts       map[string]int
	totalRetries      int
	validationRetries int
	iteration         int
	lastFinishReason  FinishReason

	configFingerprint string
}

// executionToolRegistry routes a per-run immutable overlay without mutating
// the agent's shared registry. It preserves custom registry execution for
// static tools and uses a normal registry only for overlay tools.
type executionToolRegistry struct {
	tools       []Tool
	byName      map[string]Tool
	owner       map[string]ToolRegistry
	hasExecutor bool
	base        ToolRegistry
}

func buildExecutionToolRegistry(base ToolRegistry, sets []Toolset) (*executionToolRegistry, error) {
	registry := &executionToolRegistry{
		byName: make(map[string]Tool),
		owner:  make(map[string]ToolRegistry),
	}

	if base != nil {
		registry.hasExecutor = true
		registry.base = base
		for _, definition := range base.Tools() {
			name := definition.Function.Name
			if name == "" {
				return nil, fmt.Errorf("agent registry contains a tool with an empty name")
			}
			if _, exists := registry.byName[name]; exists {
				return nil, fmt.Errorf("agent registry contains duplicate tool %q", name)
			}
			registry.byName[name] = definition
			registry.owner[name] = base
			registry.tools = append(registry.tools, definition)
		}
	}

	if len(sets) == 0 {
		return registry, nil
	}
	overlay := agentictool.NewRegistry()
	for _, set := range sets {
		if set == nil {
			return nil, fmt.Errorf("run toolset must not be nil")
		}
		definitions, handlers := set.ToolsAndHandlers()
		if len(definitions) != len(handlers) {
			return nil, fmt.Errorf("run toolset returned %d definitions for %d handlers", len(definitions), len(handlers))
		}
		for index, definition := range definitions {
			name := definition.Function.Name
			if name == "" {
				return nil, fmt.Errorf("run toolset contains a tool with an empty name")
			}
			if _, exists := registry.byName[name]; exists {
				return nil, fmt.Errorf("run tool overlay duplicates registered tool %q", name)
			}
			if err := overlay.Register(definition, handlers[index]); err != nil {
				return nil, fmt.Errorf("register run tool %q: %w", name, err)
			}
			registry.byName[name] = definition
			registry.owner[name] = overlay
			registry.hasExecutor = true
			registry.tools = append(registry.tools, definition)
		}
	}
	return registry, nil
}

func (r *executionToolRegistry) Count() int {
	if r == nil {
		return 0
	}
	return len(r.tools)
}

func (r *executionToolRegistry) Tools() []Tool {
	if r == nil {
		return nil
	}
	return append([]Tool(nil), r.tools...)
}

func (r *executionToolRegistry) Get(name string) (ToolHandler, bool) {
	if r == nil {
		return nil, false
	}
	owner, ok := r.owner[name]
	if !ok {
		return nil, false
	}
	return owner.Get(name)
}

func (r *executionToolRegistry) Tool(name string) (Tool, bool) {
	if r == nil {
		return Tool{}, false
	}
	definition, ok := r.byName[name]
	return definition, ok
}

func (r *executionToolRegistry) Execute(ctx context.Context, call ToolUse, deps any) (ToolExecutionResult, error) {
	if r == nil {
		return makeToolResultError(call, fmt.Sprintf("Unknown tool: %s", call.Name), nil), nil
	}
	owner, ok := r.owner[call.Name]
	if !ok {
		if r.base != nil {
			return r.base.Execute(ctx, call, deps)
		}
		return makeToolResultError(call, fmt.Sprintf("Unknown tool: %s", call.Name), nil), nil
	}
	return owner.Execute(ctx, call, deps)
}

func (c *agentCore) prepareLoop(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*loopState, error) {
	if err := c.preflight(ctx, deps); err != nil {
		return nil, err
	}
	return c.prepareLoopAfterPreflight(ctx, prompt, deps, opts...)
}

func (c *agentCore) prepareLoopAfterPreflight(ctx context.Context, prompt string, deps dependencyEnvelope, opts ...RunOption) (*loopState, error) {
	message := NewTextMessage(RoleUser, prompt)
	return c.prepareLoopForDrive(ctx, DriveInput{Mode: DriveStart, Prompt: &message}, deps, opts...)
}

func (c *agentCore) prepareLoopForDrive(ctx context.Context, input DriveInput, deps dependencyEnvelope, opts ...RunOption) (*loopState, error) {
	options := applyRunOptions(opts)
	history := append(cloneMessages(input.History), cloneMessages(options.messages)...)
	if err := validateDriveInput(input, history); err != nil {
		return nil, err
	}
	var prompt *Message
	if input.Mode == DriveStart {
		promptCopy := cloneMessages([]Message{*input.Prompt})[0]
		prompt = &promptCopy
	}
	return c.prepareLoopWithHistory(ctx, history, len(history), prompt, deps, options)
}

func (c *agentCore) prepareLoopForResume(ctx context.Context, history []Message, deps dependencyEnvelope, opts ...RunOption) (*loopState, error) {
	options := applyRunOptions(opts)
	if len(options.messages) > 0 {
		return nil, fmt.Errorf("%w: WithMessages is not valid with Resume", ErrDriveInput)
	}
	history = cloneMessages(history)
	if len(history) == 0 {
		return nil, fmt.Errorf("%w: Resume requires non-empty history", ErrDriveInput)
	}
	if _, err := inspectTranscript(history); err != nil {
		return nil, err
	}
	return c.prepareLoopWithHistory(ctx, history, len(history), nil, deps, options)
}

func (c *agentCore) prepareLoopWithHistory(
	ctx context.Context,
	history []Message,
	historyLen int,
	prompt *Message,
	deps dependencyEnvelope,
	options *runOptions,
) (*loopState, error) {

	systemPrompt, err := c.resolveSystemPrompt(ctx, deps)
	if err != nil {
		return nil, fmt.Errorf("system prompt: %w", err)
	}
	if c.systemPromptSuffix != "" {
		if systemPrompt != "" {
			systemPrompt += "\n\n" + c.systemPromptSuffix
		} else {
			systemPrompt = c.systemPromptSuffix
		}
	}

	hasSystemPrompt := false
	for _, message := range history {
		if message.Role == RoleSystem {
			hasSystemPrompt = true
			break
		}
	}
	if !hasSystemPrompt && systemPrompt != "" {
		history = append([]Message{NewTextMessage(RoleSystem, systemPrompt)}, history...)
	}
	if prompt != nil {
		history = append(history, *prompt)
	}

	registry, err := buildExecutionToolRegistry(c.registry, options.toolsets)
	if err != nil {
		return nil, err
	}

	maxIterations := c.config.maxIterations
	if options.maxIterations != nil {
		maxIterations = *options.maxIterations
	}
	usageLimits := c.config.usageLimits
	if options.usageLimits != nil {
		usageLimits = options.usageLimits
	}
	endStrategy := c.config.endStrategy
	if options.endStrategy != nil {
		endStrategy = *options.endStrategy
	}
	historyProcessor := c.config.historyProcessor
	if options.historyProcessor != nil {
		historyProcessor = options.historyProcessor
	}
	if options.toolCancellationGrace == nil {
		grace := time.Second
		options.toolCancellationGrace = &grace
	}

	calls, results := transcriptToolState(history)
	state := &loopState{
		ctx:              ctx,
		deps:             deps,
		messages:         history,
		historyLen:       historyLen,
		maxIterations:    maxIterations,
		usageLimits:      usageLimits,
		endStrategy:      endStrategy,
		historyProcessor: historyProcessor,
		options:          options,
		registry:         registry,
		allToolCalls:     calls,
		allToolResults:   results,
	}
	state.configFingerprint = c.executionFingerprint(state)
	return state, nil
}

func (c *agentCore) buildRequest(ls *loopState, stream bool) (*ChatRequest, error) {
	requestMessages := ls.messages
	if ls.historyProcessor != nil {
		var err error
		requestMessages, err = ls.historyProcessor.Process(ls.ctx, ls.messages)
		if err != nil {
			return nil, fmt.Errorf("history processor: %w", err)
		}
	}
	request := &ChatRequest{
		Model:       c.model.Name(),
		Messages:    requestMessages,
		Temperature: firstNonNil(ls.options.temperature, c.config.temperature),
		MaxTokens:   firstNonNil(ls.options.maxTokens, c.config.maxTokens),
		TopP:        firstNonNil(ls.options.topP, c.config.topP),
		Stream:      stream,
		PromptCache: clonePromptCache(firstNonNil(ls.options.promptCache, c.config.promptCache)),
	}
	if ls.registry.Count() > 0 {
		availableTools := ls.registry.Tools()
		if c.config.toolPrepareFunc != nil {
			var err error
			availableTools, err = c.config.toolPrepareFunc(ls.ctx, ls.deps, availableTools)
			if err != nil {
				return nil, fmt.Errorf("tool prepare: %w", err)
			}
		}
		request.Tools = availableTools
		request.ToolChoice = firstNonNil(ls.options.toolChoice, c.config.toolChoice)
	}
	request.ResponseFormat = c.responseFormat
	request.Thinking = c.config.thinking
	return request, nil
}

func clonePromptCache(value *PromptCacheConfig) *PromptCacheConfig {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func (ls *loopState) checkPreRequestLimits() error {
	if ls.usageLimits == nil {
		return nil
	}
	return ls.usageLimits.checkBeforeRequest(ls.totalUsage)
}

func (ls *loopState) checkPostResponseLimits() error {
	if ls.usageLimits == nil {
		return nil
	}
	return ls.usageLimits.checkAfterResponse(ls.totalUsage)
}

func (ls *loopState) checkToolCallLimits(pendingCalls int) error {
	if ls.usageLimits == nil {
		return nil
	}
	return ls.usageLimits.checkBeforeToolCalls(ls.totalUsage.ToolCalls, pendingCalls)
}

func (c *agentCore) resolveMaxRetries(handler ToolHandler, globalDefault int) int {
	if configurable, ok := handler.(interface{ ToolConfig() *ToolConfig }); ok {
		if cfg := configurable.ToolConfig(); cfg != nil && cfg.MaxRetries != nil {
			return *cfg.MaxRetries
		}
	}
	return globalDefault
}

func addToolExecutionState(ctx context.Context, ls *loopState) context.Context {
	var retryCounts map[string]int
	if len(ls.retryCounts) > 0 {
		retryCounts = make(map[string]int, len(ls.retryCounts))
		for name, count := range ls.retryCounts {
			retryCounts[name] = count
		}
	}
	parent := ls.messages
	if len(parent) > 0 {
		parent = parent[:len(parent)-1]
	}
	return core.WithToolExecutionState(ctx, core.ToolExecutionState{
		Messages:    cloneMessages(parent),
		RetryCounts: retryCounts,
	})
}

func validateResumeInput(ctx context.Context, call ToolUse, definition Tool, input map[string]any) error {
	if err := agentictool.ValidateInput(input, definition); err != nil {
		return fmt.Errorf("%w: override input for tool %q: %v", ErrResumeDecision, call.ID, err)
	}
	return nil
}
