package agentic

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"
)

type HandoffInputFilter int

const (
	HandoffFullHistory HandoffInputFilter = iota
	HandoffLastMessage
	HandoffSummary
)

type handoffConfig struct {
	inputFilter  HandoffInputFilter
	maxTokens    *int
	systemPrompt string
	summaryFunc  HandoffSummaryFunc
}

type HandoffSummaryFunc func(context.Context, []Message) (string, error)
type HandoffOption func(*handoffConfig)

func WithHandoffFilter(filter HandoffInputFilter) HandoffOption {
	return func(c *handoffConfig) { c.inputFilter = filter }
}

func WithHandoffSystemPrompt(prompt string) HandoffOption {
	return func(c *handoffConfig) { c.systemPrompt = prompt }
}

func WithHandoffMaxTokens(n int) HandoffOption {
	return func(c *handoffConfig) { c.maxTokens = &n }
}

func WithHandoffSummaryFunc(fn HandoffSummaryFunc) HandoffOption {
	return func(c *handoffConfig) { c.summaryFunc = fn }
}

// Handoff delegates to a child whose dependencies are already bound.
type Handoff struct {
	name        string
	description string
	run         func(context.Context, string, ...RunOption) (any, error)
	config      handoffConfig
}

func NewHandoff[O any](name, description string, child Runner[O], opts ...HandoffOption) *Handoff {
	return &Handoff{
		name:        name,
		description: description,
		config:      applyHandoffOptions(opts),
		run: func(ctx context.Context, prompt string, runOpts ...RunOption) (any, error) {
			result, err := child.Run(ctx, prompt, runOpts...)
			if err != nil {
				return nil, err
			}
			return result.Output, nil
		},
	}
}

// HandoffWithDeps derives child dependencies from the parent run's exact D.
type HandoffWithDeps[D any] struct {
	name        string
	description string
	run         func(context.Context, dependencyEnvelope, string, ...RunOption) (any, error)
	config      handoffConfig
}

func NewHandoffWithDeps[ParentD, ChildD, O any](
	name, description string,
	child *TypedAgentWithDeps[O, ChildD],
	mapper func(RunContext[ParentD]) (ChildD, error),
	opts ...HandoffOption,
) *HandoffWithDeps[ParentD] {
	return newMappedHandoff(name, description, child.Run, mapper, opts...)
}

// NewTextHandoffWithDeps is the text-output equivalent of NewHandoffWithDeps.
func NewTextHandoffWithDeps[ParentD, ChildD any](
	name, description string,
	child *AgentWithDeps[ChildD],
	mapper func(RunContext[ParentD]) (ChildD, error),
	opts ...HandoffOption,
) *HandoffWithDeps[ParentD] {
	return newMappedHandoff(name, description, child.Run, mapper, opts...)
}

func newMappedHandoff[ParentD, ChildD, O any](
	name, description string,
	run func(context.Context, string, ChildD, ...RunOption) (*Result[O], error),
	mapper func(RunContext[ParentD]) (ChildD, error),
	opts ...HandoffOption,
) *HandoffWithDeps[ParentD] {
	return &HandoffWithDeps[ParentD]{
		name:        name,
		description: description,
		config:      applyHandoffOptions(opts),
		run: func(ctx context.Context, envelope dependencyEnvelope, prompt string, runOpts ...RunOption) (any, error) {
			parentDeps, err := core.ExtractDependency[ParentD](envelope)
			if err != nil {
				return nil, err
			}
			childDeps, err := mapper(RunContext[ParentD]{Ctx: ctx, Deps: parentDeps})
			if err != nil {
				return nil, fmt.Errorf("map handoff dependencies: %w", err)
			}
			result, err := run(ctx, prompt, childDeps, runOpts...)
			if err != nil {
				return nil, err
			}
			return result.Output, nil
		},
	}
}

func NewIdentityHandoff[O, D any](name, description string, child *TypedAgentWithDeps[O, D], opts ...HandoffOption) *HandoffWithDeps[D] {
	return NewHandoffWithDeps(name, description, child, func(ctx RunContext[D]) (D, error) { return ctx.Deps, nil }, opts...)
}

func NewIdentityTextHandoff[D any](name, description string, child *AgentWithDeps[D], opts ...HandoffOption) *HandoffWithDeps[D] {
	return NewTextHandoffWithDeps(name, description, child, func(ctx RunContext[D]) (D, error) { return ctx.Deps, nil }, opts...)
}

func applyHandoffOptions(opts []HandoffOption) handoffConfig {
	config := handoffConfig{inputFilter: HandoffLastMessage}
	for _, opt := range opts {
		opt(&config)
	}
	return config
}

type handoffInput struct {
	Task string `json:"task" description:"The task to delegate to the sub-agent"`
}

type handoffHandler struct {
	handoff *Handoff
}

func (h *handoffHandler) Name() string { return h.handoff.name }

func (h *handoffHandler) Execute(ctx context.Context, input map[string]interface{}, _ any) (interface{}, error) {
	task, runOpts, err := prepareHandoffRun(ctx, input, &h.handoff.config)
	if err != nil {
		return nil, err
	}
	result, err := h.handoff.run(ctx, task, runOpts...)
	if err != nil {
		return nil, fmt.Errorf("handoff to %q: %w", h.handoff.name, err)
	}
	return result, nil
}

type handoffWithDepsHandler[D any] struct {
	handoff *HandoffWithDeps[D]
}

func (h *handoffWithDepsHandler[D]) Name() string { return h.handoff.name }

func (h *handoffWithDepsHandler[D]) Execute(ctx context.Context, input map[string]interface{}, deps any) (interface{}, error) {
	task, runOpts, err := prepareHandoffRun(ctx, input, &h.handoff.config)
	if err != nil {
		return nil, err
	}
	envelope, ok := deps.(dependencyEnvelope)
	if !ok {
		return nil, fmt.Errorf("invalid dependency envelope: got %T", deps)
	}
	result, err := h.handoff.run(ctx, envelope, task, runOpts...)
	if err != nil {
		return nil, fmt.Errorf("handoff to %q: %w", h.handoff.name, err)
	}
	return result, nil
}

func prepareHandoffRun(ctx context.Context, input map[string]interface{}, config *handoffConfig) (string, []RunOption, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return "", nil, fmt.Errorf("marshal input: %w", err)
	}
	var typedInput handoffInput
	if err := json.Unmarshal(data, &typedInput); err != nil {
		return "", nil, fmt.Errorf("unmarshal handoff input: %w", err)
	}
	var runOpts []RunOption
	if config.maxTokens != nil {
		runOpts = append(runOpts, WithRunMaxTokens(*config.maxTokens))
	}
	executionState, _ := core.ToolExecutionStateFromContext(ctx)
	switch config.inputFilter {
	case HandoffFullHistory:
		history := withoutSystemMessages(executionState.Messages)
		if len(history) > 0 {
			runOpts = append(runOpts, WithMessages(history...))
		}
	case HandoffSummary:
		summary, err := summarizeHandoffHistory(ctx, executionState.Messages, config)
		if err != nil {
			return "", nil, fmt.Errorf("summarize handoff history: %w", err)
		}
		if summary != "" {
			typedInput.Task = "Conversation context:\n" + summary + "\n\nTask:\n" + typedInput.Task
		}
	}
	prompt := typedInput.Task
	if config.systemPrompt != "" {
		prompt = config.systemPrompt + "\n\n" + prompt
	}
	return prompt, runOpts, nil
}

func summarizeHandoffHistory(ctx context.Context, messages []Message, config *handoffConfig) (string, error) {
	messages = withoutSystemMessages(messages)
	if config.summaryFunc != nil {
		return config.summaryFunc(ctx, append([]Message(nil), messages...))
	}
	return compactHandoffHistory(messages), nil
}

func withoutSystemMessages(messages []Message) []Message {
	filtered := make([]Message, 0, len(messages))
	for _, message := range messages {
		if message.Role != RoleSystem {
			filtered = append(filtered, message)
		}
	}
	return filtered
}

func compactHandoffHistory(messages []Message) string {
	const maxMessages = 8
	filtered := withoutSystemMessages(messages)
	if len(filtered) > maxMessages {
		filtered = filtered[len(filtered)-maxMessages:]
	}
	var lines []string
	for _, message := range filtered {
		if text := strings.TrimSpace(message.GetTextContent()); text != "" {
			lines = append(lines, fmt.Sprintf("%s: %s", message.Role, clipHandoffText(text, 500)))
		}
		for _, toolUse := range message.GetToolUses() {
			lines = append(lines, fmt.Sprintf("assistant called %s", toolUse.Name))
		}
		for _, result := range message.GetToolResults() {
			lines = append(lines, fmt.Sprintf("tool result: %s", clipHandoffText(result.Content, 500)))
		}
	}
	return strings.Join(lines, "\n")
}

func clipHandoffText(text string, max int) string {
	runes := []rune(text)
	if len(runes) <= max {
		return text
	}
	return string(runes[:max]) + "…"
}

func handoffTool(name, description string) Tool {
	tool, err := NewToolFromStruct(name, description, handoffInput{})
	if err != nil {
		panic(fmt.Sprintf("failed to create handoff tool: %v", err))
	}
	return tool
}

func AddHandoff[T toolTarget](target T, handoff *Handoff) T {
	target.registerTool(handoffTool(handoff.name, handoff.description), &handoffHandler{handoff: handoff})
	return target
}

func AddHandoffWithDeps[D any, T dependencyToolTarget[D]](target T, handoff *HandoffWithDeps[D]) T {
	target.registerTool(handoffTool(handoff.name, handoff.description), &handoffWithDepsHandler[D]{handoff: handoff})
	return target
}

func (a *Agent) AddHandoff(h *Handoff) *Agent                       { return AddHandoff(a, h) }
func (a *AgentWithDeps[D]) AddHandoff(h *Handoff) *AgentWithDeps[D] { return AddHandoff(a, h) }
func (a *TypedAgent[O]) AddHandoff(h *Handoff) *TypedAgent[O]       { return AddHandoff(a, h) }
func (a *TypedAgentWithDeps[O, D]) AddHandoff(h *Handoff) *TypedAgentWithDeps[O, D] {
	return AddHandoff(a, h)
}
func (a *AgentWithDeps[D]) AddHandoffWithDeps(h *HandoffWithDeps[D]) *AgentWithDeps[D] {
	return AddHandoffWithDeps(a, h)
}
func (a *TypedAgentWithDeps[O, D]) AddHandoffWithDeps(h *HandoffWithDeps[D]) *TypedAgentWithDeps[O, D] {
	return AddHandoffWithDeps(a, h)
}
