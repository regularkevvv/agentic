package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
)

// ApprovalFunc is called before an approval-gated tool executes.
// It receives the tool call details and returns whether to proceed.
type ApprovalFunc func(ctx context.Context, toolCall core.ToolUse) (approved bool, err error)

// ChannelToolOption configures channel-backed tool behavior.
type ChannelToolOption func(*channelToolConfig)

type channelToolConfig struct {
	approvalFunc ApprovalFunc
	timeout      time.Duration
}

// WithApproval sets a human-in-the-loop approval function.
func WithApproval(fn ApprovalFunc) ChannelToolOption {
	return func(c *channelToolConfig) {
		c.approvalFunc = fn
	}
}

// WithChannelTimeout sets the maximum wait time for a channel-backed tool.
func WithChannelTimeout(d time.Duration) ChannelToolOption {
	return func(c *channelToolConfig) {
		c.timeout = d
	}
}

// channelHandler waits for a single result from a channel-backed tool.
type channelHandler[TInput any, TOutput any] struct {
	name    string
	handler func(ctx context.Context, input TInput) (<-chan TOutput, error)
	config  *channelToolConfig
	toolCfg *core.ToolConfig
}

func (h *channelHandler[TInput, TOutput]) Name() string {
	return h.name
}

func (h *channelHandler[TInput, TOutput]) ToolConfig() *core.ToolConfig {
	return h.toolCfg
}

func (h *channelHandler[TInput, TOutput]) Execute(
	ctx context.Context,
	input map[string]interface{},
	deps any,
) (interface{}, error) {
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("marshal input: %w", err)
	}

	var typedInput TInput
	if err := json.Unmarshal(inputJSON, &typedInput); err != nil {
		return nil, fmt.Errorf("unmarshal to %T: %w", typedInput, err)
	}

	// Run approval if configured
	if h.config.approvalFunc != nil {
		toolCall := core.ToolUse{
			Name:  h.name,
			Input: input,
		}
		approved, err := h.config.approvalFunc(ctx, toolCall)
		if err != nil {
			return nil, fmt.Errorf("approval: %w", err)
		}
		if !approved {
			return nil, &core.ModelRetry{Message: fmt.Sprintf("Tool %q was rejected by approval", h.name)}
		}
	}

	// Start the channel-backed handler.
	resultCh, err := h.handler(ctx, typedInput)
	if err != nil {
		return nil, err
	}

	// Wait with optional timeout. The agent run remains active while waiting;
	// this is deliberately a channel-backed tool, not resumable execution.
	if h.config.timeout > 0 {
		timer := time.NewTimer(h.config.timeout)
		defer timer.Stop()
		select {
		case result, ok := <-resultCh:
			if !ok {
				return nil, fmt.Errorf("channel tool %q: channel closed without result", h.name)
			}
			return result, nil
		case <-timer.C:
			return nil, fmt.Errorf("channel tool %q: timed out after %v", h.name, h.config.timeout)
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// No timeout: wait until a result or cancellation.
	select {
	case result, ok := <-resultCh:
		if !ok {
			return nil, fmt.Errorf("channel tool %q: channel closed without result", h.name)
		}
		return result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// ChannelTool creates a tool whose handler returns a result channel. The
// current agent run waits for one value, its deadline, or cancellation.
func ChannelTool[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (<-chan TOutput, error),
	opts ...ChannelToolOption,
) (core.Tool, core.ToolHandler, error) {
	var zeroInput TInput
	tool, err := NewToolFromStruct(name, description, zeroInput)
	if err != nil {
		return core.Tool{}, nil, err
	}

	cfg := &channelToolConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	h := &channelHandler[TInput, TOutput]{
		name:    name,
		handler: handler,
		config:  cfg,
	}

	return tool, h, nil
}

// MustChannelTool is like ChannelTool but panics on error.
func MustChannelTool[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (<-chan TOutput, error),
	opts ...ChannelToolOption,
) (core.Tool, core.ToolHandler) {
	tool, h, err := ChannelTool(name, description, handler, opts...)
	if err != nil {
		panic(err)
	}
	return tool, h
}

// approvalHandler wraps a synchronous tool handler with approval logic.
type approvalHandler[TInput any, TOutput any] struct {
	name     string
	handler  func(ctx context.Context, input TInput) (TOutput, error)
	approval ApprovalFunc
	config   *channelToolConfig
	toolCfg  *core.ToolConfig
}

func (h *approvalHandler[TInput, TOutput]) Name() string {
	return h.name
}

func (h *approvalHandler[TInput, TOutput]) ToolConfig() *core.ToolConfig {
	return h.toolCfg
}

func (h *approvalHandler[TInput, TOutput]) Execute(
	ctx context.Context,
	input map[string]interface{},
	deps any,
) (interface{}, error) {
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("marshal input: %w", err)
	}

	var typedInput TInput
	if err := json.Unmarshal(inputJSON, &typedInput); err != nil {
		return nil, fmt.Errorf("unmarshal to %T: %w", typedInput, err)
	}

	// Run approval
	toolCall := core.ToolUse{
		Name:  h.name,
		Input: input,
	}
	approved, err := h.approval(ctx, toolCall)
	if err != nil {
		return nil, fmt.Errorf("approval: %w", err)
	}
	if !approved {
		return nil, &core.ModelRetry{Message: fmt.Sprintf("Tool %q was rejected by approval", h.name)}
	}

	// Execute the handler
	return h.handler(ctx, typedInput)
}

// ApprovalTool creates a tool that requires approval before execution.
func ApprovalTool[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (TOutput, error),
	approvalFn ApprovalFunc,
	opts ...ChannelToolOption,
) (core.Tool, core.ToolHandler, error) {
	var zeroInput TInput
	tool, err := NewToolFromStruct(name, description, zeroInput)
	if err != nil {
		return core.Tool{}, nil, err
	}

	cfg := &channelToolConfig{approvalFunc: approvalFn}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.approvalFunc == nil {
		return core.Tool{}, nil, fmt.Errorf("approval function cannot be nil")
	}

	h := &approvalHandler[TInput, TOutput]{
		name:     name,
		handler:  handler,
		approval: cfg.approvalFunc,
		config:   cfg,
	}

	return tool, h, nil
}

// MustApprovalTool is like ApprovalTool but panics on error.
func MustApprovalTool[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (TOutput, error),
	approvalFn ApprovalFunc,
	opts ...ChannelToolOption,
) (core.Tool, core.ToolHandler) {
	tool, h, err := ApprovalTool(name, description, handler, approvalFn, opts...)
	if err != nil {
		panic(err)
	}
	return tool, h
}
