package tool

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/regularkevvv/agentic/internal/core"
)

// PlainToolHandler implements ToolHandler for tools that don't need dependencies.
type PlainToolHandler[TInput any, TOutput any] struct {
	name    string
	handler func(input TInput) (TOutput, error)
	config  *core.ToolConfig
}

// ContextToolHandler implements ToolHandler for ordinary Go handlers that
// need cancellation, deadlines, tracing, or other request-scoped context.
type ContextToolHandler[TInput any, TOutput any] struct {
	name    string
	handler func(ctx context.Context, input TInput) (TOutput, error)
	config  *core.ToolConfig
}

// Execute implements ToolHandler.
func (h *PlainToolHandler[TInput, TOutput]) Execute(
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

	return h.handler(typedInput)
}

// Name implements ToolHandler.
func (h *PlainToolHandler[TInput, TOutput]) Name() string {
	return h.name
}

// ToolConfig implements ConfigurableToolHandler.
func (h *PlainToolHandler[TInput, TOutput]) ToolConfig() *core.ToolConfig {
	return h.config
}

func (h *ContextToolHandler[TInput, TOutput]) Execute(
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
	return h.handler(ctx, typedInput)
}

func (h *ContextToolHandler[TInput, TOutput]) Name() string {
	return h.name
}

func (h *ContextToolHandler[TInput, TOutput]) ToolConfig() *core.ToolConfig {
	return h.config
}

// ToolPlain creates a plain tool (no dependencies) from a struct and handler function.
func ToolPlain[TInput any, TOutput any](
	name, description string,
	handler func(input TInput) (TOutput, error),
	opts ...ToolOption,
) (core.Tool, core.ToolHandler, error) {
	var zeroInput TInput
	tool, err := NewToolFromStruct(name, description, zeroInput)
	if err != nil {
		return core.Tool{}, nil, err
	}

	cfg := applyToolOptions(opts)

	h := &PlainToolHandler[TInput, TOutput]{
		name:    name,
		handler: handler,
		config:  cfg,
	}

	return tool, h, nil
}

// ToolWithContext creates a tool whose handler receives context.Context. This
// is the recommended form for I/O and any work that should honor cancellation.
func ToolWithContext[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (TOutput, error),
	opts ...ToolOption,
) (core.Tool, core.ToolHandler, error) {
	var zeroInput TInput
	tool, err := NewToolFromStruct(name, description, zeroInput)
	if err != nil {
		return core.Tool{}, nil, err
	}
	return tool, &ContextToolHandler[TInput, TOutput]{
		name:    name,
		handler: handler,
		config:  applyToolOptions(opts),
	}, nil
}

// MustToolPlain is like ToolPlain but panics on error.
func MustToolPlain[TInput any, TOutput any](
	name, description string,
	handler func(input TInput) (TOutput, error),
	opts ...ToolOption,
) (core.Tool, core.ToolHandler) {
	tool, h, err := ToolPlain(name, description, handler, opts...)
	if err != nil {
		panic(err)
	}
	return tool, h
}

// MustToolWithContext is like ToolWithContext but panics on error.
func MustToolWithContext[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (TOutput, error),
	opts ...ToolOption,
) (core.Tool, core.ToolHandler) {
	tool, h, err := ToolWithContext(name, description, handler, opts...)
	if err != nil {
		panic(err)
	}
	return tool, h
}
