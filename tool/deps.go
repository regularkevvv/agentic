package tool

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/regularkevvv/agentic/internal/core"
)

// DepsToolHandler implements ToolHandler for tools that need access to agent dependencies.
type DepsToolHandler[TInput any, TOutput any, DepsT any] struct {
	name    string
	handler func(ctx core.RunContext[DepsT], input TInput) (TOutput, error)
	config  *core.ToolConfig
}

// Execute implements ToolHandler.
func (h *DepsToolHandler[TInput, TOutput, DepsT]) Execute(
	ctx context.Context,
	input map[string]interface{},
	agentDeps any,
) (interface{}, error) {
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("marshal input: %w", err)
	}

	var typedInput TInput
	if err := json.Unmarshal(inputJSON, &typedInput); err != nil {
		return nil, fmt.Errorf("unmarshal to %T: %w", typedInput, err)
	}

	deps, err := core.ExtractDependency[DepsT](agentDeps)
	if err != nil {
		return nil, err
	}

	runCtx := core.RunContext[DepsT]{
		Ctx:      ctx,
		Deps:     deps,
		ToolName: h.name,
	}
	if state, ok := core.ToolExecutionStateFromContext(ctx); ok {
		runCtx.Retry = state.RetryCounts[h.name]
	}

	return h.handler(runCtx, typedInput)
}

// Name implements ToolHandler.
func (h *DepsToolHandler[TInput, TOutput, DepsT]) Name() string {
	return h.name
}

// ToolConfig implements ConfigurableToolHandler.
func (h *DepsToolHandler[TInput, TOutput, DepsT]) ToolConfig() *core.ToolConfig {
	return h.config
}

// ToolWithDeps creates a tool that has access to agent-level dependencies.
func ToolWithDeps[TInput any, TOutput any, DepsT any](
	name, description string,
	handler func(ctx core.RunContext[DepsT], input TInput) (TOutput, error),
	opts ...ToolOption,
) (core.Tool, core.ToolHandler, error) {
	var zeroInput TInput
	tool, err := NewToolFromStruct(name, description, zeroInput)
	if err != nil {
		return core.Tool{}, nil, err
	}

	cfg := applyToolOptions(opts)

	h := &DepsToolHandler[TInput, TOutput, DepsT]{
		name:    name,
		handler: handler,
		config:  cfg,
	}

	return tool, h, nil
}

// MustToolWithDeps is like ToolWithDeps but panics on error.
func MustToolWithDeps[TInput any, TOutput any, DepsT any](
	name, description string,
	handler func(ctx core.RunContext[DepsT], input TInput) (TOutput, error),
	opts ...ToolOption,
) (core.Tool, core.ToolHandler) {
	tool, h, err := ToolWithDeps[TInput, TOutput, DepsT](name, description, handler, opts...)
	if err != nil {
		panic(err)
	}
	return tool, h
}
