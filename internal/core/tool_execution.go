package core

import "context"

type toolExecutionStateKey struct{}

// ToolExecutionState carries run data needed by built-in composite handlers.
// It is internal execution plumbing rather than user dependency injection.
type ToolExecutionState struct {
	Messages    []Message
	RetryCounts map[string]int
}

// WithToolExecutionState attaches a snapshot of the parent run to a tool call.
func WithToolExecutionState(ctx context.Context, state ToolExecutionState) context.Context {
	return context.WithValue(ctx, toolExecutionStateKey{}, state)
}

// ToolExecutionStateFromContext returns state attached by the agent loop.
func ToolExecutionStateFromContext(ctx context.Context) (ToolExecutionState, bool) {
	state, ok := ctx.Value(toolExecutionStateKey{}).(ToolExecutionState)
	return state, ok
}
