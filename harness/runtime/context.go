// Package runtime carries harness-owned tool services through the standard Go
// context without changing application dependency values.
package runtime

import (
	"context"

	"github.com/regularkevvv/agentic/harness/env"
)

type ToolUpdate struct {
	Kind    string
	Payload []byte
}

type ToolRuntime struct {
	Environment env.Environment
	SessionID   string
	Emit        func(ToolUpdate)
}

type contextKey struct{}

// WithContext attaches runtime services. Agentic adds ToolCallContext to a
// descendant context immediately before invoking each tool handler.
func WithContext(ctx context.Context, value ToolRuntime) context.Context {
	return context.WithValue(ctx, contextKey{}, value)
}

func FromContext(ctx context.Context) (ToolRuntime, bool) {
	value, ok := ctx.Value(contextKey{}).(ToolRuntime)
	return value, ok
}
