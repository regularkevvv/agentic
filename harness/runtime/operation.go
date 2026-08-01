package runtime

import (
	"context"

	agentic "github.com/regularkevvv/agentic"
)

// Operation is a capability-neutral durable fact emitted from inside a tool
// handler. Payload is an opaque, versioned capability wire value.
type Operation struct {
	ID           string
	Kind         string
	Phase        string
	ParentCallID string
	Payload      []byte
}

// OperationRuntime is the session-owned write-ahead and result-projection
// port available to composite tools. It intentionally exposes no journal or
// artifact implementation.
type OperationRuntime interface {
	RecordOperation(context.Context, Operation) error
	ProjectToolResult(context.Context, agentic.ToolUse, agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error)
}
