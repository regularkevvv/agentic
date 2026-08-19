// Package runtime carries harness-owned tool services through the standard Go
// context without changing application dependency values.
package runtime

import (
	"context"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/event"
)

type ToolUpdate struct {
	// Kind is a capability-owned, non-sensitive presentation label. Harness
	// bounds it before observation; Payload remains opaque and is never included
	// in the default presentation projection.
	Kind    string
	Payload []byte
}

// Scope identifies one session in an in-process parent/child topology.
// SessionID is assigned by Harness when a session is opened. ParentSessionID,
// Agent, and Depth are empty/zero for an ordinary top-level session.
type Scope struct {
	SessionID       string
	ParentSessionID string
	Agent           string
	Depth           int
}

// UsageCharge is durable child usage attributed to a parent session.
type UsageCharge struct {
	SessionID string
	Agent     string
	Usage     agentic.Usage
}

// BudgetLease coordinates child usage with the parent session's cumulative
// budget. Budgeted parents serialize leases so remaining limits cannot be
// oversubscribed; unbounded parents may issue concurrent leases. A lease must
// be closed even when child execution fails.
type BudgetLease interface {
	Limits() *agentic.UsageLimits
	Commit(context.Context, UsageCharge) error
	Close()
}

// CaptureRuntime is the generic parent-session port used by topology
// capabilities. It exposes immutable execution policy and explicit mutation
// methods without coupling session core to a concrete subagent package.
type CaptureRuntime interface {
	History() []agentic.Message
	Toolsets() []agentic.Toolset
	ToolGate() agentic.ToolGate
	DelegationTools() []string
	Scope() Scope
	AcquireBudget(context.Context) (BudgetLease, error)
	ProjectEvent(context.Context, event.Record) error
}

type ToolRuntime struct {
	Environment env.Environment
	SessionID   string
	Scope       Scope
	Capture     CaptureRuntime
	Operations  OperationRuntime
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
