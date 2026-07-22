package core

import "encoding/json"

// ExecutionStatus describes the durable state reached by an agent execution.
// It is intentionally independent from a provider finish reason: a run can be
// suspended after a successful tool-call response, for example.
type ExecutionStatus uint8

const (
	ExecutionCompleted ExecutionStatus = iota
	ExecutionSuspended
	ExecutionStopped
	ExecutionInterrupted
	ExecutionFailed
)

// Suspension is a serializable description of work which has been admitted
// but not completed. Payload is versioned by its producer; callers should
// treat it as opaque except when persisting it for a later Resume call.
type Suspension struct {
	ID           string
	Kind         string
	FrontierHash string
	Payload      json.RawMessage
}

// ExecutionSnapshot is the non-generic final or partial state associated with
// an agent-owned StreamResult. Provider-owned streams do not set a snapshot.
type ExecutionSnapshot struct {
	Status      ExecutionStatus
	Messages    []Message
	ToolCalls   []ToolUse
	ToolResults []ToolExecutionResult
	Usage       Usage
	Suspension  *Suspension
}
