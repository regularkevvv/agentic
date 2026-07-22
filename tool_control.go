package agentic

import (
	"context"
	"encoding/json"
)

// ToolGate preflights an entire regular-tool batch before any handler starts.
// Output tools are framework calls and are not passed to the gate.
type ToolGate interface {
	EvaluateBatch(context.Context, []ToolUse) (ToolBatchDecision, error)
}

// ToolGateFunc adapts a function to ToolGate.
type ToolGateFunc func(context.Context, []ToolUse) (ToolBatchDecision, error)

func (f ToolGateFunc) EvaluateBatch(ctx context.Context, calls []ToolUse) (ToolBatchDecision, error) {
	return f(ctx, calls)
}

// ToolBatchDecision contains exactly one disposition for every input call in
// the same order. A suspension applies atomically to the entire batch.
type ToolBatchDecision struct {
	Calls    []ToolDisposition
	Deferral *ToolDeferral
}

// ToolDisposition describes one gated call.
type ToolDisposition struct {
	Kind     ToolDispositionKind
	Result   *ToolExecutionResult
	Continue bool
}

// ToolDeferral is serializable gate-owned metadata carried in Suspension.
type ToolDeferral struct {
	Kind    string
	Payload json.RawMessage
}

// ToolDispositionKind is the permitted gate disposition.
type ToolDispositionKind uint8

const (
	ToolDispositionInvalid ToolDispositionKind = iota
	ToolDispositionExecute
	ToolDispositionReturn
	ToolDispositionSuspend
)

// ToolResultProcessor projects a handler result before it is committed to the
// transcript and formatted for the model. It cannot change call identity or
// turn a handler error into success.
type ToolResultProcessor interface {
	Process(context.Context, ToolUse, ToolExecutionResult) (ToolExecutionResult, error)
}

// ToolResultProcessorFunc adapts a function to ToolResultProcessor.
type ToolResultProcessorFunc func(context.Context, ToolUse, ToolExecutionResult) (ToolExecutionResult, error)

func (f ToolResultProcessorFunc) Process(ctx context.Context, call ToolUse, result ToolExecutionResult) (ToolExecutionResult, error) {
	return f(ctx, call, result)
}

type allowAllToolGate struct{}

func (allowAllToolGate) EvaluateBatch(_ context.Context, calls []ToolUse) (ToolBatchDecision, error) {
	dispositions := make([]ToolDisposition, len(calls))
	for i := range dispositions {
		dispositions[i].Kind = ToolDispositionExecute
	}
	return ToolBatchDecision{Calls: dispositions}, nil
}
