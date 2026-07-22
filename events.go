package agentic

import "context"

// EventNature separates transient provider previews from committed execution
// state and lifecycle transitions.
type EventNature uint8

const (
	EventPreview EventNature = iota
	EventAuthoritative
	EventLifecycle
)

// EventType identifies a concrete canonical event kind.
type EventType uint8

const (
	EventTypeTextPreview EventType = iota
	EventTypeThinkingPreview
	EventTypeToolCallPreview
	EventTypeToolArgumentPreview
	EventTypeAssistantCommitted
	EventTypeToolBatchPlanned
	EventTypeToolStarted
	EventTypeToolResultCommitted
	EventTypeOutputValidated
	EventTypeTurnMessagesInjected
	EventTypeRunStarted
	EventTypeTurnStarted
	EventTypeTurnEnded
	EventTypeRunSuspended
	EventTypeRunCompleted
	EventTypeRunInterrupted
	EventTypeRunError
	EventTypeRunEnded
)

// Event is emitted by the one execution fold at deterministic preview, commit,
// and lifecycle points. EventSink is synchronous by design: a sink is a
// low-level execution participant rather than a best-effort subscriber.
type Event interface {
	Nature() EventNature
	Type() EventType
	TurnIndex() int
}

// EventSink receives canonical events synchronously.
type EventSink interface {
	Emit(context.Context, Event) error
}

// EventSinkFunc adapts a function to EventSink.
type EventSinkFunc func(context.Context, Event) error

func (f EventSinkFunc) Emit(ctx context.Context, event Event) error {
	return f(ctx, event)
}

type eventBase struct {
	nature EventNature
	typ    EventType
	turn   int
}

func (e eventBase) Nature() EventNature { return e.nature }
func (e eventBase) Type() EventType     { return e.typ }
func (e eventBase) TurnIndex() int      { return e.turn }

func newEventBase(nature EventNature, typ EventType, turn int) eventBase {
	return eventBase{nature: nature, typ: typ, turn: turn}
}

// TextPreviewEvent carries a transient text delta from a streaming transport.
type TextPreviewEvent struct {
	eventBase
	Delta string
}

// ThinkingPreviewEvent carries a transient thinking delta from a streaming
// transport, including provider replay metadata when available.
type ThinkingPreviewEvent struct {
	eventBase
	Delta        string
	Signature    string
	ProviderName string
	ThinkingID   string
}

// ToolCallPreviewEvent announces a model-emitted tool call before the complete
// assistant message is committed. It exists so the legacy streaming adapter
// can preserve its tool-call-start projection without treating that preview as
// an admitted handler start.
type ToolCallPreviewEvent struct {
	eventBase
	Call ToolUse
}

// ToolArgumentPreviewEvent carries a transient JSON-argument delta.
type ToolArgumentPreviewEvent struct {
	eventBase
	ToolCallID string
	Delta      string
}

// AssistantCommittedEvent is emitted once a complete assistant message has
// been appended to the authoritative transcript.
type AssistantCommittedEvent struct {
	eventBase
	Message Message
}

// ToolBatchPlannedEvent is emitted after call classification and before any
// regular tool handler starts.
type ToolBatchPlannedEvent struct {
	eventBase
	Calls []ToolUse
}

// ToolStartedEvent is emitted immediately before an admitted handler starts.
type ToolStartedEvent struct {
	eventBase
	Call    ToolUse
	Attempt int
}

// ToolResultCommittedEvent is emitted after a paired tool result is appended.
type ToolResultCommittedEvent struct {
	eventBase
	Result ToolExecutionResult
}

// OutputValidatedEvent records a completion candidate that has passed parsing
// and all configured validators.
type OutputValidatedEvent struct {
	eventBase
	Candidate CompletionCandidate
}

// TurnMessagesInjectedEvent records user messages injected by a turn hook.
type TurnMessagesInjectedEvent struct {
	eventBase
	Messages []Message
}

type RunStartedEvent struct{ eventBase }
type TurnStartedEvent struct{ eventBase }

// TurnEndedEvent records a fully committed turn, including validation retry
// and tool-only turns. Candidate is zero when the turn has no valid output.
type TurnEndedEvent struct {
	eventBase
	Candidate CompletionCandidate
	Usage     Usage
}

type RunSuspendedEvent struct {
	eventBase
	Suspension Suspension
}

type RunCompletedEvent struct {
	eventBase
	Usage        Usage
	FinishReason FinishReason
}

type RunInterruptedEvent struct{ eventBase }

type RunErrorEvent struct {
	eventBase
	Error error
}

type RunEndedEvent struct {
	eventBase
	Status ExecutionStatus
}

// CompletionSource describes how a validated completion was supplied.
type CompletionSource uint8

const (
	CompletionNone CompletionSource = iota
	CompletionText
	CompletionOutputTool
)

// CompletionCandidate is intentionally output-type agnostic. Hooks need to
// know whether a valid candidate exists and where it came from; the typed
// value remains in the typed Result returned to the caller.
type CompletionCandidate struct {
	Source     CompletionSource
	ToolCallID string
}

// Valid reports whether this is a parsed and validated completion candidate.
func (c CompletionCandidate) Valid() bool { return c.Source != CompletionNone }

// Turn is the committed state visible to a per-run turn hook. Slices and
// messages are defensive copies owned by the hook invocation.
type Turn struct {
	Index     int
	Messages  []Message
	Assistant Message
	Results   []ToolExecutionResult
	Usage     Usage
	Candidate CompletionCandidate
}

// TurnAction controls terminal arbitration after a committed turn.
type TurnAction uint8

const (
	TurnDefault TurnAction = iota
	TurnContinue
	TurnStop
)

// TurnDecision is returned by a TurnHook. Inject is valid only with
// TurnContinue and contains one or more user messages.
type TurnDecision struct {
	Action TurnAction
	Inject []Message
}

// TurnHook runs after turn commit and candidate validation, but before the
// single terminal-arbitration decision.
type TurnHook func(context.Context, Turn) (TurnDecision, error)
