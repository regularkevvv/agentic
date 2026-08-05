package sessionloop

import "encoding/json"

// EventNature distinguishes authoritative facts from lossy previews.
// Authoritative data is never inferred from previews (law L5).
type EventNature string

// Event natures.
const (
	EventAuthoritative EventNature = "authoritative"
	EventPreview       EventNature = "preview"
)

// EventKind names one event in the protocol vocabulary. Unknown kinds are
// allowed only as explicitly namespaced extensions — a kind carrying a
// dot-separated namespace prefix outside the standard vocabulary — and the
// standard typed payloads stay empty on such events.
type EventKind string

// Standard event kinds.
const (
	EventCommandAccepted EventKind = "command.accepted"
	EventEntryCommitted  EventKind = "entry.committed"
	EventQueueAccepted   EventKind = "queue.accepted"
	EventQueueDrained    EventKind = "queue.drained"
	EventQueueCancelled  EventKind = "queue.cancelled"
	EventRunStarted      EventKind = "run.started"
	EventRunSuspended    EventKind = "run.suspended"
	EventRunSettled      EventKind = "run.settled"
	EventSessionState    EventKind = "session.state"
	EventUsage           EventKind = "usage"
	EventPreviewDelta    EventKind = "preview.delta"
)

// Event is the envelope carrying common identity and exactly one relevant
// typed payload. Every authoritative event has a non-zero position when
// replay is advertised. Preview events may repeat the latest durable
// position and use Ordinal for arrival order. Dropped counts preview events
// lost from this subscriber's stream since its previous delivery.
type Event struct {
	Position   Position
	Ordinal    uint64
	Nature     EventNature
	Kind       EventKind
	SessionID  SessionID
	RunID      RunID
	CommandID  CommandID
	State      State
	Entry      *Entry
	Queue      *QueuedInput
	Suspension *Suspension
	Outcome    *RunOutcome
	Usage      *Usage
	Preview    *Preview
	Dropped    uint64
}

// Clone returns a deep, copy-owned copy of the event.
func (e Event) Clone() Event {
	clone := e
	if e.Entry != nil {
		entry := e.Entry.Clone()
		clone.Entry = &entry
	}
	if e.Queue != nil {
		queue := e.Queue.Clone()
		clone.Queue = &queue
	}
	if e.Suspension != nil {
		suspension := e.Suspension.Clone()
		clone.Suspension = &suspension
	}
	if e.Outcome != nil {
		outcome := e.Outcome.Clone()
		clone.Outcome = &outcome
	}
	if e.Usage != nil {
		usage := *e.Usage
		clone.Usage = &usage
	}
	if e.Preview != nil {
		preview := *e.Preview
		clone.Preview = &preview
	}
	return clone
}

// QueuedInput is one durable queue item awaiting a future drain.
type QueuedInput struct {
	ID        QueueID
	Kind      CommandKind
	CommandID CommandID
	Position  Position
	Content   []Block
}

// Clone returns a deep, copy-owned copy of the queued input.
func (q QueuedInput) Clone() QueuedInput {
	clone := q
	clone.Content = cloneBlocks(q.Content)
	return clone
}

// Usage is cumulative resource accounting for a session.
type Usage struct {
	PromptTokens        int64
	CompletionTokens    int64
	TotalTokens         int64
	CacheReadTokens     int64
	CacheCreationTokens int64
	ReasoningTokens     int64
	Requests            int64
	ToolCalls           int64
}

// PreviewKind names one lossy progress-update shape.
type PreviewKind string

// Standard preview kinds.
const (
	PreviewText     PreviewKind = "text"
	PreviewThinking PreviewKind = "thinking"
	PreviewTool     PreviewKind = "tool"
)

// Preview is a transient, lossy progress update. It never becomes transcript
// truth merely by being observed (law L5).
type Preview struct {
	Kind       PreviewKind
	Text       string
	ToolCallID string
}

// Suspension is a durable pause of the active run awaiting resolution. It
// carries safe display strings only, never provider payloads: opaque
// provider state stays in the concrete adapter (law L12).
type Suspension struct {
	ID          string
	Kind        string
	Description string
	Decisions   []SuspensionDecision
}

// Clone returns a deep, copy-owned copy of the suspension.
func (s Suspension) Clone() Suspension {
	clone := s
	if s.Decisions != nil {
		clone.Decisions = make([]SuspensionDecision, len(s.Decisions))
		copy(clone.Decisions, s.Decisions)
	}
	return clone
}

// SuspensionDecision is one required decision inside a suspension, described
// with safe display strings only.
type SuspensionDecision struct {
	ID         string
	Name       string
	Capability string
	Action     string
	Resource   string
}

// ResolutionAction names one caller decision on a suspension decision.
type ResolutionAction string

// Standard resolution actions.
const (
	ResolutionApprove        ResolutionAction = "approve"
	ResolutionDeny           ResolutionAction = "deny"
	ResolutionExternalResult ResolutionAction = "external_result"
)

// Resolution answers a suspension. It always targets the exact suspension ID
// of the current run; resolution continues the same run identity (law L10).
type Resolution struct {
	SuspensionID string
	Decisions    []ResolutionDecision
}

// Clone returns a deep, copy-owned copy of the resolution.
func (r Resolution) Clone() Resolution {
	clone := r
	if r.Decisions != nil {
		clone.Decisions = make([]ResolutionDecision, len(r.Decisions))
		for index, decision := range r.Decisions {
			clone.Decisions[index] = decision.Clone()
		}
	}
	return clone
}

// ResolutionDecision is one caller decision answering a suspension decision.
type ResolutionDecision struct {
	ID     string
	Action ResolutionAction
	Data   json.RawMessage
	Reason string
}

// Clone returns a deep, copy-owned copy of the decision.
func (d ResolutionDecision) Clone() ResolutionDecision {
	clone := d
	clone.Data = cloneRaw(d.Data)
	return clone
}

// RunOutcomeKind names the singular settled outcome of a run (law L10).
type RunOutcomeKind string

// Standard run outcomes. Suspension is a durable pause, not settlement.
const (
	RunCompleted   RunOutcomeKind = "completed"
	RunInterrupted RunOutcomeKind = "interrupted"
	RunFailed      RunOutcomeKind = "failed"
)

// RunOutcome is the singular settled outcome of one run. Failure is a
// sanitized description; Output is optional structured output present only
// when the host advertises CapabilityStructuredOutput and an
// application-owned projector produced it.
type RunOutcome struct {
	RunID   RunID
	Kind    RunOutcomeKind
	Failure string
	Output  json.RawMessage
}

// Clone returns a deep, copy-owned copy of the outcome.
func (o RunOutcome) Clone() RunOutcome {
	clone := o
	clone.Output = cloneRaw(o.Output)
	return clone
}
