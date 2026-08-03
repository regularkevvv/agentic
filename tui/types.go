package tui

import (
	"errors"
	"strings"
)

type State string

const (
	StateIdle         State = "idle"
	StateRunning      State = "running"
	StateClosing      State = "closing"
	StateSuspended    State = "suspended"
	StateInterrupting State = "interrupting"
	StateFaulted      State = "faulted"
	StateClosed       State = "closed"
)

type Input struct{ Text string }

func (i Input) Validate() error {
	if strings.TrimSpace(i.Text) == "" {
		return errors.New("input text is empty")
	}
	return nil
}

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Thinking struct {
	Text         string
	ProviderName string
	ThinkingID   string
	Redacted     bool
}

type ToolState string

const (
	ToolPreview ToolState = "preview"
	ToolPlanned ToolState = "planned"
	ToolRunning ToolState = "running"
	ToolDone    ToolState = "done"
	ToolError   ToolState = "error"
)

type Tool struct {
	CallID  string
	Name    string
	State   ToolState
	Attempt int
	Summary string
}

type Entry struct {
	Role     Role
	Text     string
	Thinking []Thinking
	Tools    []Tool
}

type QueueKind string

const (
	QueueSteer    QueueKind = "steer"
	QueueFollowUp QueueKind = "follow_up"
	QueueNextTurn QueueKind = "next_turn"
)

type QueuedInput struct {
	ID   string
	Kind QueueKind
	Text string
}

type Usage struct {
	PromptTokens        int
	CompletionTokens    int
	TotalTokens         int
	CacheReadTokens     int
	CacheCreationTokens int
	ReasoningTokens     int
	Requests            int
	ToolCalls           int
}

func (u Usage) CacheHitPercent() float64 {
	if u.PromptTokens <= 0 || u.CacheReadTokens <= 0 {
		return 0
	}
	return 100 * float64(u.CacheReadTokens) / float64(u.PromptTokens)
}

type Approval struct {
	CallID         string
	ToolName       string
	Capability     string
	Action         string
	ResourceScheme string
	// CanonicalResource is the validated policy identity. Default renderers
	// must not display it because it can contain normalized sensitive details.
	CanonicalResource string
	// ResourceDisplay is the capability-owned bounded label safe for operator
	// presentation.
	ResourceDisplay string
}

type Suspension struct {
	ID          string
	Kind        string
	Supported   bool
	Approvals   []Approval
	Description string
}

type Snapshot struct {
	SessionID    string
	Cursor       uint64
	State        State
	Transcript   []Entry
	Pending      []QueuedInput
	Suspension   *Suspension
	Usage        Usage
	ProfileLabel string
	Workspace    string
	Execution    string
}

type DecisionAction string

const (
	DecisionApprove DecisionAction = "approve"
	DecisionDeny    DecisionAction = "deny"
)

type Decision struct {
	CallID string
	Action DecisionAction
	Reason string
}

type Resolution struct {
	SuspensionID string
	Decisions    []Decision
	Prompt       *Input
}

func (r Resolution) Validate() error {
	if r.SuspensionID == "" {
		return errors.New("resolution suspension ID is empty")
	}
	seen := make(map[string]bool, len(r.Decisions))
	for _, decision := range r.Decisions {
		if decision.CallID == "" || seen[decision.CallID] {
			return errors.New("resolution contains an empty or duplicate call ID")
		}
		if decision.Action != DecisionApprove && decision.Action != DecisionDeny {
			return errors.New("resolution decision action is invalid")
		}
		seen[decision.CallID] = true
	}
	if r.Prompt != nil {
		if err := r.Prompt.Validate(); err != nil {
			return err
		}
	}
	return nil
}

type EventKind string

const (
	EventTextDelta          EventKind = "text.delta"
	EventThinkingDelta      EventKind = "thinking.delta"
	EventAssistantCommitted EventKind = "assistant.committed"
	EventToolPlanned        EventKind = "tool.planned"
	EventToolStarted        EventKind = "tool.started"
	EventToolResult         EventKind = "tool.result"
	EventMessagesInjected   EventKind = "messages.injected"
	EventRunStarted         EventKind = "run.started"
	EventTurnStarted        EventKind = "turn.started"
	EventTurnEnded          EventKind = "turn.ended"
	EventRunSuspended       EventKind = "run.suspended"
	EventRunCompleted       EventKind = "run.completed"
	EventRunInterrupted     EventKind = "run.interrupted"
	EventRunFailed          EventKind = "run.failed"
	EventRunEnded           EventKind = "run.ended"
	EventQueueAccepted      EventKind = "queue.accepted"
	EventQueueDrained       EventKind = "queue.drained"
	EventQueueCancelled     EventKind = "queue.cancelled"
	EventSessionCreated     EventKind = "session.created"
	EventSessionRecovered   EventKind = "session.recovered"
	EventSessionFaulted     EventKind = "session.faulted"
	EventSessionClosed      EventKind = "session.closed"
	EventUsage              EventKind = "usage"
)

type Event struct {
	Cursor     uint64
	Ordinal    uint64
	Durable    bool
	SessionID  string
	ParentID   string
	Agent      string
	Depth      int
	Turn       int
	Kind       EventKind
	TextDelta  string
	Entry      *Entry
	Entries    []Entry
	Thinking   *Thinking
	Tool       *Tool
	Tools      []Tool
	Usage      *Usage
	Suspension *Suspension
	Failure    string
	Queue      *QueuedInput
	State      State
	Dropped    uint64
}
