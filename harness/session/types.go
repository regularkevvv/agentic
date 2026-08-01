// Package session implements the harness single-flight runtime state machine.
package session

import (
	"errors"
	"fmt"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
)

type State uint8

const (
	Idle State = iota
	Running
	Closing
	Suspended
	Interrupting
	Faulted
	Closed
)

func (s State) String() string {
	switch s {
	case Idle:
		return "idle"
	case Running:
		return "running"
	case Closing:
		return "closing"
	case Suspended:
		return "suspended"
	case Interrupting:
		return "interrupting"
	case Faulted:
		return "faulted"
	case Closed:
		return "closed"
	default:
		return fmt.Sprintf("state(%d)", s)
	}
}

type QueueKind string

const (
	QueueSteer    QueueKind = "steer"
	QueueFollowUp QueueKind = "follow_up"
	QueueNextTurn QueueKind = "next_turn"
)

type QueueEntry struct {
	ID       string          `json:"id"`
	Kind     QueueKind       `json:"kind"`
	Message  agentic.Message `json:"message"`
	Accepted time.Time       `json:"accepted"`
}

type QueueReceipt struct {
	ID     string    `json:"id"`
	Kind   QueueKind `json:"kind"`
	Cursor uint64    `json:"cursor"`
}

type Snapshot struct {
	Cursor     uint64
	State      State
	Messages   []agentic.Message
	Pending    []QueueEntry
	Suspension *agentic.Suspension
	Usage      agentic.Usage
}

type Option func(*options) error

type options struct {
	budget         *agentic.UsageLimits
	drainAll       bool
	initialHistory []agentic.Message
}

func WithBudget(limits agentic.UsageLimits) Option {
	return func(options *options) error {
		copy := cloneLimits(limits)
		options.budget = &copy
		return nil
	}
}

func WithDrainAll(enabled bool) Option {
	return func(options *options) error {
		options.drainAll = enabled
		return nil
	}
}

// WithInitialHistory copies a complete, protocol-valid transcript into a new
// session before its first prompt. It is primarily used by explicit capture
// policies; recovery never reapplies the option.
func WithInitialHistory(messages ...agentic.Message) Option {
	return func(options *options) error {
		options.initialHistory = cloneMessages(messages)
		return nil
	}
}

type SubscribeOptions = event.SubscribeOptions
type Subscription = event.Subscription
type Event = event.Record
type ResumeRequest = harnessruntime.ResumeRequest
type ToolResolution = harnessruntime.ToolResolution
type ResolutionAction = harnessruntime.ResolutionAction

const (
	ResolutionInvalid        = harnessruntime.ResolutionInvalid
	ResolutionApprove        = harnessruntime.ResolutionApprove
	ResolutionDeny           = harnessruntime.ResolutionDeny
	ResolutionExternalResult = harnessruntime.ResolutionExternalResult
)

var (
	ErrSessionBusy              = errors.New("session is not idle")
	ErrRunClosing               = errors.New("run is closing")
	ErrTurnNotSteerable         = errors.New("turn is not steerable")
	ErrNotRunning               = errors.New("session is not running or suspended")
	ErrSessionSuspended         = errors.New("session is suspended")
	ErrSessionFaulted           = errors.New("session is faulted")
	ErrCommitProjectionMismatch = errors.New("agentic commit projection mismatch")
	ErrBudgetExceeded           = errors.New("session budget exceeded")
	ErrInvalidMessage           = errors.New("session input must be a user message")
	ErrInvalidResumeRequest     = harnessruntime.ErrInvalidResumeRequest
	ErrIndeterminateTool        = errors.New("indeterminate tool cannot be executed automatically")
	ErrInvalidInitialHistory    = errors.New("initial session history is not protocol-valid")
	ErrSessionOpen              = store.ErrSessionOpen
	ErrSessionClosed            = errors.New("session is closed")
)

type FaultError struct {
	SessionID string
	Cause     error
}

func (e *FaultError) Error() string {
	return fmt.Sprintf("%s: %s: %v", ErrSessionFaulted, e.SessionID, e.Cause)
}

func (e *FaultError) Unwrap() error { return e.Cause }

func (e *FaultError) Is(target error) bool { return target == ErrSessionFaulted }

type BudgetError struct {
	Cause error
}

func (e *BudgetError) Error() string {
	if e.Cause == nil {
		return ErrBudgetExceeded.Error()
	}
	return ErrBudgetExceeded.Error() + ": " + e.Cause.Error()
}

func (e *BudgetError) Unwrap() error        { return e.Cause }
func (e *BudgetError) Is(target error) bool { return target == ErrBudgetExceeded }
