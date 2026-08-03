// Package harness provides durable, policy-composable Agentic sessions.
package harness

import (
	"context"
	"errors"
	"sync"
	"time"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/artifact"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/codec"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/repair"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/session"
	"github.com/regularkevvv/agentic/harness/store"
)

// RuntimeConfig is the explicit low-level substrate assembly. Capability
// policy is layered on it by Builder.
type RuntimeConfig struct {
	Sessions              store.Repository
	Codec                 codec.Codec
	Events                event.Factory
	Environments          env.Factory
	ResultProcessors      artifact.ProcessorFactory
	Clock                 harnessruntime.Clock
	IDs                   harnessruntime.IDGenerator
	ToolCancellationGrace time.Duration
	Scope                 harnessruntime.Scope
}

// Harness has immutable execution configuration and is safe for concurrent
// session creation. The internal registry only prevents duplicate live owners
// of one session ID in this process.
type Harness[O any] struct {
	driver       agentic.Driver[O]
	sessions     store.Repository
	codec        codec.Codec
	events       event.Factory
	environments env.Factory
	processors   artifact.ProcessorFactory
	clock        harnessruntime.Clock
	ids          harnessruntime.IDGenerator
	grace        time.Duration
	capabilities []string
	toolsets     []agentic.Toolset
	toolGate     agentic.ToolGate
	context      contextpolicy.Projector
	eventPolicy  []event.Middleware
	lifecycle    []harnessruntime.LifecycleHook
	resume       harnessruntime.ResumePlanner
	instructions harnessruntime.ExchangeInstructionProvider
	scope        harnessruntime.Scope
	delegation   []string

	mu      sync.Mutex
	live    map[string]*Session[O]
	opening map[string]bool
}

// Session is the root-package facade over the single-flight session runtime.
type Session[O any] struct {
	*session.Session[O]
	owner *Harness[O]
}

type SessionState = session.State
type QueueKind = session.QueueKind
type QueueEntry = session.QueueEntry
type QueueReceipt = session.QueueReceipt
type Snapshot = session.Snapshot
type SessionOption = session.Option
type ResumeRequest = session.ResumeRequest
type ToolResolution = session.ToolResolution
type ResolutionAction = session.ResolutionAction
type SubscribeOptions = event.SubscribeOptions
type Subscription = event.Subscription
type Event = event.Record

const (
	SessionIdle         = session.Idle
	SessionRunning      = session.Running
	SessionClosing      = session.Closing
	SessionSuspended    = session.Suspended
	SessionInterrupting = session.Interrupting
	SessionFaulted      = session.Faulted
	SessionClosed       = session.Closed

	SteerQueue    = session.QueueSteer
	FollowUpQueue = session.QueueFollowUp
	NextTurnQueue = session.QueueNextTurn

	ResolutionInvalid        = session.ResolutionInvalid
	ResolutionApprove        = session.ResolutionApprove
	ResolutionDeny           = session.ResolutionDeny
	ResolutionExternalResult = session.ResolutionExternalResult
)

var (
	ErrSessionBusy              = session.ErrSessionBusy
	ErrRunClosing               = session.ErrRunClosing
	ErrTurnNotSteerable         = session.ErrTurnNotSteerable
	ErrNotRunning               = session.ErrNotRunning
	ErrSessionSuspended         = session.ErrSessionSuspended
	ErrSessionFaulted           = session.ErrSessionFaulted
	ErrCommitProjectionMismatch = session.ErrCommitProjectionMismatch
	ErrBudgetExceeded           = session.ErrBudgetExceeded
	ErrInvalidMessage           = session.ErrInvalidMessage
	ErrInvalidResumeRequest     = session.ErrInvalidResumeRequest
	ErrIndeterminateTool        = session.ErrIndeterminateTool
	ErrInvalidInitialHistory    = session.ErrInvalidInitialHistory
	ErrSessionOpen              = session.ErrSessionOpen
	ErrSessionClosed            = session.ErrSessionClosed
)

func WithBudget(limits agentic.UsageLimits) SessionOption { return session.WithBudget(limits) }
func WithDrainAll(enabled bool) SessionOption             { return session.WithDrainAll(enabled) }
func WithInitialHistory(messages ...agentic.Message) SessionOption {
	return session.WithInitialHistory(messages...)
}

// NewRuntime validates the released root Driver capability and constructs the
// policy-neutral low-level runtime.
func NewRuntime[O any](runner agentic.Runner[O], config RuntimeConfig) (*Harness[O], error) {
	return newHarness(runner, config, capability.Plan{})
}

func newHarness[O any](runner agentic.Runner[O], config RuntimeConfig, plan capability.Plan) (*Harness[O], error) {
	driver, err := agentic.RequireDriver(runner)
	if err != nil {
		return nil, err
	}
	if config.Sessions == nil {
		return nil, errors.New("harness session repository is required")
	}
	if config.Codec == nil {
		return nil, errors.New("harness payload codec is required")
	}
	if config.Events == nil {
		return nil, errors.New("harness event factory is required")
	}
	if config.Environments == nil {
		return nil, errors.New("harness environment factory is required")
	}
	if config.ResultProcessors == nil {
		return nil, errors.New("harness result-processor factory is required")
	}
	if config.Clock == nil {
		return nil, errors.New("harness clock is required")
	}
	if config.IDs == nil {
		return nil, errors.New("harness ID generator is required")
	}
	if config.ToolCancellationGrace < 0 {
		return nil, errors.New("tool cancellation grace cannot be negative")
	}
	if config.Scope.Depth < 0 {
		return nil, errors.New("harness scope depth cannot be negative")
	}
	grace := config.ToolCancellationGrace
	if grace == 0 {
		grace = time.Second
	}
	contextProjector := plan.ContextPolicy()
	if contextProjector == nil {
		contextProjector = contextpolicy.Passthrough()
	}
	resumePlanner := plan.ResumePlanner()
	if resumePlanner == nil {
		resumePlanner = harnessruntime.DefaultResumePlanner()
	}
	return &Harness[O]{
		driver:       driver,
		sessions:     config.Sessions,
		codec:        config.Codec,
		events:       config.Events,
		environments: config.Environments,
		processors:   config.ResultProcessors,
		clock:        config.Clock,
		ids:          config.IDs,
		grace:        grace,
		capabilities: plan.IDs(),
		toolsets:     plan.Toolsets(),
		toolGate:     plan.ToolGate(),
		context:      contextProjector,
		eventPolicy:  plan.EventMiddleware(),
		lifecycle:    plan.LifecycleHooks(),
		resume:       resumePlanner,
		instructions: plan.ExchangeInstructionProvider(),
		scope:        config.Scope,
		delegation:   plan.DelegationTools(),
		live:         make(map[string]*Session[O]),
		opening:      make(map[string]bool),
	}, nil
}

// Capabilities returns the stable topological order frozen at Build time.
func (h *Harness[O]) Capabilities() []string {
	return append([]string(nil), h.capabilities...)
}

func (h *Harness[O]) NewSession(ctx context.Context, opts ...SessionOption) (*Session[O], error) {
	id, err := h.ids.New("sess")
	if err != nil {
		return nil, err
	}
	created, err := session.New(ctx, h.sessionConfig(id), opts...)
	if err != nil {
		return nil, err
	}
	wrapped := &Session[O]{Session: created, owner: h}
	h.mu.Lock()
	h.live[id] = wrapped
	h.mu.Unlock()
	return wrapped, nil
}

// ResumeSession reopens durable state. An existing healthy in-process owner is
// rejected; a Faulted owner is replaced after normal log recovery succeeds.
func (h *Harness[O]) ResumeSession(ctx context.Context, id string) (*Session[O], error) {
	h.mu.Lock()
	if h.opening[id] {
		h.mu.Unlock()
		return nil, session.ErrSessionOpen
	}
	current := h.live[id]
	h.opening[id] = true
	h.mu.Unlock()
	defer func() {
		h.mu.Lock()
		delete(h.opening, id)
		h.mu.Unlock()
	}()
	if current != nil {
		state := current.State()
		if state != session.Faulted && state != session.Closed {
			return nil, session.ErrSessionOpen
		}
	}
	if current != nil {
		if err := current.Session.Close(ctx); err != nil {
			return nil, err
		}
	}
	recovered, err := session.Recover(ctx, h.sessionConfig(id))
	if err != nil {
		return nil, err
	}
	wrapped := &Session[O]{Session: recovered, owner: h}
	h.mu.Lock()
	h.live[id] = wrapped
	h.mu.Unlock()
	return wrapped, nil
}

func (h *Harness[O]) sessionConfig(id string) session.Config[O] {
	scope := h.scope
	scope.SessionID = id
	return session.Config[O]{
		ID:                    id,
		Driver:                h.driver,
		Repository:            h.sessions,
		Codec:                 h.codec,
		Events:                h.events,
		Environments:          h.environments,
		ResultProcessors:      h.processors,
		Clock:                 h.clock,
		IDs:                   h.ids,
		ToolCancellationGrace: h.grace,
		Toolsets:              append([]agentic.Toolset(nil), h.toolsets...),
		ToolGate:              h.toolGate,
		Context:               h.context,
		EventMiddleware:       append([]event.Middleware(nil), h.eventPolicy...),
		LifecycleHooks:        append([]harnessruntime.LifecycleHook(nil), h.lifecycle...),
		ResumePlanner:         h.resume,
		Instructions:          h.instructions,
		Scope:                 scope,
		DelegationTools:       append([]string(nil), h.delegation...),
	}
}

func (s *Session[O]) Close(ctx context.Context) error {
	err := s.Session.Close(ctx)
	if err == nil && s.State() == session.Closed && s.owner != nil {
		s.owner.mu.Lock()
		if s.owner.live[s.ID()] == s {
			delete(s.owner.live, s.ID())
		}
		s.owner.mu.Unlock()
	}
	return err
}

type FrontierMode = repair.FrontierMode
type PendingState = repair.PendingState
type PendingCall = repair.PendingCall
type PendingCalls = repair.PendingCalls

const (
	CloseInterruptedFrontier = repair.CloseInterruptedFrontier
	PreserveDeferredFrontier = repair.PreserveDeferredFrontier
	PendingUnknown           = repair.PendingUnknown
	PendingPlanned           = repair.PendingPlanned
	PendingStarted           = repair.PendingStarted
	PendingIndeterminate     = repair.PendingIndeterminate
)

func Repair(mode FrontierMode, pending PendingCalls) agentic.HistoryProcessor {
	return repair.Repair(mode, pending)
}
