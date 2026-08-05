package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/permission"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/session"
	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// SessionLoopOption configures the sessionloop host constructor.
type SessionLoopOption[O any] func(*sessionLoopOptions[O]) error

type sessionLoopOptions[O any] struct {
	outputProjector     func(O) (json.RawMessage, error)
	suspensionProjector session.SuspensionProjector
	sessionOptions      []SessionOption
}

// WithSessionLoopOutputProjector installs the application-owned conversion
// of a completed run's typed output into one JSON value. Without it the host
// does not advertise output.structured and typed output stays on the legacy
// Harness API.
func WithSessionLoopOutputProjector[O any](projector func(O) (json.RawMessage, error)) SessionLoopOption[O] {
	return func(options *sessionLoopOptions[O]) error {
		if projector == nil {
			return errors.New("sessionloop output projector must not be nil")
		}
		options.outputProjector = projector
		return nil
	}
}

// WithSessionLoopSessionOptions forwards session options (budget, drain-all,
// initial history) to every session the host creates.
func WithSessionLoopSessionOptions[O any](options ...SessionOption) SessionLoopOption[O] {
	return func(target *sessionLoopOptions[O]) error {
		for _, option := range options {
			if option == nil {
				return errors.New("sessionloop session option must not be nil")
			}
		}
		target.sessionOptions = append(target.sessionOptions, options...)
		return nil
	}
}

// WithSessionLoopSuspensionProjector overrides the host's projection of
// durable suspensions into safe display values.
func WithSessionLoopSuspensionProjector[O any](
	projector func(agentic.Suspension) (sessionloop.Suspension, error),
) SessionLoopOption[O] {
	return func(options *sessionLoopOptions[O]) error {
		if projector == nil {
			return errors.New("sessionloop suspension projector must not be nil")
		}
		options.suspensionProjector = projector
		return nil
	}
}

// sessionLoopHost adapts an application-assembled Harness to the neutral
// sessionloop protocol. It never creates a model, provider, store,
// permission policy, capability, tool, memory system, or terminal; it only
// opens sessions on the runtime it was given (plan 8.6).
type sessionLoopHost[O any] struct {
	runtime *Harness[O]
	options sessionLoopOptions[O]
}

// NewSessionLoopHost returns the provider-neutral sessionloop.Host view over
// an already assembled Harness runtime.
func NewSessionLoopHost[O any](runtime *Harness[O], options ...SessionLoopOption[O]) (sessionloop.Host, error) {
	if runtime == nil {
		return nil, errors.New("sessionloop host requires a harness runtime")
	}
	settings := sessionLoopOptions[O]{suspensionProjector: ProjectPermissionSuspension}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("sessionloop host option must not be nil")
		}
		if err := option(&settings); err != nil {
			return nil, err
		}
	}
	return &sessionLoopHost[O]{runtime: runtime, options: settings}, nil
}

// ProjectPermissionSuspension is the root default suspension projection:
// permission deferrals expose one typed decision per approval; suspensions
// with an unsupported deferral kind stay resolvable elsewhere and project a
// generic description with no decisions.
func ProjectPermissionSuspension(value agentic.Suspension) (sessionloop.Suspension, error) {
	result := sessionloop.Suspension{
		ID:          value.ID,
		Kind:        value.Kind,
		Description: "The run is durably paused awaiting operator resolution.",
	}
	approvals, err := permission.InspectSuspension(value)
	if err != nil {
		if errors.Is(err, harnessruntime.ErrUnsupportedDeferral) {
			result.Description = "This suspension has no registered terminal presenter; use the owning application to resolve it."
			return result, nil
		}
		return sessionloop.Suspension{}, fmt.Errorf("inspect permission suspension: %w", err)
	}
	result.Description = "The run is paused for permission approval."
	result.Decisions = make([]sessionloop.SuspensionDecision, len(approvals))
	for index, approval := range approvals {
		result.Decisions[index] = sessionloop.SuspensionDecision{
			ID:         approval.CallID,
			Name:       approval.ToolName,
			Capability: approval.Request.Capability,
			Action:     approval.Request.Action,
			Resource:   approval.Request.CanonicalResource.Display,
		}
	}
	return result, nil
}

// NewSession creates a fresh durable session. SessionOptions.Meta is
// correlation-only and deliberately ignored: it is never persisted and never
// becomes model-visible content (documented).
func (h *sessionLoopHost[O]) NewSession(ctx context.Context, _ sessionloop.SessionOptions) (sessionloop.Session, error) {
	created, err := h.runtime.NewSession(ctx, h.options.sessionOptions...)
	if err != nil {
		return nil, mapSessionLoopHostError(err)
	}
	return h.view(created)
}

// OpenSession reopens durable session state through the existing exclusive
// ResumeSession path.
func (h *sessionLoopHost[O]) OpenSession(ctx context.Context, id sessionloop.SessionID) (sessionloop.Session, error) {
	if id == "" {
		return nil, fmt.Errorf("session ID must not be empty: %w", sessionloop.ErrInvalidCommand)
	}
	opened, err := h.runtime.ResumeSession(ctx, string(id))
	if err != nil {
		return nil, mapSessionLoopHostError(err)
	}
	return h.view(opened)
}

// view wraps one root session. CloseRoot closes the root wrapper — not only
// the embedded session — so the in-process ownership registry releases
// exactly as today.
func (h *sessionLoopHost[O]) view(root *Session[O]) (sessionloop.Session, error) {
	view, err := session.NewLoopView(root.Session, session.LoopConfig[O]{
		CloseRoot:           root.Close,
		OutputProjector:     h.options.outputProjector,
		SuspensionProjector: h.options.suspensionProjector,
	})
	if err != nil {
		_ = root.Close(context.Background())
		return nil, err
	}
	return view, nil
}

func mapSessionLoopHostError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, ErrSessionOpen) {
		return fmt.Errorf("%w: %w", sessionloop.ErrSessionOpen, err)
	}
	return err
}
