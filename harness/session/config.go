package session

import (
	"errors"
	"time"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/artifact"
	"github.com/regularkevvv/agentic/harness/codec"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
)

type Config[O any] struct {
	ID                    string
	Driver                agentic.Driver[O]
	Repository            store.Repository
	Codec                 codec.Codec
	Events                event.Factory
	Environments          env.Factory
	ResultProcessors      artifact.ProcessorFactory
	Clock                 harnessruntime.Clock
	IDs                   harnessruntime.IDGenerator
	ToolCancellationGrace time.Duration
	Toolsets              []agentic.Toolset
	ToolGate              agentic.ToolGate
	Context               contextpolicy.Projector
	EventMiddleware       []event.Middleware
	LifecycleHooks        []harnessruntime.LifecycleHook
	ResumePlanner         harnessruntime.ResumePlanner
	Instructions          harnessruntime.ExchangeInstructionProvider
	Scope                 harnessruntime.Scope
	DelegationTools       []string
}

func (c Config[O]) validate() error {
	if c.ID == "" {
		return errors.New("session ID is required")
	}
	if c.Driver == nil {
		return errors.New("session driver is required")
	}
	if c.Repository == nil {
		return errors.New("session repository is required")
	}
	if c.Codec == nil {
		return errors.New("session payload codec is required")
	}
	if c.Events == nil {
		return errors.New("session event factory is required")
	}
	if c.Environments == nil {
		return errors.New("session environment factory is required")
	}
	if c.ResultProcessors == nil {
		return errors.New("session result-processor factory is required")
	}
	if c.Clock == nil {
		return errors.New("session clock is required")
	}
	if c.IDs == nil {
		return errors.New("session ID generator is required")
	}
	for _, middleware := range c.EventMiddleware {
		if middleware == nil {
			return errors.New("session event middleware must not be nil")
		}
	}
	for _, hook := range c.LifecycleHooks {
		if hook == nil {
			return errors.New("session lifecycle hook must not be nil")
		}
	}
	if c.ToolCancellationGrace < 0 {
		return errors.New("tool cancellation grace cannot be negative")
	}
	scope := c.Scope
	if scope.SessionID == "" {
		scope.SessionID = c.ID
	}
	if err := validateScope(c.ID, scope); err != nil {
		return err
	}
	return nil
}

func validateScope(sessionID string, scope harnessruntime.Scope) error {
	if scope.Depth < 0 {
		return errors.New("session scope depth cannot be negative")
	}
	if scope.SessionID != sessionID {
		return errors.New("session scope ID does not match session ID")
	}
	if scope.ParentSessionID == scope.SessionID {
		return errors.New("session scope cannot be its own parent")
	}
	if scope.Depth == 0 && scope.ParentSessionID != "" {
		return errors.New("top-level session scope cannot have a parent")
	}
	if scope.Depth > 0 && (scope.ParentSessionID == "" || scope.Agent == "") {
		return errors.New("child session scope requires a parent and agent")
	}
	return nil
}

type persistedOptions struct {
	Budget   *agentic.UsageLimits
	DrainAll bool
}

type sessionCreatedPayload struct {
	Options persistedOptions
	Scope   *harnessruntime.Scope
}

type runOpenedPayload struct {
	ID           string
	Mode         string
	Recovery     bool
	Limits       *agentic.UsageLimits
	Instructions string
}

type runClosedPayload struct {
	ID     string
	Status agentic.ExecutionStatus
	Error  string
}

type messagePayload struct {
	Message agentic.Message
	Source  string
	QueueID string
}

type queueMutationPayload struct {
	ID     string
	Reason string
	Entry  *QueueEntry
}

type usagePayload struct {
	Run     agentic.Usage
	Session agentic.Usage
}

type interruptMarkerPayload struct {
	Message string
}

type contextMessagePayload struct {
	After   int
	Message agentic.Message
}

type compactionPayload struct {
	Compaction contextpolicy.Compaction
}

type resolutionAcceptedPayload struct {
	SuspensionID string
	Request      harnessruntime.ResumeRequest
}

type childUsagePayload struct {
	Charge  harnessruntime.UsageCharge
	Session agentic.Usage
}
