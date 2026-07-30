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
	return nil
}

type persistedOptions struct {
	Budget   *agentic.UsageLimits
	DrainAll bool
}

type sessionCreatedPayload struct {
	Options persistedOptions
}

type runOpenedPayload struct {
	ID       string
	Mode     string
	Recovery bool
	Limits   *agentic.UsageLimits
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
