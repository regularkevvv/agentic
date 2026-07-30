package harness

import (
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/capability"
)

type Capability = capability.Capability
type Ordering = capability.Ordering
type Registry = capability.Registry
type CapabilityFunc = capability.Func

// Option configures a capability Builder.
type Option func(*builderOptions) error

type builderOptions struct {
	runtime      *RuntimeConfig
	capabilities []Capability
}

// WithRuntime supplies the explicit substrate assembly used by a Builder.
func WithRuntime(config RuntimeConfig) Option {
	return func(options *builderOptions) error {
		copy := config
		options.runtime = &copy
		return nil
	}
}

// WithCapabilities adds ordinary graph nodes in stable option order.
func WithCapabilities(capabilities ...Capability) Option {
	return func(options *builderOptions) error {
		for _, current := range capabilities {
			if current == nil {
				return errors.New("capability must not be nil")
			}
			options.capabilities = append(options.capabilities, current)
		}
		return nil
	}
}

// Builder resolves an immutable capability graph around an already-bound
// Agentic runner.
type Builder[O any] struct {
	runner agentic.Runner[O]
	config builderOptions
	err    error
}

// New creates a capability builder. Validation is deferred to Build so option
// assembly remains ergonomic.
func New[O any](runner agentic.Runner[O], options ...Option) *Builder[O] {
	builder := &Builder[O]{runner: runner}
	for _, option := range options {
		if option == nil {
			builder.err = errors.New("harness option must not be nil")
			break
		}
		if err := option(&builder.config); err != nil {
			builder.err = err
			break
		}
	}
	return builder
}

// Build validates the Driver, compiles a stable capability DAG, and freezes
// every contribution into a concurrent-session-safe Harness.
func (b *Builder[O]) Build() (*Harness[O], error) {
	if b == nil {
		return nil, errors.New("harness builder is nil")
	}
	if b.err != nil {
		return nil, b.err
	}
	if b.config.runtime == nil {
		return nil, errors.New("harness runtime configuration is required")
	}
	plan, err := capability.Compile(b.config.capabilities...)
	if err != nil {
		return nil, fmt.Errorf("compile harness capabilities: %w", err)
	}
	return newHarness(b.runner, *b.config.runtime, plan)
}
