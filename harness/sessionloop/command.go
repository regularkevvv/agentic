package sessionloop

import (
	"encoding/json"
	"fmt"
)

// CommandKind names one caller request. The three delivery commands stay
// distinct because they carry different timing and recovery semantics: steer
// affects the active run at its next steerable boundary, follow-up continues
// the active run after its current candidate boundary, and next-turn waits
// for a future start and survives interruption of the current run.
type CommandKind string

// Standard command kinds.
const (
	CommandStart     CommandKind = "start"
	CommandSteer     CommandKind = "steer"
	CommandFollowUp  CommandKind = "follow_up"
	CommandNextTurn  CommandKind = "next_turn"
	CommandResolve   CommandKind = "resolve"
	CommandInterrupt CommandKind = "interrupt"
)

// Command is one caller request to a session.
//
// ID is always present in an accepted receipt and in the events caused by
// the command; a host may generate it when omitted. IdempotencyKey is
// meaningful only when CapabilityIdempotentDispatch is advertised; without
// that capability it is rejected rather than accepted and ignored.
type Command struct {
	ID             CommandID
	Kind           CommandKind
	RunID          RunID
	Input          *Input
	Resolution     *Resolution
	IdempotencyKey string
}

// Clone returns a deep, copy-owned copy of the command.
func (c Command) Clone() Command {
	clone := c
	if c.Input != nil {
		input := c.Input.Clone()
		clone.Input = &input
	}
	if c.Resolution != nil {
		resolution := c.Resolution.Clone()
		clone.Resolution = &resolution
	}
	return clone
}

// Validate enforces the structural matrix of the protocol: which of RunID,
// Input, and Resolution each command kind requires or forbids. Every failure
// wraps ErrInvalidCommand with the specific reason. Structural validation is
// deterministic and centralized so every host rejects the same commands.
func (c Command) Validate() error {
	switch c.Kind {
	case CommandStart:
		if c.RunID != "" {
			return fmt.Errorf("start command must not target a run: %w", ErrInvalidCommand)
		}
		if c.Input == nil {
			return fmt.Errorf("start command requires an input: %w", ErrInvalidCommand)
		}
		if c.Resolution != nil {
			return fmt.Errorf("start command must not carry a resolution: %w", ErrInvalidCommand)
		}
	case CommandSteer, CommandFollowUp:
		if c.RunID == "" {
			return fmt.Errorf("%s command requires the target run: %w", c.Kind, ErrInvalidCommand)
		}
		if c.Input == nil {
			return fmt.Errorf("%s command requires an input: %w", c.Kind, ErrInvalidCommand)
		}
		if c.Resolution != nil {
			return fmt.Errorf("%s command must not carry a resolution: %w", c.Kind, ErrInvalidCommand)
		}
	case CommandNextTurn:
		if c.RunID != "" {
			return fmt.Errorf("next_turn command is session-targeted and must not name a run: %w", ErrInvalidCommand)
		}
		if c.Input == nil {
			return fmt.Errorf("next_turn command requires an input: %w", ErrInvalidCommand)
		}
		if c.Resolution != nil {
			return fmt.Errorf("next_turn command must not carry a resolution: %w", ErrInvalidCommand)
		}
	case CommandResolve:
		if c.RunID == "" {
			return fmt.Errorf("resolve command requires the target run: %w", ErrInvalidCommand)
		}
		if c.Resolution == nil {
			return fmt.Errorf("resolve command requires a resolution: %w", ErrInvalidCommand)
		}
		if err := c.Resolution.validate(); err != nil {
			return err
		}
	case CommandInterrupt:
		if c.RunID == "" {
			return fmt.Errorf("interrupt command requires the target run: %w", ErrInvalidCommand)
		}
		if c.Input != nil {
			return fmt.Errorf("interrupt command must not carry an input: %w", ErrInvalidCommand)
		}
		if c.Resolution != nil {
			return fmt.Errorf("interrupt command must not carry a resolution: %w", ErrInvalidCommand)
		}
	default:
		return fmt.Errorf("unknown command kind %q: %w", c.Kind, ErrInvalidCommand)
	}
	return nil
}

func (r Resolution) validate() error {
	if r.SuspensionID == "" {
		return fmt.Errorf("resolution requires the exact suspension ID: %w", ErrInvalidCommand)
	}
	for index, decision := range r.Decisions {
		if decision.ID == "" {
			return fmt.Errorf("resolution decision %d requires the decision ID: %w", index, ErrInvalidCommand)
		}
		switch decision.Action {
		case ResolutionApprove, ResolutionDeny, ResolutionExternalResult:
		default:
			return fmt.Errorf("resolution decision %q has unknown action %q: %w", decision.ID, decision.Action, ErrInvalidCommand)
		}
		if len(decision.Data) > 0 && !json.Valid(decision.Data) {
			return fmt.Errorf("resolution decision %q data must be one complete JSON value: %w", decision.ID, ErrInvalidCommand)
		}
	}
	return nil
}

// ValidateCommand validates the command structurally and then against the
// advertised capability set. Structural validation runs first, so an invalid
// command reports ErrInvalidCommand even when the host also lacks the
// capability. Capability failures wrap ErrUnsupported: an IdempotencyKey
// requires CapabilityIdempotentDispatch, steer/follow_up/next_turn require
// their input capabilities, interrupt requires CapabilityInterrupt, and
// resolve requires CapabilitySuspensionResolve. Start is a baseline protocol
// requirement and is always allowed.
func ValidateCommand(command Command, capabilities Capabilities) error {
	if err := command.Validate(); err != nil {
		return err
	}
	if command.IdempotencyKey != "" && !capabilities.Supports(CapabilityIdempotentDispatch) {
		return fmt.Errorf("idempotency keys require capability %q: %w", CapabilityIdempotentDispatch, ErrUnsupported)
	}
	required := map[CommandKind]Capability{
		CommandSteer:     CapabilitySteer,
		CommandFollowUp:  CapabilityFollowUp,
		CommandNextTurn:  CapabilityNextTurn,
		CommandInterrupt: CapabilityInterrupt,
		CommandResolve:   CapabilitySuspensionResolve,
	}
	if capability, ok := required[command.Kind]; ok && !capabilities.Supports(capability) {
		return fmt.Errorf("%s commands require capability %q: %w", command.Kind, capability, ErrUnsupported)
	}
	return nil
}

// AcceptanceGuarantee declares what a receipt promises (law L3).
type AcceptanceGuarantee string

const (
	// AcceptanceAccepted means the owning host accepted the command, but
	// crash replay is not promised.
	AcceptanceAccepted AcceptanceGuarantee = "accepted"

	// AcceptanceDurable means the command and required input facts are
	// crash-durable and replayable before the receipt is returned. A host
	// must not claim it unless it can actually replay the accepted fact
	// after its own failure boundary.
	AcceptanceDurable AcceptanceGuarantee = "durable"
)

// Receipt is evidence that the host accepted a command. It is not the run
// result (law L2). For start, resolve, and interrupt, RunID identifies the
// affected run. For queued input, QueueID identifies the durable queue item.
// A durable receipt has a non-zero position at or after every fact required
// to reconstruct the acceptance.
type Receipt struct {
	CommandID CommandID
	SessionID SessionID
	RunID     RunID
	QueueID   QueueID
	Position  Position
	Guarantee AcceptanceGuarantee
}
