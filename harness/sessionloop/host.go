package sessionloop

import (
	"context"
	"slices"
)

// SessionID identifies one durable session across handles and processes.
type SessionID string

// RunID identifies one application-level exchange inside a session. At most
// one run is active per session (law L1).
type RunID string

// CommandID identifies one dispatched command in its receipt and in every
// event the command causes. A host may generate it when omitted.
type CommandID string

// QueueID identifies one durable queued input awaiting a future drain.
type QueueID string

// EntryID identifies one authoritative committed transcript entry.
type EntryID string

// State is the observable session state.
type State string

// The common session state machine. Closing and interrupting remain
// observable transitional states; suspension is a durable pause of the same
// run, not settlement.
const (
	StateIdle         State = "idle"
	StateRunning      State = "running"
	StateClosing      State = "closing"
	StateSuspended    State = "suspended"
	StateInterrupting State = "interrupting"
	StateFaulted      State = "faulted"
	StateClosed       State = "closed"
)

// Capability names one optional protocol behavior a host advertises.
// Capabilities are honest (law L9): calling an unsupported operation returns
// ErrUnsupported, and an adapter never simulates a missing behavior by
// rewriting ordinary user content. Unknown capability strings survive round
// trips unchanged.
type Capability string

// Standard capabilities. New, open, start, snapshot, authoritative committed
// entries, authoritative settlement, and close are baseline protocol
// requirements and therefore are not capabilities.
const (
	CapabilityDurableAcceptance  Capability = "acceptance.durable"
	CapabilityReplay             Capability = "events.replay"
	CapabilityPreview            Capability = "events.preview"
	CapabilitySteer              Capability = "input.steer"
	CapabilityFollowUp           Capability = "input.follow_up"
	CapabilityNextTurn           Capability = "input.next_turn"
	CapabilityInterrupt          Capability = "run.interrupt"
	CapabilitySuspensionResolve  Capability = "run.suspension.resolve"
	CapabilityIdempotentDispatch Capability = "dispatch.idempotent"
	CapabilityDetailedTools      Capability = "content.tools.detailed"
	CapabilityStructuredOutput   Capability = "output.structured"
)

// Capabilities is the advertised capability set of one opened session handle.
// It is immutable for the handle's lifetime.
type Capabilities []Capability

// NewCapabilities returns a defensive, deduplicated, sorted capability set.
// Unknown capability strings are preserved so they survive round trips.
func NewCapabilities(capabilities ...Capability) Capabilities {
	set := make(Capabilities, 0, len(capabilities))
	for _, capability := range capabilities {
		if !slices.Contains(set, capability) {
			set = append(set, capability)
		}
	}
	slices.Sort(set)
	return set
}

// Supports reports whether the set advertises the capability.
func (c Capabilities) Supports(capability Capability) bool {
	return slices.Contains(c, capability)
}

// Clone returns an independent copy of the set.
func (c Capabilities) Clone() Capabilities {
	if c == nil {
		return nil
	}
	return slices.Clone(c)
}

// SessionOptions carries application metadata for a new session. Metadata is
// for correlation, not instruction smuggling: implementations must not
// translate unknown metadata into model-visible text.
type SessionOptions struct {
	Meta map[string]string
}

// Clone returns an independent copy of the options.
func (o SessionOptions) Clone() SessionOptions {
	return SessionOptions{Meta: cloneMeta(o.Meta)}
}

// Host creates or opens session handles. It does not choose a provider,
// model, tools, memory, permissions, or storage; those remain
// application-owned assembly.
type Host interface {
	NewSession(context.Context, SessionOptions) (Session, error)
	OpenSession(context.Context, SessionID) (Session, error)
}

// Session is a long-lived conversation and command receiver. It can contain
// zero or many runs over its lifetime.
//
// Dispatch returns when the command is accepted according to the receipt's
// guarantee (law L2); the answer and final outcome arrive through events and
// snapshots. The Dispatch context controls validation and acceptance only:
// after a successful receipt, canceling it does not cancel the run (law L4).
// Snapshot returns a copy-owned authoritative view (law L7). Close is
// idempotent and releases the handle, never the durable session (law L11).
type Session interface {
	ID() SessionID
	Capabilities() Capabilities
	Dispatch(context.Context, Command) (Receipt, error)
	Snapshot(context.Context) (Snapshot, error)
	Subscribe(context.Context, SubscribeOptions) (Stream, error)
	Close(context.Context) error
}
