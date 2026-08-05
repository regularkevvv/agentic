package sessionloop

import "errors"

// The sentinels below are the portable error categories of the protocol.
// Adapters wrap them around concrete causes so generic callers can use
// errors.Is against the category while implementation-specific callers keep
// their own sentinels. Error strings are not part of the contract; only the
// errors.Is identity is.
var (
	// ErrUnsupported reports that the host does not advertise the capability
	// required by the operation. Capabilities are immutable for one opened
	// session handle, and a missing behavior must fail explicitly rather than
	// be emulated by rewriting user content (law L9).
	ErrUnsupported = errors.New("sessionloop: operation not supported by this host")

	// ErrInvalidCommand reports that a command is structurally invalid: a
	// field required by its kind is missing, a forbidden field is present, or
	// carried content fails validation. Structural validation precedes
	// capability and state checks, so an invalid command reports this
	// sentinel even when the host also lacks the capability.
	ErrInvalidCommand = errors.New("sessionloop: invalid command")

	// ErrSessionBusy reports a start dispatched while the session already has
	// an active or suspended run. A session has at most one active run
	// (law L1); queue a next-turn input instead of starting.
	ErrSessionBusy = errors.New("sessionloop: session already has an active run")

	// ErrSessionOpen reports that an exclusive-writer implementation refused
	// to open a second concurrent handle to the same session.
	ErrSessionOpen = errors.New("sessionloop: session is already open elsewhere")

	// ErrSessionClosed reports an operation on a closed session handle.
	// Closing releases the handle, not the durable session (law L11).
	ErrSessionClosed = errors.New("sessionloop: session is closed")

	// ErrSessionFaulted reports that a session-level invariant or storage
	// failure moved the session to the faulted state; runs can no longer be
	// accepted on this handle.
	ErrSessionFaulted = errors.New("sessionloop: session is faulted")

	// ErrStaleRun reports a targeted command (steer, follow-up, resolve,
	// interrupt) whose RunID no longer identifies the active run. Targeted
	// commands never cross runs (law L8).
	ErrStaleRun = errors.New("sessionloop: command targets a run that is no longer active")

	// ErrSuspended reports an operation that requires an actively running run
	// while the current run is durably suspended awaiting resolution.
	ErrSuspended = errors.New("sessionloop: run is suspended")

	// ErrNotRunning reports a run-targeted command dispatched while the
	// session has no active run.
	ErrNotRunning = errors.New("sessionloop: session has no active run")

	// ErrLagged reports that a subscriber fell behind and authoritative
	// events were lost from its stream. The consumer must discard speculative
	// state, load a snapshot, and subscribe after its position (law L7).
	ErrLagged = errors.New("sessionloop: subscriber lagged behind authoritative events")

	// ErrUnknownPosition reports a subscribe-after position the host cannot
	// replay from. Positions are opaque and must be passed back unchanged
	// (law L6); a forged or foreign position reports this sentinel.
	ErrUnknownPosition = errors.New("sessionloop: unknown replay position")

	// ErrCommandConflict reports a command that is structurally valid and
	// supported but conflicts with the session's current activity, such as
	// resolving a run that is not suspended.
	ErrCommandConflict = errors.New("sessionloop: command conflicts with current session activity")
)
