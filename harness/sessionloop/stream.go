package sessionloop

import "context"

// Position is an opaque replay token plus a monotonic sequence inside one
// session history. Consumers may compare positions for equality and use the
// numeric sequence for monotonicity checks; they must pass the opaque token
// back unchanged and must not parse it (law L6).
type Position struct {
	Sequence uint64
	Token    string
}

// IsZero reports whether the position is the zero position. A zero position
// means the event is not replayable (law L6).
func (p Position) IsZero() bool {
	return p.Sequence == 0 && p.Token == ""
}

// Equal reports whether both positions are identical.
func (p Position) Equal(other Position) bool {
	return p == other
}

// SubscribeOptions configures one event stream.
//
// After requests replay of authoritative events strictly after the given
// position; the zero position replays from the beginning of the available
// history. Preview opts in to lossy preview events (law L5). Buffer bounds
// the subscriber's queue of undelivered events; implementations choose a
// default when it is not positive.
type SubscribeOptions struct {
	After   Position
	Preview bool
	Buffer  int
}

// Stream delivers ordered events for one subscription.
//
// Next uses its context only for that wait. It returns io.EOF after the
// stream closed normally, an error wrapping ErrLagged when the subscriber
// fell terminally behind authoritative events, and the context's error when
// the wait is canceled. Close is idempotent (law L4).
type Stream interface {
	Next(context.Context) (Event, error)
	Close() error
}
