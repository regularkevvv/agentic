package runtime

import "time"

// Clock supplies wall-clock observations to the session state machine.
type Clock interface {
	Now() time.Time
}

// IDGenerator supplies opaque runtime identifiers. Prefix identifies the
// domain noun ("sess", "run", "queue", or "recovery"), not an encoding.
type IDGenerator interface {
	New(prefix string) (string, error)
}
