// Acceptance lookup separates journal facts from command delivery and execution.

package sessionloop

import "context"

// AcceptanceReader looks up an exact keyed command without dispatching it.
// A found receipt must survive the host's advertised failure boundary. Same-key
// different semantics return ErrCommandConflict. Absence is not completion and
// does not cancel an in-flight acceptance; retry Dispatch with the same key.
// Durable actor hosts implement this in addition to Session.
type AcceptanceReader interface {
	Acceptance(context.Context, Command) (Receipt, bool, error)
}

// Rejection is a terminal command resolution, not a failed run or transient I/O
// error. Stable reasons avoid persisting arbitrary error strings or credentials.
type Rejection string

const (
	RejectionStaleRun       Rejection = "stale_run"
	RejectionNotRunning     Rejection = "not_running"
	RejectionUnsupported    Rejection = "unsupported"
	RejectionInvalidCommand Rejection = "invalid_command"
)

func (r Rejection) Valid() bool {
	switch r {
	case RejectionStaleRun, RejectionNotRunning, RejectionUnsupported, RejectionInvalidCommand:
		return true
	default:
		return false
	}
}

// RejectionRecorder atomically records acceptance plus a terminal rejection in
// the same journal, without executing the command or altering any active run.
// Repeating an already accepted identity returns its original receipt; it must
// never overwrite acceptance with rejection. Rejected receipts survive reopen
// and are returned by AcceptanceReader. This is an optional actor-host port.
type RejectionRecorder interface {
	Reject(context.Context, Command, Rejection) (Receipt, error)
}
