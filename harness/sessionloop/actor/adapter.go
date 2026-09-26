// Runtime assembly ports connect the small application API to a lawful adapter.
// See spec/SessionContract/Adapter.lean for the corresponding proof obligations.

package actor

import (
	"context"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// Ownership grants one authoritative writer. Acquire atomically records a fresh
// generation AND unfinished execution before Open. Renewal preserves identity.
// Release is called only after durable quiescence and successful handle closure;
// it clears execution-needed state, never pending input. On errors the Worker
// leaves the grant to expire, preserving the recovery obligation.
type Ownership interface {
	Acquire(context.Context, ActorID, string, time.Duration) (Lease, error)
	Renew(context.Context, Lease, time.Duration) (Lease, error)
	Release(context.Context, Lease) error
}

// Adapter is infrastructure, not an application repository. Its mailbox,
// ownership and session journal must share a coherent authority protocol.
// A preflight lease check followed by an unguarded write is NOT sufficient.
type Adapter interface {
	Mailbox
	Ownership
	// Guarantee is immutable for the adapter's lifetime.
	Guarantee() sessionloop.AcceptanceGuarantee
	// Receive waits for claimable work: pending input OR unfinished execution.
	// It must fairly rediscover retained work without a future submission, wake
	// signal or repair cron, including after ownership expires. It is safe for
	// concurrent receivers; Acquire resolves any competing discoveries.
	Receive(context.Context) (ActorID, error)
	// Pending returns only undelivered commands, ordered strictly by Sequence,
	// after the exclusive cursor. Pagination must not hide later steering
	// behind a start command waiting for the active run to finish.
	Pending(context.Context, Lease, uint64, int) ([]Command, error)
	// Acknowledge deletes delivery state, not identity or journal history. The
	// Worker passes a receipt checked against the opened journal. This mutation
	// must atomically validate the lease. Retries after deletion are harmless.
	Acknowledge(context.Context, Lease, CommandID, sessionloop.Receipt) error
}

// Session adds acceptance lookup to the neutral session protocol. This lookup
// is essential after commit/reply-loss: a retried Start must find its receipt
// even if the recovered session is no longer idle. RejectionRecorder resolves
// terminally invalid commands in that same journal before mailbox deletion.
type Session interface {
	sessionloop.Session
	sessionloop.AcceptanceReader
	sessionloop.RejectionRecorder
	sessionloop.RecoveryCloser
}

// SessionOpener restores the SAME session under the grant. Assembly persists
// its actor-to-journal binding before returning and binds authority to every
// journal commit, including asynchronous execution and recovery writes. Missing
// or corrupt history is an error, never permission to create a replacement.
// Creation is allowed only for a genuinely new, durably bound session.
type SessionOpener interface {
	Open(context.Context, Lease) (Session, error)
}

type SessionOpenerFunc func(context.Context, Lease) (Session, error)

func (f SessionOpenerFunc) Open(ctx context.Context, lease Lease) (Session, error) {
	return f(ctx, lease)
}

// EventSink projects observations. It is not the execution journal and does not
// decide acceptance or settlement. Projections need their own replay/dedup policy.
type EventSink interface {
	Observe(context.Context, Lease, sessionloop.Event) error
}
type SnapshotSink interface {
	ObserveSnapshot(context.Context, Lease, sessionloop.Snapshot) error
}
type EventSinkFunc func(context.Context, Lease, sessionloop.Event) error

func (f EventSinkFunc) Observe(ctx context.Context, lease Lease, event sessionloop.Event) error {
	return f(ctx, lease, event)
}
