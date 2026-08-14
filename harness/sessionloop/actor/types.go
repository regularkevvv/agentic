// Package actor runs durable sessionloop conversations as passivating actors.
// Storage, leases, notification transport, authentication, and host assembly
// remain application-owned.
package actor

import (
	"context"
	"errors"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

type ActorID string
type CommandID string
type Fence uint64

type CommandState string

const (
	CommandPending    CommandState = "pending"
	CommandDispatched CommandState = "dispatched"
	CommandSettled    CommandState = "settled"
	CommandFailed     CommandState = "failed"
)

type Command struct {
	ID       CommandID
	ActorID  ActorID
	Sequence uint64
	Command  sessionloop.Command
	State    CommandState
	Receipt  *sessionloop.Receipt
	Outcome  *sessionloop.RunOutcome
	Error    string
	Created  time.Time
}

type Submission struct {
	ID        CommandID
	ActorID   ActorID
	Sequence  uint64
	Duplicate bool
}

type Lease struct {
	ActorID ActorID
	Owner   string
	Fence   Fence
	Expires time.Time
}

var (
	ErrLeaseHeld       = errors.New("session actor: lease is held elsewhere")
	ErrLeaseLost       = errors.New("session actor: lease was lost")
	ErrCommandConflict = errors.New("session actor: command identity conflict")
	ErrCommandNotFound = errors.New("session actor: command not found")
)

// CommandStore is the durable mailbox. Enqueue must be idempotent by Command
// ID: the same ID and semantic command returns Duplicate=true, while a
// different command under that ID returns ErrCommandConflict.
type CommandStore interface {
	Enqueue(context.Context, Command) (Submission, error)
	Pending(context.Context, Lease, int) ([]Command, error)
	MarkDispatched(context.Context, Lease, CommandID, sessionloop.Receipt) error
	MarkSettled(context.Context, Lease, CommandID, *sessionloop.RunOutcome) error
	MarkFailed(context.Context, Lease, CommandID, error) error
	ReadyActors(context.Context, int) ([]ActorID, error)
}

// LeaseStore grants one fenced writer per actor. Every mailbox mutation after
// activation carries the lease, allowing adapters to reject stale pods.
type LeaseStore interface {
	Acquire(context.Context, ActorID, string, time.Duration) (Lease, error)
	Renew(context.Context, Lease, time.Duration) (Lease, error)
	Release(context.Context, Lease) error
}

// Doorbell is a lossy wake-up path. CommandStore remains authoritative and a
// Supervisor periodically scans ReadyActors, so dropped notifications never
// drop commands.
type Doorbell interface {
	Ring(context.Context, ActorID) error
	Subscribe(context.Context) (Subscription, error)
}

type Subscription interface {
	Next(context.Context) (ActorID, error)
	Close() error
}

// SessionOpener translates an application actor identity into one already
// assembled SessionLoop session. It is where an application binds tenants,
// Chacla realms, providers, permissions, and durable harness state.
type SessionOpener interface {
	Open(context.Context, ActorID) (sessionloop.Session, error)
}

type SessionOpenerFunc func(context.Context, ActorID) (sessionloop.Session, error)

func (f SessionOpenerFunc) Open(ctx context.Context, id ActorID) (sessionloop.Session, error) {
	return f(ctx, id)
}

// EventSink receives authoritative and preview events while an actor owns the
// session. Implementations commonly persist authoritative projections and
// publish previews to transient subscribers.
type EventSink interface {
	Observe(context.Context, Lease, sessionloop.Event) error
}

// SnapshotSink is an optional EventSink extension. A supervisor calls it
// immediately after activation so an application projection also reflects
// recovery that happened while no actor owned the session (for example, a
// process dying during an active run).
type SnapshotSink interface {
	ObserveSnapshot(context.Context, Lease, sessionloop.Snapshot) error
}

type EventSinkFunc func(context.Context, Lease, sessionloop.Event) error

func (f EventSinkFunc) Observe(ctx context.Context, lease Lease, event sessionloop.Event) error {
	return f(ctx, lease, event)
}
