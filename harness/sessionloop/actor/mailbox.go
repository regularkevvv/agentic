// Package actor separates submission from independently scheduled session work.
// Applications use Mailbox and Worker. Assembly uses the adapter ports; their
// laws and failure assumptions are specified in spec/CONTRACT.md.
package actor

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

type ActorID string
type CommandID string
type Fence uint64

// Command is immutable delivery intent, not execution history. Sequence is
// assigned by the mailbox. ID is scoped to ActorID and survives mailbox cleanup.
type Command struct {
	ID       CommandID
	ActorID  ActorID
	Sequence uint64
	Command  sessionloop.Command
}

// Normalize fixes dispatch identity before submission; retries never choose a
// new kind, run target or idempotency key based on the current session snapshot.
func (c Command) Normalize() (Command, error) {
	if c.ID == "" || c.ActorID == "" {
		return Command{}, fmt.Errorf("actor and command IDs are required: %w", sessionloop.ErrInvalidCommand)
	}
	if (c.Command.ID != "" && c.Command.ID != sessionloop.CommandID(c.ID)) ||
		(c.Command.IdempotencyKey != "" && c.Command.IdempotencyKey != string(c.ID)) {
		return Command{}, ErrCommandConflict
	}
	c.Command = c.Command.Clone()
	c.Command.ID = sessionloop.CommandID(c.ID)
	c.Command.IdempotencyKey = string(c.ID)
	if err := c.Command.Validate(); err != nil {
		return Command{}, err
	}
	if c.Command.Input != nil {
		if err := sessionloop.ValidateInput(*c.Command.Input); err != nil {
			return Command{}, err
		}
	}
	return c, nil
}

// Submission acknowledges retained work, not harness acceptance or completion.
// Guarantee states the adapter's failure boundary, including for duplicate IDs.
type Submission struct {
	ID        CommandID
	ActorID   ActorID
	Sequence  uint64
	Duplicate bool
	Guarantee sessionloop.AcceptanceGuarantee
}

// Mailbox never opens a session or runs a worker. Success means the exact input
// is retained and discoverable under the declared guarantee. Same-ID retries
// return the original submission; changed semantics return ErrCommandConflict.
type Mailbox interface {
	Submit(context.Context, Command) (Submission, error)
}

// Worker independently executes accepted work until canceled. Polling, receive,
// notifications and distributed scheduling are implementation choices, not
// additional application-facing contracts.
type Worker interface{ Run(context.Context) error }

// Lease is one scoped mutation grant, not a connection or another service.
// Only runtime assembly/adapters inspect it; applications submit commands.
type Lease struct {
	ActorID ActorID
	Owner   string
	Fence   Fence
	// Expires is informational. Validate against the current stored deadline,
	// not this snapshot: renewal preserves ActorID/Owner/Fence identity.
	Expires time.Time
}

var (
	ErrLeaseHeld           = errors.New("session actor: lease is held elsewhere")
	ErrLeaseLost           = errors.New("session actor: lease was lost")
	ErrNoWork              = errors.New("session actor: no recoverable work")
	ErrCommandConflict     = errors.New("session actor: command identity conflict")
	ErrCommandNotFound     = errors.New("session actor: command not found")
	ErrInvalidReceipt      = errors.New("session actor: receipt does not prove acceptance")
	ErrGenerationExhausted = errors.New("session actor: ownership generation exhausted")
	ErrWorkerRunning       = errors.New("session actor: worker already running")
)
