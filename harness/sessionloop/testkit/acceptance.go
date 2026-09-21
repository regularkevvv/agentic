// Acceptance lookup gives worker tests the same replay seam as a journal-backed
// harness. State survives handle restart, not destruction of the testkit Host.

package testkit

import (
	"context"
	"encoding/json"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func commandSemantics(command sessionloop.Command) string {
	command.ID, command.IdempotencyKey = "", ""
	encoded, _ := json.Marshal(command)
	return string(encoded)
}

func (s *session) Acceptance(ctx context.Context, command sessionloop.Command) (sessionloop.Receipt, bool, error) {
	if err := ctx.Err(); err != nil {
		return sessionloop.Receipt{}, false, err
	}
	if err := command.Validate(); err != nil {
		return sessionloop.Receipt{}, false, err
	}
	if command.IdempotencyKey == "" {
		return sessionloop.Receipt{}, false, sessionloop.ErrInvalidCommand
	}
	s.state.mu.Lock()
	defer s.state.mu.Unlock()
	if s.closed {
		return sessionloop.Receipt{}, false, sessionloop.ErrSessionClosed
	}
	receipt, found := s.state.keys[command.IdempotencyKey]
	if !found {
		return sessionloop.Receipt{}, false, nil
	}
	if s.state.keyCommands[command.IdempotencyKey] != commandSemantics(command) ||
		(command.ID != "" && receipt.CommandID != command.ID) {
		return sessionloop.Receipt{}, false, sessionloop.ErrCommandConflict
	}
	return receipt, true, nil
}

var _ sessionloop.AcceptanceReader = (*session)(nil)

func (s *session) Reject(ctx context.Context, command sessionloop.Command, reason sessionloop.Rejection) (sessionloop.Receipt, error) {
	if err := ctx.Err(); err != nil {
		return sessionloop.Receipt{}, err
	}
	if err := command.Validate(); err != nil {
		return sessionloop.Receipt{}, err
	}
	if command.ID == "" || command.IdempotencyKey == "" || !reason.Valid() {
		return sessionloop.Receipt{}, sessionloop.ErrInvalidCommand
	}
	if !s.state.host.idempotent {
		return sessionloop.Receipt{}, sessionloop.ErrUnsupported
	}
	s.state.mu.Lock()
	defer s.state.mu.Unlock()
	if s.closed {
		return sessionloop.Receipt{}, sessionloop.ErrSessionClosed
	}
	if receipt, found := s.state.keys[command.IdempotencyKey]; found {
		if s.state.keyCommands[command.IdempotencyKey] != commandSemantics(command) || receipt.CommandID != command.ID {
			return sessionloop.Receipt{}, sessionloop.ErrCommandConflict
		}
		return receipt, nil
	}
	position := s.state.appendLocked(sessionloop.Event{Kind: "testkit.command.rejected", CommandID: command.ID})
	receipt := sessionloop.Receipt{CommandID: command.ID, SessionID: s.ID(), Position: position, Guarantee: sessionloop.AcceptanceDurable, Rejection: reason}
	if s.state.keys == nil {
		s.state.keys = make(map[string]sessionloop.Receipt)
		s.state.keyCommands = make(map[string]string)
	}
	s.state.keys[command.IdempotencyKey] = receipt
	s.state.keyCommands[command.IdempotencyKey] = commandSemantics(command)
	return receipt, nil
}

var _ sessionloop.RejectionRecorder = (*session)(nil)
