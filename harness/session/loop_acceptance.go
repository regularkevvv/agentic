// Actor recovery reads the harness's existing acceptance journal; there is no
// second command log and lookup never starts or steers execution.

package session

import (
	"context"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// Acceptance implements sessionloop.AcceptanceReader from restored journal
// receipts. It runs before dispatch eligibility checks after an ambiguous retry.
func (v *LoopView[O]) Acceptance(ctx context.Context, command sessionloop.Command) (sessionloop.Receipt, bool, error) {
	if err := ctx.Err(); err != nil {
		return sessionloop.Receipt{}, false, err
	}
	// A rejected unsupported command must still be queryable on this host.
	if err := command.Validate(); err != nil {
		return sessionloop.Receipt{}, false, err
	}
	if command.IdempotencyKey == "" {
		return sessionloop.Receipt{}, false, fmt.Errorf("acceptance lookup requires an idempotency key: %w", sessionloop.ErrInvalidCommand)
	}
	v.lifecycleMu.RLock()
	defer v.lifecycleMu.RUnlock()
	if v.isClosed() {
		return sessionloop.Receipt{}, false, loopClosedError()
	}
	digest, err := loopCommandDigest(command)
	if err != nil {
		return sessionloop.Receipt{}, false, err
	}
	v.mu.Lock()
	accepted, found := v.idempotency[command.IdempotencyKey]
	v.mu.Unlock()
	if !found {
		return sessionloop.Receipt{}, false, nil
	}
	if accepted.digest != digest || (command.ID != "" && accepted.receipt.CommandID != command.ID) {
		return sessionloop.Receipt{}, false, sessionloop.ErrCommandConflict
	}
	return accepted.receipt, true, nil
}

var _ sessionloop.AcceptanceReader = (*LoopView[any])(nil)

// Reject persists a terminal command resolution in the acceptance journal.
// It does not invent a run, rewrite a transcript or overwrite a prior receipt.
func (v *LoopView[O]) Reject(ctx context.Context, command sessionloop.Command, reason sessionloop.Rejection) (sessionloop.Receipt, error) {
	if err := ctx.Err(); err != nil {
		return sessionloop.Receipt{}, err
	}
	if err := command.Validate(); err != nil {
		return sessionloop.Receipt{}, err
	}
	if command.ID == "" || command.IdempotencyKey == "" || !reason.Valid() {
		return sessionloop.Receipt{}, sessionloop.ErrInvalidCommand
	}
	v.lifecycleMu.RLock()
	defer v.lifecycleMu.RUnlock()
	if v.isClosed() {
		return sessionloop.Receipt{}, loopClosedError()
	}
	digest, err := loopCommandDigest(command)
	if err != nil {
		return sessionloop.Receipt{}, err
	}
	receipt, owned, claim, err := v.claimIdempotencyKey(ctx, command.IdempotencyKey, digest)
	if err != nil {
		return sessionloop.Receipt{}, err
	}
	if !owned {
		if receipt.CommandID != command.ID {
			return sessionloop.Receipt{}, sessionloop.ErrCommandConflict
		}
		return receipt, nil
	}
	defer v.releaseIdempotencyClaim(command.IdempotencyKey, claim)
	marker := v.commandAcceptance(command, command.ID, digest)
	marker.Rejection = string(reason)
	entry, err := pending(v.inner.codecRef(), kindCommandAccepted, marker)
	if err != nil {
		return sessionloop.Receipt{}, err
	}
	v.inner.mu.Lock()
	if v.inner.state == Faulted || v.inner.state == Closed {
		v.inner.mu.Unlock()
		return sessionloop.Receipt{}, sessionloop.ErrSessionFaulted
	}
	commit, err := v.inner.journal.Append(ctx, v.inner.cursor, entry)
	if err == nil {
		v.inner.cursor = commit.Cursor
	}
	v.inner.mu.Unlock()
	if err != nil {
		return sessionloop.Receipt{}, mapLoopError(err)
	}
	v.inner.publishOwn(commit.Entries, agentic.EventAuthoritative)
	receipt = sessionloop.Receipt{CommandID: command.ID, SessionID: v.ID(), Position: loopPosition(commit.Cursor), Guarantee: sessionloop.AcceptanceDurable, Rejection: reason}
	v.rememberCommandAcceptance(marker, receipt)
	return receipt, nil
}

var _ sessionloop.RejectionRecorder = (*LoopView[any])(nil)
