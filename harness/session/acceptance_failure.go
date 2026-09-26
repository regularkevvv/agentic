package session

// Acceptance failure invalidates the live view, not durable history. Append's
// error cannot distinguish rollback from a committed batch whose reply was
// lost. Only closing and reconstructing the session can resolve that ambiguity.

import (
	"context"

	"github.com/regularkevvv/agentic/harness/store"
)

// appendAcceptanceLocked runs with s.mu held. Invalidation precedes unlock,
// so concurrent journal readers cannot expose a committed but unattributed key.
// Even a known pre-commit adapter error is treated conservatively: Journal does
// not expose a portable proof that an error means no durable effect.
func (s *Session[O]) appendAcceptanceLocked(ctx context.Context, entries ...store.PendingEntry) (store.Commit, error) {
	// Before invoking storage we know there cannot have been a durable effect.
	if err := ctx.Err(); err != nil {
		return store.Commit{}, err
	}
	commit, err := s.journal.Append(ctx, s.cursor, entries...)
	if err == nil {
		return commit, nil
	}
	s.acceptanceFault = err
	if cancel := s.faultLocked(err); cancel != nil {
		// runCancel is created with context.WithCancel, never a user callback.
		cancel()
	}
	return store.Commit{}, s.acceptanceFaultLocked()
}

func (s *Session[O]) acceptanceFaultLocked() error {
	if s.acceptanceFault == nil {
		return nil
	}
	return &FaultError{SessionID: s.id, Cause: s.acceptanceFault}
}

// availability returns the state-change signal and error from the same locked
// observation. A stream cannot miss a fault between checking and waiting.
func (v *LoopView[O]) availability() (<-chan struct{}, error) {
	if v.isClosed() {
		return nil, loopClosedError()
	}
	return v.projectionAvailability()
}

// Existing streams retain their normal drain/EOF behavior on Close. Only an
// unresolved acceptance error invalidates projection of their buffered records.
func (v *LoopView[O]) projectionAvailability() (<-chan struct{}, error) {
	v.inner.mu.Lock()
	defer v.inner.mu.Unlock()
	return v.inner.stateChange, mapLoopError(v.inner.acceptanceFaultLocked())
}
