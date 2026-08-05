package session

import (
	"context"

	agentic "github.com/regularkevvv/agentic"
)

func (s *Session[O]) Interrupt(ctx context.Context) error {
	if _, err := s.requestInterrupt(ctx); err != nil {
		return err
	}
	return s.WaitForIdle(ctx)
}

// requestInterrupt is the request half of Interrupt: state validation, the
// transition to Interrupting, the run.interrupted mark, and the run-context
// cancellation (or, for a Suspended session, the spawned finishInterrupt),
// WITHOUT the settlement wait. The legacy Interrupt composes it with
// WaitForIdle; there is no durable interrupt-request fact, so the returned
// run ID is in-memory evidence only.
func (s *Session[O]) requestInterrupt(_ context.Context) (string, error) {
	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return "", err
	}
	runID := ""
	if s.run != nil {
		runID = s.run.id
	}
	switch s.state {
	case Closed:
		s.mu.Unlock()
		return "", ErrSessionClosed
	case Idle:
		s.mu.Unlock()
		return "", ErrNotRunning
	case Interrupting:
		s.mu.Unlock()
		return runID, nil
	case Running, Closing:
		s.transitionLocked(Interrupting)
		if s.run != nil {
			s.run.interrupted = true
		}
		cancel := s.runCancel
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return runID, nil
	case Suspended:
		s.transitionLocked(Interrupting)
		if s.run != nil {
			s.run.interrupted = true
		}
		s.mu.Unlock()
		go func() {
			_ = s.finishInterrupt(&agentic.Execution[O]{Status: agentic.ExecutionInterrupted})
		}()
		return runID, nil
	default:
		s.mu.Unlock()
		return "", ErrNotRunning
	}
}
