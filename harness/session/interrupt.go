package session

import (
	"context"

	agentic "github.com/regularkevvv/agentic"
)

func (s *Session[O]) Interrupt(ctx context.Context) error {
	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return err
	}
	switch s.state {
	case Closed:
		s.mu.Unlock()
		return ErrSessionClosed
	case Idle:
		s.mu.Unlock()
		return ErrNotRunning
	case Interrupting:
		s.mu.Unlock()
		return s.WaitForIdle(ctx)
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
		return s.WaitForIdle(ctx)
	case Suspended:
		s.transitionLocked(Interrupting)
		if s.run != nil {
			s.run.interrupted = true
		}
		s.mu.Unlock()
		go func() {
			_ = s.finishInterrupt(&agentic.Execution[O]{Status: agentic.ExecutionInterrupted})
		}()
		return s.WaitForIdle(ctx)
	default:
		s.mu.Unlock()
		return ErrNotRunning
	}
}
