package session

import (
	"context"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/store"
)

func (s *Session[O]) Interrupt(ctx context.Context) error {
	if _, err := s.requestInterrupt(ctx, ""); err != nil {
		return err
	}
	return s.WaitForIdle(ctx)
}

// requestInterrupt is the request half of Interrupt: state validation, the
// transition to Interrupting, the run.interrupted mark, and the run-context
// cancellation (or, for a Suspended session, the spawned finishInterrupt),
// WITHOUT the settlement wait. The legacy Interrupt composes it with
// WaitForIdle. SessionLoop may additionally supply a durable command marker;
// the legacy API does not, preserving its existing journal shape.
//
// A non-empty expectedRunID pins the interrupt to one run identity (law L8):
// it is revalidated under s.mu before any transition, so an interrupt aimed
// at a run that settled concurrently fails with errStaleRunTarget instead of
// canceling its successor. The legacy Interrupt and the view's Close pass
// "" (session-targeted, no check) — zero behavior change for them.
func (s *Session[O]) requestInterrupt(ctx context.Context, expectedRunID string) (string, error) {
	return s.requestInterruptCommand(ctx, expectedRunID, nil)
}

func (s *Session[O]) requestInterruptCommand(
	ctx context.Context,
	expectedRunID string,
	command *loopCommandAcceptedPayload,
) (string, error) {
	var publish []store.Entry
	defer func() {
		if len(publish) > 0 {
			s.publishOwn(publish, agentic.EventAuthoritative)
		}
	}()
	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return "", err
	}
	if err := s.staleTargetLocked(expectedRunID); err != nil {
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
	case Running, Closing, Interrupting, Suspended:
	default:
		s.mu.Unlock()
		return "", ErrNotRunning
	}
	if command != nil {
		accepted := *command
		accepted.RunID = runID
		entry, err := pending(s.codec, kindCommandAccepted, accepted)
		if err != nil {
			s.mu.Unlock()
			return "", err
		}
		commit, err := s.journal.Append(ctx, s.cursor, entry)
		if err != nil {
			s.mu.Unlock()
			return "", err
		}
		s.cursor = commit.Cursor
		publish = commit.Entries
	}
	switch s.state {
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
