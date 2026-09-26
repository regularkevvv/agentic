package session

import (
	"context"
	"errors"

	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

var errExecutionAbandoned = errors.New("session execution abandoned for recovery")

// abandonRun invalidates volatile execution under the same mutex used by every
// journal-emitting path. Unlike Interrupt it writes no cancellation or closure.
func (s *Session[O]) abandonRun() {
	s.mu.Lock()
	if s.state == Closed {
		s.mu.Unlock()
		return
	}
	cancel := s.faultLocked(errExecutionAbandoned)
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (s *Session[O]) joinRecovery(ctx context.Context) error {
	s.mu.Lock()
	done := s.recoveryDone
	s.mu.Unlock()
	if done == nil {
		return nil
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Abandon is recovery cleanup for an unexposed native handle. SessionLoop also
// joins its dispatch goroutines before calling the owning root closer.
func (s *Session[O]) Abandon(ctx context.Context) error {
	s.abandonRun()
	if err := s.joinRecovery(ctx); err != nil {
		return err
	}
	return s.Close(ctx)
}

// Close releases the session's journal lease, event hub, and environment.
// Durable state is untouched and may later be reopened through ResumeSession.
func (s *Session[O]) Close(ctx context.Context) error {
	s.closeMu.Lock()
	defer s.closeMu.Unlock()

	s.mu.Lock()
	switch s.state {
	case Running, Closing, Interrupting:
		s.mu.Unlock()
		return ErrSessionBusy
	}
	runClosingHook := !s.closingHookDone
	if s.state != Closed {
		s.transitionLocked(Closed)
	}
	s.mu.Unlock()

	var closingHookErr error
	if runClosingHook {
		closingHookErr = harnessruntime.RunLifecycleHooks(ctx, s.lifecycle, harnessruntime.LifecycleEvent{
			Phase:     harnessruntime.LifecycleSessionClosing,
			SessionID: s.id,
		})
		if closingHookErr == nil {
			s.closingHookDone = true
		}
	}
	if !s.busClosed && s.bus != nil {
		s.bus.Close()
		s.busClosed = true
	}
	var journalErr, environmentErr error
	if !s.journalClosed && s.journal != nil {
		journalErr = s.journal.Close(ctx)
		if journalErr == nil {
			s.journalClosed = true
		}
	}
	if !s.environmentClosed && s.environment != nil {
		environmentErr = s.environment.Close(ctx)
		if environmentErr == nil {
			s.environmentClosed = true
		}
	}
	cleanupErr := errors.Join(closingHookErr, journalErr, environmentErr)
	if cleanupErr == nil && !s.closedHookDone {
		if err := harnessruntime.RunLifecycleHooks(ctx, s.lifecycle, harnessruntime.LifecycleEvent{
			Phase:     harnessruntime.LifecycleSessionClosed,
			SessionID: s.id,
		}); err != nil {
			return err
		}
		s.closedHookDone = true
	}
	return cleanupErr
}
