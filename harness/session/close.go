package session

import (
	"context"
	"errors"
)

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
	if s.state != Closed {
		s.transitionLocked(Closed)
	}
	s.mu.Unlock()

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
	return errors.Join(journalErr, environmentErr)
}
