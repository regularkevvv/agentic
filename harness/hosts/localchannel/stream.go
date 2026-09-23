// Package localchannel provides bounded live projections with journal-derived replay.
// Slow consumers get ErrLagged; channels never hold authoritative execution work.
package localchannel

import (
	"context"
	"io"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

type stream struct {
	owner           *handle
	options         sessionloop.SubscribeOptions
	replay, pending []sessionloop.Event
	dropped         uint64
	ordinal         uint64
	err             error
	ended, closed   bool
	changed         chan struct{}
}

func (s *handle) Subscribe(ctx context.Context, options sessionloop.SubscribeOptions) (sessionloop.Stream, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.host.mu.Lock()
	defer s.host.mu.Unlock()
	if s.closed {
		return nil, sessionloop.ErrSessionClosed
	}
	if options.After.Sequence > s.state.snapshot.Position.Sequence {
		return nil, sessionloop.ErrUnknownPosition
	}
	if options.Buffer <= 0 {
		options.Buffer = 256
	}
	sub := &stream{owner: s, options: options, changed: make(chan struct{})}
	for _, event := range s.state.history {
		if event.Position.Sequence > options.After.Sequence {
			sub.replay = append(sub.replay, event.Clone())
		}
	}
	s.state.streams[sub] = struct{}{}
	return sub, nil
}

func (s *stream) signalLocked()       { close(s.changed); s.changed = make(chan struct{}) }
func (s *stream) endLocked(err error) { s.ended, s.err = true, err; s.signalLocked() }

func (s *stream) deliverLocked(event sessionloop.Event) {
	if event.Nature == sessionloop.EventAuthoritative && !event.Position.IsZero() && event.Position.Sequence <= s.options.After.Sequence {
		return
	}
	if s.ended || s.closed || (event.Nature == sessionloop.EventPreview && !s.options.Preview) {
		return
	}
	if len(s.pending) >= s.options.Buffer {
		if event.Nature == sessionloop.EventPreview {
			s.dropped++
		} else {
			s.endLocked(sessionloop.ErrLagged)
		}
		return
	}
	value := event.Clone()
	if value.Nature == sessionloop.EventPreview {
		s.ordinal++
		value.Ordinal = s.ordinal
	}
	value.Dropped += s.dropped
	s.dropped = 0
	s.pending = append(s.pending, value)
	s.signalLocked()
}

func (s *stream) Next(ctx context.Context) (sessionloop.Event, error) {
	h := s.owner.host
	h.mu.Lock()
	defer h.mu.Unlock()
	for {
		if err := ctx.Err(); err != nil {
			return sessionloop.Event{}, err
		}
		if s.closed {
			return sessionloop.Event{}, io.EOF
		}
		if s.err != nil {
			return sessionloop.Event{}, s.err
		}
		if len(s.replay) > 0 {
			event := s.replay[0]
			s.replay = s.replay[1:]
			return event, nil
		}
		if len(s.pending) > 0 {
			event := s.pending[0]
			s.pending = s.pending[1:]
			return event, nil
		}
		if s.ended {
			return sessionloop.Event{}, io.EOF
		}
		changed := s.changed
		h.mu.Unlock()
		select {
		case <-ctx.Done():
		case <-changed:
		}
		h.mu.Lock()
	}
}

func (s *stream) Close() error {
	s.owner.host.mu.Lock()
	defer s.owner.host.mu.Unlock()
	if !s.closed {
		s.closed = true
		s.replay, s.pending = nil, nil
		delete(s.owner.state.streams, s)
		s.signalLocked()
	}
	return nil
}
