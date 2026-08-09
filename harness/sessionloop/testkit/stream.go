package testkit

import (
	"context"
	"fmt"
	"io"
	"sync"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

const defaultStreamBuffer = 64

// stream is one subscription. It is safe for one consumer: Next may be
// called from a single goroutine while the engine delivers concurrently.
//
// Replayed history is preloaded without a bound; the buffer limit applies to
// live deliveries. A full buffer drops previews (counted into the next
// delivered event's Dropped) and terminally fails the stream with ErrLagged
// when an authoritative event cannot be buffered, because authoritative
// facts must never be silently lost (law L5).
type stream struct {
	state   *sessionState
	limit   int
	preview bool

	mu      sync.Mutex
	queue   []sessionloop.Event
	dropped uint64
	lagged  bool
	ended   bool
	closed  bool
	signal  chan struct{}
}

func newStream(state *sessionState, limit int, preview bool) *stream {
	return &stream{
		state:   state,
		limit:   limit,
		preview: preview,
		signal:  make(chan struct{}, 1),
	}
}

func (s *stream) deliver(event sessionloop.Event, authoritative bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.lagged || s.ended {
		return
	}
	if !authoritative && !s.preview {
		return
	}
	if len(s.queue) >= s.limit {
		if authoritative {
			s.lagged = true
			s.wakeLocked()
		} else {
			s.dropped++
		}
		return
	}
	if s.dropped > 0 {
		event.Dropped += s.dropped
		s.dropped = 0
	}
	s.queue = append(s.queue, event)
	s.wakeLocked()
}

// end marks the session closed: remaining buffered events stay readable and
// Next reports io.EOF afterwards.
func (s *stream) end() {
	s.mu.Lock()
	s.ended = true
	s.wakeLocked()
	s.mu.Unlock()
}

func (s *stream) wakeLocked() {
	select {
	case s.signal <- struct{}{}:
	default:
	}
}

// Next returns the next event. It reports io.EOF after the stream closed
// normally, an error wrapping sessionloop.ErrLagged after the subscriber
// fell terminally behind, and the context's error when the wait is canceled.
func (s *stream) Next(ctx context.Context) (sessionloop.Event, error) {
	for {
		if err := ctx.Err(); err != nil {
			return sessionloop.Event{}, err
		}
		s.mu.Lock()
		switch {
		case s.closed:
			s.mu.Unlock()
			return sessionloop.Event{}, io.EOF
		case s.lagged:
			s.mu.Unlock()
			return sessionloop.Event{}, fmt.Errorf("testkit: authoritative events were lost for this subscriber: %w", sessionloop.ErrLagged)
		case len(s.queue) > 0:
			event := s.queue[0]
			s.queue = s.queue[1:]
			s.mu.Unlock()
			return event, nil
		case s.ended:
			s.mu.Unlock()
			return sessionloop.Event{}, io.EOF
		}
		s.mu.Unlock()
		select {
		case <-ctx.Done():
			return sessionloop.Event{}, ctx.Err()
		case <-s.signal:
		}
	}
}

// Close is idempotent and detaches the subscription.
func (s *stream) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.wakeLocked()
	s.mu.Unlock()

	s.state.mu.Lock()
	delete(s.state.subs, s)
	s.state.mu.Unlock()
	return nil
}
