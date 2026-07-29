package event

import (
	"fmt"
	"sync"
)

type ErrSubscriberLagged struct {
	LastCursor uint64
}

func (e *ErrSubscriberLagged) Error() string {
	return fmt.Sprintf("subscriber lagged behind authoritative events after cursor %d", e.LastCursor)
}

type SubscribeOptions struct {
	AfterCursor uint64
	Buffer      int
	Preview     bool
}

// Subscription is independent from execution. Err receives one lag error when
// disconnected for authoritative backpressure; both channels then close.
type Subscription struct {
	Events <-chan Record
	Err    <-chan error
	close  func()
	once   sync.Once
}

// NewSubscription allows event-hub adapters to expose the common public
// subscription shape without leaking their internal subscriber type.
func NewSubscription(events <-chan Record, errs <-chan error, closeFn func()) *Subscription {
	return &Subscription{Events: events, Err: errs, close: closeFn}
}

func (s *Subscription) Close() {
	if s == nil {
		return
	}
	s.once.Do(func() {
		if s.close != nil {
			s.close()
		}
	})
}
