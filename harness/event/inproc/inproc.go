// Package inproc provides the bounded, nonblocking process-local event hub.
package inproc

import (
	"context"
	"sync"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/event"
)

type Factory struct{}

func NewFactory() Factory { return Factory{} }

func (Factory) Open(ctx context.Context, history []event.Record) (event.Hub, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return New(history), nil
}

type subscriber struct {
	id           uint64
	events       chan event.Record
	errors       chan error
	preview      bool
	dropped      uint64
	lastCursor   uint64
	disconnected bool
}

// Hub fans one producer out to bounded subscribers without ever blocking it.
type Hub struct {
	mu          sync.Mutex
	nextID      uint64
	cursor      uint64
	history     []event.Record
	subscribers map[uint64]*subscriber
	closed      bool
}

func New(history []event.Record) *Hub {
	hub := &Hub{subscribers: make(map[uint64]*subscriber)}
	for _, record := range history {
		if record.Nature == agentic.EventPreview || record.Cursor == 0 {
			continue
		}
		cloned := event.Clone(record)
		hub.history = append(hub.history, cloned)
		if cloned.Cursor > hub.cursor {
			hub.cursor = cloned.Cursor
		}
	}
	return hub
}

func (h *Hub) Cursor() uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.cursor
}

func (h *Hub) PublishDurable(record event.Record) {
	if record.Nature == agentic.EventPreview || record.Cursor == 0 {
		panic("PublishDurable requires a durable event")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return
	}
	if record.Cursor > h.cursor {
		h.cursor = record.Cursor
	}
	record = event.Clone(record)
	h.history = append(h.history, record)
	h.publishLocked(record)
}

func (h *Hub) PublishPreview(record event.Record) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return
	}
	record.Nature = agentic.EventPreview
	record.Cursor = h.cursor
	h.publishLocked(event.Clone(record))
}

func (h *Hub) Subscribe(options event.SubscribeOptions) *event.Subscription {
	if options.Buffer <= 0 {
		options.Buffer = 64
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.nextID++
	sub := &subscriber{
		id:         h.nextID,
		events:     make(chan event.Record, options.Buffer),
		errors:     make(chan error, 1),
		preview:    options.Preview,
		lastCursor: options.AfterCursor,
	}
	if h.closed {
		sub.disconnected = true
		sub.errors <- nil
		close(sub.events)
		close(sub.errors)
		return event.NewSubscription(sub.events, sub.errors, nil)
	}
	h.subscribers[sub.id] = sub
	for _, record := range h.history {
		if record.Cursor <= options.AfterCursor {
			continue
		}
		if !h.deliverLocked(sub, record) {
			break
		}
	}
	return event.NewSubscription(sub.events, sub.errors, func() {
		h.mu.Lock()
		defer h.mu.Unlock()
		if current := h.subscribers[sub.id]; current != nil {
			h.closeLocked(current, nil)
		}
	})
}

func (h *Hub) Close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return
	}
	h.closed = true
	for _, sub := range h.subscribers {
		h.closeLocked(sub, nil)
	}
}

func (h *Hub) publishLocked(record event.Record) {
	for _, sub := range h.subscribers {
		if record.Nature == agentic.EventPreview && !sub.preview {
			continue
		}
		h.deliverLocked(sub, record)
	}
}

func (h *Hub) deliverLocked(sub *subscriber, record event.Record) bool {
	if sub.disconnected {
		return false
	}
	if record.Nature != agentic.EventPreview && record.Cursor <= sub.lastCursor {
		return true
	}
	if sub.dropped > 0 {
		record.Dropped.Preview += sub.dropped
	}
	select {
	case sub.events <- event.Clone(record):
		if sub.dropped > 0 {
			sub.dropped = 0
		}
		if record.Nature != agentic.EventPreview {
			sub.lastCursor = record.Cursor
		}
		return true
	default:
		if record.Nature == agentic.EventPreview {
			sub.dropped++
			return true
		}
		h.closeLocked(sub, &event.ErrSubscriberLagged{LastCursor: sub.lastCursor})
		return false
	}
}

func (h *Hub) closeLocked(sub *subscriber, terminal error) {
	if sub.disconnected {
		return
	}
	sub.disconnected = true
	delete(h.subscribers, sub.id)
	sub.errors <- terminal
	close(sub.events)
	close(sub.errors)
}

var _ event.Factory = Factory{}
var _ event.Hub = (*Hub)(nil)
