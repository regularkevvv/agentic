package app

import (
	"context"
	"errors"
	"time"

	uit "github.com/regularkevvv/agentic/tui"
)

type bridgeUpdate struct {
	events   []uit.Event
	snapshot *uit.Snapshot
	err      error
}

type eventBridge struct {
	updates chan bridgeUpdate
	cancel  context.CancelFunc
}

func startBridge(parent context.Context, session uit.Session, initial uit.Snapshot, previewHz, buffer int) *eventBridge {
	ctx, cancel := context.WithCancel(parent)
	bridge := &eventBridge{updates: make(chan bridgeUpdate, 4), cancel: cancel}
	go bridge.run(ctx, session, initial, previewHz, buffer)
	return bridge
}

func (b *eventBridge) Close() {
	if b != nil && b.cancel != nil {
		b.cancel()
	}
}

func (b *eventBridge) run(ctx context.Context, session uit.Session, initial uit.Snapshot, previewHz, buffer int) {
	defer close(b.updates)
	if session == nil {
		b.send(ctx, bridgeUpdate{err: errors.New("event bridge requires a session")})
		return
	}
	if previewHz <= 0 {
		previewHz = 60
	}
	if buffer <= 0 {
		buffer = 256
	}
	interval := time.Second / time.Duration(previewHz)
	after := initial.Cursor
	for ctx.Err() == nil {
		source := session.Subscribe(uit.SubscribeOptions{AfterCursor: after, Buffer: buffer, Preview: true})
		if source == nil {
			b.send(ctx, bridgeUpdate{err: errors.New("session returned a nil subscription")})
			return
		}
		resync, terminal := b.consume(ctx, source, interval, &after)
		source.Close()
		if terminal != nil {
			if !errors.Is(terminal, context.Canceled) {
				b.send(ctx, bridgeUpdate{err: terminal})
			}
			return
		}
		if !resync {
			return
		}
		snapshot, err := session.Snapshot(ctx)
		if err != nil {
			b.send(ctx, bridgeUpdate{err: err})
			return
		}
		after = snapshot.Cursor
		if !b.send(ctx, bridgeUpdate{snapshot: &snapshot}) {
			return
		}
	}
}

func (b *eventBridge) consume(ctx context.Context, source uit.Subscription, interval time.Duration, after *uint64) (bool, error) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	events, failures := source.Events(), source.Errors()
	batch := make([]uit.Event, 0, 32)
	flush := func() bool {
		if len(batch) == 0 {
			return true
		}
		copyOfBatch := append([]uit.Event(nil), batch...)
		batch = batch[:0]
		return b.send(ctx, bridgeUpdate{events: copyOfBatch})
	}
	for events != nil || failures != nil {
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case event, ok := <-events:
			if !ok {
				events = nil
				continue
			}
			if event.Durable && event.Cursor > *after {
				*after = event.Cursor
			}
			if event.Dropped > 0 {
				flush()
				return true, nil
			}
			batch = append(batch, event)
		case err, ok := <-failures:
			if !ok {
				failures = nil
				continue
			}
			if err != nil {
				flush()
				return true, nil
			}
		case <-ticker.C:
			if !flush() {
				return false, ctx.Err()
			}
		}
	}
	if !flush() {
		return false, ctx.Err()
	}
	return false, nil
}

func (b *eventBridge) send(ctx context.Context, update bridgeUpdate) bool {
	select {
	case <-ctx.Done():
		return false
	case b.updates <- update:
		return true
	}
}
