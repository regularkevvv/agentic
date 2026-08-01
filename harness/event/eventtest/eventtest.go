// Package eventtest provides a reusable conformance suite for event hubs.
package eventtest

import (
	"context"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/event"
)

type Factory func(context.Context, []event.Record) (event.Hub, error)

func Run(t *testing.T, factory Factory) {
	t.Helper()
	t.Run("preview_gap", func(t *testing.T) {
		hub := open(t, factory, nil)
		defer hub.Close()
		sub := hub.Subscribe(event.SubscribeOptions{Buffer: 1, Preview: true})
		defer sub.Close()
		hub.PublishPreview(preview("one"))
		hub.PublishPreview(preview("dropped"))
		if got := <-sub.Events; got.Name != "one" || !got.Dropped.Empty() {
			t.Fatalf("first event = %#v", got)
		}
		hub.PublishPreview(preview("three"))
		if got := <-sub.Events; got.Name != "three" || got.Dropped.Preview != 1 {
			t.Fatalf("event after gap = %#v", got)
		}
	})

	t.Run("authoritative_lag_isolated", func(t *testing.T) {
		hub := open(t, factory, nil)
		defer hub.Close()
		lagged := hub.Subscribe(event.SubscribeOptions{Buffer: 1})
		healthy := hub.Subscribe(event.SubscribeOptions{Buffer: 4})
		defer healthy.Close()
		hub.PublishDurable(durable(1))
		hub.PublishDurable(durable(2))
		if got := <-lagged.Events; got.Cursor != 1 {
			t.Fatalf("lagged cursor = %d", got.Cursor)
		}
		var lag *event.ErrSubscriberLagged
		if err := <-lagged.Err; !errors.As(err, &lag) || lag.LastCursor != 1 {
			t.Fatalf("lag error = %#v", err)
		}
		if first, second := <-healthy.Events, <-healthy.Events; first.Cursor != 1 || second.Cursor != 2 {
			t.Fatalf("healthy cursors = %d, %d", first.Cursor, second.Cursor)
		}
	})

	t.Run("historical_resubscribe", func(t *testing.T) {
		hub := open(t, factory, []event.Record{durable(1), durable(2), durable(3)})
		defer hub.Close()
		sub := hub.Subscribe(event.SubscribeOptions{AfterCursor: 1, Buffer: 2, Preview: true})
		defer sub.Close()
		if first, second := <-sub.Events, <-sub.Events; first.Cursor != 2 || second.Cursor != 3 {
			t.Fatalf("replayed cursors = %d, %d", first.Cursor, second.Cursor)
		}
		hub.PublishPreview(preview("live"))
		if got := <-sub.Events; got.Cursor != 3 {
			t.Fatalf("preview cursor = %d", got.Cursor)
		}
	})

	t.Run("live_resubscribe_starts_strictly_after_cursor", func(t *testing.T) {
		hub := open(t, factory, nil)
		defer hub.Close()
		// A snapshot may observe a durable append immediately before the hub
		// publishes it. The subsequent live publication at the snapshot cursor
		// must not be redelivered to this subscriber.
		sub := hub.Subscribe(event.SubscribeOptions{AfterCursor: 1, Buffer: 1})
		defer sub.Close()
		hub.PublishDurable(durable(1))
		select {
		case got := <-sub.Events:
			t.Fatalf("received cursor %d at requested cursor 1", got.Cursor)
		default:
		}
		hub.PublishDurable(durable(2))
		if got := <-sub.Events; got.Cursor != 2 {
			t.Fatalf("first event after cursor 1 = %d", got.Cursor)
		}
	})
}

func open(t *testing.T, factory Factory, history []event.Record) event.Hub {
	t.Helper()
	hub, err := factory(context.Background(), history)
	if err != nil {
		t.Fatal(err)
	}
	if hub == nil {
		t.Fatal("factory returned nil hub")
	}
	return hub
}

func preview(name string) event.Record {
	return event.Record{Nature: agentic.EventPreview, Source: "eventtest", Name: name}
}

func durable(cursor uint64) event.Record {
	return event.Record{Cursor: cursor, Nature: agentic.EventAuthoritative, Source: "eventtest", Name: "commit"}
}
