package testkit

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func internalContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func preview(text string) sessionloop.Event {
	return sessionloop.Event{Nature: sessionloop.EventPreview, Kind: sessionloop.EventPreviewDelta, Preview: &sessionloop.Preview{Kind: sessionloop.PreviewText, Text: text}}
}

func authoritative(sequence uint64) sessionloop.Event {
	return sessionloop.Event{
		Nature:   sessionloop.EventAuthoritative,
		Kind:     sessionloop.EventSessionState,
		Position: sessionloop.Position{Sequence: sequence, Token: positionToken(sequence)},
	}
}

func internalStream(limit int, withPreviews bool) *stream {
	state := &sessionState{subs: make(map[*stream]struct{})}
	subscriber := newStream(state, limit, withPreviews)
	state.subs[subscriber] = struct{}{}
	return subscriber
}

func TestDroppedPreviewsAreCountedIntoTheNextDeliveredEvent(t *testing.T) {
	t.Parallel()
	subscriber := internalStream(1, true)
	subscriber.deliver(preview("kept"), false)
	subscriber.deliver(preview("dropped-1"), false)
	subscriber.deliver(preview("dropped-2"), false)

	kept, err := subscriber.Next(internalContext(t))
	if err != nil || kept.Preview.Text != "kept" || kept.Dropped != 0 {
		t.Fatalf("first delivery = %#v, %v; want the kept preview with no drops", kept, err)
	}

	subscriber.deliver(authoritative(1), true)
	next, err := subscriber.Next(internalContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if next.Dropped != 2 {
		t.Fatalf("Dropped = %d, want the 2 previews lost since the previous delivery", next.Dropped)
	}
}

func TestPreviewsAreInvisibleToSubscribersThatDidNotOptIn(t *testing.T) {
	t.Parallel()
	subscriber := internalStream(4, false)
	subscriber.deliver(preview("ignored"), false)
	subscriber.deliver(authoritative(1), true)
	event, err := subscriber.Next(internalContext(t))
	if err != nil || event.Nature != sessionloop.EventAuthoritative {
		t.Fatalf("delivery = %#v, %v; previews must be filtered without counting as drops", event, err)
	}
	if event.Dropped != 0 {
		t.Fatalf("Dropped = %d; filtered previews are not losses", event.Dropped)
	}
}

func TestAuthoritativeOverflowIsTerminalEvenWithBufferedEvents(t *testing.T) {
	t.Parallel()
	subscriber := internalStream(1, false)
	subscriber.deliver(authoritative(1), true)
	subscriber.deliver(authoritative(2), true)
	if _, err := subscriber.Next(internalContext(t)); !errors.Is(err, sessionloop.ErrLagged) {
		t.Fatalf("Next after authoritative overflow = %v, want ErrLagged before any buffered event", err)
	}
	subscriber.deliver(authoritative(3), true)
	if _, err := subscriber.Next(internalContext(t)); !errors.Is(err, sessionloop.ErrLagged) {
		t.Fatalf("a lagged stream accepted later deliveries: %v", err)
	}
}

func TestDeliveriesAfterCloseOrEndAreIgnored(t *testing.T) {
	t.Parallel()
	subscriber := internalStream(4, true)
	subscriber.end()
	subscriber.deliver(authoritative(1), true)
	if _, err := subscriber.Next(internalContext(t)); err == nil {
		t.Fatal("an ended stream delivered an event appended after the end")
	}

	closed := internalStream(4, true)
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	closed.deliver(authoritative(1), true)
	if _, err := closed.Next(internalContext(t)); err == nil {
		t.Fatal("a closed stream delivered an event appended after Close")
	}
}
