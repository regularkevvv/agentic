package inproc

import (
	"context"
	"errors"
	"sync"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/eventtest"
)

func TestHubConformance(t *testing.T) {
	t.Parallel()
	eventtest.Run(t, NewFactory().Open)
}

func preview(name string) event.Record {
	return event.Record{Nature: agentic.EventPreview, Source: "test", Name: name}
}

func durable(cursor uint64) event.Record {
	return event.Record{Cursor: cursor, Nature: agentic.EventAuthoritative, Source: "test", Name: "commit"}
}

func TestPreviewDropReportsGap(t *testing.T) {
	t.Parallel()
	bus := New(nil)
	sub := bus.Subscribe(event.SubscribeOptions{Buffer: 1, Preview: true})
	defer sub.Close()
	bus.PublishPreview(preview("one"))
	bus.PublishPreview(preview("dropped"))
	if got := <-sub.Events; got.Name != "one" || !got.Dropped.Empty() {
		t.Fatalf("first event = %#v", got)
	}
	bus.PublishPreview(preview("three"))
	got := <-sub.Events
	if got.Name != "three" || got.Dropped.Preview != 1 {
		t.Fatalf("event after gap = %#v", got)
	}
}

func TestAuthoritativeLagDisconnectsOnlySubscriber(t *testing.T) {
	t.Parallel()
	bus := New(nil)
	lagged := bus.Subscribe(event.SubscribeOptions{Buffer: 1})
	healthy := bus.Subscribe(event.SubscribeOptions{Buffer: 4})
	defer healthy.Close()
	bus.PublishDurable(durable(1))
	bus.PublishDurable(durable(2))

	if got := <-lagged.Events; got.Cursor != 1 {
		t.Fatalf("lagged first cursor = %d", got.Cursor)
	}
	if _, ok := <-lagged.Events; ok {
		t.Fatal("lagged Events remained open")
	}
	err, ok := <-lagged.Err
	if !ok {
		t.Fatal("lagged Err closed without terminal error")
	}
	var lag *event.ErrSubscriberLagged
	if !errors.As(err, &lag) || lag.LastCursor != 1 {
		t.Fatalf("lag error = %#v", err)
	}
	if _, ok := <-lagged.Err; ok {
		t.Fatal("lagged Err delivered more than once")
	}

	if first, second := <-healthy.Events, <-healthy.Events; first.Cursor != 1 || second.Cursor != 2 {
		t.Fatalf("healthy cursors = %d, %d", first.Cursor, second.Cursor)
	}
}

func TestSnapshotCursorAndHistoricalResubscribe(t *testing.T) {
	t.Parallel()
	bus := New(nil)
	for cursor := uint64(1); cursor <= 3; cursor++ {
		bus.PublishDurable(durable(cursor))
	}
	if got := bus.Cursor(); got != 3 {
		t.Fatalf("cursor = %d", got)
	}
	sub := bus.Subscribe(event.SubscribeOptions{AfterCursor: 1, Buffer: 2, Preview: true})
	defer sub.Close()
	if first, second := <-sub.Events, <-sub.Events; first.Cursor != 2 || second.Cursor != 3 {
		t.Fatalf("replayed cursors = %d, %d", first.Cursor, second.Cursor)
	}
	bus.PublishPreview(preview("live"))
	if got := <-sub.Events; got.Cursor != 3 || got.Name != "live" {
		t.Fatalf("live preview = %#v", got)
	}
}

func TestHistoricalReplayLag(t *testing.T) {
	t.Parallel()
	bus := New([]event.Record{durable(1), durable(2)})
	sub := bus.Subscribe(event.SubscribeOptions{Buffer: 1})
	if got := <-sub.Events; got.Cursor != 1 {
		t.Fatalf("cursor = %d", got.Cursor)
	}
	var lag *event.ErrSubscriberLagged
	if err := <-sub.Err; !errors.As(err, &lag) || lag.LastCursor != 1 {
		t.Fatalf("lag = %v", err)
	}
}

func TestBusConcurrentPublishSubscribeAndClose(t *testing.T) {
	t.Parallel()
	bus := New(nil)
	const workers = 24
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func(index int) {
			defer wg.Done()
			sub := bus.Subscribe(event.SubscribeOptions{Buffer: 8, Preview: true})
			bus.PublishPreview(preview("race"))
			if index%3 == 0 {
				bus.PublishDurable(durable(uint64(index + 1)))
			}
			sub.Close()
		}(i)
	}
	wg.Wait()
}

func TestFactoryHistoryFilteringClosedHubAndDuplicateCursors(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewFactory().Open(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Open = %v", err)
	}
	payload := []byte("original")
	bus := New([]event.Record{
		preview("ignored"),
		{Nature: agentic.EventAuthoritative, Source: "zero"},
		{Cursor: 2, Nature: agentic.EventAuthoritative, Source: "history", Payload: payload},
		{Cursor: 1, Nature: agentic.EventAuthoritative, Source: "history"},
	})
	payload[0] = 'X'
	if bus.Cursor() != 2 {
		t.Fatalf("history cursor = %d", bus.Cursor())
	}
	noPreview := bus.Subscribe(event.SubscribeOptions{})
	if got := <-noPreview.Events; got.Cursor != 2 || string(got.Payload) != "original" {
		t.Fatalf("history clone = %#v", got)
	}
	bus.PublishPreview(preview("hidden"))
	select {
	case got := <-noPreview.Events:
		t.Fatalf("preview reached disabled subscriber: %#v", got)
	default:
	}
	bus.PublishDurable(durable(2))
	select {
	case got := <-noPreview.Events:
		t.Fatalf("duplicate cursor reached subscriber: %#v", got)
	default:
	}
	noPreview.Close()
	noPreview.Close()

	assertPanics := func(record event.Record) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatal("invalid durable publish did not panic")
			}
		}()
		bus.PublishDurable(record)
	}
	assertPanics(preview("bad"))
	assertPanics(event.Record{Nature: agentic.EventAuthoritative})

	bus.Close()
	bus.Close()
	bus.PublishPreview(preview("closed"))
	bus.PublishDurable(durable(3))
	closed := bus.Subscribe(event.SubscribeOptions{})
	if _, ok := <-closed.Events; ok {
		t.Fatal("closed subscription events remained open")
	}
	if err, ok := <-closed.Err; !ok || err != nil {
		t.Fatalf("closed subscription terminal = %v, %t", err, ok)
	}
	if _, ok := <-closed.Err; ok {
		t.Fatal("closed subscription error channel remained open")
	}
}
