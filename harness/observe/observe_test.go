package observe

import (
	"errors"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/event"
)

func TestUsageProjectionAndCacheRate(t *testing.T) {
	t.Parallel()
	value := UsageFromAgentic(agentic.Usage{
		PromptTokens: 200, CompletionTokens: 20, TotalTokens: 220,
		CacheReadTokens: 150, CacheCreationTokens: 10, ReasoningTokens: 5,
		Requests: 2, ToolCalls: 3,
	})
	if value.CacheHitPercent() != 75 || value.TotalTokens != 220 || value.Requests != 2 || value.ToolCalls != 3 {
		t.Fatalf("usage = %#v", value)
	}
	if (Usage{}).CacheHitPercent() != 0 || (Usage{PromptTokens: 1}).CacheHitPercent() != 0 {
		t.Fatal("empty cache rate was nonzero")
	}
}

func TestProjectSubscriptionCopiesAndForwardsFailures(t *testing.T) {
	t.Parallel()
	records := make(chan event.Record, 2)
	failures := make(chan error, 1)
	var closed atomic.Int32
	source := event.NewSubscription(records, failures, func() { closed.Add(1) })
	projected := ProjectSubscription(source, func(record event.Record) (Event, error) {
		record.Payload[0] = 'x'
		return Event{Cursor: record.Cursor, TextDelta: string(record.Payload)}, nil
	})
	payload := []byte("abc")
	records <- event.Record{Cursor: 2, Payload: payload}
	value := <-projected.Events()
	if value.TextDelta != "xbc" || string(payload) != "abc" {
		t.Fatalf("projection=%#v original=%q", value, payload)
	}
	want := errors.New("lag")
	failures <- want
	if err := <-projected.Errors(); !errors.Is(err, want) {
		t.Fatalf("error = %v", err)
	}
	projected.Close()
	projected.Close()
	if closed.Load() != 1 {
		t.Fatalf("source closes = %d", closed.Load())
	}
}

func TestProjectSubscriptionProjectionErrorAndNilEdges(t *testing.T) {
	t.Parallel()
	records := make(chan event.Record, 1)
	source := event.NewSubscription(records, make(chan error), nil)
	want := errors.New("project")
	projected := ProjectSubscription(source, func(event.Record) (Event, error) { return Event{}, want })
	records <- event.Record{}
	if err := <-projected.Errors(); !errors.Is(err, want) {
		t.Fatalf("error = %v", err)
	}
	projected.Close()

	for _, value := range []Subscription{ProjectSubscription(nil, func(event.Record) (Event, error) { return Event{}, nil }), ProjectSubscription(source, nil)} {
		if _, ok := <-value.Events(); ok {
			t.Fatal("nil edge retained event channel")
		}
		value.Close()
	}
	var nilSubscription *subscription
	nilSubscription.Close()
}

func TestProjectSubscriptionClosesAcrossEveryChannelState(t *testing.T) {
	t.Parallel()

	closedRecords := make(chan event.Record)
	closedErrors := make(chan error)
	close(closedRecords)
	close(closedErrors)
	drained := ProjectSubscription(
		event.NewSubscription(closedRecords, closedErrors, nil),
		func(event.Record) (Event, error) { return Event{}, nil },
	)
	if _, ok := <-drained.Events(); ok {
		t.Fatal("closed source retained event channel")
	}

	nilFailure := make(chan error, 1)
	nilFailure <- nil
	fromNilFailure := ProjectSubscription(
		event.NewSubscription(make(chan event.Record), nilFailure, nil),
		func(event.Record) (Event, error) { return Event{}, nil },
	)
	if _, ok := <-fromNilFailure.Errors(); ok {
		t.Fatal("nil source failure retained error channel")
	}

	idle := ProjectSubscription(
		event.NewSubscription(make(chan event.Record), make(chan error), nil),
		func(event.Record) (Event, error) { return Event{}, nil },
	)
	idle.Close()
	if _, ok := <-idle.Events(); ok {
		t.Fatal("closed idle subscription retained event channel")
	}
}

func TestProjectSubscriptionCloseUnblocksPendingDelivery(t *testing.T) {
	t.Parallel()
	records := make(chan event.Record, 1)
	projectedRecord := make(chan struct{})
	projected := ProjectSubscription(
		event.NewSubscription(records, make(chan error), nil),
		func(event.Record) (Event, error) {
			close(projectedRecord)
			return Event{TextDelta: "blocked"}, nil
		},
	)
	records <- event.Record{}
	<-projectedRecord
	projected.Close()
	if _, ok := <-projected.Events(); ok {
		t.Fatal("closed delivery retained event channel")
	}
}
