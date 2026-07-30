package event

import (
	"context"
	"errors"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

type middlewareTestEvent struct{}

func (middlewareTestEvent) Nature() agentic.EventNature { return agentic.EventLifecycle }
func (middlewareTestEvent) Type() agentic.EventType     { return agentic.EventTypeRunStarted }
func (middlewareTestEvent) TurnIndex() int              { return 0 }

func TestMiddlewareChainOrderAndValidation(t *testing.T) {
	t.Parallel()
	var order []string
	sink := agentic.EventSinkFunc(func(context.Context, agentic.Event) error {
		order = append(order, "sink")
		return nil
	})
	first := MiddlewareHandler(func(ctx context.Context, value agentic.Event, next agentic.EventSink) error {
		order = append(order, "first:before")
		err := next.Emit(ctx, value)
		order = append(order, "first:after")
		return err
	})
	second := MiddlewareHandler(func(ctx context.Context, value agentic.Event, next agentic.EventSink) error {
		order = append(order, "second:before")
		err := next.Emit(ctx, value)
		order = append(order, "second:after")
		return err
	})
	chained, err := Chain(sink, first, second)
	if err != nil {
		t.Fatal(err)
	}
	if err := chained.Emit(context.Background(), middlewareTestEvent{}); err != nil {
		t.Fatal(err)
	}
	want := []string{"first:before", "second:before", "sink", "second:after", "first:after"}
	if !reflect.DeepEqual(order, want) {
		t.Fatalf("order=%v want=%v", order, want)
	}
	if _, err := Chain(nil); err == nil {
		t.Fatal("nil sink succeeded")
	}
	if _, err := Chain(sink, MiddlewareFunc(func(agentic.EventSink) agentic.EventSink { return nil })); err == nil {
		t.Fatal("nil wrapped sink succeeded")
	}
	if _, err := Chain(sink, nil); err == nil {
		t.Fatal("nil middleware succeeded")
	}
}

func TestMiddlewareErrorPropagates(t *testing.T) {
	t.Parallel()
	want := errors.New("stop")
	chained, err := Chain(
		agentic.EventSinkFunc(func(context.Context, agentic.Event) error { return nil }),
		MiddlewareHandler(func(context.Context, agentic.Event, agentic.EventSink) error { return want }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := chained.Emit(context.Background(), middlewareTestEvent{}); !errors.Is(err, want) {
		t.Fatalf("error=%v", err)
	}
}
