package event

import (
	"context"
	"errors"

	agentic "github.com/regularkevvv/agentic"
)

// Middleware wraps the synchronous Agentic commit sink for one session.
// Public subscription delivery remains independent and nonblocking.
type Middleware interface {
	Wrap(agentic.EventSink) agentic.EventSink
}

// MiddlewareFunc adapts a wrapping function to Middleware.
type MiddlewareFunc func(agentic.EventSink) agentic.EventSink

func (f MiddlewareFunc) Wrap(next agentic.EventSink) agentic.EventSink {
	return f(next)
}

// MiddlewareHandler is a conventional around-handler adapter.
type MiddlewareHandler func(context.Context, agentic.Event, agentic.EventSink) error

func (h MiddlewareHandler) Wrap(next agentic.EventSink) agentic.EventSink {
	return agentic.EventSinkFunc(func(ctx context.Context, value agentic.Event) error {
		return h(ctx, value, next)
	})
}

// Chain wraps sink in registration order. The first middleware is the
// outermost observer.
func Chain(sink agentic.EventSink, middleware ...Middleware) (agentic.EventSink, error) {
	if sink == nil {
		return nil, errors.New("event sink is required")
	}
	result := sink
	for index := len(middleware) - 1; index >= 0; index-- {
		if middleware[index] == nil {
			return nil, errors.New("event middleware must not be nil")
		}
		result = middleware[index].Wrap(result)
		if result == nil {
			return nil, errors.New("event middleware returned a nil sink")
		}
	}
	return result, nil
}
