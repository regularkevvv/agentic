package runtime

import "context"

// LifecyclePhase identifies a harness-owned session lifecycle boundary.
type LifecyclePhase string

const (
	LifecycleSessionOpened    LifecyclePhase = "session.opened"
	LifecycleSessionRecovered LifecyclePhase = "session.recovered"
	LifecycleSessionClosing   LifecyclePhase = "session.closing"
	LifecycleSessionClosed    LifecyclePhase = "session.closed"
)

// LifecycleEvent is delivered to capability lifecycle hooks. Hooks are
// session-scoped observers and never replace the durable event stream.
type LifecycleEvent struct {
	Phase     LifecyclePhase
	SessionID string
}

// LifecycleHook participates in an ordered capability lifecycle chain.
// Implementations shared by an immutable Harness must be safe for concurrent
// sessions.
type LifecycleHook interface {
	HandleLifecycle(context.Context, LifecycleEvent) error
}

// LifecycleHookFunc adapts a function to LifecycleHook.
type LifecycleHookFunc func(context.Context, LifecycleEvent) error

func (f LifecycleHookFunc) HandleLifecycle(ctx context.Context, event LifecycleEvent) error {
	return f(ctx, event)
}

// RunLifecycleHooks invokes hooks in registration order.
func RunLifecycleHooks(ctx context.Context, hooks []LifecycleHook, event LifecycleEvent) error {
	for _, hook := range hooks {
		if err := hook.HandleLifecycle(ctx, event); err != nil {
			return err
		}
	}
	return nil
}
