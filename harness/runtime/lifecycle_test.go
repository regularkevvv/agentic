package runtime

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestRunLifecycleHooksInOrder(t *testing.T) {
	t.Parallel()
	var order []int
	hooks := []LifecycleHook{
		LifecycleHookFunc(func(context.Context, LifecycleEvent) error {
			order = append(order, 1)
			return nil
		}),
		LifecycleHookFunc(func(context.Context, LifecycleEvent) error {
			order = append(order, 2)
			return nil
		}),
	}
	if err := RunLifecycleHooks(context.Background(), hooks, LifecycleEvent{Phase: LifecycleSessionOpened, SessionID: "s"}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []int{1, 2}) {
		t.Fatalf("order=%v", order)
	}
	want := errors.New("stop")
	if err := RunLifecycleHooks(context.Background(), []LifecycleHook{
		LifecycleHookFunc(func(context.Context, LifecycleEvent) error { return want }),
		LifecycleHookFunc(func(context.Context, LifecycleEvent) error {
			t.Fatal("hook after error ran")
			return nil
		}),
	}, LifecycleEvent{}); !errors.Is(err, want) {
		t.Fatalf("error=%v", err)
	}
}
