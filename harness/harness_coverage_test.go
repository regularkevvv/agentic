package harness

import (
	"context"
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/env"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
)

type harnessIDsFunc func(string) (string, error)

func (f harnessIDsFunc) New(prefix string) (string, error) { return f(prefix) }

type closeFailLease struct {
	env.Lease
	err error
}

func (l *closeFailLease) Close(context.Context) error { return l.err }

func TestRuntimeValidationFacadeOptionsAndRepair(t *testing.T) {
	driver := &facadeDriver{}
	valid := runtimeConfig(t)
	cases := []struct {
		name   string
		mutate func(*RuntimeConfig)
		want   string
	}{
		{"sessions", func(c *RuntimeConfig) { c.Sessions = nil }, "repository"},
		{"codec", func(c *RuntimeConfig) { c.Codec = nil }, "codec"},
		{"events", func(c *RuntimeConfig) { c.Events = nil }, "event"},
		{"environments", func(c *RuntimeConfig) { c.Environments = nil }, "environment"},
		{"processors", func(c *RuntimeConfig) { c.ResultProcessors = nil }, "result-processor"},
		{"clock", func(c *RuntimeConfig) { c.Clock = nil }, "clock"},
		{"ids", func(c *RuntimeConfig) { c.IDs = nil }, "ID generator"},
		{"grace", func(c *RuntimeConfig) { c.ToolCancellationGrace = -1 }, "negative"},
		{"scope", func(c *RuntimeConfig) { c.Scope.Depth = -1 }, "depth"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			config := valid
			test.mutate(&config)
			if _, err := NewRuntime[string](driver, config); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want containing %q", err, test.want)
			}
		})
	}
	max := 2
	if WithBudget(agentic.UsageLimits{MaxRequests: &max}) == nil ||
		WithDrainAll(true) == nil ||
		WithInitialHistory(agentic.NewTextMessage(agentic.RoleUser, "history")) == nil {
		t.Fatal("facade options returned nil")
	}
	processor := Repair(CloseInterruptedFrontier, PendingCalls{})
	if processed, err := processor.Process(context.Background(), nil); err != nil || len(processed) != 0 {
		t.Fatalf("repair facade = %#v, %v", processed, err)
	}
}

func TestNewSessionIDAndConstructionFailuresAndOpeningGuard(t *testing.T) {
	boom := errors.New("boom")
	config := runtimeConfig(t)
	config.IDs = harnessIDsFunc(func(string) (string, error) { return "", boom })
	runtime, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.NewSession(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("ID error = %v", err)
	}

	config = runtimeConfig(t)
	config.IDs = harnessIDsFunc(func(string) (string, error) { return "", nil })
	runtime, err = NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.NewSession(context.Background()); err == nil {
		t.Fatal("invalid generated ID created a session")
	}

	config = runtimeConfig(t)
	runtime, err = NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	runtime.mu.Lock()
	runtime.opening["already"] = true
	runtime.mu.Unlock()
	if _, err := runtime.ResumeSession(context.Background(), "already"); !errors.Is(err, ErrSessionOpen) {
		t.Fatalf("opening guard = %v", err)
	}
}

func TestResumeClosedOwnerPropagatesCleanupFailure(t *testing.T) {
	boom := errors.New("close")
	config := runtimeConfig(t)
	config.Environments = env.FactoryFunc(func(context.Context, string) (env.Lease, error) {
		lease, err := envmemory.New("/", nil)
		if err != nil {
			return nil, err
		}
		return &closeFailLease{Lease: lease, err: boom}, nil
	})
	runtime, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	current, err := runtime.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := current.Close(context.Background()); !errors.Is(err, boom) ||
		current.State() != SessionClosed {
		t.Fatalf("initial close = %v, state=%s", err, current.State())
	}
	if _, err := runtime.ResumeSession(context.Background(), current.ID()); !errors.Is(err, boom) {
		t.Fatalf("resume cleanup error = %v", err)
	}
}
