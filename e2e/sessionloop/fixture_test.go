// The fixture wires only public ports. Fault injection wraps acknowledgment;
// storage, ownership, dispatch, execution, tools and replay remain real code.
package sessionloop_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor/localchannel"
	"github.com/regularkevvv/agentic/harness/store"
)

var errInjected = errors.New("injected acknowledgment failure")

type localFixture struct {
	ctx     context.Context
	config  harness.DefaultConfig
	model   *scriptedModel
	id      actor.ActorID
	mailbox *localchannel.Store
	adapter *observedAdapter
	errors  chan error
}

func newLocalFixture(t *testing.T) *localFixture {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	t.Cleanup(cancel)
	f := &localFixture{
		ctx: ctx, mailbox: localchannel.New(), model: &scriptedModel{entered: make(chan struct{})},
		errors: make(chan error, 32),
		config: harness.DefaultConfig{
			WorkspaceRoot: t.TempDir(), SessionDir: filepath.Join(t.TempDir(), "sessions"),
			ContextWindowTokens: 16_384, PromptCacheRetention: agentic.PromptCacheShort,
		},
	}
	if err := os.WriteFile(filepath.Join(f.config.WorkspaceRoot, "fixture.txt"), []byte("local-e2e-content"), 0o600); err != nil {
		t.Fatal(err)
	}
	f.adapter = &observedAdapter{Adapter: f.mailbox, receipts: make(chan sessionloop.Receipt, 32),
		attempts: make(chan sessionloop.Receipt, 32), released: make(chan retirement, 32)}
	host, err := f.host(nil)
	if err != nil {
		t.Fatal(err)
	}
	s, err := host.NewSession(ctx, sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// Identity binding: the actor ID IS the pre-created journal ID. Open never
	// creates a replacement if that ID cannot be restored.
	f.id = actor.ActorID(s.ID())
	if err := s.Close(ctx); err != nil {
		t.Fatal(err)
	}
	return f
}

func (f *localFixture) host(authority store.Authority) (sessionloop.Host, error) {
	assembly, err := harness.AssembleDefault(f.config)
	if err != nil {
		return nil, err
	}
	if authority != nil {
		assembly.Runtime.Sessions = store.WithAuthority(assembly.Runtime.Sessions, authority)
	}
	runtime, err := harness.New(agentic.NewAgent("Read the requested file and follow user instructions.", f.model),
		harness.WithRuntime(assembly.Runtime), harness.WithCapabilities(assembly.Capabilities...)).Build()
	if err != nil {
		return nil, err
	}
	return harness.NewSessionLoopHost(runtime)
}

func (f *localFixture) startWorker(t *testing.T, owner string) func() {
	t.Helper()
	opener := actor.SessionOpenerFunc(func(ctx context.Context, lease actor.Lease) (actor.Session, error) {
		host, err := f.host(f.mailbox.Authority(lease))
		if err != nil {
			return nil, err
		}
		s, err := host.OpenSession(ctx, sessionloop.SessionID(lease.ActorID))
		if err != nil {
			return nil, err
		}
		return s.(actor.Session), nil
	})
	w, err := actor.NewWorker(actor.Config{
		Owner: owner, Adapter: f.adapter, SessionOpener: opener,
		MaxActors: 1, BatchSize: 1, LeaseTTL: time.Second,
		PollInterval: time.Millisecond, RetryInterval: time.Millisecond,
		OnError: func(_ actor.ActorID, err error) { f.errors <- err },
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(f.ctx)
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			cancel()
			select {
			case err := <-done:
				if !errors.Is(err, context.Canceled) {
					t.Errorf("worker stopped with %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Error("worker did not stop")
			}
		})
	}
	t.Cleanup(stop)
	return stop
}

func (f *localFixture) command(id, text string) actor.Command {
	return actor.Command{ActorID: f.id, ID: actor.CommandID(id), Command: sessionloop.Command{
		Kind: sessionloop.CommandStart, Input: &sessionloop.Input{
			Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: text}},
		},
	}}
}

func (f *localFixture) submit(t *testing.T, command actor.Command) {
	t.Helper()
	submission, err := f.mailbox.Submit(f.ctx, command)
	if err != nil || submission.Guarantee != sessionloop.AcceptanceAccepted {
		t.Fatalf("local mailbox must not promise process-crash durability: %+v, %v", submission, err)
	}
}

func (f *localFixture) ack(t *testing.T, id sessionloop.CommandID) sessionloop.Receipt {
	t.Helper()
	r := await(t, f.ctx, f.adapter.receipts)
	if r.CommandID != id || r.Guarantee != sessionloop.AcceptanceDurable || r.Rejection != "" {
		t.Fatalf("acknowledgment=%+v, want durable %s", r, id)
	}
	return r
}

func (f *localFixture) retired(t *testing.T) {
	t.Helper()
	r := await(t, f.ctx, f.adapter.released)
	if r.pending != 0 {
		t.Fatalf("accepted mailbox inputs were retained: %d", r.pending)
	}
}

func (f *localFixture) noErrors(t *testing.T) {
	t.Helper()
	select {
	case err := <-f.errors:
		t.Fatalf("unexpected worker error: %v", err)
	default:
	}
}

func (f *localFixture) reopen(t *testing.T) actor.Session {
	t.Helper()
	host, err := f.host(nil)
	if err != nil {
		t.Fatal(err)
	}
	s, err := host.OpenSession(f.ctx, sessionloop.SessionID(f.id))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(context.Background()); err != nil {
			t.Error(err)
		}
	})
	return s.(actor.Session)
}

type retirement struct{ pending int }

type observedAdapter struct {
	actor.Adapter
	receipts chan sessionloop.Receipt
	attempts chan sessionloop.Receipt
	released chan retirement
	failure  string
	fail     atomic.Bool
}

func (a *observedAdapter) Acknowledge(ctx context.Context, lease actor.Lease, id actor.CommandID, receipt sessionloop.Receipt) error {
	a.attempts <- receipt
	if a.failure == "before deletion" && a.fail.CompareAndSwap(true, false) {
		return errInjected
	}
	if err := a.Adapter.Acknowledge(ctx, lease, id, receipt); err != nil {
		return err
	}
	if a.failure == "after deletion" && a.fail.CompareAndSwap(true, false) {
		return errInjected
	}
	a.receipts <- receipt
	return nil
}

func (a *observedAdapter) Release(ctx context.Context, lease actor.Lease) error {
	pending, err := a.Pending(ctx, lease, 0, 100)
	if err != nil {
		return err
	}
	if err := a.Adapter.Release(ctx, lease); err != nil {
		return err
	}
	a.released <- retirement{pending: len(pending)}
	return nil
}

func await[T any](t *testing.T, ctx context.Context, channel <-chan T) T {
	t.Helper()
	select {
	case value := <-channel:
		return value
	case <-ctx.Done():
		t.Fatal(fmt.Errorf("waiting for local e2e: %w", ctx.Err()))
		var zero T
		return zero
	}
}
