// End-to-end local assembly connects the Worker to the real Agentic harness,
// its original replay journal, and the same mutation authority as the mailbox.
package harness

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	actormemory "github.com/regularkevvv/agentic/harness/sessionloop/actor/memory"
	"github.com/regularkevvv/agentic/harness/store"
)

type actorDoneAdapter struct {
	actor.Adapter
	done chan struct{}
}

func (a *actorDoneAdapter) Release(ctx context.Context, l actor.Lease) error {
	err := a.Adapter.Release(ctx, l)
	if err == nil {
		a.done <- struct{}{}
	}
	return err
}

func TestActorWorkerUsesHarnessJournalAndRestoresSameSession(t *testing.T) {
	config := runtimeConfig(t)
	base := config.Sessions
	runtime, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	host, err := NewSessionLoopHost(runtime)
	if err != nil {
		t.Fatal(err)
	}
	seed, err := host.NewSession(t.Context(), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	boundID := seed.ID()
	if err := seed.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	mailbox := actormemory.New()
	adapter := &actorDoneAdapter{Adapter: mailbox, done: make(chan struct{}, 8)}
	opener := actor.SessionOpenerFunc(func(ctx context.Context, l actor.Lease) (actor.Session, error) {
		owned := config
		owned.Sessions = store.WithAuthority(base, mailbox.Authority(l))
		r, err := NewRuntime[string](&facadeDriver{}, owned)
		if err != nil {
			return nil, err
		}
		h, err := NewSessionLoopHost(r)
		if err != nil {
			return nil, err
		}
		s, err := h.OpenSession(ctx, boundID)
		if err != nil {
			return nil, err
		}
		return s.(actor.Session), nil
	})
	worker, err := actor.NewWorker(actor.Config{Owner: "local", Adapter: adapter, SessionOpener: opener, PollInterval: time.Millisecond, MaxActors: 1})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- worker.Run(ctx) }()
	for _, id := range []actor.CommandID{"one", "two"} {
		command := actor.Command{ActorID: "conversation", ID: id, Command: sessionloop.Command{Kind: sessionloop.CommandStart,
			Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: string(id)}}}}}
		if _, err := mailbox.Submit(t.Context(), command); err != nil {
			t.Fatal(err)
		}
		select {
		case <-adapter.done:
		case <-time.After(3 * time.Second):
			t.Fatal("harness did not retire")
		}
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	s, err := host.OpenSession(t.Context(), boundID)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(t.Context())
	snapshot, err := s.Snapshot(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, entry := range snapshot.Entries {
		if entry.Role == sessionloop.RoleUser {
			count++
		}
	}
	if count != 2 {
		t.Fatalf("lost session continuity: %+v", snapshot.Entries)
	}
}
