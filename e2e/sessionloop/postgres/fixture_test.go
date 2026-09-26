// Fixtures assemble the real native Harness and shared actor Worker through
// public ports. Only the model is deterministic; SQL, replay and tools are real.
package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
)

func database(t *testing.T) *Store {
	t.Helper()
	dsn := os.Getenv("AGENTIC_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("run bash run.sh for real PostgreSQL and PgBouncer coverage")
	}
	s, err := Connect(t.Context(), dsn, "e2e_"+strings.ReplaceAll(uuid.NewString(), "-", ""))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Migrate(t.Context()); err != nil {
		s.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		// This exact randomly allocated schema belongs to this test alone.
		if _, err := s.pool.Exec(ctx, "DROP SCHEMA "+pgx.Identifier{s.schema}.Sanitize()+" CASCADE"); err != nil {
			t.Error(err)
		}
		s.Close()
	})
	return s
}

func reconnect(t *testing.T, s *Store) *Store {
	t.Helper()
	next, err := Connect(t.Context(), os.Getenv("AGENTIC_POSTGRES_DSN"), s.schema)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(next.Close)
	return next
}

func command(id actor.ActorID, key, text string) actor.Command {
	return actor.Command{ActorID: id, ID: actor.CommandID(key), Command: sessionloop.Command{
		Kind: sessionloop.CommandStart, Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: text}}}}}
}

func submit(t *testing.T, s *Store, c actor.Command) actor.Submission {
	t.Helper()
	r, err := s.Submit(t.Context(), c)
	if err != nil || r.Guarantee != sessionloop.AcceptanceDurable {
		t.Fatalf("submit=%+v %v", r, err)
	}
	return r
}

func host(config harness.DefaultConfig, model agentic.Model, repo store.Repository) (sessionloop.Host, error) {
	a, err := harness.AssembleDefault(config)
	if err != nil {
		return nil, err
	}
	a.Runtime.Sessions = repo
	h, err := harness.New(agentic.NewAgent("Follow the user's instructions.", model), harness.WithRuntime(a.Runtime), harness.WithCapabilities(a.Capabilities...)).Build()
	if err != nil {
		return nil, err
	}
	return harness.NewSessionLoopHost(h)
}

type fixture struct {
	store         *Store
	config        harness.DefaultConfig
	model         *model
	id            actor.ActorID
	repository    func(actor.Lease, store.Repository) store.Repository
	onWorkerError func(error)
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{store: database(t), model: &model{entered: make(chan struct{}, 32)}, config: harness.DefaultConfig{
		WorkspaceRoot: t.TempDir(), SessionDir: filepath.Join(t.TempDir(), "unused"),
		ContextWindowTokens: 16384, PromptCacheRetention: agentic.PromptCacheShort}}
	h, err := host(f.config, f.model, f.store.Bootstrap())
	if err != nil {
		t.Fatal(err)
	}
	s, err := h.NewSession(t.Context(), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	f.id = actor.ActorID(s.ID())
	if err := s.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	return f
}

type observer struct {
	actor.Adapter
	beforeAck     func()
	afterAck      func()
	beforeRelease func()
	afterRelease  func()
	acks          chan sessionloop.Receipt
	releases      chan struct{}
}

func observe(s actor.Adapter) *observer {
	return &observer{Adapter: s, acks: make(chan sessionloop.Receipt, 32), releases: make(chan struct{}, 32)}
}

func (a *observer) Acknowledge(ctx context.Context, lease actor.Lease, id actor.CommandID, r sessionloop.Receipt) error {
	if a.beforeAck != nil {
		a.beforeAck()
	}
	if err := a.Adapter.Acknowledge(ctx, lease, id, r); err != nil {
		return err
	}
	if a.afterAck != nil {
		a.afterAck()
	}
	a.acks <- r
	return nil
}

func (a *observer) Release(ctx context.Context, lease actor.Lease) error {
	if a.beforeRelease != nil {
		a.beforeRelease()
	}
	if err := a.Adapter.Release(ctx, lease); err != nil {
		return err
	}
	if a.afterRelease != nil {
		a.afterRelease()
	}
	a.releases <- struct{}{}
	return nil
}

func (f *fixture) worker(t *testing.T, s *Store, a actor.Adapter) func() {
	t.Helper()
	opener := actor.SessionOpenerFunc(func(ctx context.Context, lease actor.Lease) (actor.Session, error) {
		repo := s.Repository(lease)
		if f.repository != nil {
			repo = f.repository(lease, repo)
		}
		h, err := host(f.config, f.model, repo)
		if err != nil {
			return nil, err
		}
		session, err := h.OpenSession(ctx, sessionloop.SessionID(lease.ActorID))
		if err != nil {
			return nil, err
		}
		return session.(actor.Session), nil
	})
	w, err := actor.NewWorker(actor.Config{Owner: uuid.NewString(), Adapter: a, SessionOpener: opener,
		LeaseTTL: time.Second, PollInterval: 5 * time.Millisecond, RetryInterval: 10 * time.Millisecond,
		MaxActors: 8, BatchSize: 1, OnError: func(_ actor.ActorID, err error) {
			if f.onWorkerError != nil {
				f.onWorkerError(err)
				return
			}
			t.Errorf("worker: %v", err)
		}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			cancel()
			err := await(t, done)
			if !errors.Is(err, context.Canceled) {
				t.Errorf("worker stop: %v", err)
			}
		})
	}
	t.Cleanup(stop)
	return stop
}

func await[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(15 * time.Second):
		t.Fatal("timed out waiting for e2e boundary")
	}
	var zero T
	return zero
}

// inspect deliberately schedules a read after worker retirement. Recovery tests
// wait for independent recovery BEFORE calling this; it cannot rescue lost work.
func (f *fixture) inspect(t *testing.T) actor.Session {
	t.Helper()
	err := f.store.transaction(t.Context(), func(tx pgx.Tx) error {
		if _, err := lock(t.Context(), tx, f.id); err != nil {
			return err
		}
		_, err := tx.Exec(t.Context(), "UPDATE sessions SET needs_execution=true WHERE id=$1", string(f.id))
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := f.store.Acquire(t.Context(), f.id, "inspect", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	h, err := host(f.config, f.model, f.store.Repository(lease))
	if err != nil {
		t.Fatal(err)
	}
	s, err := h.OpenSession(t.Context(), sessionloop.SessionID(f.id))
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

type request struct {
	messages []json.RawMessage
	cache    string
}
type model struct {
	mu       sync.Mutex
	requests []request
	entered  chan struct{}
	gate     <-chan struct{}
	readFile bool
	toolSeen bool
}

func (*model) Name() string { return "e2e:postgres" }
func (m *model) Request(ctx context.Context, req *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	encoded, err := json.Marshal(req.Messages)
	if err != nil {
		return nil, err
	}
	var r request
	if err := json.Unmarshal(encoded, &r.messages); err != nil {
		return nil, err
	}
	if req.PromptCache != nil {
		r.cache = req.PromptCache.Key
	}
	m.mu.Lock()
	m.requests = append(m.requests, r)
	n := len(m.requests)
	m.mu.Unlock()
	if m.entered != nil {
		m.entered <- struct{}{}
	}
	if m.gate != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-m.gate:
		}
	}
	// Request attempts are not committed turns: a lost worker can issue the
	// same history again. Derive the scripted response from that history.
	hasCall := false
	for _, message := range req.Messages {
		for _, call := range message.GetToolUses() {
			if call.ID == "read-fixture" {
				hasCall = true
			}
		}
	}
	if m.readFile && !hasCall {
		return &agentic.ChatResponse{Model: m.Name(), FinishReason: agentic.FinishReasonToolCalls,
			Message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "read-fixture", Name: "read_file", Input: map[string]any{"path": "fixture.txt"}})}, nil
	}
	if m.readFile && hasCall {
		found := false
		for _, message := range req.Messages {
			for _, result := range message.GetToolResults() {
				if result.ToolUseID == "read-fixture" && !result.IsError && strings.Contains(result.Content, "postgres-e2e-content") {
					found = true
				}
			}
		}
		if !found || !strings.Contains(string(encoded), "steering") {
			return nil, errors.New("missing real tool result or steering in next request")
		}
		m.mu.Lock()
		m.toolSeen = true
		m.mu.Unlock()
	}
	return &agentic.ChatResponse{Model: m.Name(), FinishReason: agentic.FinishReasonStop,
		Message: agentic.NewTextMessage(agentic.RoleAssistant, fmt.Sprintf("done %d", n))}, nil
}
