package session

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"strings"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/artifact"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/env"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type errorCodec struct {
	base      codec.Codec
	encodeErr error
	decodeErr error
}

func (c errorCodec) Encode(value any) ([]byte, error) {
	if c.encodeErr != nil {
		return nil, c.encodeErr
	}
	return c.base.Encode(value)
}

func (c errorCodec) Decode(payload []byte, value any) error {
	if c.decodeErr != nil {
		return c.decodeErr
	}
	return c.base.Decode(payload, value)
}

type fixedClock struct{ value time.Time }

func (c fixedClock) Now() time.Time { return c.value }

type idsFunc func(string) (string, error)

func (f idsFunc) New(prefix string) (string, error) { return f(prefix) }

type repositoryStub struct {
	create func(context.Context, string, ...store.PendingEntry) (store.Journal, store.Commit, error)
	open   func(context.Context, string) (store.Journal, error)
}

func (r repositoryStub) Create(
	ctx context.Context,
	id string,
	entries ...store.PendingEntry,
) (store.Journal, store.Commit, error) {
	return r.create(ctx, id, entries...)
}

func (r repositoryStub) Open(ctx context.Context, id string) (store.Journal, error) {
	if r.open == nil {
		return nil, store.ErrSessionNotFound
	}
	return r.open(ctx, id)
}

type journalStub struct {
	id     string
	load   func(context.Context) (store.Snapshot, error)
	append func(context.Context, store.Cursor, ...store.PendingEntry) (store.Commit, error)
	close  func(context.Context) error
}

func (j *journalStub) SessionID() string { return j.id }

func (j *journalStub) Load(ctx context.Context) (store.Snapshot, error) {
	if j.load == nil {
		return store.Snapshot{}, nil
	}
	return j.load(ctx)
}

func (j *journalStub) Append(
	ctx context.Context,
	cursor store.Cursor,
	entries ...store.PendingEntry,
) (store.Commit, error) {
	if j.append == nil {
		return store.Commit{}, errors.New("append rejected")
	}
	return j.append(ctx, cursor, entries...)
}

func (j *journalStub) Close(ctx context.Context) error {
	if j.close == nil {
		return nil
	}
	return j.close(ctx)
}

type leaseStub struct {
	close func(context.Context) error
}

func (l *leaseStub) Files() env.FileSystem { return l }
func (l *leaseStub) Shell() (env.Shell, bool) {
	return nil, false
}
func (l *leaseStub) Close(ctx context.Context) error {
	if l.close == nil {
		return nil
	}
	return l.close(ctx)
}
func (l *leaseStub) CanonicalPath(context.Context, string) (env.CanonicalResource, error) {
	return env.CanonicalResource{Scheme: "stub", ID: "root"}, nil
}
func (l *leaseStub) ReadFile(context.Context, string) ([]byte, error) {
	return nil, fs.ErrNotExist
}
func (l *leaseStub) WriteFile(context.Context, string, []byte, fs.FileMode) error {
	return nil
}
func (l *leaseStub) MkdirAll(context.Context, string, fs.FileMode) error { return nil }
func (l *leaseStub) ReadDir(context.Context, string) ([]env.DirEntry, error) {
	return nil, nil
}
func (l *leaseStub) Stat(context.Context, string) (env.FileInfo, error) {
	return env.FileInfo{}, fs.ErrNotExist
}
func (l *leaseStub) Remove(context.Context, string) error { return nil }

func passthroughProcessor() agentic.ToolResultProcessor {
	return agentic.ToolResultProcessorFunc(func(
		_ context.Context,
		_ agentic.ToolUse,
		result agentic.ToolExecutionResult,
	) (agentic.ToolExecutionResult, error) {
		return result, nil
	})
}

func TestConfigValidationCoversEveryRequiredPort(t *testing.T) {
	if _, err := New[string](context.Background(), Config[string]{}); err == nil {
		t.Fatal("New accepted an invalid config")
	}
	valid := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	cases := []struct {
		name   string
		mutate func(*Config[string])
		want   string
	}{
		{"id", func(c *Config[string]) { c.ID = "" }, "ID"},
		{"driver", func(c *Config[string]) { c.Driver = nil }, "driver"},
		{"repository", func(c *Config[string]) { c.Repository = nil }, "repository"},
		{"codec", func(c *Config[string]) { c.Codec = nil }, "codec"},
		{"events", func(c *Config[string]) { c.Events = nil }, "event"},
		{"environment", func(c *Config[string]) { c.Environments = nil }, "environment"},
		{"processor", func(c *Config[string]) { c.ResultProcessors = nil }, "result-processor"},
		{"clock", func(c *Config[string]) { c.Clock = nil }, "clock"},
		{"ids", func(c *Config[string]) { c.IDs = nil }, "ID generator"},
		{"event middleware", func(c *Config[string]) {
			c.EventMiddleware = []event.Middleware{nil}
		}, "middleware"},
		{"lifecycle hook", func(c *Config[string]) {
			c.LifecycleHooks = []harnessruntime.LifecycleHook{nil}
		}, "lifecycle"},
		{"grace", func(c *Config[string]) { c.ToolCancellationGrace = -time.Second }, "negative"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			config := valid
			test.mutate(&config)
			if err := config.validate(); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validate error = %v, want containing %q", err, test.want)
			}
		})
	}
	if err := valid.validate(); err != nil {
		t.Fatalf("valid config: %v", err)
	}
}

func TestAcceptanceErrorRejectsSteeringWhileSuspended(t *testing.T) {
	session := newRunningSession(t)
	session.mu.Lock()
	session.transitionLocked(Suspended)
	err := session.acceptanceErrorLocked(QueueSteer)
	session.mu.Unlock()
	if !errors.Is(err, ErrSessionSuspended) {
		t.Fatalf("suspended steer acceptance = %v", err)
	}
}

func TestValueHelpersAndErrors(t *testing.T) {
	states := []State{Idle, Running, Closing, Suspended, Interrupting, Faulted, Closed, State(99)}
	want := []string{"idle", "running", "closing", "suspended", "interrupting", "faulted", "closed", "state(99)"}
	for index, state := range states {
		if got := state.String(); got != want[index] {
			t.Fatalf("%d.String() = %q", state, got)
		}
	}
	settings := options{}
	if err := WithDrainAll(true)(&settings); err != nil || !settings.drainAll {
		t.Fatalf("WithDrainAll = %#v, %v", settings, err)
	}
	limit := 3
	if err := WithBudget(agentic.UsageLimits{MaxRequests: &limit})(&settings); err != nil {
		t.Fatal(err)
	}
	limit = 9
	if settings.budget == nil || *settings.budget.MaxRequests != 3 {
		t.Fatalf("budget was not cloned: %#v", settings.budget)
	}
	cause := errors.New("boom")
	fault := &FaultError{SessionID: "s", Cause: cause}
	if !errors.Is(fault, ErrSessionFaulted) || !errors.Is(fault, cause) ||
		!strings.Contains(fault.Error(), "s") || fault.Unwrap() != cause {
		t.Fatalf("fault behavior = %v", fault)
	}
	budget := &BudgetError{}
	if budget.Error() != ErrBudgetExceeded.Error() || !errors.Is(budget, ErrBudgetExceeded) ||
		budget.Unwrap() != nil {
		t.Fatalf("empty budget behavior = %v", budget)
	}
	budget = &BudgetError{Cause: cause}
	if !strings.Contains(budget.Error(), cause.Error()) || budget.Unwrap() != cause {
		t.Fatalf("caused budget behavior = %v", budget)
	}
}

func TestEntryEncodingFailureAndDefensiveResults(t *testing.T) {
	base := jsoncodec.New()
	failure := errors.New("codec failure")
	if _, err := pending(errorCodec{base: base, encodeErr: failure}, "x", struct{}{}); !errors.Is(err, failure) {
		t.Fatalf("pending error = %v", err)
	}
	if _, err := decodePayload[struct{}](errorCodec{base: base, decodeErr: failure}, store.Entry{
		Kind: "x", Seq: 7, Payload: []byte("{}"),
	}); !errors.Is(err, failure) || !strings.Contains(err.Error(), "sequence 7") {
		t.Fatalf("decode error = %v", err)
	}
	batch := newEntryBatch(errorCodec{base: base, encodeErr: failure}, 2)
	batch.Add("first", struct{}{})
	batch.Add("ignored", struct{}{})
	if entries, err := batch.Result(); entries != nil || !errors.Is(err, failure) {
		t.Fatalf("failed batch = %#v, %v", entries, err)
	}
	good := newEntryBatch(base, 1)
	good.Add("first", struct{ Value string }{"original"})
	entries, err := good.Result()
	if err != nil || len(entries) != 1 {
		t.Fatalf("good batch = %#v, %v", entries, err)
	}
	entries[0].Payload[0] = 'x'
	again, err := good.Result()
	if err != nil || len(again) != 1 || again[0].Payload[0] == 'x' {
		t.Fatalf("batch exposed payload alias: %#v, %v", again, err)
	}
}

func TestHelperEdgeCases(t *testing.T) {
	unencodable := agentic.NewToolUseMessage(agentic.ToolUse{
		ID: "bad", Name: "bad", Input: map[string]any{"channel": make(chan int)},
	})
	cloned := cloneMessages([]agentic.Message{unencodable})
	if len(cloned) != 1 {
		t.Fatalf("fallback clone = %#v", cloned)
	}
	previous := agentic.Usage{RequestUsages: []agentic.RequestUsage{{}}}
	if _, err := usageDelta(agentic.Usage{}, previous); err == nil {
		t.Fatal("request-usage regression was accepted")
	}
	messages := []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "base")}
	markers := []contextMarker{
		{after: 0, message: agentic.NewTextMessage(agentic.RoleUser, "before")},
		{after: 9, message: agentic.NewTextMessage(agentic.RoleUser, "tail")},
	}
	projected := providerHistory(messages, markers)
	if len(projected) != 3 || projected[2].GetTextContent() != "tail" {
		t.Fatalf("provider history tail = %#v", projected)
	}
	shiftContextMarkers(markers, 2)
	if markers[0].after != 2 || markers[1].after != 11 {
		t.Fatalf("shifted markers = %#v", markers)
	}
}

func TestNewRejectsOptionAndFactoryFailures(t *testing.T) {
	boom := errors.New("boom")
	newConfig := func() Config[string] {
		return sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	}
	if _, err := New(context.Background(), newConfig(), nil); err == nil {
		t.Fatal("nil option was accepted")
	}
	if _, err := New(context.Background(), newConfig(), func(*options) error { return boom }); !errors.Is(err, boom) {
		t.Fatalf("option error = %v", err)
	}

	config := newConfig()
	config.Environments = env.FactoryFunc(func(context.Context, string) (env.Lease, error) {
		return nil, boom
	})
	if _, err := New(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("environment error = %v", err)
	}
	config = newConfig()
	config.Environments = env.FactoryFunc(func(context.Context, string) (env.Lease, error) {
		return nil, nil
	})
	if _, err := New(context.Background(), config); err == nil {
		t.Fatal("nil environment was accepted")
	}

	config = newConfig()
	config.ResultProcessors = artifact.ProcessorFactoryFunc(func(context.Context, string) (agentic.ToolResultProcessor, error) {
		return nil, boom
	})
	if _, err := New(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("processor error = %v", err)
	}
	config = newConfig()
	config.ResultProcessors = artifact.ProcessorFactoryFunc(func(context.Context, string) (agentic.ToolResultProcessor, error) {
		return nil, nil
	})
	if _, err := New(context.Background(), config); err == nil {
		t.Fatal("nil processor was accepted")
	}

	config = newConfig()
	config.Events = event.FactoryFunc(func(context.Context, []event.Record) (event.Hub, error) {
		return nil, boom
	})
	if _, err := New(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("event error = %v", err)
	}
	config = newConfig()
	config.Events = event.FactoryFunc(func(context.Context, []event.Record) (event.Hub, error) {
		return nil, nil
	})
	if _, err := New(context.Background(), config); err == nil {
		t.Fatal("nil event hub was accepted")
	}

	config = newConfig()
	config.Codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	if _, err := New(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("creation encoding error = %v", err)
	}
	config = newConfig()
	config.Repository = repositoryStub{create: func(
		context.Context,
		string,
		...store.PendingEntry,
	) (store.Journal, store.Commit, error) {
		return nil, store.Commit{}, boom
	}}
	if _, err := New(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("repository create error = %v", err)
	}
	config = newConfig()
	config.Repository = repositoryStub{create: func(
		_ context.Context,
		id string,
		_ ...store.PendingEntry,
	) (store.Journal, store.Commit, error) {
		return &journalStub{id: id}, store.Commit{}, nil
	}}
	if _, err := New(context.Background(), config); err == nil {
		t.Fatal("empty creation commit was accepted")
	}

	config = newConfig()
	config.EventMiddleware = []event.Middleware{event.MiddlewareFunc(func(agentic.EventSink) agentic.EventSink {
		return nil
	})}
	if _, err := New(context.Background(), config); err == nil {
		t.Fatal("nil middleware sink was accepted")
	}
	config = newConfig()
	config.LifecycleHooks = []harnessruntime.LifecycleHook{
		harnessruntime.LifecycleHookFunc(func(context.Context, harnessruntime.LifecycleEvent) error {
			return boom
		}),
	}
	if _, err := New(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("opened lifecycle error = %v", err)
	}
}

func TestSessionStateAndAcceptanceEdgeCases(t *testing.T) {
	config := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if session.ID() != config.ID || session.State() != Idle {
		t.Fatalf("identity/state = %q/%s", session.ID(), session.State())
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := session.Snapshot(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled snapshot = %v", err)
	}
	if _, err := session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleAssistant, "bad")); !errors.Is(err, ErrInvalidMessage) {
		t.Fatalf("invalid acceptance = %v", err)
	}
	if _, err := session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "bad state")); !errors.Is(err, ErrNotRunning) {
		t.Fatalf("idle steer = %v", err)
	}
	if _, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "queued")); err != nil {
		t.Fatal(err)
	}
	snapshot, err := session.Snapshot(context.Background())
	if err != nil || len(snapshot.Pending) != 1 {
		t.Fatalf("snapshot = %#v, %v", snapshot, err)
	}
	snapshot.Pending[0].Message = agentic.NewTextMessage(agentic.RoleUser, "mutated")
	again, _ := session.Snapshot(context.Background())
	if again.Pending[0].Message.GetTextContent() != "queued" {
		t.Fatal("snapshot queue aliases internal memory")
	}

	session.mu.Lock()
	session.transitionLocked(Running)
	session.mu.Unlock()
	waitCtx, stop := context.WithCancel(context.Background())
	stop()
	if err := session.WaitForIdle(waitCtx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled wait = %v", err)
	}
	session.mu.Lock()
	session.faultLocked(errors.New("fault"))
	session.mu.Unlock()
	if err := session.WaitForIdle(context.Background()); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted wait = %v", err)
	}
	if _, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "faulted")); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted acceptance = %v", err)
	}
	session.mu.Lock()
	session.fault = nil
	session.transitionLocked(Closed)
	session.mu.Unlock()
	if err := session.WaitForIdle(context.Background()); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("closed wait = %v", err)
	}
	if _, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "closed")); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("closed acceptance = %v", err)
	}
}

func TestAcceptanceIDCodecAppendAndSecondCheckFailures(t *testing.T) {
	makeRunning := func() *Session[string] {
		config := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
		session, err := New(context.Background(), config)
		if err != nil {
			t.Fatal(err)
		}
		session.mu.Lock()
		session.run = &activeRun{id: "run"}
		session.transitionLocked(Running)
		session.mu.Unlock()
		return session
	}
	boom := errors.New("boom")
	session := makeRunning()
	session.ids = idsFunc(func(string) (string, error) { return "", boom })
	if _, err := session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "x")); !errors.Is(err, boom) {
		t.Fatalf("ID error = %v", err)
	}

	session = makeRunning()
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	if _, err := session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "x")); !errors.Is(err, boom) {
		t.Fatalf("codec error = %v", err)
	}

	session = makeRunning()
	session.ids = idsFunc(func(string) (string, error) {
		session.mu.Lock()
		session.transitionLocked(Closing)
		session.mu.Unlock()
		return "queue_1", nil
	})
	if _, err := session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "x")); !errors.Is(err, ErrRunClosing) {
		t.Fatalf("second state check = %v", err)
	}

	session = makeRunning()
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	if _, err := session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "x")); !errors.Is(err, boom) {
		t.Fatalf("append error = %v", err)
	}
	if len(session.queue) != 0 {
		t.Fatalf("append failure mutated queue: %#v", session.queue)
	}
}

func TestQueueHelpersAndFaultDefault(t *testing.T) {
	session := &Session[string]{
		state:       Idle,
		stateChange: make(chan struct{}),
		queue: []QueueEntry{
			{ID: "one", Kind: QueueSteer},
			{ID: "two", Kind: QueueNextTurn},
			{ID: "three", Kind: QueueFollowUp},
		},
	}
	if _, ok := session.removeQueueLocked("missing"); ok {
		t.Fatal("removed missing queue item")
	}
	removed, ok := session.removeQueueLocked("two")
	if !ok || removed.ID != "two" {
		t.Fatalf("removed = %#v, %v", removed, ok)
	}
	filtered := session.queueByKindsLocked(QueueSteer, QueueFollowUp)
	if len(filtered) != 2 || filtered[0].ID != "one" || filtered[1].ID != "three" {
		t.Fatalf("filtered = %#v", filtered)
	}
	session.faultLocked(nil)
	if session.state != Faulted || !errors.Is(session.fault, ErrSessionFaulted) {
		t.Fatalf("default fault = %s/%v", session.state, session.fault)
	}
}

func TestCloseStateAndCleanupFailures(t *testing.T) {
	boomHook := errors.New("hook")
	boomJournal := errors.New("journal")
	boomEnvironment := errors.New("environment")
	hub := inproc.New(nil)
	session := &Session[string]{
		id:          "close",
		state:       Running,
		stateChange: make(chan struct{}),
		bus:         hub,
		journal: &journalStub{id: "close", close: func(context.Context) error {
			return boomJournal
		}},
		environment: &leaseStub{close: func(context.Context) error {
			return boomEnvironment
		}},
		lifecycle: []harnessruntime.LifecycleHook{
			harnessruntime.LifecycleHookFunc(func(_ context.Context, value harnessruntime.LifecycleEvent) error {
				if value.Phase == harnessruntime.LifecycleSessionClosing {
					return boomHook
				}
				return nil
			}),
		},
	}
	if err := session.Close(context.Background()); !errors.Is(err, ErrSessionBusy) {
		t.Fatalf("busy close = %v", err)
	}
	session.mu.Lock()
	session.transitionLocked(Idle)
	session.mu.Unlock()
	err := session.Close(context.Background())
	if !errors.Is(err, boomHook) || !errors.Is(err, boomJournal) || !errors.Is(err, boomEnvironment) {
		t.Fatalf("joined close errors = %v", err)
	}
	if session.State() != Closed {
		t.Fatalf("closed state = %s", session.State())
	}

	closedHookErr := errors.New("closed hook")
	session = &Session[string]{
		id:          "closed-hook",
		state:       Idle,
		stateChange: make(chan struct{}),
		bus:         inproc.New(nil),
		journal:     &journalStub{id: "closed-hook"},
		environment: &leaseStub{},
		lifecycle: []harnessruntime.LifecycleHook{
			harnessruntime.LifecycleHookFunc(func(_ context.Context, value harnessruntime.LifecycleEvent) error {
				if value.Phase == harnessruntime.LifecycleSessionClosed {
					return closedHookErr
				}
				return nil
			}),
		},
	}
	if err := session.Close(context.Background()); !errors.Is(err, closedHookErr) {
		t.Fatalf("closed hook error = %v", err)
	}
	session.lifecycle = nil
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("idempotent retry = %v", err)
	}
}

func TestContextViewCloneHelpers(t *testing.T) {
	session := &Session[string]{
		messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "base")},
		contextMarkers: []contextMarker{
			{after: 0, message: agentic.NewTextMessage(agentic.RoleUser, "context")},
		},
		run: &activeRun{contextMarkerCount: 99},
	}
	systemMessage := agentic.NewTextMessage(agentic.RoleSystem, "system")
	expected, full := session.contextViewsLocked([]agentic.Message{systemMessage})
	if len(expected) != 3 || len(full) != 3 || expected[0].Role != agentic.RoleSystem {
		t.Fatalf("context views = %#v / %#v", expected, full)
	}
	cloned := cloneContextMarkers(session.contextMarkers)
	cloned[0].message = agentic.NewTextMessage(agentic.RoleUser, "changed")
	if session.contextMarkers[0].message.GetTextContent() != "context" {
		t.Fatal("context marker clone aliased")
	}
	if cloneContextCompaction(nil) != nil {
		t.Fatal("nil compaction did not remain nil")
	}
	compaction := &contextpolicy.Compaction{
		Cut:     1,
		Summary: agentic.NewTextMessage(agentic.RoleUser, "summary"),
	}
	compactionCopy := cloneContextCompaction(compaction)
	compactionCopy.Summary = agentic.NewTextMessage(agentic.RoleUser, "changed")
	if compaction.Summary.GetTextContent() != "summary" {
		t.Fatal("compaction clone aliased")
	}
}

func TestUtilityFixturesRemainValid(t *testing.T) {
	memoryEnvironment, err := envmemory.New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	if memoryEnvironment.Files() == nil {
		t.Fatal("memory environment fixture has no filesystem")
	}
	if got := (fixedClock{value: time.Unix(1, 0)}).Now(); !got.Equal(time.Unix(1, 0)) {
		t.Fatal(got)
	}
	if _, err := (repositoryStub{create: func(
		context.Context,
		string,
		...store.PendingEntry,
	) (store.Journal, store.Commit, error) {
		return nil, store.Commit{}, nil
	}}).Open(context.Background(), "missing"); !errors.Is(err, store.ErrSessionNotFound) {
		t.Fatal(err)
	}
	if got := fmt.Sprint(passthroughProcessor()); got == "" {
		t.Fatal("processor fixture has no representation")
	}
}
