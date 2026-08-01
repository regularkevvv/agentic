package session

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type syntheticEvent struct {
	nature agentic.EventNature
	typ    agentic.EventType
	turn   int
}

func (e syntheticEvent) Nature() agentic.EventNature { return e.nature }
func (e syntheticEvent) Type() agentic.EventType     { return e.typ }
func (e syntheticEvent) TurnIndex() int              { return e.turn }

func newRunningSession(t *testing.T) *Session[string] {
	t.Helper()
	config := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	current, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	current.mu.Lock()
	current.run = &activeRun{
		id:      "run",
		started: make(map[string]bool),
		results: make(map[string]bool),
	}
	current.transitionLocked(Running)
	current.mu.Unlock()
	return current
}

func TestProjectHistoryFailureAndStateBoundaries(t *testing.T) {
	boom := errors.New("boom")
	session := newRunningSession(t)
	session.mu.Lock()
	session.faultLocked(boom)
	session.mu.Unlock()
	if _, err := session.projectHistory(context.Background(), nil); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted projection = %v", err)
	}

	session = newRunningSession(t)
	session.mu.Lock()
	session.run = nil
	session.mu.Unlock()
	if _, err := session.projectHistory(context.Background(), nil); !errors.Is(err, ErrCommitProjectionMismatch) {
		t.Fatalf("missing run projection = %v", err)
	}

	session = newRunningSession(t)
	session.messages = []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "durable")}
	canceled := false
	session.runCancel = func() { canceled = true }
	if _, err := session.projectHistory(context.Background(), []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "different"),
	}); !errors.Is(err, ErrCommitProjectionMismatch) || !canceled || session.State() != Faulted {
		t.Fatalf("mismatch projection = %v, canceled=%v, state=%s", err, canceled, session.State())
	}

	session = newRunningSession(t)
	session.context = contextpolicy.ProjectorFunc(func(context.Context, contextpolicy.ProjectionRequest) (contextpolicy.Projection, error) {
		return contextpolicy.Projection{}, boom
	})
	if _, err := session.projectHistory(context.Background(), nil); !errors.Is(err, boom) {
		t.Fatalf("projector error = %v", err)
	}

	session = newRunningSession(t)
	session.context = contextpolicy.ProjectorFunc(func(_ context.Context, request contextpolicy.ProjectionRequest) (contextpolicy.Projection, error) {
		session.mu.Lock()
		session.faultLocked(boom)
		session.mu.Unlock()
		return contextpolicy.Projection{Messages: request.Messages}, nil
	})
	if _, err := session.projectHistory(context.Background(), nil); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("post-project fault = %v", err)
	}

	session = newRunningSession(t)
	session.context = contextpolicy.ProjectorFunc(func(_ context.Context, request contextpolicy.ProjectionRequest) (contextpolicy.Projection, error) {
		session.mu.Lock()
		session.transitionLocked(Idle)
		session.mu.Unlock()
		return contextpolicy.Projection{Messages: request.Messages}, nil
	})
	if _, err := session.projectHistory(context.Background(), nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("post-project state change = %v", err)
	}

	session = newRunningSession(t)
	session.runCancel = func() {}
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	session.context = contextpolicy.ProjectorFunc(func(_ context.Context, request contextpolicy.ProjectionRequest) (contextpolicy.Projection, error) {
		return contextpolicy.Projection{
			Messages:         request.Messages,
			DurableAdditions: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "context")},
		}, nil
	})
	if _, err := session.projectHistory(context.Background(), nil); !errors.Is(err, boom) || session.State() != Faulted {
		t.Fatalf("projection encode error = %v, state=%s", err, session.State())
	}

	session = newRunningSession(t)
	session.runCancel = func() {}
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	session.context = contextpolicy.ProjectorFunc(func(_ context.Context, request contextpolicy.ProjectionRequest) (contextpolicy.Projection, error) {
		return contextpolicy.Projection{
			Messages:         request.Messages,
			DurableAdditions: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "context")},
		}, nil
	})
	if _, err := session.projectHistory(context.Background(), nil); !errors.Is(err, boom) || session.State() != Faulted {
		t.Fatalf("projection append error = %v, state=%s", err, session.State())
	}

	session = newRunningSession(t)
	session.context = contextpolicy.ProjectorFunc(func(context.Context, contextpolicy.ProjectionRequest) (contextpolicy.Projection, error) {
		return contextpolicy.Projection{Messages: []agentic.Message{
			agentic.NewToolUseMessage(agentic.ToolUse{ID: "call", Name: "tool"}),
			agentic.NewToolResultMessageFor("call", "tool", "one", false),
			agentic.NewToolResultMessageFor("call", "tool", "two", false),
		}}, nil
	})
	if _, err := session.projectHistory(context.Background(), nil); err == nil {
		t.Fatal("invalid projected frontier was accepted")
	}
}

func TestPromptPreflightAndPersistenceFailures(t *testing.T) {
	boom := errors.New("boom")
	config := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleAssistant, "bad")); !errors.Is(err, ErrInvalidMessage) {
		t.Fatalf("invalid prompt = %v", err)
	}
	for _, test := range []struct {
		state State
		want  error
	}{
		{Faulted, ErrSessionFaulted},
		{Suspended, ErrSessionSuspended},
		{Closed, ErrSessionClosed},
		{Running, ErrSessionBusy},
	} {
		session.mu.Lock()
		session.state = test.state
		session.fault = boom
		session.mu.Unlock()
		if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); !errors.Is(err, test.want) {
			t.Fatalf("%s prompt = %v", test.state, err)
		}
	}

	session, err = New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}))
	if err != nil {
		t.Fatal(err)
	}
	session.ids = idsFunc(func(string) (string, error) { return "", boom })
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); !errors.Is(err, boom) {
		t.Fatalf("run ID failure = %v", err)
	}

	session, err = New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}))
	if err != nil {
		t.Fatal(err)
	}
	session.ids = idsFunc(func(string) (string, error) {
		session.mu.Lock()
		session.transitionLocked(Running)
		session.mu.Unlock()
		return "run_id", nil
	})
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); !errors.Is(err, ErrSessionBusy) {
		t.Fatalf("second prompt check = %v", err)
	}

	session, err = New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}))
	if err != nil {
		t.Fatal(err)
	}
	zero := 0
	session.budget = &agentic.UsageLimits{MaxTotalTokens: &zero}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("prompt budget = %v", err)
	}

	session, err = New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}))
	if err != nil {
		t.Fatal(err)
	}
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); !errors.Is(err, boom) {
		t.Fatalf("prompt encoding = %v", err)
	}

	session, err = New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}))
	if err != nil {
		t.Fatal(err)
	}
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); !errors.Is(err, boom) {
		t.Fatalf("prompt append = %v", err)
	}
}

func TestRunOptionsToolRuntimeAndToolUpdateEdges(t *testing.T) {
	session := newRunningSession(t)
	session.toolsets = []agentic.Toolset{nil}
	session.toolGate = agentic.ToolGateFunc(func(context.Context, []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
		return agentic.ToolBatchDecision{}, nil
	})
	limit := 1
	options := session.runOptions(&agentic.UsageLimits{MaxRequests: &limit})
	if len(options) != 8 {
		t.Fatalf("run options = %d", len(options))
	}
	runtimeValue, ok := harnessruntime.FromContext(session.withToolRuntime(context.Background()))
	if !ok || runtimeValue.SessionID != session.id || runtimeValue.Environment != session.environment || runtimeValue.Emit == nil {
		t.Fatalf("tool runtime = %#v, %v", runtimeValue, ok)
	}

	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: errors.New("encode")}
	session.emitToolUpdate(harnessruntime.ToolUpdate{Kind: "ignored"})

	session.codec = jsoncodec.New()
	session.run = nil
	subscription := session.Subscribe(event.SubscribeOptions{
		AfterCursor: session.bus.Cursor(),
		Preview:     true,
		Buffer:      1,
	})
	defer subscription.Close()
	session.emitToolUpdate(harnessruntime.ToolUpdate{Kind: "progress"})
	select {
	case record := <-subscription.Events:
		if record.Turn != 0 || record.Ordinal != 0 || record.Name != "progress" {
			t.Fatalf("tool update = %#v", record)
		}
	case <-time.After(time.Second):
		t.Fatal("tool update was not published")
	}
}

func TestTurnHookStateDrainAndStorageFailures(t *testing.T) {
	boom := errors.New("boom")
	for _, test := range []struct {
		name  string
		state State
		want  error
	}{
		{"faulted", Faulted, ErrSessionFaulted},
		{"interrupting", Interrupting, context.Canceled},
		{"idle", Idle, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			session := newRunningSession(t)
			session.mu.Lock()
			session.state = test.state
			session.fault = boom
			session.mu.Unlock()
			_, err := session.turnHook(context.Background(), agentic.Turn{})
			if test.state == Idle {
				if err == nil || !strings.Contains(err.Error(), "idle") {
					t.Fatalf("idle hook = %v", err)
				}
			} else if !errors.Is(err, test.want) {
				t.Fatalf("%s hook = %v", test.state, err)
			}
		})
	}

	session := newRunningSession(t)
	decision, err := session.turnHook(context.Background(), agentic.Turn{})
	if err != nil || decision.Action != agentic.TurnDefault || session.State() != Running {
		t.Fatalf("noncandidate hook = %#v, %v, %s", decision, err, session.State())
	}
	session = newRunningSession(t)
	decision, err = session.turnHook(context.Background(), agentic.Turn{
		Candidate: agentic.CompletionCandidate{Source: agentic.CompletionText},
	})
	if err != nil || decision.Action != agentic.TurnDefault || session.State() != Closing {
		t.Fatalf("candidate hook = %#v, %v, %s", decision, err, session.State())
	}

	session = newRunningSession(t)
	session.drainAll = true
	session.queue = []QueueEntry{
		{ID: "one", Kind: QueueSteer, Message: agentic.NewTextMessage(agentic.RoleUser, "one")},
		{ID: "two", Kind: QueueSteer, Message: agentic.NewTextMessage(agentic.RoleUser, "two")},
	}
	decision, err = session.turnHook(context.Background(), agentic.Turn{})
	if err != nil || len(decision.Inject) != 2 || len(session.queue) != 0 {
		t.Fatalf("drain-all hook = %#v, %v, queue=%#v", decision, err, session.queue)
	}

	session = newRunningSession(t)
	session.queue = []QueueEntry{{ID: "one", Kind: QueueSteer, Message: agentic.NewTextMessage(agentic.RoleUser, "one")}}
	session.runCancel = func() {}
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	if _, err := session.turnHook(context.Background(), agentic.Turn{}); !errors.Is(err, boom) || session.State() != Faulted {
		t.Fatalf("hook encoding = %v, %s", err, session.State())
	}

	session = newRunningSession(t)
	session.queue = []QueueEntry{{ID: "one", Kind: QueueSteer, Message: agentic.NewTextMessage(agentic.RoleUser, "one")}}
	session.runCancel = func() {}
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	if _, err := session.turnHook(context.Background(), agentic.Turn{}); !errors.Is(err, boom) || session.State() != Faulted {
		t.Fatalf("hook append = %v, %s", err, session.State())
	}
}

func TestEmitPreviewDurableAndFailureEdges(t *testing.T) {
	boom := errors.New("boom")
	session := newRunningSession(t)
	if err := session.Emit(context.Background(), nil); err == nil {
		t.Fatal("nil event was accepted")
	}

	session = newRunningSession(t)
	session.codec = selectiveCodec{
		base:       jsoncodec.New(),
		rejectType: reflect.TypeOf(syntheticEvent{}),
		err:        boom,
	}
	if err := session.Emit(context.Background(), syntheticEvent{
		nature: agentic.EventPreview,
		typ:    agentic.EventTypeTextPreview,
	}); !errors.Is(err, boom) || session.State() != Running {
		t.Fatalf("preview encode failure = %v, %s", err, session.State())
	}

	session = newRunningSession(t)
	session.runCancel = func() {}
	if err := session.Emit(context.Background(), syntheticEvent{
		nature: agentic.EventAuthoritative,
		typ:    agentic.EventType(99),
	}); err == nil || session.State() != Faulted {
		t.Fatalf("unsupported durable event = %v, %s", err, session.State())
	}
	if _, err := agenticEntryKind(agentic.EventType(99)); err == nil {
		t.Fatal("unknown event kind was accepted")
	}

	session = newRunningSession(t)
	session.mu.Lock()
	session.faultLocked(boom)
	session.mu.Unlock()
	if err := session.Emit(context.Background(), syntheticEvent{
		nature: agentic.EventAuthoritative,
		typ:    agentic.EventTypeAssistantCommitted,
	}); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted emit = %v", err)
	}

	session = newRunningSession(t)
	session.mu.Lock()
	session.run = nil
	session.mu.Unlock()
	if err := session.Emit(context.Background(), syntheticEvent{
		nature: agentic.EventAuthoritative,
		typ:    agentic.EventTypeAssistantCommitted,
	}); err == nil || !strings.Contains(err.Error(), "active session run") {
		t.Fatalf("runless emit = %v", err)
	}

	session = newRunningSession(t)
	session.runCancel = func() {}
	session.codec = selectiveCodec{
		base:       jsoncodec.New(),
		rejectType: reflect.TypeOf(event.Record{}),
		err:        boom,
	}
	if err := session.Emit(context.Background(), syntheticEvent{
		nature: agentic.EventAuthoritative,
		typ:    agentic.EventTypeAssistantCommitted,
	}); !errors.Is(err, boom) || session.State() != Faulted {
		t.Fatalf("durable entry encoding = %v, %s", err, session.State())
	}

	session = newRunningSession(t)
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	if err := session.Emit(context.Background(), syntheticEvent{
		nature: agentic.EventAuthoritative,
		typ:    agentic.EventTypeAssistantCommitted,
	}); !errors.Is(err, boom) || session.State() != Faulted {
		t.Fatalf("durable append = %v, %s", err, session.State())
	}

	session = newRunningSession(t)
	subscription := session.Subscribe(event.SubscribeOptions{
		AfterCursor: session.bus.Cursor(),
		Preview:     true,
		Buffer:      2,
	})
	defer subscription.Close()
	if err := session.Emit(context.Background(), syntheticEvent{
		nature: agentic.EventPreview,
		typ:    agentic.EventTypeTextPreview,
		turn:   4,
	}); err != nil {
		t.Fatal(err)
	}
	record := <-subscription.Events
	if record.Turn != 4 || record.Ordinal != 1 {
		t.Fatalf("preview record = %#v", record)
	}
}

func TestApplyAgenticEventLockedAllMutations(t *testing.T) {
	session := newRunningSession(t)
	call := agentic.ToolUse{ID: "call", Name: "tool"}
	session.applyAgenticEventLocked(&agentic.AssistantCommittedEvent{
		Message: agentic.NewTextMessage(agentic.RoleAssistant, "answer"),
	})
	session.applyAgenticEventLocked(&agentic.ToolBatchPlannedEvent{Calls: []agentic.ToolUse{call}})
	session.run.started = nil
	session.applyAgenticEventLocked(&agentic.ToolStartedEvent{Call: call})
	session.run.results = nil
	session.applyAgenticEventLocked(&agentic.ToolResultCommittedEvent{Result: agentic.ToolExecutionResult{
		ToolUseID: call.ID, ToolName: call.Name, Content: map[string]any{"ok": true},
	}})
	session.run.pendingInjectionIDs = []string{"queue"}
	session.applyAgenticEventLocked(&agentic.TurnMessagesInjectedEvent{
		Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "injected")},
	})
	session.run.previewOrdinal = 9
	session.applyAgenticEventLocked(&agentic.TurnStartedEvent{})
	session.run.resumeInProgress = true
	session.applyAgenticEventLocked(&agentic.RunStartedEvent{})
	session.applyAgenticEventLocked(&agentic.RunSuspendedEvent{
		Suspension: agentic.Suspension{ID: "suspension"},
	})
	if len(session.messages) != 3 || len(session.run.expected) != 3 ||
		len(session.run.planned) != 1 || !session.run.started[call.ID] ||
		!session.run.results[call.ID] || session.run.pendingInjectionIDs != nil ||
		session.run.previewOrdinal != 0 || !session.run.resumeEventSeen ||
		session.suspension == nil || session.suspension.ID != "suspension" {
		t.Fatalf("agentic event state = messages=%#v run=%#v suspension=%#v", session.messages, session.run, session.suspension)
	}
}

func TestPublishOwnRecordClonesPayload(t *testing.T) {
	entry := store.Entry{Seq: 3, Kind: "kind", Payload: []byte("payload")}
	record := ownRecord(entry, agentic.EventLifecycle)
	entry.Payload[0] = 'x'
	if record.Cursor != 3 || record.Name != "kind" || string(record.Payload) != "payload" {
		t.Fatalf("own record = %#v", record)
	}
	session := newRunningSession(t)
	session.publishOwn(nil, agentic.EventAuthoritative)
	_ = inproc.New(nil)
}
