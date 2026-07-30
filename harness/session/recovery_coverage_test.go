package session

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/artifact"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type selectiveCodec struct {
	base       codec.Codec
	rejectType reflect.Type
	err        error
}

func (c selectiveCodec) Encode(value any) ([]byte, error) {
	if c.rejectType != nil && reflect.TypeOf(value) == c.rejectType {
		return nil, c.err
	}
	return c.base.Encode(value)
}

func (c selectiveCodec) Decode(payload []byte, value any) error {
	return c.base.Decode(payload, value)
}

func encodedEntry(t *testing.T, payloadCodec codec.Codec, seq uint64, kind string, value any) store.Entry {
	t.Helper()
	payload, err := codec.Encode(payloadCodec, value)
	if err != nil {
		t.Fatal(err)
	}
	id := "entry_" + kind
	if seq > 0 {
		id += "_" + strings.Repeat("x", int(seq))
	}
	return store.Entry{
		Schema:     store.CurrentSchema,
		Seq:        seq,
		ID:         id,
		Kind:       kind,
		Payload:    payload,
		Durability: store.DurabilitySync,
	}
}

func createdEntry(t *testing.T, payloadCodec codec.Codec) store.Entry {
	t.Helper()
	return encodedEntry(t, payloadCodec, 1, kindSessionCreated, sessionCreatedPayload{})
}

func TestFoldRejectsEveryMalformedPayloadKind(t *testing.T) {
	payloadCodec := jsoncodec.New()
	kinds := []string{
		kindRunOpened,
		kindMessage,
		kindSystemMessage,
		kindRepair,
		kindQueueAccepted,
		kindQueueDrained,
		kindQueueCancelled,
		kindUsageCommitted,
		kindRecoverySuspension,
		kindInterruptMarker,
		kindContextMessage,
		kindCompaction,
	}
	for _, kind := range kinds {
		t.Run(kind, func(t *testing.T) {
			entries := []store.Entry{
				createdEntry(t, payloadCodec),
				{Schema: store.CurrentSchema, Seq: 2, ID: "bad", ParentID: "entry_session.created_x", Kind: kind, Payload: []byte("{")},
			}
			if _, _, err := fold(payloadCodec, entries); err == nil {
				t.Fatalf("fold accepted malformed %s payload", kind)
			}
		})
	}
	if _, _, err := fold(payloadCodec, nil); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("missing creation error = %v", err)
	}
	entries := []store.Entry{
		createdEntry(t, payloadCodec),
		encodedEntry(t, payloadCodec, 2, kindQueueAccepted, queueMutationPayload{ID: "queue"}),
	}
	if _, _, err := fold(payloadCodec, entries); err == nil {
		t.Fatal("accepted queue mutation without an entry")
	}
	entries = []store.Entry{
		createdEntry(t, payloadCodec),
		encodedEntry(t, payloadCodec, 2, kindContextMessage, contextMessagePayload{
			After:   1,
			Message: agentic.NewTextMessage(agentic.RoleUser, "bad"),
		}),
	}
	if _, _, err := fold(payloadCodec, entries); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("invalid marker position = %v", err)
	}
}

func TestFoldRejectsMalformedAgenticRecordsAndPayloads(t *testing.T) {
	payloadCodec := jsoncodec.New()
	if _, _, err := fold(payloadCodec, []store.Entry{
		createdEntry(t, payloadCodec),
		{Schema: store.CurrentSchema, Seq: 2, ID: "bad", Kind: kindAssistantCommitted, Payload: []byte("{")},
	}); err == nil {
		t.Fatal("malformed durable event record was accepted")
	}
	eventTypes := []struct {
		kind string
		typ  agentic.EventType
	}{
		{kindAssistantCommitted, agentic.EventTypeAssistantCommitted},
		{kindToolBatchPlanned, agentic.EventTypeToolBatchPlanned},
		{kindToolStarted, agentic.EventTypeToolStarted},
		{kindToolResult, agentic.EventTypeToolResultCommitted},
		{kindMessagesInjected, agentic.EventTypeTurnMessagesInjected},
		{kindTurnEnded, agentic.EventTypeTurnEnded},
		{kindRunSuspended, agentic.EventTypeRunSuspended},
	}
	for _, test := range eventTypes {
		t.Run(test.kind, func(t *testing.T) {
			record := event.Record{Type: test.typ, Payload: []byte("{")}
			if err := applyRecoveredEvent(payloadCodec, &foldedState{}, record); err == nil {
				t.Fatalf("apply accepted malformed %v payload", test.typ)
			}
		})
	}
	if _, _, err := fold(payloadCodec, []store.Entry{
		createdEntry(t, payloadCodec),
		encodedEntry(t, payloadCodec, 2, kindMessagesInjected, event.Record{
			Type:    agentic.EventTypeTurnMessagesInjected,
			Payload: []byte("{"),
		}),
	}); err == nil {
		t.Fatal("fold accepted malformed injected-message payload")
	}
	if _, _, err := fold(payloadCodec, []store.Entry{
		createdEntry(t, payloadCodec),
		encodedEntry(t, payloadCodec, 2, kindAssistantCommitted, event.Record{
			Type:    agentic.EventTypeAssistantCommitted,
			Payload: []byte("{"),
		}),
	}); err == nil {
		t.Fatal("fold accepted an event rejected by recovered projection")
	}
}

func TestFoldReconstructsEveryDurableStateMutation(t *testing.T) {
	payloadCodec := jsoncodec.New()
	maxRequests := 8
	systemMessage := agentic.NewTextMessage(agentic.RoleSystem, "system")
	prompt := agentic.NewTextMessage(agentic.RoleUser, "prompt")
	next := agentic.NewTextMessage(agentic.RoleUser, "next")
	repairMessage := agentic.NewToolResultMessageFor("call", "tool", "abandoned", true)
	queueOne := QueueEntry{ID: "q1", Kind: QueueSteer, Message: agentic.NewTextMessage(agentic.RoleUser, "one")}
	queueTwo := QueueEntry{ID: "q2", Kind: QueueFollowUp, Message: agentic.NewTextMessage(agentic.RoleUser, "two")}
	queueThree := QueueEntry{ID: "q3", Kind: QueueNextTurn, Message: next}
	compaction := contextpolicy.Compaction{
		Version: 1,
		Cut:     1,
		Summary: agentic.NewTextMessage(agentic.RoleUser, "summary"),
	}
	suspension := agentic.Suspension{ID: "suspension", Kind: "test", Payload: []byte(`"payload"`)}
	usage := agentic.Usage{TotalTokens: 7, Requests: 2}
	entries := []store.Entry{
		encodedEntry(t, payloadCodec, 1, kindSessionCreated, sessionCreatedPayload{
			Options: persistedOptions{
				Budget:   &agentic.UsageLimits{MaxRequests: &maxRequests},
				DrainAll: true,
			},
		}),
		encodedEntry(t, payloadCodec, 2, kindQueueAccepted, queueMutationPayload{ID: queueOne.ID, Entry: &queueOne}),
		encodedEntry(t, payloadCodec, 3, kindQueueAccepted, queueMutationPayload{ID: queueTwo.ID, Entry: &queueTwo}),
		encodedEntry(t, payloadCodec, 4, kindQueueAccepted, queueMutationPayload{ID: queueThree.ID, Entry: &queueThree}),
		encodedEntry(t, payloadCodec, 5, kindQueueDrained, queueMutationPayload{ID: queueOne.ID}),
		encodedEntry(t, payloadCodec, 6, kindQueueCancelled, queueMutationPayload{ID: queueTwo.ID}),
		encodedEntry(t, payloadCodec, 7, kindRunOpened, runOpenedPayload{ID: "run", Mode: "continue"}),
		encodedEntry(t, payloadCodec, 8, kindMessage, messagePayload{Message: next, Source: string(QueueNextTurn), QueueID: queueThree.ID}),
		encodedEntry(t, payloadCodec, 9, kindMessage, messagePayload{Message: prompt, Source: "prompt"}),
		encodedEntry(t, payloadCodec, 10, kindSystemMessage, messagePayload{Message: systemMessage}),
		encodedEntry(t, payloadCodec, 11, kindRepair, messagePayload{Message: repairMessage}),
		encodedEntry(t, payloadCodec, 12, kindUsageCommitted, usagePayload{Run: usage, Session: usage}),
		encodedEntry(t, payloadCodec, 13, kindInterruptMarker, interruptMarkerPayload{Message: "interrupted"}),
		encodedEntry(t, payloadCodec, 14, kindContextMessage, contextMessagePayload{
			After:   1,
			Message: agentic.NewTextMessage(agentic.RoleUser, "durable context"),
		}),
		encodedEntry(t, payloadCodec, 15, kindCompaction, compactionPayload{Compaction: compaction}),
		encodedEntry(t, payloadCodec, 16, kindRecoverySuspension, event.SuspensionPayload{Suspension: suspension}),
		encodedEntry(t, payloadCodec, 17, kindFault, struct{ Error string }{"fault"}),
		encodedEntry(t, payloadCodec, 18, kindBranchMoved, struct{}{}),
		encodedEntry(t, payloadCodec, 19, kindRecovered, struct{ State string }{"faulted"}),
	}
	state, history, err := fold(payloadCodec, entries)
	if err != nil {
		t.Fatal(err)
	}
	if state.state != Faulted || state.run == nil || state.run.mode != agentic.DriveContinue ||
		state.budget == nil || *state.budget.MaxRequests != maxRequests || !state.drainAll {
		t.Fatalf("folded scalar state = %#v", state)
	}
	if len(state.messages) != 4 || state.messages[0].Role != agentic.RoleSystem ||
		len(state.run.history) == 0 || state.run.history[0].Role != agentic.RoleSystem ||
		len(state.run.expected) != 2 {
		t.Fatalf("folded messages/run = %#v / %#v", state.messages, state.run)
	}
	if len(state.unapplied) != 1 || state.unapplied[0].ID != queueOne.ID ||
		len(state.queue) != 1 || state.queue[0].ID != queueThree.ID || len(state.contextMarkers) != 2 ||
		state.compaction == nil || state.suspension == nil || state.usage.TotalTokens != 7 ||
		len(history) != len(entries) {
		t.Fatalf("folded collections = %#v", state)
	}

	entries = append(entries,
		encodedEntry(t, payloadCodec, 20, kindRunClosed, runClosedPayload{ID: "run"}),
	)
	state, _, err = fold(payloadCodec, entries)
	if err != nil || state.run != nil || state.state != Idle || state.suspension != nil {
		t.Fatalf("closed fold = %#v, %v", state, err)
	}
}

func TestApplyRecoveredEventCoversRunAndProjectionState(t *testing.T) {
	payloadCodec := jsoncodec.New()
	state := foldedState{
		state: Suspended,
		run: &activeRun{
			started:             make(map[string]bool),
			results:             make(map[string]bool),
			pendingInjectionIDs: []string{"q"},
		},
	}
	call := agentic.ToolUse{ID: "call", Name: "tool"}
	usage := agentic.Usage{TotalTokens: 9}
	records := []event.Record{
		recordWithPayload(t, payloadCodec, agentic.EventTypeAssistantCommitted,
			event.AssistantPayload{Message: agentic.NewTextMessage(agentic.RoleAssistant, "answer")}),
		recordWithPayload(t, payloadCodec, agentic.EventTypeToolBatchPlanned,
			event.ToolBatchPayload{Calls: []agentic.ToolUse{call}}),
		recordWithPayload(t, payloadCodec, agentic.EventTypeToolStarted,
			event.ToolStartedPayload{Call: call}),
		recordWithPayload(t, payloadCodec, agentic.EventTypeToolResultCommitted,
			event.ToolResultPayload{ToolUseID: call.ID, ToolName: call.Name, Content: "done"}),
		recordWithPayload(t, payloadCodec, agentic.EventTypeTurnMessagesInjected,
			event.MessagesPayload{Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "injected")}}),
		recordWithPayload(t, payloadCodec, agentic.EventTypeTurnEnded,
			event.TurnEndedPayload{RunUsage: usage}),
		{Type: agentic.EventTypeRunStarted},
		recordWithPayload(t, payloadCodec, agentic.EventTypeRunSuspended,
			event.SuspensionPayload{Suspension: agentic.Suspension{ID: "again"}}),
	}
	records[5].SessionUsed = &usage
	for _, record := range records {
		if err := applyRecoveredEvent(payloadCodec, &state, record); err != nil {
			t.Fatal(err)
		}
	}
	if len(state.messages) != 3 || len(state.run.expected) != 3 ||
		len(state.run.planned) != 1 || !state.run.started[call.ID] ||
		!state.run.results[call.ID] || state.run.pendingInjectionIDs != nil ||
		state.usage.TotalTokens != 9 || state.run.lastUsage.TotalTokens != 9 ||
		!state.run.resumeEventSeen || state.state != Suspended || state.suspension.ID != "again" {
		t.Fatalf("recovered event state = %#v", state)
	}
}

func recordWithPayload(
	t *testing.T,
	payloadCodec codec.Codec,
	eventType agentic.EventType,
	payload any,
) event.Record {
	t.Helper()
	encoded, err := codec.Encode(payloadCodec, payload)
	if err != nil {
		t.Fatal(err)
	}
	return event.Record{Type: eventType, Payload: encoded}
}

func createClosedSession(t *testing.T) (Config[string], *storememory.Repository) {
	t.Helper()
	repository := storememory.New()
	config := sessionConfig(t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
	current, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if err := current.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	return config, repository
}

func TestRecoverRejectsFactoryJournalEventAndHookFailures(t *testing.T) {
	boom := errors.New("boom")
	base, repository := createClosedSession(t)

	invalid := base
	invalid.ID = ""
	if _, err := Recover(context.Background(), invalid); err == nil {
		t.Fatal("invalid recovery config was accepted")
	}
	config := base
	config.Environments = env.FactoryFunc(func(context.Context, string) (env.Lease, error) {
		return nil, boom
	})
	if _, err := Recover(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("environment error = %v", err)
	}
	config = base
	config.Environments = env.FactoryFunc(func(context.Context, string) (env.Lease, error) {
		return nil, nil
	})
	if _, err := Recover(context.Background(), config); err == nil {
		t.Fatal("nil recovery environment was accepted")
	}
	config = base
	config.ResultProcessors = artifact.ProcessorFactoryFunc(func(context.Context, string) (agentic.ToolResultProcessor, error) {
		return nil, boom
	})
	if _, err := Recover(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("processor error = %v", err)
	}
	config = base
	config.ResultProcessors = artifact.ProcessorFactoryFunc(func(context.Context, string) (agentic.ToolResultProcessor, error) {
		return nil, nil
	})
	if _, err := Recover(context.Background(), config); err == nil {
		t.Fatal("nil recovery processor was accepted")
	}
	config = base
	config.Repository = repositoryStub{
		create: func(context.Context, string, ...store.PendingEntry) (store.Journal, store.Commit, error) {
			return nil, store.Commit{}, boom
		},
		open: func(context.Context, string) (store.Journal, error) { return nil, boom },
	}
	if _, err := Recover(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("open error = %v", err)
	}
	config = base
	config.Repository = repositoryStub{
		create: func(context.Context, string, ...store.PendingEntry) (store.Journal, store.Commit, error) {
			return nil, store.Commit{}, boom
		},
		open: func(context.Context, string) (store.Journal, error) {
			return &journalStub{id: base.ID, load: func(context.Context) (store.Snapshot, error) {
				return store.Snapshot{}, boom
			}}, nil
		},
	}
	if _, err := Recover(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("load error = %v", err)
	}
	config = base
	config.Repository = repositoryStub{
		create: func(context.Context, string, ...store.PendingEntry) (store.Journal, store.Commit, error) {
			return nil, store.Commit{}, boom
		},
		open: func(context.Context, string) (store.Journal, error) {
			return &journalStub{id: base.ID, load: func(context.Context) (store.Snapshot, error) {
				return store.Snapshot{Entries: []store.Entry{{
					Schema: store.CurrentSchema, Seq: 1, ID: "bad", Kind: kindSessionCreated, Payload: []byte("{"),
				}}}, nil
			}}, nil
		},
	}
	if _, err := Recover(context.Background(), config); err == nil {
		t.Fatal("corrupt fold was accepted")
	}

	config = base
	config.Events = event.FactoryFunc(func(context.Context, []event.Record) (event.Hub, error) {
		return nil, boom
	})
	if _, err := Recover(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("event factory error = %v", err)
	}
	config = base
	config.Events = event.FactoryFunc(func(context.Context, []event.Record) (event.Hub, error) {
		return nil, nil
	})
	if _, err := Recover(context.Background(), config); err == nil {
		t.Fatal("nil recovery event hub was accepted")
	}
	config = base
	config.EventMiddleware = []event.Middleware{event.MiddlewareFunc(func(agentic.EventSink) agentic.EventSink {
		return nil
	})}
	if _, err := Recover(context.Background(), config); err == nil {
		t.Fatal("nil recovery middleware sink was accepted")
	}
	config = base
	config.LifecycleHooks = []harnessruntime.LifecycleHook{
		harnessruntime.LifecycleHookFunc(func(context.Context, harnessruntime.LifecycleEvent) error {
			return boom
		}),
	}
	if _, err := Recover(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("recovery lifecycle error = %v", err)
	}

	config = base
	config.Codec = selectiveCodec{
		base:       jsoncodec.New(),
		rejectType: reflect.TypeOf(struct{ State string }{}),
		err:        boom,
	}
	if _, err := Recover(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("recovered entry encoding error = %v", err)
	}
	failing := &failingRepository{base: repository}
	failing.fail(kindRecovered, boom)
	config = base
	config.Repository = failing
	if _, err := Recover(context.Background(), config); !errors.Is(err, boom) {
		t.Fatalf("recovered append error = %v", err)
	}
}

func TestRecoverReopensFaultedLogThroughDurableRecovery(t *testing.T) {
	config, repository := createClosedSession(t)
	journal, err := repository.Open(context.Background(), config.ID)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	faultEntry, err := pending(config.Codec, kindFault, struct{ Error string }{Error: "fault"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := journal.Append(context.Background(), snapshot.Cursor, faultEntry); err != nil {
		t.Fatal(err)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = recovered.Close(context.Background()) }()
	if recovered.State() != Idle {
		t.Fatalf("recovered faulted state = %s", recovered.State())
	}
}

func bareRecoverySession(t *testing.T) *Session[string] {
	t.Helper()
	environment, err := envmemoryForRecovery()
	if err != nil {
		t.Fatal(err)
	}
	return &Session[string]{
		id:          "recovery",
		driver:      &countingDriver{},
		codec:       jsoncodec.New(),
		environment: environment,
		processor:   passthroughProcessor(),
		clock:       fixedClock{},
		ids: idsFunc(func(prefix string) (string, error) {
			return prefix + "_id", nil
		}),
		bus:         inproc.New(nil),
		context:     contextpolicy.Passthrough(),
		eventSink:   agentic.EventSinkFunc(func(context.Context, agentic.Event) error { return nil }),
		state:       Running,
		stateChange: make(chan struct{}),
		run: &activeRun{
			id:      "old",
			started: make(map[string]bool),
			results: make(map[string]bool),
		},
	}
}

func envmemoryForRecovery() (env.Lease, error) {
	return &leaseStub{}, nil
}

func TestRecoverOpenRunFailureBoundaries(t *testing.T) {
	boom := errors.New("boom")
	input := QueueEntry{ID: "queue", Message: agentic.NewTextMessage(agentic.RoleUser, "queued")}

	session := bareRecoverySession(t)
	session.recoveryInputs = []QueueEntry{input}
	session.codec = selectiveCodec{
		base:       jsoncodec.New(),
		rejectType: reflect.TypeOf(event.MessagesPayload{}),
		err:        boom,
	}
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("recovery input payload error = %v", err)
	}

	session = bareRecoverySession(t)
	session.recoveryInputs = []QueueEntry{input}
	session.codec = selectiveCodec{
		base:       jsoncodec.New(),
		rejectType: reflect.TypeOf(event.Record{}),
		err:        boom,
	}
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("recovery event entry error = %v", err)
	}

	session = bareRecoverySession(t)
	session.recoveryInputs = []QueueEntry{input}
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("recovery input append error = %v", err)
	}

	session = bareRecoverySession(t)
	session.messages = []agentic.Message{
		agentic.NewToolUseMessage(agentic.ToolUse{ID: "duplicate", Name: "tool"}),
		agentic.NewToolResultMessageFor("duplicate", "tool", "one", false),
		agentic.NewToolResultMessageFor("duplicate", "tool", "two", false),
	}
	if err := session.recoverOpenRun(context.Background()); err == nil {
		t.Fatal("invalid recovery frontier was accepted")
	}

	call := agentic.ToolUse{ID: "indeterminate", Name: "tool"}
	session = bareRecoverySession(t)
	session.messages = []agentic.Message{agentic.NewToolUseMessage(call)}
	session.run.started[call.ID] = true
	session.ids = idsFunc(func(string) (string, error) { return "", boom })
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("recovery suspension ID error = %v", err)
	}

	session = bareRecoverySession(t)
	session.messages = []agentic.Message{agentic.NewToolUseMessage(agentic.ToolUse{
		ID: "indeterminate", Name: "tool", Input: map[string]any{"bad": make(chan int)},
	})}
	session.run.started["indeterminate"] = true
	if err := session.recoverOpenRun(context.Background()); err == nil {
		t.Fatalf("recovery frontier encoding error = %v", err)
	}

	session = bareRecoverySession(t)
	session.messages = []agentic.Message{agentic.NewToolUseMessage(call)}
	session.run.started[call.ID] = true
	session.codec = selectiveCodec{
		base:       jsoncodec.New(),
		rejectType: reflect.TypeOf(event.SuspensionPayload{}),
		err:        boom,
	}
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("recovery suspension encoding error = %v", err)
	}

	session = bareRecoverySession(t)
	session.messages = []agentic.Message{agentic.NewToolUseMessage(call)}
	session.run.started[call.ID] = true
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("recovery suspension append error = %v", err)
	}

	session = bareRecoverySession(t)
	session.ids = idsFunc(func(string) (string, error) { return "", boom })
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("recovery run ID error = %v", err)
	}

	session = bareRecoverySession(t)
	zero := 0
	session.budget = &agentic.UsageLimits{MaxRequests: &zero}
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("recovery budget error = %v", err)
	}

	session = bareRecoverySession(t)
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("recovery batch encoding error = %v", err)
	}

	session = bareRecoverySession(t)
	session.journal = &journalStub{id: session.id, append: func(
		context.Context,
		store.Cursor,
		...store.PendingEntry,
	) (store.Commit, error) {
		return store.Commit{}, boom
	}}
	if err := session.recoverOpenRun(context.Background()); !errors.Is(err, boom) {
		t.Fatalf("recovery batch append error = %v", err)
	}
}

func TestContinueRecoveredEarlyAndBudgetFailure(t *testing.T) {
	session := bareRecoverySession(t)
	session.state = Idle
	session.continueRecovered()

	session = bareRecoverySession(t)
	zero := 0
	session.budget = &agentic.UsageLimits{MaxRequests: &zero}
	session.run.limits = nil
	session.journal = &journalStub{id: session.id, append: func(
		_ context.Context,
		cursor store.Cursor,
		entries ...store.PendingEntry,
	) (store.Commit, error) {
		committed := make([]store.Entry, len(entries))
		for index, entry := range entries {
			committed[index] = store.Entry{
				Schema: store.CurrentSchema, Seq: cursor.Seq + uint64(index) + 1,
				ID: entry.Kind, Kind: entry.Kind, Payload: entry.Payload,
			}
		}
		return store.NewCommit(committed, cursor), nil
	}}
	session.continueRecovered()
	if session.State() != Idle {
		t.Fatalf("budget failure did not finish run: %s", session.State())
	}
}
