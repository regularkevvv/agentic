package session

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/repair"
	"github.com/regularkevvv/agentic/harness/store"
)

func executionWith(
	status agentic.ExecutionStatus,
	messages []agentic.Message,
	usage agentic.Usage,
) *agentic.Execution[string] {
	return &agentic.Execution[string]{
		Status: status,
		Result: &agentic.Result[string]{Messages: messages, Usage: usage},
	}
}

func TestFinishExecutionFaultInterruptNoRunAndResumeValidation(t *testing.T) {
	boom := errors.New("boom")
	session := newRunningSession(t)
	canceled := false
	session.runCancel = func() { canceled = true }
	session.mu.Lock()
	session.fault = boom
	session.transitionLocked(Faulted)
	session.mu.Unlock()
	execution := executionWith(agentic.ExecutionFailed, nil, agentic.Usage{})
	if returned, err := session.finishExecution(execution, nil); returned != execution ||
		!errors.Is(err, ErrSessionFaulted) || !canceled {
		t.Fatalf("fault finish = %#v, %v, canceled=%v", returned, err, canceled)
	}

	session = newRunningSession(t)
	execution = &agentic.Execution[string]{Status: agentic.ExecutionInterrupted}
	if returned, err := session.finishExecution(execution, nil); returned != execution ||
		!errors.Is(err, context.Canceled) || session.State() != Idle {
		t.Fatalf("interrupt finish = %#v, %v, state=%s", returned, err, session.State())
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
	if _, err := session.finishExecution(
		&agentic.Execution[string]{Status: agentic.ExecutionInterrupted},
		boom,
	); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("interrupt storage finish = %v", err)
	}

	session = newRunningSession(t)
	session.mu.Lock()
	session.run = nil
	session.mu.Unlock()
	if _, err := session.finishExecution(nil, nil); err == nil || !strings.Contains(err.Error(), "active session run") {
		t.Fatalf("runless finish = %v", err)
	}

	for _, validationErr := range []error{
		agentic.ErrDriveInput,
		agentic.ErrTranscriptInvalid,
		agentic.ErrSuspensionVersion,
		agentic.ErrSuspensionMismatch,
		agentic.ErrResumeDecision,
	} {
		if !isResumeValidationError(validationErr) {
			t.Fatalf("resume validation error not recognized: %v", validationErr)
		}
	}
	if isResumeValidationError(boom) {
		t.Fatal("ordinary error recognized as resume validation")
	}
	session = newRunningSession(t)
	session.run.resumeInProgress = true
	if _, err := session.finishExecution(nil, agentic.ErrResumeDecision); !errors.Is(err, agentic.ErrResumeDecision) ||
		session.State() != Suspended || session.run.resumeInProgress {
		t.Fatalf("resume validation finish = %v, state=%s run=%#v", err, session.State(), session.run)
	}
}

func TestFinishSuspendedValidationUsageEncodingAndAppendFailures(t *testing.T) {
	boom := errors.New("boom")
	session := newRunningSession(t)
	if _, err := session.finishExecution(&agentic.Execution[string]{
		Status: agentic.ExecutionSuspended,
	}, nil); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("incomplete suspension = %v", err)
	}

	session = newRunningSession(t)
	bad := executionWith(agentic.ExecutionSuspended, []agentic.Message{
		agentic.NewTextMessage(agentic.RoleAssistant, "uncommitted"),
	}, agentic.Usage{})
	bad.Suspension = &agentic.Suspension{ID: "suspension"}
	if _, err := session.finishExecution(bad, nil); !errors.Is(err, ErrSessionFaulted) ||
		!errors.Is(err, ErrCommitProjectionMismatch) {
		t.Fatalf("suspension projection mismatch = %v", err)
	}

	session = newRunningSession(t)
	session.run.lastUsage = agentic.Usage{TotalTokens: 2}
	backwards := executionWith(agentic.ExecutionSuspended, nil, agentic.Usage{TotalTokens: 1})
	backwards.Suspension = &agentic.Suspension{ID: "suspension"}
	if _, err := session.finishExecution(backwards, nil); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("suspension usage regression = %v", err)
	}

	session = newRunningSession(t)
	session.runCancel = func() {}
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	encodeFailure := executionWith(agentic.ExecutionSuspended, nil, agentic.Usage{TotalTokens: 1})
	encodeFailure.Suspension = &agentic.Suspension{ID: "suspension"}
	if _, err := session.finishExecution(encodeFailure, nil); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("suspension encoding failure = %v", err)
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
	appendFailure := executionWith(agentic.ExecutionSuspended, nil, agentic.Usage{TotalTokens: 1})
	appendFailure.Suspension = &agentic.Suspension{ID: "suspension"}
	if _, err := session.finishExecution(appendFailure, nil); !errors.Is(err, ErrSessionFaulted) ||
		!errors.Is(err, boom) {
		t.Fatalf("suspension append failure = %v", err)
	}
}

func TestFinishSuspendedPersistsUsageSystemAndSuspension(t *testing.T) {
	session := newRunningSession(t)
	systemMessage := agentic.NewTextMessage(agentic.RoleSystem, "system")
	execution := executionWith(agentic.ExecutionSuspended, []agentic.Message{systemMessage}, agentic.Usage{
		PromptTokens: 3,
		TotalTokens:  3,
		Requests:     1,
	})
	execution.Suspension = &agentic.Suspension{ID: "suspension", Payload: []byte(`{}`)}
	returned, err := session.finishExecution(execution, nil)
	if err != nil || returned != execution || session.State() != Suspended ||
		session.suspension == nil || session.suspension.ID != "suspension" ||
		session.usage.TotalTokens != 3 || session.run.lastUsage.TotalTokens != 3 ||
		len(session.messages) != 1 || session.messages[0].Role != agentic.RoleSystem ||
		len(session.run.history) != 1 || session.run.history[0].Role != agentic.RoleSystem {
		t.Fatalf("suspended finish = %#v, %v, state=%s usage=%#v messages=%#v run=%#v",
			returned, err, session.State(), session.usage, session.messages, session.run)
	}
}

func TestFinishOrdinaryUsageBudgetQueueEncodingAndAppendPaths(t *testing.T) {
	boom := errors.New("boom")
	session := newRunningSession(t)
	session.run.lastUsage = agentic.Usage{TotalTokens: 2}
	if _, err := session.finishExecution(
		executionWith(agentic.ExecutionFailed, nil, agentic.Usage{TotalTokens: 1}),
		boom,
	); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("ordinary usage regression = %v", err)
	}

	session = newRunningSession(t)
	session.runCancel = func() {}
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	if _, err := session.finishExecution(nil, boom); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("ordinary encoding failure = %v", err)
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
	if _, err := session.finishExecution(nil, boom); !errors.Is(err, ErrSessionFaulted) ||
		!errors.Is(err, boom) {
		t.Fatalf("ordinary append failure = %v", err)
	}

	session = newRunningSession(t)
	limit := 1
	session.budget = &agentic.UsageLimits{MaxRequests: &limit}
	session.queue = []QueueEntry{
		{ID: "steer", Kind: QueueSteer},
		{ID: "follow", Kind: QueueFollowUp},
		{ID: "next", Kind: QueueNextTurn},
	}
	limitErr := &agentic.UsageLimitExceededError{LimitName: "requests", Current: 2, Max: 1}
	execution := executionWith(agentic.ExecutionFailed, nil, agentic.Usage{TotalTokens: 4})
	returned, err := session.finishExecution(execution, limitErr)
	if returned != execution || !errors.Is(err, ErrBudgetExceeded) || session.State() != Idle ||
		len(session.queue) != 1 || session.queue[0].Kind != QueueNextTurn ||
		session.usage.TotalTokens != 4 {
		t.Fatalf("budget finish = %#v, %v, state=%s queue=%#v usage=%#v",
			returned, err, session.State(), session.queue, session.usage)
	}

	session = newRunningSession(t)
	systemMessage := agentic.NewTextMessage(agentic.RoleSystem, "system")
	execution = executionWith(agentic.ExecutionCompleted, []agentic.Message{systemMessage}, agentic.Usage{})
	if _, err := session.finishExecution(execution, nil); err != nil ||
		session.State() != Idle || len(session.messages) != 1 || session.messages[0].Role != agentic.RoleSystem {
		t.Fatalf("system completion = %v, state=%s messages=%#v", err, session.State(), session.messages)
	}
}

func TestValidateProjectionAndPersistFaultEdges(t *testing.T) {
	boom := errors.New("boom")
	session := newRunningSession(t)
	if system, err := session.validateProjectionLocked(nil); err != nil || system != nil {
		t.Fatalf("nil projection = %#v, %v", system, err)
	}
	session.run.history = []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "history")}
	if _, err := session.validateProjectionLocked(executionWith(
		agentic.ExecutionCompleted,
		nil,
		agentic.Usage{},
	)); !errors.Is(err, ErrCommitProjectionMismatch) {
		t.Fatalf("short projection = %v", err)
	}
	session.run.history = nil
	session.run.expected = []agentic.Message{agentic.NewTextMessage(agentic.RoleAssistant, "expected")}
	if _, err := session.validateProjectionLocked(executionWith(
		agentic.ExecutionCompleted,
		[]agentic.Message{agentic.NewTextMessage(agentic.RoleAssistant, "actual")},
		agentic.Usage{},
	)); !errors.Is(err, ErrCommitProjectionMismatch) {
		t.Fatalf("suffix projection = %v", err)
	}

	session = newRunningSession(t)
	session.runCancel = func() {}
	session.codec = selectiveCodec{
		base:       jsoncodec.New(),
		rejectType: reflect.TypeOf(struct{ Error string }{}),
		err:        boom,
	}
	if err := session.persistFaultLocked(boom); !errors.Is(err, ErrSessionFaulted) ||
		session.State() != Faulted {
		t.Fatalf("fault with encoding failure = %v, %s", err, session.State())
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
	if err := session.persistFaultLocked(boom); !errors.Is(err, ErrSessionFaulted) ||
		session.State() != Faulted {
		t.Fatalf("fault with append failure = %v, %s", err, session.State())
	}
}

func TestFinishInterruptFailureAndCompletePaths(t *testing.T) {
	boom := errors.New("boom")
	session := newRunningSession(t)
	session.mu.Lock()
	session.faultLocked(boom)
	session.mu.Unlock()
	if err := session.finishInterrupt(nil); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted interrupt finish = %v", err)
	}

	session = newRunningSession(t)
	session.run = nil
	if err := session.finishInterrupt(nil); err == nil || !strings.Contains(err.Error(), "active run") {
		t.Fatalf("runless interrupt finish = %v", err)
	}

	session = newRunningSession(t)
	if err := session.finishInterrupt(executionWith(agentic.ExecutionInterrupted, []agentic.Message{
		agentic.NewTextMessage(agentic.RoleAssistant, "uncommitted"),
	}, agentic.Usage{})); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("interrupt projection mismatch = %v", err)
	}

	session = newRunningSession(t)
	session.messages = []agentic.Message{
		agentic.NewToolUseMessage(agentic.ToolUse{ID: "call", Name: "tool"}),
		agentic.NewToolResultMessageFor("call", "tool", "one", false),
		agentic.NewToolResultMessageFor("call", "tool", "two", false),
	}
	if err := session.finishInterrupt(nil); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("interrupt repair failure = %v", err)
	}

	session = newRunningSession(t)
	session.run.lastUsage = agentic.Usage{TotalTokens: 2}
	if err := session.finishInterrupt(executionWith(agentic.ExecutionInterrupted, nil, agentic.Usage{
		TotalTokens: 1,
	})); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("interrupt usage regression = %v", err)
	}

	session = newRunningSession(t)
	session.runCancel = func() {}
	session.codec = errorCodec{base: jsoncodec.New(), encodeErr: boom}
	if err := session.finishInterrupt(nil); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("interrupt encoding failure = %v", err)
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
	if err := session.finishInterrupt(nil); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("interrupt append failure = %v", err)
	}

	session = newRunningSession(t)
	callPlanned := agentic.ToolUse{ID: "planned", Name: "tool"}
	callStarted := agentic.ToolUse{ID: "started", Name: "tool"}
	session.messages = []agentic.Message{agentic.NewToolUseMessage(callPlanned, callStarted)}
	session.run.started[callStarted.ID] = true
	pending := session.pendingCallsLocked()
	if len(pending.Calls) != 2 || pending.Calls[0].State != repair.PendingPlanned ||
		pending.Calls[1].State != repair.PendingIndeterminate {
		t.Fatalf("pending calls = %#v", pending)
	}

	session = newRunningSession(t)
	session.queue = []QueueEntry{
		{ID: "steer", Kind: QueueSteer},
		{ID: "follow", Kind: QueueFollowUp},
		{ID: "next", Kind: QueueNextTurn},
	}
	systemMessage := agentic.NewTextMessage(agentic.RoleSystem, "system")
	execution := executionWith(agentic.ExecutionInterrupted, []agentic.Message{systemMessage}, agentic.Usage{
		TotalTokens: 5,
	})
	if err := session.finishInterrupt(execution); err != nil || session.State() != Idle ||
		len(session.queue) != 1 || session.queue[0].Kind != QueueNextTurn ||
		session.usage.TotalTokens != 5 || len(session.messages) != 1 ||
		session.messages[0].Role != agentic.RoleSystem || len(session.contextMarkers) != 1 {
		t.Fatalf("complete interrupt = %v, state=%s queue=%#v usage=%#v messages=%#v markers=%#v",
			err, session.State(), session.queue, session.usage, session.messages, session.contextMarkers)
	}
}

func TestInterruptPublicStateMachineEdges(t *testing.T) {
	boom := errors.New("boom")
	session := newRunningSession(t)
	session.mu.Lock()
	session.faultLocked(boom)
	session.mu.Unlock()
	if err := session.Interrupt(context.Background()); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted interrupt = %v", err)
	}

	for _, test := range []struct {
		state State
		want  error
	}{
		{Closed, ErrSessionClosed},
		{Idle, ErrNotRunning},
		{State(99), ErrNotRunning},
	} {
		session = newRunningSession(t)
		session.mu.Lock()
		session.state = test.state
		session.mu.Unlock()
		if err := session.Interrupt(context.Background()); !errors.Is(err, test.want) {
			t.Fatalf("%s interrupt = %v", test.state, err)
		}
	}

	session = newRunningSession(t)
	session.mu.Lock()
	session.transitionLocked(Interrupting)
	session.mu.Unlock()
	go func() {
		session.mu.Lock()
		session.transitionLocked(Idle)
		session.mu.Unlock()
	}()
	if err := session.Interrupt(context.Background()); err != nil {
		t.Fatalf("already interrupting = %v", err)
	}

	session = newRunningSession(t)
	session.mu.Lock()
	session.transitionLocked(Suspended)
	session.mu.Unlock()
	if err := session.Interrupt(context.Background()); err != nil || session.State() != Idle {
		t.Fatalf("suspended interrupt = %v, state=%s", err, session.State())
	}

	session = newRunningSession(t)
	session.runCancel = func() {
		go func() {
			_ = session.finishInterrupt(&agentic.Execution[string]{Status: agentic.ExecutionInterrupted})
		}()
	}
	if err := session.Interrupt(context.Background()); err != nil || session.State() != Idle {
		t.Fatalf("running interrupt = %v, state=%s", err, session.State())
	}
}
