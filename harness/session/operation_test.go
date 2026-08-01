package session

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestRuntimeOperationIsDurableCursorFactAndRecoveryNeutral(t *testing.T) {
	repository := storememory.New()
	config := sessionConfig(t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	initial, _ := session.Snapshot(context.Background())
	subscription := session.Subscribe(SubscribeOptions{AfterCursor: initial.Cursor, Buffer: 4})
	defer subscription.Close()
	if err := session.RecordOperation(context.Background(), harnessruntime.Operation{}); err == nil {
		t.Fatal("invalid operation succeeded")
	}
	session.mu.Lock()
	session.transitionLocked(Running)
	session.mu.Unlock()
	operation := harnessruntime.Operation{ID: "op-1", Kind: "test.operation", Phase: "planned", ParentCallID: "outer", Payload: []byte("payload")}
	if err := session.RecordOperation(context.Background(), operation); err != nil {
		t.Fatal(err)
	}
	operation.Payload[0] = 'X'
	select {
	case record := <-subscription.Events:
		if record.Name != kindRuntimeOperation || record.Cursor <= initial.Cursor {
			t.Fatalf("record = %#v", record)
		}
	case <-time.After(time.Second):
		t.Fatal("operation event was not published")
	}
	loaded, err := session.journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, entry := range loaded.Entries {
		if entry.Kind != kindRuntimeOperation {
			continue
		}
		decoded, err := decodePayload[harnessruntime.Operation](session.codec, entry)
		if err != nil || string(decoded.Payload) != "payload" {
			t.Fatalf("decoded operation = %#v, %v", decoded, err)
		}
		found = true
	}
	if !found {
		t.Fatal("runtime operation missing from journal")
	}
	session.mu.Lock()
	session.transitionLocked(Idle)
	session.mu.Unlock()
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil || recovered.State() != Idle {
		t.Fatalf("recovered state=%v err=%v", recovered.State(), err)
	}
}

func TestRuntimeOperationFailureFaultsAndProjectedResultSpills(t *testing.T) {
	base := storememory.New()
	repository := &failingRepository{base: base}
	artifacts := artifactmemory.New()
	config := sessionConfig(t, &countingDriver{}, repository, artifacts, spill.Config{Threshold: 16, Head: 4, Tail: 4})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	session.mu.Lock()
	session.transitionLocked(Running)
	session.mu.Unlock()
	call := agentic.ToolUse{ID: "nested", Name: "tool"}
	projected, err := session.ProjectToolResult(context.Background(), call, agentic.ToolExecutionResult{
		ToolUseID: "nested", ToolName: "tool", Content: strings.Repeat("x", 100),
	})
	if err != nil || !strings.Contains(projected.Content.(string), "harness artifact") || artifacts.Count(session.ID()) != 1 {
		t.Fatalf("projected=%#v err=%v count=%d", projected, err, artifacts.Count(session.ID()))
	}
	repository.fail(kindRuntimeOperation, errors.New("disk failed"))
	err = session.RecordOperation(context.Background(), harnessruntime.Operation{ID: "op", Kind: "test", Phase: "started"})
	if !errors.Is(err, ErrSessionFaulted) || session.State() != Faulted {
		t.Fatalf("operation failure=%v state=%s", err, session.State())
	}
	if err := session.RecordOperation(context.Background(), harnessruntime.Operation{ID: "again", Kind: "test", Phase: "result"}); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted operation = %v", err)
	}
}

func TestProjectToolResultRejectsIdentityMismatchAndClosedRuntime(t *testing.T) {
	config := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	session.mu.Lock()
	session.transitionLocked(Running)
	session.mu.Unlock()
	_, err = session.ProjectToolResult(context.Background(), agentic.ToolUse{ID: "one", Name: "tool"}, agentic.ToolExecutionResult{ToolUseID: "other", ToolName: "tool"})
	if !errors.Is(err, ErrSessionFaulted) || session.State() != Faulted {
		t.Fatalf("identity mismatch=%v state=%s", err, session.State())
	}

	closed, err := New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}))
	if err != nil {
		t.Fatal(err)
	}
	if err := closed.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := closed.RecordOperation(context.Background(), harnessruntime.Operation{ID: "op", Kind: "test", Phase: "planned"}); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("closed operation = %v", err)
	}
}

func TestRuntimeOperationStateCodecAndCancellationFrontiers(t *testing.T) {
	config := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	idle, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	operation := harnessruntime.Operation{ID: "op", Kind: "test", Phase: "planned"}
	if err := idle.RecordOperation(context.Background(), operation); err == nil || !strings.Contains(err.Error(), "running session") {
		t.Fatalf("idle operation = %v", err)
	}

	encodeFailure, err := New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}))
	if err != nil {
		t.Fatal(err)
	}
	encodeFailure.mu.Lock()
	encodeFailure.transitionLocked(Running)
	encodeFailure.codec = errorCodec{base: encodeFailure.codec, encodeErr: errors.New("encode failed")}
	encodeFailure.mu.Unlock()
	if err := encodeFailure.RecordOperation(context.Background(), operation); !errors.Is(err, ErrSessionFaulted) || encodeFailure.State() != Faulted {
		t.Fatalf("encode failure = %v state=%s", err, encodeFailure.State())
	}
	if err := encodeFailure.faultOperation(errors.New("again")); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("already faulted operation = %v", err)
	}

	base := storememory.New()
	repository := &failingRepository{base: base}
	appendFailure, err := New(context.Background(), sessionConfig(t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{}))
	if err != nil {
		t.Fatal(err)
	}
	canceled := false
	appendFailure.mu.Lock()
	appendFailure.transitionLocked(Running)
	appendFailure.runCancel = func() { canceled = true }
	appendFailure.mu.Unlock()
	repository.fail(kindRuntimeOperation, errors.New("append failed"))
	if err := appendFailure.RecordOperation(context.Background(), operation); !errors.Is(err, ErrSessionFaulted) || !canceled {
		t.Fatalf("append failure = %v canceled=%v", err, canceled)
	}

	closed, err := New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}))
	if err != nil {
		t.Fatal(err)
	}
	if err := closed.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := closed.faultOperation(errors.New("closed")); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("closed fault operation = %v", err)
	}
}

func TestProjectedResultProcessorFailureFrontiers(t *testing.T) {
	newRunning := func(t *testing.T, processor agentic.ToolResultProcessor) *Session[string] {
		t.Helper()
		session, err := New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}))
		if err != nil {
			t.Fatal(err)
		}
		session.mu.Lock()
		session.transitionLocked(Running)
		session.processor = processor
		session.mu.Unlock()
		return session
	}
	call := agentic.ToolUse{ID: "nested", Name: "tool"}
	result := agentic.ToolExecutionResult{ToolUseID: "nested", ToolName: "tool", Content: "value"}

	processorError := errors.New("processor failed")
	failing := newRunning(t, agentic.ToolResultProcessorFunc(func(context.Context, agentic.ToolUse, agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error) {
		return agentic.ToolExecutionResult{}, processorError
	}))
	if _, err := failing.ProjectToolResult(context.Background(), call, result); !errors.Is(err, ErrSessionFaulted) || failing.State() != Faulted {
		t.Fatalf("processor failure = %v state=%s", err, failing.State())
	}

	canceledContext, cancel := context.WithCancel(context.Background())
	cancel()
	canceledSession := newRunning(t, agentic.ToolResultProcessorFunc(func(context.Context, agentic.ToolUse, agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error) {
		return agentic.ToolExecutionResult{}, processorError
	}))
	if _, err := canceledSession.ProjectToolResult(canceledContext, call, result); !errors.Is(err, context.Canceled) || canceledSession.State() != Running {
		t.Fatalf("canceled projection = %v state=%s", err, canceledSession.State())
	}

	for name, processor := range map[string]agentic.ToolResultProcessor{
		"identity": agentic.ToolResultProcessorFunc(func(_ context.Context, _ agentic.ToolUse, current agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error) {
			current.ToolName = "other"
			return current, nil
		}),
		"error disposition": agentic.ToolResultProcessorFunc(func(_ context.Context, _ agentic.ToolUse, current agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error) {
			current.IsError = true
			return current, nil
		}),
	} {
		t.Run(name, func(t *testing.T) {
			session := newRunning(t, processor)
			if _, err := session.ProjectToolResult(context.Background(), call, result); !errors.Is(err, ErrSessionFaulted) {
				t.Fatalf("invalid projection = %v", err)
			}
		})
	}
}
