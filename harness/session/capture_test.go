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
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type callbackCodec struct {
	base   codec.Codec
	encode func(any) ([]byte, error)
}

func (c callbackCodec) Encode(value any) ([]byte, error) {
	if c.encode != nil {
		return c.encode(value)
	}
	return c.base.Encode(value)
}

func (c callbackCodec) Decode(payload []byte, value any) error {
	return c.base.Decode(payload, value)
}

func TestInitialHistoryIsCopiedPersistedAndValidated(t *testing.T) {
	repository := storememory.New()
	config := sessionConfig(t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
	history := []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "prior question"),
		agentic.NewTextMessage(agentic.RoleAssistant, "prior answer"),
	}
	current, err := New(context.Background(), config, WithInitialHistory(history...))
	if err != nil {
		t.Fatal(err)
	}
	history[0] = agentic.NewTextMessage(agentic.RoleUser, "mutated")
	snapshot, err := current.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Messages) != 2 || snapshot.Messages[0].GetTextContent() != "prior question" {
		t.Fatalf("initial history = %#v", snapshot.Messages)
	}
	if err := current.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer recovered.Close(context.Background())
	recoveredSnapshot, _ := recovered.Snapshot(context.Background())
	if !messagesEqual(snapshot.Messages, recoveredSnapshot.Messages) {
		t.Fatalf("recovered history = %#v", recoveredSnapshot.Messages)
	}

	open := agentic.NewToolUseMessage(agentic.ToolUse{ID: "open", Name: "tool"})
	if _, err := New(context.Background(), sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{}),
		WithInitialHistory(open)); !errors.Is(err, ErrInvalidInitialHistory) {
		t.Fatalf("open initial frontier = %v", err)
	}
	duplicate := []agentic.Message{
		agentic.NewToolUseMessage(
			agentic.ToolUse{ID: "same", Name: "one"},
			agentic.ToolUse{ID: "same", Name: "two"},
		),
		agentic.NewToolResultMessageFor("same", "one", "one", false),
		agentic.NewToolResultMessageFor("same", "two", "two", false),
	}
	if _, err := validateInitialHistory(duplicate); !errors.Is(err, ErrInvalidInitialHistory) {
		t.Fatalf("duplicate initial frontier = %v", err)
	}
}

func TestCaptureRuntimeSnapshotsOpenFrontierAndPolicy(t *testing.T) {
	current := newRunningSession(t)
	defer current.Close(context.Background())
	gate := agentic.ToolGateFunc(func(_ context.Context, calls []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
		return agentic.ToolBatchDecision{Calls: make([]agentic.ToolDisposition, len(calls))}, nil
	})
	toolset := agentic.NewToolset()
	current.mu.Lock()
	current.scope = harnessruntime.Scope{SessionID: current.id, ParentSessionID: "parent", Agent: "child", Depth: 1}
	current.toolsets = []agentic.Toolset{toolset}
	current.toolGate = gate
	current.delegation = []string{"delegate"}
	current.messages = []agentic.Message{
		agentic.NewTextMessage(agentic.RoleUser, "task"),
		agentic.NewToolUseMessage(
			agentic.ToolUse{ID: "call-1", Name: "delegate"},
			agentic.ToolUse{ID: "call-2", Name: "delegate"},
		),
		agentic.NewToolResultMessageFor("call-1", "delegate", "first", false),
	}
	current.run.results = map[string]bool{"call-1": true}
	current.mu.Unlock()

	history := current.History()
	if len(history) != 1 || history[0].GetTextContent() != "task" {
		t.Fatalf("captured partial frontier = %#v", history)
	}
	if len(current.Toolsets()) != 1 || current.ToolGate() == nil {
		t.Fatal("capture policy was not exposed")
	}
	names := current.DelegationTools()
	names[0] = "mutated"
	if current.DelegationTools()[0] != "delegate" {
		t.Fatal("delegation names were not copied")
	}
	if scope := current.Scope(); scope.ParentSessionID != "parent" || scope.Depth != 1 {
		t.Fatalf("capture scope = %#v", scope)
	}

	current.mu.Lock()
	current.messages = append(current.messages, agentic.NewToolResultMessageFor("call-2", "delegate", "second", false))
	current.run.results["call-2"] = true
	current.mu.Unlock()
	if history := current.History(); len(history) != 4 {
		t.Fatalf("completed frontier was omitted: %#v", history)
	}
}

func TestCaptureHistoryCoversIdleEmptyAndNonToolFrontiers(t *testing.T) {
	current := newRunningSession(t)
	defer current.Close(context.Background())

	current.mu.Lock()
	current.messages = nil
	current.mu.Unlock()
	if history := current.History(); len(history) != 0 {
		t.Fatalf("empty history = %#v", history)
	}

	current.mu.Lock()
	current.run = nil
	current.messages = []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "idle")}
	current.mu.Unlock()
	if history := current.History(); len(history) != 1 {
		t.Fatalf("idle history = %#v", history)
	}

	current.mu.Lock()
	current.run = &activeRun{}
	current.messages = []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "running")}
	current.mu.Unlock()
	if history := current.History(); len(history) != 1 {
		t.Fatalf("non-assistant frontier = %#v", history)
	}

	current.mu.Lock()
	current.messages = []agentic.Message{agentic.NewTextMessage(agentic.RoleAssistant, "complete")}
	current.mu.Unlock()
	if history := current.History(); len(history) != 1 {
		t.Fatalf("assistant text frontier = %#v", history)
	}
}

func TestChildBudgetLeaseCommitValidationAndSerialization(t *testing.T) {
	current := newRunningSession(t)
	defer current.Close(context.Background())
	current.mu.Lock()
	current.budget = &agentic.UsageLimits{MaxRequests: agentic.IntPtr(3)}
	current.usage = agentic.Usage{Requests: 1}
	current.mu.Unlock()

	lease, err := current.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if limits := lease.Limits(); limits == nil || limits.MaxRequests == nil || *limits.MaxRequests != 2 {
		t.Fatalf("child limits = %#v", limits)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := current.AcquireBudget(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled concurrent budget acquisition = %v", err)
	}
	if err := lease.Commit(context.Background(), harnessruntime.UsageCharge{
		SessionID: "child",
		Agent:     "worker",
		Usage:     agentic.Usage{Requests: 1, TotalTokens: 5},
	}); err != nil {
		t.Fatal(err)
	}
	if err := lease.Commit(context.Background(), harnessruntime.UsageCharge{SessionID: "child"}); err == nil {
		t.Fatal("duplicate child usage commit succeeded")
	}
	lease.Close()
	lease.Close()
	if snapshot, _ := current.Snapshot(context.Background()); snapshot.Usage.Requests != 2 {
		t.Fatalf("charged usage = %#v", snapshot.Usage)
	}

	second, err := current.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	second.Close()
	if err := second.Commit(context.Background(), harnessruntime.UsageCharge{SessionID: "child"}); err == nil {
		t.Fatal("closed child budget lease committed")
	}

	invalidLease, err := current.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := invalidLease.Commit(context.Background(), harnessruntime.UsageCharge{}); err == nil {
		t.Fatal("invalid child usage committed through a lease")
	}
	invalidLease.Close()

	for _, charge := range []harnessruntime.UsageCharge{
		{},
		{SessionID: "child", Usage: agentic.Usage{Requests: -1}},
	} {
		if err := validateUsageCharge(charge); err == nil {
			t.Fatalf("invalid usage charge accepted: %#v", charge)
		}
	}

	current.mu.Lock()
	current.transitionLocked(Idle)
	current.mu.Unlock()
	if _, err := current.AcquireBudget(context.Background()); err == nil {
		t.Fatal("idle parent issued a child budget")
	}
	current.mu.Lock()
	current.faultLocked(errors.New("fault"))
	current.mu.Unlock()
	if _, err := current.AcquireBudget(context.Background()); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted parent budget = %v", err)
	}

	exhausted := newRunningSession(t)
	defer exhausted.Close(context.Background())
	exhausted.mu.Lock()
	exhausted.budget = &agentic.UsageLimits{MaxRequests: agentic.IntPtr(1)}
	exhausted.usage = agentic.Usage{Requests: 1}
	exhausted.mu.Unlock()
	if _, err := exhausted.AcquireBudget(context.Background()); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("exhausted parent budget = %v", err)
	}

	for _, state := range []State{Idle, Closed} {
		inactive := newRunningSession(t)
		inactiveLease, err := inactive.AcquireBudget(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		inactive.mu.Lock()
		inactive.transitionLocked(state)
		inactive.mu.Unlock()
		err = inactiveLease.Commit(context.Background(), harnessruntime.UsageCharge{SessionID: "child"})
		inactiveLease.Close()
		if err == nil {
			t.Fatalf("%s parent accepted a late child usage commit", state)
		}
		if closeErr := inactive.Close(context.Background()); closeErr != nil {
			t.Fatal(closeErr)
		}
	}
}

func TestChildBudgetExhaustionAndStorageFailuresAreDurableFaults(t *testing.T) {
	current := newRunningSession(t)
	defer current.Close(context.Background())
	current.mu.Lock()
	current.budget = &agentic.UsageLimits{MaxRequests: agentic.IntPtr(1)}
	current.mu.Unlock()
	lease, err := current.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	err = lease.Commit(context.Background(), harnessruntime.UsageCharge{
		SessionID: "child",
		Usage:     agentic.Usage{Requests: 1},
	})
	lease.Close()
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("post-response child budget exhaustion = %v", err)
	}
	if snapshot, _ := current.Snapshot(context.Background()); snapshot.Usage.Requests != 1 {
		t.Fatalf("exhausting child usage was not committed: %#v", snapshot.Usage)
	}

	encodeFailure := newRunningSession(t)
	defer encodeFailure.Close(context.Background())
	lease, err = encodeFailure.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	encodeCanceled := false
	encodeFailure.mu.Lock()
	encodeFailure.runCancel = func() { encodeCanceled = true }
	encodeFailure.mu.Unlock()
	encodeFailure.codec = errorCodec{base: jsoncodec.New(), encodeErr: errors.New("encode")}
	err = lease.Commit(context.Background(), harnessruntime.UsageCharge{SessionID: "child"})
	lease.Close()
	if !errors.Is(err, ErrSessionFaulted) || encodeFailure.State() != Faulted || !encodeCanceled {
		t.Fatalf("encode failure = %v, state=%s, canceled=%v", err, encodeFailure.State(), encodeCanceled)
	}

	appendFailure := newRunningSession(t)
	defer appendFailure.Close(context.Background())
	originalJournal := appendFailure.journal
	lease, err = appendFailure.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	appendCanceled := false
	appendFailure.mu.Lock()
	appendFailure.runCancel = func() { appendCanceled = true }
	appendFailure.mu.Unlock()
	appendFailure.journal = &journalStub{id: appendFailure.id, append: func(context.Context, store.Cursor, ...store.PendingEntry) (store.Commit, error) {
		return store.Commit{}, errors.New("append")
	}}
	err = lease.Commit(context.Background(), harnessruntime.UsageCharge{SessionID: "child"})
	lease.Close()
	appendFailure.journal = originalJournal
	if !errors.Is(err, ErrSessionFaulted) || appendFailure.State() != Faulted || !appendCanceled {
		t.Fatalf("append failure = %v, state=%s, canceled=%v", err, appendFailure.State(), appendCanceled)
	}

	faultedCommit := newRunningSession(t)
	defer faultedCommit.Close(context.Background())
	lease, err = faultedCommit.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	faultedCommit.mu.Lock()
	faultedCommit.faultLocked(errors.New("fault before child commit"))
	faultedCommit.mu.Unlock()
	err = lease.Commit(context.Background(), harnessruntime.UsageCharge{SessionID: "child"})
	lease.Close()
	if !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted child usage commit = %v", err)
	}
}

func TestProjectedChildEventsPersistAndValidateTopology(t *testing.T) {
	repository := storememory.New()
	config := sessionConfig(t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
	config.Scope = harnessruntime.Scope{Agent: "root"}
	current, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	subscription := current.Subscribe(event.SubscribeOptions{Buffer: 16, Preview: true})
	child := event.Record{
		Nature:    agentic.EventAuthoritative,
		SessionID: "child",
		ParentID:  current.ID(),
		Agent:     "worker",
		Depth:     1,
		Source:    "agentic",
		Name:      "child.commit",
	}
	if err := current.ProjectEvent(context.Background(), child); err != nil {
		t.Fatal(err)
	}
	if err := current.ProjectEvent(context.Background(), event.Record{
		Nature:    agentic.EventPreview,
		SessionID: "child",
		ParentID:  current.ID(),
		Agent:     "worker",
		Depth:     1,
		Name:      "child.preview",
	}); err != nil {
		t.Fatal(err)
	}
	if err := current.ProjectEvent(context.Background(), event.Record{
		Nature:    agentic.EventLifecycle,
		SessionID: "grandchild",
		ParentID:  "child",
		Agent:     "nested",
		Depth:     2,
		Name:      "grandchild.lifecycle",
	}); err != nil {
		t.Fatal(err)
	}
	var first event.Record
	for first.SessionID != "child" || first.Nature == agentic.EventPreview {
		first = <-subscription.Events
	}
	second := <-subscription.Events
	if first.Cursor == 0 || first.SessionID != "child" || second.Nature != agentic.EventPreview ||
		second.Cursor != first.Cursor {
		t.Fatalf("projected events = %#v, %#v", first, second)
	}
	cursor := first.Cursor
	subscription.Close()
	if err := current.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer recovered.Close(context.Background())
	replay := recovered.Subscribe(event.SubscribeOptions{AfterCursor: cursor - 1, Buffer: 4})
	defer replay.Close()
	record := <-replay.Events
	if record.Cursor != cursor || record.SessionID != "child" || record.ParentID != config.ID {
		t.Fatalf("recovered child event = %#v", record)
	}

	invalid := newRunningSession(t)
	defer invalid.Close(context.Background())
	for _, record := range []event.Record{
		{Nature: agentic.EventAuthoritative},
		{Nature: agentic.EventAuthoritative, SessionID: invalid.ID()},
		{Nature: agentic.EventAuthoritative, SessionID: "child", ParentID: invalid.ID(), Depth: 1},
		{Nature: agentic.EventAuthoritative, SessionID: "child", ParentID: invalid.ID(), Agent: "worker", Depth: 0},
		{Nature: agentic.EventAuthoritative, SessionID: "descendant", ParentID: "other", Agent: "worker", Depth: 1},
		{Nature: agentic.EventNature(99), SessionID: "child", ParentID: invalid.ID(), Agent: "worker", Depth: 1},
	} {
		if err := invalid.ProjectEvent(context.Background(), record); err == nil {
			t.Fatalf("invalid projected event accepted: %#v", record)
		}
	}
	if err := invalid.ProjectEvent(context.Background(), event.Record{
		Nature: agentic.EventPreview, SessionID: "child", Agent: "worker", Depth: 1,
	}); err != nil {
		t.Fatalf("default direct parent ID = %v", err)
	}
	invalid.mu.Lock()
	invalid.transitionLocked(Closed)
	invalid.mu.Unlock()
	if err := invalid.ProjectEvent(context.Background(), event.Record{
		Nature: agentic.EventAuthoritative, SessionID: "child", ParentID: invalid.ID(), Agent: "worker", Depth: 1,
	}); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("closed projection = %v", err)
	}
	if err := invalid.ProjectEvent(context.Background(), event.Record{
		Nature: agentic.EventPreview, SessionID: "child", ParentID: invalid.ID(), Agent: "worker", Depth: 1,
	}); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("closed preview projection = %v", err)
	}
}

func TestProjectedEventAppendFailureFaultsParent(t *testing.T) {
	current := newRunningSession(t)
	defer current.Close(context.Background())
	appendCanceled := false
	current.mu.Lock()
	current.runCancel = func() { appendCanceled = true }
	current.mu.Unlock()
	original := current.journal
	current.journal = &journalStub{id: current.id, append: func(context.Context, store.Cursor, ...store.PendingEntry) (store.Commit, error) {
		return store.Commit{}, errors.New("project append")
	}}
	err := current.ProjectEvent(context.Background(), event.Record{
		Nature: agentic.EventLifecycle, SessionID: "child", ParentID: current.ID(), Agent: "worker", Depth: 1,
	})
	current.journal = original
	if !errors.Is(err, ErrSessionFaulted) || current.State() != Faulted || !appendCanceled {
		t.Fatalf("project append failure = %v, state=%s, canceled=%v", err, current.State(), appendCanceled)
	}

	faulted := newRunningSession(t)
	defer faulted.Close(context.Background())
	faulted.mu.Lock()
	faulted.faultLocked(errors.New("fault"))
	faulted.mu.Unlock()
	if err := faulted.ProjectEvent(context.Background(), event.Record{
		Nature: agentic.EventLifecycle, SessionID: "child", ParentID: faulted.ID(), Agent: "worker", Depth: 1,
	}); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted projection = %v", err)
	}

	previewFault := newRunningSession(t)
	defer previewFault.Close(context.Background())
	previewFault.mu.Lock()
	previewFault.faultLocked(errors.New("preview fault"))
	previewFault.mu.Unlock()
	if err := previewFault.ProjectEvent(context.Background(), event.Record{
		Nature: agentic.EventPreview, SessionID: "child", ParentID: previewFault.ID(), Agent: "worker", Depth: 1,
	}); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted preview projection = %v", err)
	}

	encodeFailure := newRunningSession(t)
	defer encodeFailure.Close(context.Background())
	encodeCanceled := false
	encodeFailure.mu.Lock()
	encodeFailure.runCancel = func() { encodeCanceled = true }
	encodeFailure.mu.Unlock()
	encodeFailure.codec = errorCodec{base: jsoncodec.New(), encodeErr: errors.New("project encode")}
	if err := encodeFailure.ProjectEvent(context.Background(), event.Record{
		Nature: agentic.EventLifecycle, SessionID: "child", ParentID: encodeFailure.ID(), Agent: "worker", Depth: 1,
	}); !errors.Is(err, ErrSessionFaulted) || encodeFailure.State() != Faulted || !encodeCanceled {
		t.Fatalf("project encode failure = %v, state=%s, canceled=%v", err, encodeFailure.State(), encodeCanceled)
	}

	faultDuringEncode := newRunningSession(t)
	defer faultDuringEncode.Close(context.Background())
	faultDuringEncode.codec = callbackCodec{
		base: jsoncodec.New(),
		encode: func(any) ([]byte, error) {
			faultDuringEncode.mu.Lock()
			faultDuringEncode.faultLocked(errors.New("concurrent fault"))
			faultDuringEncode.mu.Unlock()
			return nil, errors.New("encode after fault")
		},
	}
	if err := faultDuringEncode.ProjectEvent(context.Background(), event.Record{
		Nature: agentic.EventLifecycle, SessionID: "child", ParentID: faultDuringEncode.ID(), Agent: "worker", Depth: 1,
	}); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("fault during project encoding = %v", err)
	}

	closedDuringEncode := newRunningSession(t)
	defer closedDuringEncode.Close(context.Background())
	closedDuringEncode.codec = callbackCodec{
		base: jsoncodec.New(),
		encode: func(any) ([]byte, error) {
			closedDuringEncode.mu.Lock()
			closedDuringEncode.transitionLocked(Closed)
			closedDuringEncode.mu.Unlock()
			return nil, errors.New("encode after close")
		},
	}
	if err := closedDuringEncode.ProjectEvent(context.Background(), event.Record{
		Nature: agentic.EventLifecycle, SessionID: "child", ParentID: closedDuringEncode.ID(), Agent: "worker", Depth: 1,
	}); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("close during project encoding = %v", err)
	}
}

func TestScopeValidationAndRecordDefaults(t *testing.T) {
	config := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	config.Scope = harnessruntime.Scope{SessionID: "wrong"}
	if err := config.validate(); err == nil || !strings.Contains(err.Error(), "scope ID") {
		t.Fatalf("mismatched scope = %v", err)
	}
	config.Scope = harnessruntime.Scope{Depth: -1}
	if err := config.validate(); err == nil || !strings.Contains(err.Error(), "depth") {
		t.Fatalf("negative scope depth = %v", err)
	}
	for _, invalid := range []harnessruntime.Scope{
		{SessionID: config.ID, ParentSessionID: "parent"},
		{SessionID: config.ID, Depth: 1},
		{SessionID: config.ID, ParentSessionID: "parent", Depth: 1},
		{SessionID: config.ID, ParentSessionID: config.ID, Agent: "worker", Depth: 1},
	} {
		config.Scope = invalid
		if err := config.validate(); err == nil {
			t.Fatalf("invalid topology scope accepted: %#v", invalid)
		}
	}
	scope := harnessruntime.Scope{SessionID: "session", ParentSessionID: "parent", Agent: "worker", Depth: 2}
	record := scopeRecord(scope, event.Record{Name: "event"})
	if record.SessionID != "session" || record.ParentID != "parent" || record.Agent != "worker" || record.Depth != 2 {
		t.Fatalf("scoped record = %#v", record)
	}
	preserved := scopeRecord(scope, event.Record{SessionID: "child", ParentID: "other", Depth: 3})
	if preserved.SessionID != "child" || preserved.ParentID != "other" || preserved.Depth != 3 {
		t.Fatalf("existing child scope was overwritten: %#v", preserved)
	}

	repository := storememory.New()
	config = sessionConfig(t, &countingDriver{}, repository, artifactmemory.New(), spill.Config{})
	config.Scope = harnessruntime.Scope{ParentSessionID: "parent", Agent: "worker", Depth: 1}
	created, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if err := created.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	config.Scope = harnessruntime.Scope{}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if scope := recovered.Scope(); scope.SessionID != config.ID ||
		scope.ParentSessionID != "parent" || scope.Agent != "worker" || scope.Depth != 1 {
		t.Fatalf("recovered durable scope = %#v", scope)
	}
	if err := recovered.Close(context.Background()); err != nil {
		t.Fatal(err)
	}

	badRepository := storememory.New()
	badConfig := sessionConfig(t, &countingDriver{}, badRepository, artifactmemory.New(), spill.Config{})
	badScope := harnessruntime.Scope{SessionID: "different"}
	entry, err := pending(badConfig.Codec, kindSessionCreated, sessionCreatedPayload{Scope: &badScope})
	if err != nil {
		t.Fatal(err)
	}
	journal, _, err := badRepository.Create(context.Background(), badConfig.ID, entry)
	if err != nil {
		t.Fatal(err)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := Recover(context.Background(), badConfig); err == nil ||
		!strings.Contains(err.Error(), "persisted session scope ID") {
		t.Fatalf("mismatched durable scope = %v", err)
	}
}

func TestBudgetLeaseCloseUnblocksWaiter(t *testing.T) {
	current := newRunningSession(t)
	defer current.Close(context.Background())
	current.mu.Lock()
	current.budget = &agentic.UsageLimits{MaxRequests: agentic.IntPtr(10)}
	current.mu.Unlock()
	first, err := current.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	acquired := make(chan harnessruntime.BudgetLease, 1)
	go func() {
		lease, _ := current.AcquireBudget(context.Background())
		acquired <- lease
	}()
	select {
	case <-acquired:
		t.Fatal("second budget lease was not serialized")
	case <-time.After(10 * time.Millisecond):
	}
	first.Close()
	select {
	case lease := <-acquired:
		if lease == nil {
			t.Fatal("serialized budget acquisition returned nil")
		}
		lease.Close()
	case <-time.After(time.Second):
		t.Fatal("closing budget lease did not unblock waiter")
	}
}

func TestUnboundedBudgetLeasesCanRunConcurrently(t *testing.T) {
	current := newRunningSession(t)
	defer current.Close(context.Background())
	first, err := current.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	second, err := current.AcquireBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	first.Close()
	second.Close()
}
