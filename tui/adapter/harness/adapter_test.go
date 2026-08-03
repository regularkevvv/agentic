package harness

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	harnesscore "github.com/regularkevvv/agentic/harness"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/observe"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
	uit "github.com/regularkevvv/agentic/tui"
)

type adapterDriver struct {
	mu         sync.Mutex
	resume     agentic.ResumeInput
	resumeErr  error
	suspension *agentic.Suspension
	block      <-chan struct{}
	entered    chan struct{}
	once       sync.Once
}

func (d *adapterDriver) Run(ctx context.Context, prompt string, options ...agentic.RunOption) (*agentic.Result[string], error) {
	message := agentic.NewTextMessage(agentic.RoleUser, prompt)
	execution, err := d.Drive(ctx, agentic.DriveInput{Mode: agentic.DriveStart, Prompt: &message}, options...)
	if execution == nil {
		return nil, err
	}
	return execution.Result, err
}

func (d *adapterDriver) Drive(ctx context.Context, input agentic.DriveInput, _ ...agentic.RunOption) (*agentic.Execution[string], error) {
	messages := append([]agentic.Message(nil), input.History...)
	if input.Prompt != nil {
		messages = append(messages, *input.Prompt)
	}
	result := &agentic.Result[string]{Messages: messages, Usage: agentic.Usage{PromptTokens: 100, CacheReadTokens: 70, TotalTokens: 100}}
	if d.block != nil {
		d.once.Do(func() { close(d.entered) })
		select {
		case <-d.block:
		case <-ctx.Done():
			return &agentic.Execution[string]{Status: agentic.ExecutionInterrupted, Result: result}, ctx.Err()
		}
	}
	if d.suspension != nil {
		return &agentic.Execution[string]{Status: agentic.ExecutionSuspended, Result: result, Suspension: d.suspension}, nil
	}
	return &agentic.Execution[string]{Status: agentic.ExecutionCompleted, Result: result}, nil
}

func (d *adapterDriver) Resume(_ context.Context, input agentic.ResumeInput, _ ...agentic.RunOption) (*agentic.Execution[string], error) {
	d.mu.Lock()
	d.resume = input
	d.mu.Unlock()
	return nil, d.resumeErr
}

func adapterRuntime(t *testing.T, driver *adapterDriver) *harnesscore.Harness[string] {
	t.Helper()
	environments, err := envmemory.NewFactory(envmemory.Config{Cwd: "/workspace"})
	if err != nil {
		t.Fatal(err)
	}
	processors, err := spill.NewFactory(artifactmemory.New(), spill.Config{})
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := harnesscore.NewRuntime[string](driver, harnesscore.RuntimeConfig{
		Sessions: storememory.New(), Codec: jsoncodec.New(), Events: inproc.NewFactory(),
		Environments: environments, ResultProcessors: processors, Clock: system.NewClock(), IDs: system.NewIDs(),
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

func TestHarnessAdapterSessionLifecycle(t *testing.T) {
	t.Parallel()
	if _, err := New[string](nil); err == nil {
		t.Fatal("nil runtime succeeded")
	}
	runtime := adapterRuntime(t, &adapterDriver{})
	if _, err := New(runtime, nil); err == nil {
		t.Fatal("nil option succeeded")
	}
	port, err := New(runtime, WithProfileLabel("work"), WithWorkspace("/workspace"))
	if err != nil {
		t.Fatal(err)
	}
	cancelledNew, cancelNew := context.WithCancel(context.Background())
	cancelNew()
	if _, err := port.NewSession(cancelledNew, uit.SessionOptions{}); err == nil {
		t.Fatal("cancelled new session succeeded")
	}
	created, err := port.NewSession(context.Background(), uit.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if created.ID() == "" {
		t.Fatal("empty ID")
	}
	subscription := created.Subscribe(uit.SubscribeOptions{Buffer: 32, Preview: true})
	snapshot, err := created.Snapshot(context.Background())
	if err != nil || snapshot.SessionID != created.ID() || snapshot.ProfileLabel != "work" || snapshot.Workspace != "/workspace" {
		t.Fatalf("snapshot = %#v, %v", snapshot, err)
	}
	if err := created.Submit(context.Background(), uit.Input{}); err == nil {
		t.Fatal("empty submit succeeded")
	}
	if err := created.Submit(context.Background(), uit.Input{Text: "hello"}); err != nil {
		t.Fatal(err)
	}
	select {
	case event := <-subscription.Events():
		if event.SessionID != created.ID() {
			t.Fatalf("observed event = %#v", event)
		}
	case <-time.After(time.Second):
		t.Fatal("adapter subscription emitted no event")
	}
	subscription.Close()
	snapshot, _ = created.Snapshot(context.Background())
	if len(snapshot.Transcript) != 1 || snapshot.Transcript[0].Text != "hello" || snapshot.Usage.CacheReadTokens != 70 {
		t.Fatalf("post-submit snapshot = %#v", snapshot)
	}
	for _, invoke := range []func(context.Context, uit.Input) error{created.Steer, created.FollowUp, created.NextTurn} {
		if err := invoke(context.Background(), uit.Input{}); err == nil {
			t.Fatal("empty queue input succeeded")
		}
	}
	if err := created.Interrupt(context.Background()); err == nil {
		t.Fatal("idle interrupt succeeded")
	}
	id := created.ID()
	if err := created.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	resumed, err := port.ResumeSession(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.ID() != id {
		t.Fatalf("resumed ID = %s", resumed.ID())
	}
	if _, err := port.ResumeSession(context.Background(), ""); err == nil {
		t.Fatal("empty resume ID succeeded")
	}
	if _, err := port.ResumeSession(context.Background(), "missing"); err == nil {
		t.Fatal("missing session resumed")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := created.Snapshot(cancelled); err == nil {
		t.Fatal("cancelled snapshot succeeded")
	}
	_ = resumed.Close(context.Background())
}

func TestHarnessAdapterQueuesAndInterruptsRunningSession(t *testing.T) {
	t.Parallel()
	release := make(chan struct{})
	driver := &adapterDriver{block: release, entered: make(chan struct{})}
	runtime := adapterRuntime(t, driver)
	port, _ := New(runtime)
	created, _ := port.NewSession(context.Background(), uit.SessionOptions{})
	done := make(chan error, 1)
	go func() { done <- created.Submit(context.Background(), uit.Input{Text: "run"}) }()
	select {
	case <-driver.entered:
	case <-time.After(time.Second):
		t.Fatal("driver did not start")
	}
	for _, invoke := range []func(context.Context, uit.Input) error{created.Steer, created.FollowUp, created.NextTurn} {
		if err := invoke(context.Background(), uit.Input{Text: "queued"}); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := created.Snapshot(context.Background())
	if err != nil || len(snapshot.Pending) != 3 {
		t.Fatalf("pending snapshot = %#v, %v", snapshot, err)
	}
	if err := created.Interrupt(context.Background()); err != nil {
		t.Fatal(err)
	}
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("submit did not stop")
	}
	_ = created.Close(context.Background())
}

func TestMappingConservativelyRedactsMessagesAndEvents(t *testing.T) {
	t.Parallel()
	toolUse := agentic.ToolUse{ID: "call", Name: "secret_tool", Input: map[string]any{"api_key": "do-not-render"}}
	toolResult := agentic.ToolResult{ToolUseID: "call", Name: "secret_tool", Content: "raw secret", IsError: true}
	messageValue := agentic.Message{Role: agentic.RoleAssistant, Content: []agentic.Part{
		{Type: agentic.ContentText, Text: "safe"},
		{Type: agentic.ContentThinking, Thinking: &agentic.ThinkingBlock{Text: "thought", ProviderName: "p", ID: "redacted_thinking", Signature: "signature", ProviderDetails: map[string]any{"secret": true}}},
		{Type: agentic.ContentToolUse, ToolUse: &toolUse},
		{Type: agentic.ContentToolResult, ToolResult: &toolResult},
		{Type: agentic.ContentToolUse}, {Type: agentic.ContentToolResult}, {Type: agentic.ContentThinking},
	}}
	mapped := message(messageValue)
	serialized, _ := json.Marshal(mapped)
	if mapped.Text != "safe" || len(mapped.Thinking) != 1 || !mapped.Thinking[0].Redacted || len(mapped.Tools) != 2 || mapped.Tools[1].State != uit.ToolError {
		t.Fatalf("mapped = %#v", mapped)
	}
	if strings.Contains(string(serialized), "do-not-render") || strings.Contains(string(serialized), "raw secret") || strings.Contains(string(serialized), "signature") {
		t.Fatalf("sensitive projection = %s", serialized)
	}
	values := messages([]agentic.Message{agentic.NewTextMessage(agentic.RoleSystem, "system"), messageValue})
	if len(values) != 1 {
		t.Fatalf("messages = %#v", values)
	}
	observed := observe.Message{Role: "assistant", Text: "safe", Thinking: []observe.Thinking{{Text: "t", Redacted: true}}, Tools: []observe.Tool{{CallID: "c", Name: "tool", State: observe.ToolDone}}}
	entry := observedMessage(observed)
	if len(entry.Thinking) != 1 || len(entry.Tools) != 1 || entry.Tools[0].State != uit.ToolDone {
		t.Fatalf("observed entry = %#v", entry)
	}
	if state(harnesscore.SessionIdle) != uit.StateIdle || usage(agentic.Usage{TotalTokens: 2}).TotalTokens != 2 {
		t.Fatal("state/usage mapping failed")
	}

	owner := snapshotOwner{snapshot: uit.Snapshot{Suspension: &uit.Suspension{ID: "s", Supported: true}}}
	observation := observe.Event{
		Cursor: 3, Ordinal: 2, Nature: agentic.EventAuthoritative, SessionID: "session", ParentID: "parent", Agent: "child", Depth: 1, Turn: 2,
		Kind: observe.KindRunSuspended, TextDelta: "delta", Message: &observed, Messages: []observe.Message{observed},
		Thinking: &observe.Thinking{Text: "think"}, Tool: &observe.Tool{CallID: "c", State: observe.ToolRunning}, Tools: []observe.Tool{{CallID: "d"}},
		Usage: &observe.Usage{PromptTokens: 2}, Suspension: &observe.Suspension{ID: "s"}, Failure: &observe.Failure{Message: "failure"},
		Queue: &observe.Queue{ID: "q", Kind: "steer", Message: &observe.Message{Text: "queued"}}, State: "suspended", Dropped: 4,
	}
	event, err := mapObservation(observation, owner)
	if err != nil || !event.Durable || event.Suspension == nil || event.Failure != "failure" || event.Queue.Text != "queued" || len(event.Entries) != 1 || len(event.Tools) != 1 {
		t.Fatalf("event = %#v, %v", event, err)
	}
	preview, _ := mapObservation(observe.Event{Nature: agentic.EventPreview, Queue: &observe.Queue{}}, owner)
	if preview.Durable || preview.Queue == nil {
		t.Fatalf("preview = %#v", preview)
	}
	_, err = mapObservation(observe.Event{Suspension: &observe.Suspension{}}, snapshotOwner{err: errors.New("snapshot")})
	if err == nil {
		t.Fatal("snapshot mapping error ignored")
	}
}

func TestSuspensionAndResolutionValidation(t *testing.T) {
	t.Parallel()
	unsupported, err := suspension(agentic.Suspension{ID: "custom", Kind: "custom"})
	if err != nil || unsupported.Supported || !strings.Contains(unsupported.Description, "no registered") {
		t.Fatalf("unsupported = %#v, %v", unsupported, err)
	}
	if _, err := suspension(agentic.Suspension{ID: "bad", Kind: harnessruntime.PermissionDeferralKind, Payload: []byte("bad")}); err == nil {
		t.Fatal("malformed permission suspension succeeded")
	}
	runtime := adapterRuntime(t, &adapterDriver{})
	port, _ := New(runtime)
	created, _ := port.NewSession(context.Background(), uit.SessionOptions{})
	if err := created.Resolve(context.Background(), uit.Resolution{}); err == nil {
		t.Fatal("invalid resolution succeeded")
	}
	if err := created.Resolve(context.Background(), uit.Resolution{SuspensionID: "missing"}); err == nil {
		t.Fatal("missing current suspension resolved")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := created.Resolve(cancelled, uit.Resolution{SuspensionID: "missing"}); err == nil {
		t.Fatal("cancelled resolution succeeded")
	}
	_ = created.Close(context.Background())

	malformedDriver := &adapterDriver{suspension: &agentic.Suspension{ID: "bad", Kind: harnessruntime.PermissionDeferralKind, Payload: []byte("bad")}}
	malformedRuntime := adapterRuntime(t, malformedDriver)
	malformedPort, _ := New(malformedRuntime)
	malformedSession, _ := malformedPort.NewSession(context.Background(), uit.SessionOptions{})
	if err := malformedSession.Submit(context.Background(), uit.Input{Text: "suspend"}); err != nil {
		t.Fatal(err)
	}
	if _, err := malformedSession.Snapshot(context.Background()); err == nil {
		t.Fatal("malformed suspension snapshot succeeded")
	}
	if err := malformedSession.Resolve(context.Background(), uit.Resolution{SuspensionID: "bad"}); err == nil {
		t.Fatal("malformed suspension resolution succeeded")
	}
	_ = malformedSession.Close(context.Background())
}

type snapshotOwner struct {
	snapshot uit.Snapshot
	err      error
}

func (o snapshotOwner) Snapshot(context.Context) (uit.Snapshot, error) { return o.snapshot, o.err }

type observeSubscription struct {
	events chan observe.Event
	errors chan error
	once   sync.Once
}

func (s *observeSubscription) Events() <-chan observe.Event { return s.events }
func (s *observeSubscription) Errors() <-chan error         { return s.errors }
func (s *observeSubscription) Close() {
	s.once.Do(func() {
		close(s.events)
		close(s.errors)
	})
}

func TestMappedSubscriptionLifecycleAndErrors(t *testing.T) {
	t.Parallel()
	nilMapped := mapSubscription(nil, snapshotOwner{})
	if _, ok := <-nilMapped.Events(); ok {
		t.Fatal("nil source events remained open")
	}
	nilMapped.Close()
	nilMapped.Close()

	source := &observeSubscription{events: make(chan observe.Event, 1), errors: make(chan error, 1)}
	mapped := mapSubscription(source, snapshotOwner{})
	source.events <- observe.Event{Kind: observe.KindTextDelta, TextDelta: "x"}
	if event := <-mapped.Events(); event.TextDelta != "x" {
		t.Fatalf("mapped event = %#v", event)
	}
	source.errors <- errors.New("lag")
	if err := <-mapped.Errors(); err == nil {
		t.Fatal("source error missing")
	}
	mapped.Close()
	mapped.Close()

	projectionSource := &observeSubscription{events: make(chan observe.Event, 1), errors: make(chan error, 1)}
	projection := mapSubscription(projectionSource, snapshotOwner{err: errors.New("snapshot")})
	projectionSource.events <- observe.Event{Suspension: &observe.Suspension{}}
	if err := <-projection.Errors(); err == nil {
		t.Fatal("projection error missing")
	}
	projection.Close()

	closedSource := &observeSubscription{events: make(chan observe.Event), errors: make(chan error)}
	closedMapped := mapSubscription(closedSource, snapshotOwner{})
	closedSource.Close()
	if _, ok := <-closedMapped.Events(); ok {
		t.Fatal("normally closed source retained mapped events")
	}
	if _, ok := <-closedMapped.Errors(); ok {
		t.Fatal("normally closed source retained mapped errors")
	}
	closedMapped.Close()
	var nilPointer *mappedSubscription
	nilPointer.Close()

	cancelSource := &observeSubscription{events: make(chan observe.Event, 1), errors: make(chan error)}
	cancelMapped := mapSubscription(cancelSource, snapshotOwner{})
	cancelSource.events <- observe.Event{Kind: observe.KindTextDelta}
	cancelMapped.Close()
	for range cancelMapped.Events() {
	}
}

func TestPermissionSuspensionMapsAndResolveReachesDriver(t *testing.T) {
	t.Parallel()
	resumeErr := errors.New("resume reached")
	driver := &adapterDriver{resumeErr: resumeErr}
	driver.suspension = permissionSuspension(t)
	runtime := adapterRuntime(t, driver)
	port, _ := New(runtime)
	created, _ := port.NewSession(context.Background(), uit.SessionOptions{})
	if err := created.Submit(context.Background(), uit.Input{Text: "suspend"}); err != nil {
		t.Fatal(err)
	}
	snapshot, err := created.Snapshot(context.Background())
	if err != nil || snapshot.Suspension == nil || !snapshot.Suspension.Supported || len(snapshot.Suspension.Approvals) != 2 {
		t.Fatalf("snapshot = %#v, %v", snapshot, err)
	}
	base := uit.Resolution{SuspensionID: snapshot.Suspension.ID}
	unknown := base
	unknown.Decisions = []uit.Decision{{CallID: "unknown", Action: uit.DecisionApprove}}
	if created.Resolve(context.Background(), unknown) == nil {
		t.Fatal("unknown approval resolved")
	}
	if created.Resolve(context.Background(), base) == nil {
		t.Fatal("incomplete approval resolved")
	}
	resolution := base
	resolution.Decisions = []uit.Decision{{CallID: "call", Action: uit.DecisionApprove}, {CallID: "call-two", Action: uit.DecisionDeny, Reason: "no"}}
	resolution.Prompt = &uit.Input{Text: "continue"}
	if err := created.Resolve(context.Background(), resolution); !errors.Is(err, resumeErr) {
		t.Fatalf("resume error = %v", err)
	}
	driver.mu.Lock()
	defer driver.mu.Unlock()
	if len(driver.resume.Decisions) != 2 || driver.resume.Decisions[0].Action != agentic.ToolResumeExecute || driver.resume.Decisions[1].Action != agentic.ToolResumeReturn || driver.resume.Prompt == nil {
		t.Fatalf("resume input = %#v", driver.resume)
	}
}

func permissionSuspension(t *testing.T) *agentic.Suspension {
	t.Helper()
	inner, err := json.Marshal(map[string]any{
		"version": 1, "required_resolution_ids": []string{"call", "call-two"},
		"requests": []map[string]any{{
			"call_id": "call", "request": map[string]any{
				"Capability": "filesystem", "Action": "write",
				"CanonicalResource": map[string]any{"Scheme": "file", "ID": "/workspace/file", "Display": "file"},
			},
		}, {
			"call_id": "call-two", "request": map[string]any{
				"Capability": "shell", "Action": "exec",
				"CanonicalResource": map[string]any{"Scheme": "command", "ID": "echo", "Display": "echo"},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	id := "suspension"
	root, err := json.Marshal(map[string]any{
		"Version": 1, "SuspensionID": id,
		"Calls":             []agentic.ToolUse{{ID: "call", Name: "write_file"}, {ID: "call-two", Name: "run_command"}},
		"ExecutableCallIDs": []string{"call", "call-two"},
		"Deferral":          agentic.ToolDeferral{Kind: harnessruntime.PermissionDeferralKind, Payload: inner},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &agentic.Suspension{ID: id, Kind: harnessruntime.PermissionDeferralKind, Payload: root}
}

var _ observe.Subscription = (*observeSubscription)(nil)

// Keep timeout behavior deterministic if a failure accidentally blocks a
// mapped subscription; no production code depends on this sentinel.
var _ = time.Second
