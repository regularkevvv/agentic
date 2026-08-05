package harness

// Real-host conformance (plan S4): the reusable sessionloop conformance
// suite runs against an in-memory application-assembled Harness with a
// ctx-aware scripted driver, plus JSONL reopen conformance, the
// durable-receipt-precedes-driver proof, and ownership-registry release.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/permission"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/conformance"
	"github.com/regularkevvv/agentic/harness/store"
	storejsonl "github.com/regularkevvv/agentic/harness/store/jsonl"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

const sessionLoopTestWait = 10 * time.Second

func sessionLoopTestContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), sessionLoopTestWait)
	t.Cleanup(cancel)
	return ctx
}

// conformanceGate implements conformance.Gate through a driver barrier: the
// next model request blocks until released. Release is idempotent.
type conformanceGate struct {
	mu      sync.Mutex
	pending chan struct{}
}

func (g *conformanceGate) HoldNextRun() func() {
	g.mu.Lock()
	defer g.mu.Unlock()
	release := make(chan struct{})
	g.pending = release
	var once sync.Once
	return func() { once.Do(func() { close(release) }) }
}

func (g *conformanceGate) take() <-chan struct{} {
	g.mu.Lock()
	defer g.mu.Unlock()
	hold := g.pending
	g.pending = nil
	return hold
}

// conformanceDriver wraps the real agent so an armed Gate blocks the next
// accepted run at Drive entry — before its first step — while staying
// ctx-aware for interruption.
type conformanceDriver struct {
	agentic.Driver[string]
	gate *conformanceGate
}

func (d *conformanceDriver) Drive(ctx context.Context, input agentic.DriveInput, options ...agentic.RunOption) (*agentic.Execution[string], error) {
	if hold := d.gate.take(); hold != nil {
		select {
		case <-ctx.Done():
			return &agentic.Execution[string]{Status: agentic.ExecutionInterrupted}, ctx.Err()
		case <-hold:
		}
	}
	return d.Driver.Drive(ctx, input, options...)
}

// conformanceModel is the ctx-aware scripted model behind the factory: it
// keys the documented conformance scenarios on the suite's fixed start texts
// (input Meta is deliberately never model-visible through this host).
type conformanceModel struct {
	mu    sync.Mutex
	calls int
}

func (m *conformanceModel) Name() string { return "test:sessionloop-conformance" }

func (m *conformanceModel) Request(ctx context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	m.calls++
	call := m.calls
	m.mu.Unlock()

	messages := request.Messages
	if len(messages) > 0 && len(messages[len(messages)-1].GetToolResults()) > 0 {
		return conformanceText("done"), nil
	}
	lastUser := ""
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role == agentic.RoleUser {
			if text := messages[index].GetTextContent(); text != "" {
				lastUser = text
				break
			}
		}
	}
	switch {
	case strings.Contains(lastUser, "use a tool"):
		return conformanceToolCall(agentic.ToolUse{
			ID: fmt.Sprintf("call-%d", call), Name: "lookup", Input: map[string]any{"query": lastUser},
		}), nil
	case strings.Contains(lastUser, "please pause"):
		return conformanceToolCall(agentic.ToolUse{
			ID: fmt.Sprintf("gate-%d", call), Name: "danger", Input: map[string]any{"value": lastUser},
		}), nil
	case strings.Contains(lastUser, "show progress"):
		return conformanceToolCall(agentic.ToolUse{
			ID: fmt.Sprintf("prog-%d", call), Name: "progress", Input: map[string]any{"note": lastUser},
		}), nil
	default:
		return conformanceText("echo: " + lastUser), nil
	}
}

func conformanceText(text string) *agentic.ChatResponse {
	return &agentic.ChatResponse{
		Model:           "test:sessionloop-conformance",
		Message:         agentic.NewTextMessage(agentic.RoleAssistant, text),
		Usage:           agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15, Requests: 1},
		FinishReason:    agentic.FinishReasonStop,
		RawFinishReason: string(agentic.FinishReasonStop),
	}
}

func conformanceToolCall(call agentic.ToolUse) *agentic.ChatResponse {
	return &agentic.ChatResponse{
		Model:           "test:sessionloop-conformance",
		Message:         agentic.NewToolUseMessage(call),
		Usage:           agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15, Requests: 1},
		FinishReason:    agentic.FinishReasonToolCalls,
		RawFinishReason: string(agentic.FinishReasonToolCalls),
	}
}

type conformanceLookupInput struct {
	Query string `json:"query"`
}

type conformanceDangerInput struct {
	Value string `json:"value"`
}

type conformanceProgressInput struct {
	Note string `json:"note"`
}

func newSessionLoopEnv(t *testing.T, sessions store.Repository) conformance.Env {
	t.Helper()
	gate := &conformanceGate{}
	model := &conformanceModel{}
	agent := agentic.NewAgent("conformance system", model)
	agentic.AddTool(agent,
		func(_ context.Context, input conformanceLookupInput) (string, error) {
			return "found: " + input.Query, nil
		},
		agentic.AutoToolName("lookup"), agentic.AutoToolDescription("Look something up"))
	agentic.AddTool(agent,
		func(_ context.Context, input conformanceDangerInput) (string, error) {
			return "authorized: " + input.Value, nil
		},
		agentic.AutoToolName("danger"), agentic.AutoToolDescription("Perform a gated action"))
	agentic.AddTool(agent,
		func(ctx context.Context, _ conformanceProgressInput) (string, error) {
			if tools, ok := harnessruntime.FromContext(ctx); ok && tools.Emit != nil {
				tools.Emit(harnessruntime.ToolUpdate{Kind: "progress"})
			}
			return "progressing", nil
		},
		agentic.AutoToolName("progress"), agentic.AutoToolDescription("Report progress"))

	policy, err := permission.New(permission.DecisionAllow,
		permission.Rule{Pattern: "tool/danger/**", Decision: permission.DecisionAsk})
	if err != nil {
		t.Fatal(err)
	}
	permissionCapability, err := permission.NewCapability(policy)
	if err != nil {
		t.Fatal(err)
	}
	config := runtimeConfig(t)
	config.Sessions = sessions
	driver := &conformanceDriver{Driver: agent, gate: gate}
	runtime, err := New[string](driver, WithRuntime(config), WithCapabilities(permissionCapability)).Build()
	if err != nil {
		t.Fatal(err)
	}
	host, err := NewSessionLoopHost(runtime,
		WithSessionLoopOutputProjector[string](func(output string) (json.RawMessage, error) {
			return json.Marshal(output)
		}))
	if err != nil {
		t.Fatal(err)
	}
	return conformance.Env{Host: host, Gate: gate}
}

// TestSessionLoopHostConformance runs the full independent conformance suite
// against the real in-memory Harness. Every advertised optional capability
// activates its case (nothing but dispatch.idempotent may skip).
func TestSessionLoopHostConformance(t *testing.T) {
	conformance.Run(t, func(t *testing.T) conformance.Env {
		return newSessionLoopEnv(t, storememory.New())
	})
}

// TestSessionLoopHostConformanceOnJSONL runs the same suite against a
// JSONL-store-backed Harness, proving durable behaviors — including a
// suspension surviving close and reopen — hold on a real persistent store,
// not only in memory.
func TestSessionLoopHostConformanceOnJSONL(t *testing.T) {
	conformance.Run(t, func(t *testing.T) conformance.Env {
		repository, err := storejsonl.New(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		return newSessionLoopEnv(t, repository)
	})
}

func TestSessionLoopHostAdvertisedCapabilitiesActivateOptionalCases(t *testing.T) {
	env := newSessionLoopEnv(t, storememory.New())
	session, err := env.Host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(context.Background()) }()
	capabilities := session.Capabilities()
	required := []sessionloop.Capability{
		sessionloop.CapabilityDurableAcceptance,
		sessionloop.CapabilityReplay,
		sessionloop.CapabilityPreview,
		sessionloop.CapabilitySteer,
		sessionloop.CapabilityFollowUp,
		sessionloop.CapabilityNextTurn,
		sessionloop.CapabilityInterrupt,
		sessionloop.CapabilitySuspensionResolve,
		sessionloop.CapabilityDetailedTools,
		sessionloop.CapabilityStructuredOutput,
	}
	for _, capability := range required {
		if !capabilities.Supports(capability) {
			t.Fatalf("capability %q not advertised: %v", capability, capabilities)
		}
	}
	if capabilities.Supports(sessionloop.CapabilityIdempotentDispatch) {
		t.Fatal("dispatch.idempotent must never be advertised without durable idempotency")
	}
}

func sessionLoopAwaitSettled(t *testing.T, stream sessionloop.Stream, runID sessionloop.RunID) (sessionloop.Event, []sessionloop.Event) {
	t.Helper()
	var seen []sessionloop.Event
	for {
		event, err := stream.Next(sessionLoopTestContext(t))
		if err != nil {
			t.Fatalf("Next failed after %d events: %v", len(seen), err)
		}
		seen = append(seen, event)
		if event.Kind == sessionloop.EventRunSettled && event.RunID == runID {
			return event, seen
		}
	}
}

func sessionLoopEntries(events []sessionloop.Event) []sessionloop.Entry {
	var entries []sessionloop.Entry
	for _, event := range events {
		if event.Kind == sessionloop.EventEntryCommitted && event.Entry != nil {
			entries = append(entries, *event.Entry)
		}
	}
	return entries
}

// TestSessionLoopHostJSONLRecoveryConformance completes a run on a JSONL
// store, closes, reopens through the host, and proves the reopened snapshot
// and the replayed stream agree before a second run works.
func TestSessionLoopHostJSONLRecoveryConformance(t *testing.T) {
	repository, err := storejsonl.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	env := newSessionLoopEnv(t, repository)
	session, err := env.Host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	id := session.ID()
	stream, err := session.Subscribe(sessionLoopTestContext(t), sessionloop.SubscribeOptions{Buffer: 256})
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := session.Dispatch(sessionLoopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandStart, Input: &sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "durable hello"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Guarantee != sessionloop.AcceptanceDurable || receipt.Position.IsZero() {
		t.Fatalf("receipt = %#v", receipt)
	}
	_, live := sessionLoopAwaitSettled(t, stream, receipt.RunID)
	liveEntries := sessionLoopEntries(live)
	_ = stream.Close()
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("Close = %v", err)
	}

	reopened, err := env.Host.OpenSession(sessionLoopTestContext(t), id)
	if err != nil {
		t.Fatalf("OpenSession = %v", err)
	}
	defer func() { _ = reopened.Close(context.Background()) }()
	snapshot, err := reopened.Snapshot(sessionLoopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	replayStream, err := reopened.Subscribe(sessionLoopTestContext(t), sessionloop.SubscribeOptions{Buffer: 256})
	if err != nil {
		t.Fatal(err)
	}
	_, replayed := sessionLoopAwaitSettled(t, replayStream, receipt.RunID)
	replayedEntries := sessionLoopEntries(replayed)
	_ = replayStream.Close()
	if !reflect.DeepEqual(snapshot.Entries, replayedEntries) {
		t.Fatalf("reopened snapshot and replay disagree:\nsnapshot %#v\nreplay   %#v", snapshot.Entries, replayedEntries)
	}
	if len(replayedEntries) != len(liveEntries) {
		t.Fatalf("replayed %d entries, live saw %d", len(replayedEntries), len(liveEntries))
	}
	for index, entry := range replayedEntries {
		if entry.CommandID != "" {
			t.Fatalf("replayed entry %d carries live-host attribution %q", index, entry.CommandID)
		}
		if entry.Origin != liveEntries[index].Origin || !reflect.DeepEqual(entry.Content, liveEntries[index].Content) {
			t.Fatalf("replayed entry %d diverged from live content", index)
		}
	}

	secondStream, err := reopened.Subscribe(sessionLoopTestContext(t), sessionloop.SubscribeOptions{
		After: snapshot.Position, Buffer: 256,
	})
	if err != nil {
		t.Fatal(err)
	}
	second, err := reopened.Dispatch(sessionLoopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandStart, Input: &sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "second run"}}},
	})
	if err != nil {
		t.Fatalf("second start after reopen = %v", err)
	}
	settled, _ := sessionLoopAwaitSettled(t, secondStream, second.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("second outcome = %#v", settled.Outcome)
	}
	_ = secondStream.Close()
}

// recordingRepository observes every durable append so tests can assert the
// journal frontier visible at driver-invocation time.
type recordingRepository struct {
	store.Repository
	mu     sync.Mutex
	kinds  []string
	cursor store.Cursor
}

type recordingJournal struct {
	store.Journal
	owner *recordingRepository
}

func (r *recordingRepository) state() ([]string, store.Cursor) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.kinds...), r.cursor
}

func (r *recordingRepository) Create(ctx context.Context, id string, entries ...store.PendingEntry) (store.Journal, store.Commit, error) {
	journal, commit, err := r.Repository.Create(ctx, id, entries...)
	if err != nil {
		return nil, store.Commit{}, err
	}
	r.mu.Lock()
	for _, entry := range commit.Entries {
		r.kinds = append(r.kinds, entry.Kind)
	}
	r.cursor = commit.Cursor
	r.mu.Unlock()
	return &recordingJournal{Journal: journal, owner: r}, commit, nil
}

func (j *recordingJournal) Append(ctx context.Context, cursor store.Cursor, entries ...store.PendingEntry) (store.Commit, error) {
	commit, err := j.Journal.Append(ctx, cursor, entries...)
	if err != nil {
		return commit, err
	}
	j.owner.mu.Lock()
	for _, entry := range commit.Entries {
		j.owner.kinds = append(j.owner.kinds, entry.Kind)
	}
	j.owner.cursor = commit.Cursor
	j.owner.mu.Unlock()
	return commit, nil
}

type receiptProbeDriver struct {
	mu     sync.Mutex
	before func() error
	drives int
}

func (d *receiptProbeDriver) Run(ctx context.Context, prompt string, options ...agentic.RunOption) (*agentic.Result[string], error) {
	message := agentic.NewTextMessage(agentic.RoleUser, prompt)
	execution, err := d.Drive(ctx, agentic.DriveInput{Mode: agentic.DriveStart, Prompt: &message}, options...)
	if execution == nil {
		return nil, err
	}
	return execution.Result, err
}

func (d *receiptProbeDriver) Drive(_ context.Context, input agentic.DriveInput, _ ...agentic.RunOption) (*agentic.Execution[string], error) {
	d.mu.Lock()
	d.drives++
	before := d.before
	d.mu.Unlock()
	if before != nil {
		if err := before(); err != nil {
			return nil, err
		}
	}
	messages := append([]agentic.Message(nil), input.History...)
	if input.Prompt != nil {
		messages = append(messages, *input.Prompt)
	}
	return &agentic.Execution[string]{Status: agentic.ExecutionCompleted, Result: &agentic.Result[string]{Messages: messages}}, nil
}

func (d *receiptProbeDriver) Resume(context.Context, agentic.ResumeInput, ...agentic.RunOption) (*agentic.Execution[string], error) {
	return nil, errors.New("unexpected Resume")
}

// TestSessionLoopDurableReceiptPrecedesDriverExecution proves the acceptance
// facts (run.opened + prompt) are journal-committed before Drive fires, and
// that the receipt position IS that acceptance commit.
func TestSessionLoopDurableReceiptPrecedesDriverExecution(t *testing.T) {
	repository := &recordingRepository{Repository: storememory.New()}
	config := runtimeConfig(t)
	config.Sessions = repository
	driver := &receiptProbeDriver{}
	runtime, err := NewRuntime[string](driver, config)
	if err != nil {
		t.Fatal(err)
	}
	host, err := NewSessionLoopHost(runtime)
	if err != nil {
		t.Fatal(err)
	}
	session, err := host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(context.Background()) }()

	var atDrive store.Cursor
	var kindsAtDrive []string
	driver.before = func() error {
		kindsAtDrive, atDrive = repository.state()
		return nil
	}
	stream, err := session.Subscribe(sessionLoopTestContext(t), sessionloop.SubscribeOptions{Buffer: 64})
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := session.Dispatch(sessionLoopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandStart, Input: &sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "prove durability"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	sessionLoopAwaitSettled(t, stream, receipt.RunID)
	_ = stream.Close()

	if driver.drives != 1 {
		t.Fatalf("driver ran %d times", driver.drives)
	}
	joined := strings.Join(kindsAtDrive, ",")
	if !strings.Contains(joined, "run.opened") || !strings.Contains(joined, "message") {
		t.Fatalf("journal at Drive time = %v, missing the acceptance batch", kindsAtDrive)
	}
	if receipt.Position.Sequence != atDrive.Seq || receipt.Position.Token != atDrive.EntryID {
		t.Fatalf("receipt position %#v is not the acceptance commit %#v", receipt.Position, atDrive)
	}
}

// TestSessionLoopCloseReleasesOwnershipRegistry proves closing the view
// releases the root Harness live registry: ResumeSession refuses while the
// view is open and succeeds after Close; double Close is idempotent.
func TestSessionLoopCloseReleasesOwnershipRegistry(t *testing.T) {
	config := runtimeConfig(t)
	driver := &receiptProbeDriver{}
	runtime, err := NewRuntime[string](driver, config)
	if err != nil {
		t.Fatal(err)
	}
	host, err := NewSessionLoopHost(runtime)
	if err != nil {
		t.Fatal(err)
	}
	session, err := host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	id := string(session.ID())

	if _, err := runtime.ResumeSession(sessionLoopTestContext(t), id); !errors.Is(err, ErrSessionOpen) {
		t.Fatalf("ResumeSession while the view is open = %v, want ErrSessionOpen", err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("Close = %v", err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("second Close = %v, want the memoized nil", err)
	}
	resumed, err := runtime.ResumeSession(sessionLoopTestContext(t), id)
	if err != nil {
		t.Fatalf("ResumeSession after view Close = %v; the live registry leaked", err)
	}
	if err := resumed.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// TestSessionLoopStreamTerminatesAfterHostClose double-checks stream
// termination through the host-assembled path.
func TestSessionLoopStreamTerminatesAfterHostClose(t *testing.T) {
	env := newSessionLoopEnv(t, storememory.New())
	session, err := env.Host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := session.Subscribe(sessionLoopTestContext(t), sessionloop.SubscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := stream.Next(sessionLoopTestContext(t)); err != nil {
			if !errors.Is(err, io.EOF) {
				t.Fatalf("stream ended with %v, want io.EOF", err)
			}
			break
		}
	}
}
