package sessionloop

// Differential proof (plan S5): two independent in-memory Harness runtimes
// are built from ONE deterministic assembly (fixed clock, counter IDs) and
// the same ctx-aware scripted model. Host A is the direct legacy adapter
// (tui/adapter/harness); host B is this sessionloop bridge over
// harness.NewSessionLoopHost. Both are driven through identical scripted
// sequences and must produce identical normalized snapshots, filtered event
// tuples, and operation error identities.
//
// Documented differential exclusions (each also marked where applied):
//
//	Snapshots
//	  - Tool.Summary: legacy uses the session's application-owned ToolSummary
//	    redactor; the protocol carries no summary, the bridge uses the name.
//	  - Suspension.Description: legacy leaves supported descriptions empty;
//	    the protocol always carries a safe display description.
//	  - Approval.ResourceScheme / Approval.CanonicalResource: the protocol
//	    deliberately transports only the safe display resource string.
//	  - Entry.Thinking: the protocol excludes thinking from the authoritative
//	    projection (law L12); scripts avoid thinking so this stays empty.
//	  - Suspension.ID: suspension identifiers are minted by the agentic
//	    core's own randomness, not the injected deterministic IDGenerator,
//	    so two independent runtimes can never agree; both are normalized to
//	    a placeholder after within-host consistency is exercised.
//	Events (tuples are filtered to the kinds both paths emit)
//	  - legacy session.created / turn.started / turn.ended / tool.planned /
//	    tool.started / output.validated have no protocol source by design.
//	  - legacy message.system carries privacy-excluded system instructions
//	    the protocol never projects (plan 8.5); the app ignores the kind.
//	  - legacy run.started and run.ended duplicates with empty State come
//	    from agentic bookkeeping records; run identity and settlement are
//	    authoritative in run.opened / run.closed, which both paths emit.
//	  - legacy run.interrupted with State "interrupting" is the
//	    interrupt.marker record; settlement stays authoritative in run.closed.
//	  - legacy resolution.accepted and protocol command.accepted are
//	    path-unique passthrough kind strings the app ignores.
//	  - Cursor is excluded for run.completed/run.interrupted/run.failed: the
//	    bridge synthesizes them at the settlement cursor while legacy reads
//	    them from separate agentic records.
//	  - queue.drained / queue.cancelled tuples compare the queue ID only:
//	    legacy drain payloads carry no kind or text.
//	  - events delivered before a lag disconnect are not compared: the two
//	    pipelines hold different in-flight depths and the app discards
//	    speculative state and resynchronizes from a snapshot on any error.
//	Previews (TestDifferentialPreviewParity: Preview-true subscriptions over
//	a streaming scripted model, compared on the preview tier only)
//	  - Cursor and Ordinal are excluded: legacy stamps the raw preview
//	    record's cursor and bus ordinal while the bridge repeats the last
//	    durable position with a projection-local ordinal.
//	  - tool previews compare call identity and state only: legacy
//	    tool.planned previews carry Name and Summary (and a batch Tools
//	    slice for call starts); the protocol preview transports only the
//	    tool call ID.
//	  - legacy tool.update previews (anonymous capability tool updates)
//	    have no bridge counterpart: the protocol preview carries no call
//	    identity for them, and the bridge drops identity-less tool previews
//	    the app reducer could never upsert.
//	Errors
//	  - compared by errors.Is class AND err.Error() text; both must match,
//	    except submit-interrupted, where the spec mandates the bridge return
//	    bare context.Canceled while legacy surfaces the driver's wrap text
//	    (the errors.Is identity still must match).
//	  - resolve bounce: when an accepted resolve fails resume validation and
//	    the session bounces straight back to suspended, legacy Resolve
//	    surfaces the concrete validation error object; the protocol's
//	    zero-position live-only session.state bounce signal deliberately
//	    carries no error identity, so the bridge returns its own boundary
//	    error ("resume attempt was rejected and the session remains
//	    suspended"). Exercised against the scripted host fake
//	    (TestResolveResumeValidationBounceIsTerminal), not by this script.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
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
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/permission"
	"github.com/regularkevvv/agentic/harness/session"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
	uit "github.com/regularkevvv/agentic/tui"
	legacyadapter "github.com/regularkevvv/agentic/tui/adapter/harness"
)

const differentialWait = 30 * time.Second

// fixedClock pins every wall-clock observation (characterization approach).
type fixedClock struct{ value time.Time }

func (c fixedClock) Now() time.Time { return c.value }

// counterIDs is a deterministic per-prefix ID generator; each runtime owns
// an independent instance so both journals mint identical identifiers.
type counterIDs struct {
	mu       sync.Mutex
	counters map[string]int
}

func (g *counterIDs) New(prefix string) (string, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.counters == nil {
		g.counters = map[string]int{}
	}
	g.counters[prefix]++
	return fmt.Sprintf("%s_d%d", prefix, g.counters[prefix]), nil
}

// runGate blocks the next model request until released, ctx-aware, so the
// script can steer or interrupt mid-run at a deterministic point.
type runGate struct {
	mu      sync.Mutex
	entered chan struct{}
	hold    chan struct{}
}

func (g *runGate) arm() (<-chan struct{}, func()) {
	g.mu.Lock()
	defer g.mu.Unlock()
	entered := make(chan struct{})
	hold := make(chan struct{})
	g.entered, g.hold = entered, hold
	var once sync.Once
	return entered, func() { once.Do(func() { close(hold) }) }
}

func (g *runGate) take() (chan struct{}, chan struct{}) {
	g.mu.Lock()
	defer g.mu.Unlock()
	entered, hold := g.entered, g.hold
	g.entered, g.hold = nil, nil
	return entered, hold
}

// scriptModel keys deterministic responses on the last user text and holds
// at an armed gate before answering.
type scriptModel struct {
	gate  *runGate
	mu    sync.Mutex
	calls int
}

func (m *scriptModel) Name() string { return "test:differential" }

func (m *scriptModel) Request(ctx context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	if entered, hold := m.gate.take(); hold != nil {
		close(entered)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-hold:
		}
	}
	m.mu.Lock()
	m.calls++
	call := m.calls
	m.mu.Unlock()
	messages := request.Messages
	lastUser := ""
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role == agentic.RoleUser {
			if text := messages[index].GetTextContent(); text != "" {
				lastUser = text
				break
			}
		}
	}
	if len(messages) > 0 && len(messages[len(messages)-1].GetToolResults()) > 0 {
		return scriptText("done"), nil
	}
	switch {
	case strings.Contains(lastUser, "please pause"):
		return scriptToolCall(agentic.ToolUse{
			ID: fmt.Sprintf("gate-%d", call), Name: "danger", Input: map[string]any{"value": lastUser},
		}), nil
	case strings.Contains(lastUser, "tool"):
		return scriptToolCall(agentic.ToolUse{
			ID: fmt.Sprintf("call-%d", call), Name: "lookup", Input: map[string]any{"query": lastUser},
		}), nil
	default:
		return scriptText("echo: " + lastUser), nil
	}
}

func scriptText(text string) *agentic.ChatResponse {
	return &agentic.ChatResponse{
		Model:           "test:differential",
		Message:         agentic.NewTextMessage(agentic.RoleAssistant, text),
		Usage:           agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15, Requests: 1},
		FinishReason:    agentic.FinishReasonStop,
		RawFinishReason: string(agentic.FinishReasonStop),
	}
}

func scriptToolCall(call agentic.ToolUse) *agentic.ChatResponse {
	return &agentic.ChatResponse{
		Model:           "test:differential",
		Message:         agentic.NewToolUseMessage(call),
		Usage:           agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15, Requests: 1},
		FinishReason:    agentic.FinishReasonToolCalls,
		RawFinishReason: string(agentic.FinishReasonToolCalls),
	}
}

// streamingScriptModel replays the scripted response through the streaming
// transport: text splits into two deltas, tool calls into a call-start plus
// two argument deltas, so both differential paths observe an identical
// deterministic preview sequence.
type streamingScriptModel struct {
	*scriptModel
}

func (m *streamingScriptModel) RequestStream(ctx context.Context, request *agentic.ChatRequest) (*agentic.StreamResult, error) {
	response, err := m.Request(ctx, request)
	if err != nil {
		return nil, err
	}
	events := make(chan agentic.StreamEvent, 16)
	go func() {
		defer close(events)
		for _, part := range response.Message.Content {
			switch part.Type {
			case agentic.ContentText:
				for _, chunk := range splitHalves(part.Text) {
					events <- agentic.StreamEvent{Type: agentic.StreamEventTextDelta, Delta: chunk}
				}
			case agentic.ContentToolUse:
				if part.ToolUse == nil {
					continue
				}
				arguments, marshalErr := json.Marshal(part.ToolUse.Input)
				if marshalErr != nil {
					events <- agentic.StreamEvent{Type: agentic.StreamEventError, Error: marshalErr}
					return
				}
				events <- agentic.StreamEvent{
					Type:    agentic.StreamEventToolCallStart,
					ToolUse: &agentic.ToolUse{ID: part.ToolUse.ID, Name: part.ToolUse.Name},
				}
				for _, chunk := range splitHalves(string(arguments)) {
					events <- agentic.StreamEvent{
						Type: agentic.StreamEventToolCallDelta, ToolCallID: part.ToolUse.ID, Delta: chunk,
					}
				}
			}
		}
		usage := response.Usage
		events <- agentic.StreamEvent{Type: agentic.StreamEventDone, Usage: &usage, FinishReason: response.FinishReason}
	}()
	return agentic.NewStreamResult(events), nil
}

func splitHalves(value string) []string {
	half := len(value) / 2
	var result []string
	for _, chunk := range []string{value[:half], value[half:]} {
		if chunk != "" {
			result = append(result, chunk)
		}
	}
	return result
}

type lookupInput struct {
	Query string `json:"query"`
}

type dangerInput struct {
	Value string `json:"value"`
}

// differentialRuntime is the single deterministic assembly both hosts share.
// With streaming enabled the scripted model serves the streaming transport,
// so runs publish deterministic preview records.
func differentialRuntime(t *testing.T, gate *runGate, streaming bool) *harnesscore.Harness[string] {
	t.Helper()
	var model agentic.Model = &scriptModel{gate: gate}
	if streaming {
		model = &streamingScriptModel{scriptModel: &scriptModel{gate: gate}}
	}
	agent := agentic.NewAgent("differential system", model)
	agentic.AddTool(agent,
		func(_ context.Context, input lookupInput) (string, error) { return "found: " + input.Query, nil },
		agentic.AutoToolName("lookup"), agentic.AutoToolDescription("Look something up"))
	agentic.AddTool(agent,
		func(_ context.Context, input dangerInput) (string, error) { return "authorized: " + input.Value, nil },
		agentic.AutoToolName("danger"), agentic.AutoToolDescription("Perform a gated action"))
	policy, err := permission.New(permission.DecisionAllow,
		permission.Rule{Pattern: "tool/danger/**", Decision: permission.DecisionAsk})
	if err != nil {
		t.Fatal(err)
	}
	capability, err := permission.NewCapability(policy)
	if err != nil {
		t.Fatal(err)
	}
	environments, err := envmemory.NewFactory(envmemory.Config{Cwd: "/workspace"})
	if err != nil {
		t.Fatal(err)
	}
	processors, err := spill.NewFactory(artifactmemory.New(), spill.Config{})
	if err != nil {
		t.Fatal(err)
	}
	config := harnesscore.RuntimeConfig{
		Sessions: storememory.New(), Codec: jsoncodec.New(), Events: inproc.NewFactory(),
		Environments: environments, ResultProcessors: processors,
		Clock:          fixedClock{value: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)},
		IDs:            &counterIDs{},
		ModelStreaming: streaming,
	}
	runtime, err := harnesscore.New[string](agent, harnesscore.WithRuntime(config), harnesscore.WithCapabilities(capability)).Build()
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

func differentialPresenter() uit.ToolPresenter {
	return uit.ToolPresenterFunc(func(tool uit.Tool) uit.ToolPresentation {
		return uit.ToolPresentation{Category: uit.ToolCategoryExplore, Title: "Safe " + tool.Name}
	})
}

// diffHost is one side of the differential.
type diffHost struct {
	name string
	host uit.Host
	gate *runGate
}

func newLegacyHost(t *testing.T) diffHost {
	return newLegacyHostStreaming(t, false)
}

func newLegacyHostStreaming(t *testing.T, streaming bool) diffHost {
	t.Helper()
	gate := &runGate{}
	runtime := differentialRuntime(t, gate, streaming)
	host, err := legacyadapter.New(runtime,
		legacyadapter.WithProfileLabel("work"), legacyadapter.WithWorkspace("/workspace"),
		legacyadapter.WithExecutionLabel("local"), legacyadapter.WithToolPresenter(differentialPresenter()))
	if err != nil {
		t.Fatal(err)
	}
	return diffHost{name: "legacy", host: host, gate: gate}
}

func newBridgeHost(t *testing.T) diffHost {
	return newBridgeHostStreaming(t, false)
}

func newBridgeHostStreaming(t *testing.T, streaming bool) diffHost {
	t.Helper()
	gate := &runGate{}
	runtime := differentialRuntime(t, gate, streaming)
	loopHost, err := harnesscore.NewSessionLoopHost(runtime)
	if err != nil {
		t.Fatal(err)
	}
	host, err := New(loopHost,
		WithProfileLabel("work"), WithWorkspace("/workspace"),
		WithExecutionLabel("local"), WithToolPresenter(differentialPresenter()))
	if err != nil {
		t.Fatal(err)
	}
	return diffHost{name: "bridge", host: host, gate: gate}
}

// stepRecord captures one operation's observable outcome.
type stepRecord struct {
	name     string
	errText  string
	errClass string
	snapshot uit.Snapshot
}

func errClass(err error) string {
	if err == nil {
		return "nil"
	}
	classes := []struct {
		name   string
		target error
	}{
		{"context.Canceled", context.Canceled},
		{"session.ErrNotRunning", session.ErrNotRunning},
		{"session.ErrSessionBusy", session.ErrSessionBusy},
		{"session.ErrSessionSuspended", session.ErrSessionSuspended},
		{"session.ErrRunClosing", session.ErrRunClosing},
		{"session.ErrSessionClosed", session.ErrSessionClosed},
		{"session.ErrSessionFaulted", session.ErrSessionFaulted},
	}
	names := []string{"error"}
	for _, class := range classes {
		if errors.Is(err, class.target) {
			names = append(names, class.name)
		}
	}
	return strings.Join(names, "|")
}

// scriptResult is everything the script observed on one host.
type scriptResult struct {
	steps       []stepRecord
	live        []uit.Event
	lagErrors   []string
	replay      []uit.Event
	finalCursor uint64
}

// runScript drives one host through the shared scripted sequence.
func runScript(t *testing.T, target diffHost) scriptResult {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), differentialWait)
	defer cancel()
	created, err := target.host.NewSession(ctx, uit.SessionOptions{})
	if err != nil {
		t.Fatalf("%s: NewSession: %v", target.name, err)
	}
	defer func() { _ = created.Close(context.Background()) }()
	live := created.Subscribe(uit.SubscribeOptions{Buffer: 4096})
	defer live.Close()

	var result scriptResult
	record := func(name string, opErr error) {
		snapshot, snapErr := created.Snapshot(ctx)
		if snapErr != nil {
			t.Fatalf("%s: %s snapshot: %v", target.name, name, snapErr)
		}
		text := ""
		if opErr != nil {
			text = opErr.Error()
		}
		result.steps = append(result.steps, stepRecord{
			name: name, errText: text, errClass: errClass(opErr), snapshot: snapshot,
		})
	}

	// Error rows while idle.
	record("steer-idle", created.Steer(ctx, uit.Input{Text: "nobody is running"}))
	record("follow-up-idle", created.FollowUp(ctx, uit.Input{Text: "nobody is running"}))
	record("interrupt-idle", created.Interrupt(ctx))

	// Plain completion.
	record("submit-hello", created.Submit(ctx, uit.Input{Text: "hello"}))

	// Queue while idle, drained by the next submit (with a tool call).
	record("next-turn-idle", created.NextTurn(ctx, uit.Input{Text: "queued for later"}))
	record("submit-tool", created.Submit(ctx, uit.Input{Text: "use a tool"}))

	// Steer mid-run at the gate; the steer drains at the tool boundary.
	entered, release := target.gate.arm()
	submitDone := make(chan error, 1)
	go func() { submitDone <- created.Submit(ctx, uit.Input{Text: "use a tool again"}) }()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatalf("%s: model never entered for steer", target.name)
	}
	record("steer-mid-run", created.Steer(ctx, uit.Input{Text: "steer note"}))
	release()
	select {
	case err := <-submitDone:
		record("submit-steered", err)
	case <-ctx.Done():
		t.Fatalf("%s: steered submit never returned", target.name)
	}

	// Suspension via the ask-gated danger tool.
	record("submit-pause", created.Submit(ctx, uit.Input{Text: "please pause"}))
	suspended, err := created.Snapshot(ctx)
	if err != nil {
		t.Fatalf("%s: suspended snapshot: %v", target.name, err)
	}
	if suspended.Suspension == nil || len(suspended.Suspension.Approvals) != 1 {
		t.Fatalf("%s: suspended snapshot = %#v", target.name, suspended.Suspension)
	}
	suspensionID := suspended.Suspension.ID
	callID := suspended.Suspension.Approvals[0].CallID

	// Queue error row while suspended.
	record("steer-suspended", created.Steer(ctx, uit.Input{Text: "still there?"}))

	// Client-side resolve validation rows (exact legacy error strings).
	record("resolve-mismatch", created.Resolve(ctx, uit.Resolution{
		SuspensionID: "wrong", Decisions: []uit.Decision{{CallID: callID, Action: uit.DecisionApprove}},
	}))
	record("resolve-unknown", created.Resolve(ctx, uit.Resolution{
		SuspensionID: suspensionID, Decisions: []uit.Decision{{CallID: "bogus", Action: uit.DecisionApprove}},
	}))
	record("resolve-incomplete", created.Resolve(ctx, uit.Resolution{SuspensionID: suspensionID}))

	// Approve and block until the resumed run completes.
	record("resolve-approve", created.Resolve(ctx, uit.Resolution{
		SuspensionID: suspensionID,
		Decisions:    []uit.Decision{{CallID: callID, Action: uit.DecisionApprove, Reason: "go"}},
	}))

	// Second suspension resolved by deny plus a continuation prompt.
	record("submit-pause-two", created.Submit(ctx, uit.Input{Text: "please pause once more"}))
	suspendedTwo, err := created.Snapshot(ctx)
	if err != nil || suspendedTwo.Suspension == nil || len(suspendedTwo.Suspension.Approvals) != 1 {
		t.Fatalf("%s: second suspension snapshot = %#v, %v", target.name, suspendedTwo.Suspension, err)
	}
	record("resolve-deny", created.Resolve(ctx, uit.Resolution{
		SuspensionID: suspendedTwo.Suspension.ID,
		Decisions:    []uit.Decision{{CallID: suspendedTwo.Suspension.Approvals[0].CallID, Action: uit.DecisionDeny, Reason: "no"}},
		Prompt:       &uit.Input{Text: "carry on without it"},
	}))

	// Interrupt a run held at the gate.
	enteredInterrupt, releaseInterrupt := target.gate.arm()
	defer releaseInterrupt()
	interruptDone := make(chan error, 1)
	go func() { interruptDone <- created.Submit(ctx, uit.Input{Text: "wait here"}) }()
	select {
	case <-enteredInterrupt:
	case <-ctx.Done():
		t.Fatalf("%s: model never entered for interrupt", target.name)
	}
	record("interrupt-running", created.Interrupt(ctx))
	select {
	case err := <-interruptDone:
		record("submit-interrupted", err)
	case <-ctx.Done():
		t.Fatalf("%s: interrupted submit never returned", target.name)
	}

	// Tail marker: a single-record commit both paths emit, so the live drain
	// below has a deterministic final cursor.
	record("next-turn-tail", created.NextTurn(ctx, uit.Input{Text: "tail marker"}))
	result.finalCursor = result.steps[len(result.steps)-1].snapshot.Cursor

	// Drain the live subscription through the tail marker.
	result.live = drainEvents(t, target.name+" live", live, result.finalCursor)

	// Subscriber lag: a replaying subscriber with a one-event buffer must be
	// disconnected with exactly one terminal error, after which both
	// channels close. Events already in flight before the disconnect are NOT
	// compared across paths: the two pipelines hold different in-flight
	// depths, and the app discards speculative state and resynchronizes from
	// a snapshot on any subscription error regardless.
	lagged := created.Subscribe(uit.SubscribeOptions{Buffer: 1})
	deadline := time.After(differentialWait)
	events, errs := lagged.Events(), lagged.Errors()
	for events != nil || errs != nil {
		select {
		case _, ok := <-events:
			if !ok {
				events = nil
			}
		case lagErr, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			var disconnect *event.ErrSubscriberLagged
			if !errors.As(lagErr, &disconnect) {
				t.Fatalf("%s: lag error = %v", target.name, lagErr)
			}
			result.lagErrors = append(result.lagErrors, "lagged")
		case <-deadline:
			t.Fatalf("%s: lagged subscriber never terminated", target.name)
		}
	}
	lagged.Close()

	// Lag recovery: snapshot then replay after zero, exactly as the app
	// bridge resubscribes; the replayed durable history must match.
	replay := created.Subscribe(uit.SubscribeOptions{Buffer: 4096})
	result.replay = drainEvents(t, target.name+" replay", replay, result.finalCursor)
	replay.Close()
	return result
}

func drainEvents(t *testing.T, name string, subscription uit.Subscription, until uint64) []uit.Event {
	t.Helper()
	var collected []uit.Event
	deadline := time.After(differentialWait)
	for {
		select {
		case value, ok := <-subscription.Events():
			if !ok {
				t.Fatalf("%s: events closed before cursor %d", name, until)
			}
			collected = append(collected, value)
			if value.Durable && value.Cursor >= until {
				return collected
			}
		case err := <-subscription.Errors():
			t.Fatalf("%s: subscription error before cursor %d: %v", name, until, err)
		case <-deadline:
			t.Fatalf("%s: no event before cursor %d; got %d events", name, until, len(collected))
		}
	}
}

// tuple is the normalized event shape compared across paths.
type tuple struct {
	Kind    string
	Cursor  uint64
	Text    string
	Role    string
	State   string
	Tool    string
	QueueID string
	Failure string
}

// legacyOnlyKinds have no protocol source by design (see the exclusion list
// at the top of this file).
var legacyOnlyKinds = map[string]bool{
	"session.created":     true,
	"turn.started":        true,
	"turn.ended":          true,
	"tool.planned":        true,
	"tool.started":        true,
	"output.validated":    true,
	"resolution.accepted": true,
	// System/driver instructions are privacy-excluded from the protocol
	// projection (plan 8.5); the app ignores this passthrough kind.
	"message.system": true,
}

// bridgeOnlyKinds are protocol passthrough kinds the legacy path never
// emits; the app ignores them.
var bridgeOnlyKinds = map[string]bool{
	"command.accepted":              true,
	"agentic.transcript.compaction": true,
}

// synthesizedCursorKinds are settled-outcome kinds whose cursor differs by
// construction (bridge synthesizes them at the settlement record).
var synthesizedCursorKinds = map[string]bool{
	"run.completed":   true,
	"run.interrupted": true,
	"run.failed":      true,
}

func filterEvents(events []uit.Event, legacy bool) []tuple {
	var result []tuple
	for _, value := range events {
		kind := string(value.Kind)
		if legacy {
			if legacyOnlyKinds[kind] {
				continue
			}
			// Agentic bookkeeping duplicates: run identity and settlement are
			// authoritative in run.opened/run.closed (State-carrying events).
			if (kind == "run.started" || kind == "run.ended") && value.State == "" {
				continue
			}
			// interrupt.marker projection; settlement is authoritative in
			// run.closed.
			if kind == "run.interrupted" && value.State == uit.StateInterrupting {
				continue
			}
		} else if bridgeOnlyKinds[kind] {
			continue
		}
		result = append(result, eventTuples(value)...)
	}
	return result
}

func eventTuples(value uit.Event) []tuple {
	base := tuple{
		Kind: string(value.Kind), Cursor: value.Cursor,
		State: string(value.State), Failure: value.Failure, Text: value.TextDelta,
	}
	if synthesizedCursorKinds[base.Kind] {
		base.Cursor = 0
	}
	if value.Queue != nil {
		base.QueueID = value.Queue.ID
		if base.Kind == "queue.accepted" {
			base.QueueID += "/" + string(value.Queue.Kind)
			base.Text = value.Queue.Text
		}
	}
	if value.Tool != nil {
		base.Tool = fmt.Sprintf("%s/%s/%s", value.Tool.CallID, value.Tool.Name, value.Tool.State)
	}
	if value.Entry != nil {
		base.Role, base.Text = string(value.Entry.Role), value.Entry.Text
		for _, tool := range value.Entry.Tools {
			base.Tool += fmt.Sprintf("%s/%s/%s;", tool.CallID, tool.Name, tool.State)
		}
	}
	if len(value.Entries) == 0 {
		return []tuple{base}
	}
	// Flatten injected batches: the legacy path emits one event carrying all
	// injected entries, the bridge one event per entry.
	result := make([]tuple, 0, len(value.Entries))
	for _, entry := range value.Entries {
		flattened := base
		flattened.Role, flattened.Text = string(entry.Role), entry.Text
		result = append(result, flattened)
	}
	return result
}

// normalizeSnapshot applies the documented lossy-field exclusions before the
// deep comparison.
func normalizeSnapshot(value uit.Snapshot) uit.Snapshot {
	result := value
	result.Transcript = make([]uit.Entry, len(value.Transcript))
	for index, entry := range value.Transcript {
		normalized := entry
		normalized.Tools = make([]uit.Tool, len(entry.Tools))
		for toolIndex, tool := range entry.Tools {
			tool.Summary = "" // documented lossy field: protocol has no summary
			normalized.Tools[toolIndex] = tool
		}
		if len(entry.Tools) == 0 {
			normalized.Tools = nil
		}
		result.Transcript[index] = normalized
	}
	if value.Suspension != nil {
		suspension := *value.Suspension
		suspension.Description = "" // legacy leaves supported descriptions empty
		if suspension.ID != "" {
			// Suspension IDs are minted by the agentic core's own randomness,
			// not the injected deterministic IDGenerator, so two independent
			// runtimes can never agree on them; each script already proves
			// within-host consistency by resolving against the observed ID.
			suspension.ID = "SUSPENSION_ID"
		}
		suspension.Approvals = make([]uit.Approval, len(value.Suspension.Approvals))
		for index, approval := range value.Suspension.Approvals {
			approval.ResourceScheme = ""    // protocol transports display only
			approval.CanonicalResource = "" // protocol transports display only
			suspension.Approvals[index] = approval
		}
		if len(value.Suspension.Approvals) == 0 {
			suspension.Approvals = nil
		}
		result.Suspension = &suspension
	}
	return result
}

func TestDifferentialLegacyAndSessionLoopBridgeMatch(t *testing.T) {
	t.Parallel()
	legacy := runScript(t, newLegacyHost(t))
	bridge := runScript(t, newBridgeHost(t))

	if len(legacy.steps) != len(bridge.steps) {
		t.Fatalf("step counts diverge: legacy %d, bridge %d", len(legacy.steps), len(bridge.steps))
	}
	for index, expected := range legacy.steps {
		actual := bridge.steps[index]
		if expected.name != actual.name {
			t.Fatalf("step %d name %q vs %q", index, expected.name, actual.name)
		}
		// submit-interrupted: the spec mandates the bridge return bare
		// context.Canceled for an interrupted run, while legacy surfaces the
		// driver's wrap text; the errors.Is identity must still match.
		compareText := expected.name != "submit-interrupted"
		if (compareText && expected.errText != actual.errText) || expected.errClass != actual.errClass {
			t.Errorf("step %s error diverged:\nlegacy %q (%s)\nbridge %q (%s)",
				expected.name, expected.errText, expected.errClass, actual.errText, actual.errClass)
		}
		left, right := normalizeSnapshot(expected.snapshot), normalizeSnapshot(actual.snapshot)
		if !reflect.DeepEqual(left, right) {
			t.Errorf("step %s snapshot diverged:\nlegacy %s\nbridge %s",
				expected.name, dumpSnapshot(left), dumpSnapshot(right))
		}
	}

	compareTuples(t, "live", filterEvents(legacy.live, true), filterEvents(bridge.live, false))
	compareTuples(t, "replay", filterEvents(legacy.replay, true), filterEvents(bridge.replay, false))

	if len(legacy.lagErrors) != 1 || len(bridge.lagErrors) != 1 {
		t.Errorf("lag errors diverged: legacy %v, bridge %v", legacy.lagErrors, bridge.lagErrors)
	}
}

// runPreviewScript drives one streaming host through a plain completion and
// a tool-calling run under a Preview-true subscription and returns every
// delivered event through the last durable cursor.
func runPreviewScript(t *testing.T, target diffHost) []uit.Event {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), differentialWait)
	defer cancel()
	created, err := target.host.NewSession(ctx, uit.SessionOptions{})
	if err != nil {
		t.Fatalf("%s: NewSession: %v", target.name, err)
	}
	defer func() { _ = created.Close(context.Background()) }()
	live := created.Subscribe(uit.SubscribeOptions{Buffer: 4096, Preview: true})
	defer live.Close()
	if err := created.Submit(ctx, uit.Input{Text: "hello"}); err != nil {
		t.Fatalf("%s: submit hello: %v", target.name, err)
	}
	if err := created.Submit(ctx, uit.Input{Text: "use a tool"}); err != nil {
		t.Fatalf("%s: submit tool: %v", target.name, err)
	}
	snapshot, err := created.Snapshot(ctx)
	if err != nil {
		t.Fatalf("%s: preview snapshot: %v", target.name, err)
	}
	return drainEvents(t, target.name+" preview", live, snapshot.Cursor)
}

// previewTuples normalizes the preview tier of one delivered sequence under
// the documented exclusions: Cursor/Ordinal are dropped, tool previews keep
// call identity and state only, and legacy tool.update previews (anonymous
// capability updates the bridge deliberately drops) are removed.
func previewTuples(events []uit.Event, legacy bool) []tuple {
	var result []tuple
	for _, value := range events {
		if value.Durable {
			continue
		}
		kind := string(value.Kind)
		if legacy && kind == "tool.update" {
			continue
		}
		base := tuple{Kind: kind, Text: value.TextDelta}
		if value.Thinking != nil {
			base.Text = value.Thinking.Text
		}
		if len(value.Tools) > 0 {
			// Legacy call-start previews batch their tools; the bridge emits
			// one identified preview per call. Flatten to per-call tuples.
			for _, tool := range value.Tools {
				flattened := base
				flattened.Tool = tool.CallID + "/" + string(tool.State)
				result = append(result, flattened)
			}
			continue
		}
		if value.Tool != nil {
			base.Tool = value.Tool.CallID + "/" + string(value.Tool.State)
		}
		result = append(result, base)
	}
	return result
}

// TestDifferentialPreviewParity proves the bridge's preview tier matches the
// legacy adapter's under a streaming scripted response: the same ordered
// text deltas and the same identified tool previews (see the preview
// exclusions documented at the top of this file).
func TestDifferentialPreviewParity(t *testing.T) {
	t.Parallel()
	legacy := runPreviewScript(t, newLegacyHostStreaming(t, true))
	bridge := runPreviewScript(t, newBridgeHostStreaming(t, true))
	expected, actual := previewTuples(legacy, true), previewTuples(bridge, false)
	if len(expected) == 0 {
		t.Fatal("streaming script produced no preview events")
	}
	compareTuples(t, "preview", expected, actual)
}

func compareTuples(t *testing.T, name string, expected, actual []tuple) {
	t.Helper()
	limit := len(expected)
	if len(actual) < limit {
		limit = len(actual)
	}
	for index := 0; index < limit; index++ {
		if expected[index] != actual[index] {
			t.Errorf("%s event %d diverged:\nlegacy %#v\nbridge %#v", name, index, expected[index], actual[index])
			return
		}
	}
	if len(expected) != len(actual) {
		t.Errorf("%s event counts diverge: legacy %d, bridge %d\nlegacy tail: %#v\nbridge tail: %#v",
			name, len(expected), len(actual), tail(expected, limit), tail(actual, limit))
	}
}

func dumpSnapshot(value uit.Snapshot) string {
	suspension := "nil"
	if value.Suspension != nil {
		suspension = fmt.Sprintf("%#v", *value.Suspension)
	}
	return fmt.Sprintf("%#v suspension=%s", value, suspension)
}

func tail(values []tuple, from int) []tuple {
	if from >= len(values) {
		return nil
	}
	return values[from:]
}
