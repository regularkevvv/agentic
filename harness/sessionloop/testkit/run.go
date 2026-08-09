package testkit

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// RunFunc drives one run. Returning nil settles the run as completed;
// returning an error settles it as failed with the error's message. When an
// interrupt was requested before the function returns, the run settles as
// interrupted regardless of the returned value (law L10).
type RunFunc func(run *RunContext) error

// EchoRunFunc is the default run behavior: emit one assistant entry echoing
// the start input text, then complete.
func EchoRunFunc() RunFunc {
	return func(run *RunContext) error {
		run.EmitAssistant("echo: " + inputText(run.Input()))
		return nil
	}
}

// ScenarioRunFunc behaves like EchoRunFunc and additionally implements the
// scripted scenarios documented by the conformance package's Factory
// contract, keyed on the start input's "conformance.scenario" metadata:
//
//   - "preview": emit two preview deltas before the assistant entry;
//   - "tools": emit a tool_call and matching tool_result entry with JSON data;
//   - "suspend": suspend once and continue after an approving resolution;
//   - "output": set a structured JSON output on the completed outcome.
func ScenarioRunFunc() RunFunc {
	return func(run *RunContext) error {
		input := run.Input()
		switch input.Meta["conformance.scenario"] {
		case "preview":
			run.EmitPreview("thinking about " + inputText(input))
			run.EmitPreview("almost done")
		case "tools":
			run.EmitToolCall("call-1", "lookup", `{"query":"conformance"}`)
			run.EmitToolResult("call-1", "lookup", "lookup complete", false)
		case "suspend":
			suspension := sessionloop.Suspension{
				ID:          "susp-1",
				Kind:        "approval",
				Description: "conformance approval gate",
				Decisions: []sessionloop.SuspensionDecision{{
					ID:         "decision-1",
					Name:       "lookup",
					Capability: "tools",
					Action:     "invoke",
					Resource:   "conformance",
				}},
			}
			if _, err := run.Suspend(suspension); err != nil {
				return err
			}
		case "output":
			run.SetOutput(`{"answer":"structured"}`)
		}
		run.EmitAssistant("echo: " + inputText(input))
		return nil
	}
}

func inputText(input sessionloop.Input) string {
	for _, block := range input.Blocks {
		if block.Kind == sessionloop.InputBlockText {
			return block.Text
		}
	}
	return ""
}

// activeRun is the engine-side state of the single active run.
type activeRun struct {
	id        sessionloop.RunID
	commandID sessionloop.CommandID
	input     sessionloop.Input

	ctx         context.Context
	cancel      context.CancelFunc
	interrupted chan struct{}
	done        chan struct{}
	hold        chan struct{}
	inputs      *inputQueue
	resolutions chan sessionloop.Resolution

	// Guarded by the owning sessionState.mu.
	interruptRequested bool
	suspension         *sessionloop.Suspension
	output             json.RawMessage
	usage              sessionloop.Usage
	previewOrdinal     uint64
}

func newActiveRun(id sessionloop.RunID, commandID sessionloop.CommandID, input sessionloop.Input, hold chan struct{}) *activeRun {
	ctx, cancel := context.WithCancel(context.Background())
	run := &activeRun{
		id:          id,
		commandID:   commandID,
		input:       input.Clone(),
		ctx:         ctx,
		cancel:      cancel,
		interrupted: make(chan struct{}),
		done:        make(chan struct{}),
		hold:        hold,
		resolutions: make(chan sessionloop.Resolution, 1),
	}
	run.inputs = newInputQueue(run.done)
	return run
}

// drive executes the run to its singular settlement. The gate hold, when
// present, blocks before the first step until released or interrupted.
func (st *sessionState) drive(run *activeRun) {
	var runErr error
	if run.hold != nil {
		select {
		case <-run.hold:
		case <-run.ctx.Done():
		}
	}
	if run.ctx.Err() == nil {
		runErr = st.host.runFunc(&RunContext{state: st, run: run})
	}
	st.settle(run, runErr)
	close(run.done)
}

// settle emits the singular settlement strictly after every entry of the
// run (law L10) and returns the session to idle.
func (st *sessionState) settle(run *activeRun, runErr error) {
	st.mu.Lock()
	defer st.mu.Unlock()
	outcome := sessionloop.RunOutcome{RunID: run.id}
	switch {
	case run.interruptRequested:
		outcome.Kind = sessionloop.RunInterrupted
	case runErr != nil:
		outcome.Kind = sessionloop.RunFailed
		outcome.Failure = runErr.Error()
	default:
		outcome.Kind = sessionloop.RunCompleted
		outcome.Output = run.output
	}
	if run.usage != (sessionloop.Usage{}) {
		usage := st.usage
		st.appendLocked(sessionloop.Event{Kind: sessionloop.EventUsage, RunID: run.id, Usage: &usage})
	}
	run.suspension = nil
	st.run = nil
	if st.closing {
		st.state = sessionloop.StateClosing
	} else {
		st.state = sessionloop.StateIdle
	}
	st.appendLocked(sessionloop.Event{Kind: sessionloop.EventRunSettled, RunID: run.id, CommandID: run.commandID, Outcome: &outcome})
	st.appendLocked(sessionloop.Event{Kind: sessionloop.EventSessionState, RunID: run.id})
	run.cancel()
}

// RunContext is the engine surface handed to a RunFunc. Every emission is
// committed durably before it returns; every boundary is copy-owned.
type RunContext struct {
	state *sessionState
	run   *activeRun
}

// Input returns a copy of the start input that began this run.
func (rc *RunContext) Input() sessionloop.Input { return rc.run.input.Clone() }

// Context is canceled when the run is interrupted or the session closes.
func (rc *RunContext) Context() context.Context { return rc.run.ctx }

// Interrupted is closed when an interrupt has been requested for this run.
func (rc *RunContext) Interrupted() <-chan struct{} { return rc.run.interrupted }

// Steered delivers accepted steer and follow-up inputs in dispatch order.
// The engine commits each input as an entry with its correct origin before
// delivering it here.
func (rc *RunContext) Steered() <-chan sessionloop.Input { return rc.run.inputs.out }

// EmitAssistant commits one assistant text entry.
func (rc *RunContext) EmitAssistant(text string) {
	delta := sessionloop.Usage{
		CompletionTokens: int64(len(text)),
		TotalTokens:      int64(len(text)),
		Requests:         1,
	}
	rc.commit(sessionloop.RoleAssistant, sessionloop.OriginAssistant, delta, sessionloop.EntryBlock{
		Kind: sessionloop.EntryBlockText,
		Text: text,
	})
}

// EmitToolCall commits one assistant tool_call entry. dataJSON must be empty
// or exactly one complete JSON value.
func (rc *RunContext) EmitToolCall(callID, name, dataJSON string) {
	rc.commit(sessionloop.RoleAssistant, sessionloop.OriginAssistant, sessionloop.Usage{ToolCalls: 1}, sessionloop.EntryBlock{
		Kind:     sessionloop.EntryBlockToolCall,
		ToolCall: &sessionloop.EntryToolCall{CallID: callID, Name: name, Data: mustRaw("EmitToolCall", dataJSON)},
	})
}

// EmitToolResult commits one tool_result entry with the textual result
// content in the block text.
func (rc *RunContext) EmitToolResult(callID, name, text string, isError bool) {
	rc.commit(sessionloop.RoleTool, sessionloop.OriginTool, sessionloop.Usage{}, sessionloop.EntryBlock{
		Kind:       sessionloop.EntryBlockToolResult,
		Text:       text,
		ToolResult: &sessionloop.EntryToolResult{CallID: callID, Name: name, IsError: isError},
	})
}

func (rc *RunContext) commit(role sessionloop.Role, origin sessionloop.EntryOrigin, delta sessionloop.Usage, blocks ...sessionloop.EntryBlock) {
	st := rc.state
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.run != rc.run {
		return
	}
	addUsage(&st.usage, delta)
	addUsage(&rc.run.usage, delta)
	st.commitEntryLocked(role, origin, rc.run.id, rc.run.commandID, blocks)
}

// EmitPreview publishes a lossy text preview delta. Previews carry the
// latest durable position, are never appended to the durable log, and may
// be dropped per subscriber (law L5).
func (rc *RunContext) EmitPreview(text string) {
	st := rc.state
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.run != rc.run {
		return
	}
	rc.run.previewOrdinal++
	event := sessionloop.Event{
		Position:  st.currentPositionLocked(),
		Ordinal:   rc.run.previewOrdinal,
		Nature:    sessionloop.EventPreview,
		Kind:      sessionloop.EventPreviewDelta,
		SessionID: st.id,
		RunID:     rc.run.id,
		State:     st.state,
		Preview:   &sessionloop.Preview{Kind: sessionloop.PreviewText, Text: text},
	}
	for subscriber := range st.subs {
		subscriber.deliver(event.Clone(), false)
	}
}

// SetOutput attaches a structured JSON output to the completed outcome.
// dataJSON must be exactly one complete JSON value.
func (rc *RunContext) SetOutput(dataJSON string) {
	st := rc.state
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.run != rc.run {
		return
	}
	rc.run.output = mustRaw("SetOutput", dataJSON)
}

// Suspend durably pauses the run and blocks until a matching resolution is
// dispatched, the run is interrupted, or the session closes. Suspension is
// a pause of the same run identity, not settlement.
func (rc *RunContext) Suspend(suspension sessionloop.Suspension) (sessionloop.Resolution, error) {
	st := rc.state
	st.mu.Lock()
	if st.run != rc.run || rc.run.interruptRequested {
		st.mu.Unlock()
		return sessionloop.Resolution{}, fmt.Errorf("testkit: run %q interrupted before suspension: %w", rc.run.id, context.Canceled)
	}
	stored := suspension.Clone()
	if stored.ID == "" {
		stored.ID = st.host.nextID("susp")
	}
	rc.run.suspension = &stored
	st.state = sessionloop.StateSuspended
	published := stored.Clone()
	st.appendLocked(sessionloop.Event{Kind: sessionloop.EventRunSuspended, RunID: rc.run.id, CommandID: rc.run.commandID, Suspension: &published})
	st.appendLocked(sessionloop.Event{Kind: sessionloop.EventSessionState, RunID: rc.run.id})
	st.mu.Unlock()

	select {
	case resolution := <-rc.run.resolutions:
		return resolution, nil
	case <-rc.run.interrupted:
		return sessionloop.Resolution{}, fmt.Errorf("testkit: run %q interrupted while suspended: %w", rc.run.id, context.Canceled)
	}
}

func addUsage(total *sessionloop.Usage, delta sessionloop.Usage) {
	total.PromptTokens += delta.PromptTokens
	total.CompletionTokens += delta.CompletionTokens
	total.TotalTokens += delta.TotalTokens
	total.CacheReadTokens += delta.CacheReadTokens
	total.CacheCreationTokens += delta.CacheCreationTokens
	total.ReasoningTokens += delta.ReasoningTokens
	total.Requests += delta.Requests
	total.ToolCalls += delta.ToolCalls
}

func mustRaw(operation, dataJSON string) json.RawMessage {
	if dataJSON == "" {
		return nil
	}
	if !json.Valid([]byte(dataJSON)) {
		panic(fmt.Sprintf("testkit: %s requires exactly one complete JSON value, got %q", operation, dataJSON))
	}
	return json.RawMessage(dataJSON)
}

// inputQueue delivers steer and follow-up inputs to the RunFunc without ever
// blocking Dispatch, preserving dispatch order.
type inputQueue struct {
	mu     sync.Mutex
	items  []sessionloop.Input
	signal chan struct{}
	out    chan sessionloop.Input
	done   <-chan struct{}
}

func newInputQueue(done <-chan struct{}) *inputQueue {
	queue := &inputQueue{
		signal: make(chan struct{}, 1),
		out:    make(chan sessionloop.Input),
		done:   done,
	}
	go queue.pump()
	return queue
}

func (q *inputQueue) push(input sessionloop.Input) {
	q.mu.Lock()
	q.items = append(q.items, input)
	q.mu.Unlock()
	select {
	case q.signal <- struct{}{}:
	default:
	}
}

func (q *inputQueue) pump() {
	for {
		q.mu.Lock()
		var item sessionloop.Input
		have := false
		if len(q.items) > 0 {
			item = q.items[0]
			q.items = q.items[1:]
			have = true
		}
		q.mu.Unlock()
		if have {
			select {
			case q.out <- item:
				continue
			case <-q.done:
				return
			}
		}
		select {
		case <-q.signal:
		case <-q.done:
			return
		}
	}
}
