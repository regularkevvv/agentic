package realtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// DelegateTool is the one function the voice model is given. Its single
// "request" argument becomes the Input of a sessionloop command.
const DelegateTool = "delegate"

// DefaultInstructions is the voice persona used when Options.Instructions is
// empty. It keeps the voice model conversational and sends everything of
// substance to the session.
const DefaultInstructions = "You are the voice of an assistant whose knowledge, memory, tools, and authority " +
	"live in a backend you reach only through the `delegate` tool. Handle greetings, small talk, and " +
	"clarifying questions yourself. For anything else — facts, the user's data, or any action — call " +
	"`delegate` with the complete request, then tell the user the result naturally and briefly. Never " +
	"invent an answer the backend did not give."

var delegateSchema = json.RawMessage(`{"type":"object","properties":{"request":{"type":"string",` +
	`"description":"The user's request, restated completely enough to act on without hearing the audio."}},` +
	`"required":["request"],"additionalProperties":false}`)

// Options configures one bridged call.
type Options struct {
	// Instructions is the voice persona. It shapes how the voice model talks
	// and when it delegates; it is not the session's system prompt.
	Instructions string
	Voice        string
	// Observe receives every call event in order, for transcripts and
	// presentation. It runs on the receive loop and must not block.
	Observe func(Event)
}

// Run bridges one live call to a session until the call ends, ctx is done,
// or the session stream fails. It configures the voice model, seeds it with
// the session's committed conversation, and answers every delegate call by
// dispatching to the session: Start when idle, Steer when a run is active
// and the session supports it.
//
// Runs belong to the session (law L4): ending the call never interrupts a
// delegated run, and its committed result is replayed into the next call.
func Run(ctx context.Context, conn Conn, session sessionloop.Session, options Options) error {
	if conn == nil || session == nil {
		return errors.New("realtime: Run requires a conn and a session")
	}
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{Buffer: 1024})
	if err != nil {
		return fmt.Errorf("realtime: subscribe: %w", err)
	}
	defer func() { _ = stream.Close() }()
	snapshot, err := session.Snapshot(ctx)
	if err != nil {
		return fmt.Errorf("realtime: snapshot: %w", err)
	}

	instructions := options.Instructions
	if instructions == "" {
		instructions = DefaultInstructions
	}
	config := Config{Instructions: instructions, Voice: options.Voice, Tools: []ToolSpec{{
		Name:        DelegateTool,
		Description: "Ask the backend to answer or act on the user's request.",
		Parameters:  delegateSchema,
	}}}
	if err := conn.Send(ctx, Action{Kind: ActionConfigure, Config: &config}); err != nil {
		return fmt.Errorf("realtime: configure: %w", err)
	}
	if history := historyTurns(snapshot.Entries); len(history) > 0 {
		if err := conn.Send(ctx, Action{Kind: ActionHistory, History: history}); err != nil {
			return fmt.Errorf("realtime: seed history: %w", err)
		}
	}

	b := &bridge{
		conn:    conn,
		session: session,
		steer:   snapshot.Capabilities.Supports(sessionloop.CapabilitySteer),
		fail:    cancel,
		runs:    make(map[sessionloop.RunID]*runState),
	}
	var handlers sync.WaitGroup
	defer handlers.Wait()
	defer cancel(nil)
	handlers.Go(func() { cancel(b.watch(ctx, stream)) })

	for {
		event, err := conn.Recv(ctx)
		if err != nil {
			if cause := context.Cause(ctx); cause != nil && !errors.Is(cause, context.Canceled) {
				return cause
			}
			return err
		}
		if options.Observe != nil {
			options.Observe(event)
		}
		switch event.Kind {
		case EventToolCall:
			if event.Call == nil {
				continue
			}
			call := *event.Call
			handlers.Go(func() { b.handle(ctx, call) })
		case EventResponseStarted:
			b.responseStarted()
		case EventResponseDone:
			b.responseDone(ctx)
		}
	}
}

type bridge struct {
	conn    Conn
	session sessionloop.Session
	steer   bool
	fail    context.CancelCauseFunc

	mu         sync.Mutex
	runs       map[sessionloop.RunID]*runState
	responding bool
	deferred   bool
}

// runState accumulates one run's authoritative outcome. A run is reported at
// its first suspension or at settlement, whichever comes first; outcome stays
// empty when it paused first.
type runState struct {
	answer   string
	outcome  sessionloop.RunOutcomeKind
	approval string
	reported bool
	ready    chan struct{}
}

func (b *bridge) run(id sessionloop.RunID) *runState {
	state, ok := b.runs[id]
	if !ok {
		state = &runState{ready: make(chan struct{})}
		b.runs[id] = state
	}
	return state
}

// watch folds authoritative session events into run states. A lagged or
// failed stream ends the call: without it, delegated answers can't be
// delivered.
func (b *bridge) watch(ctx context.Context, stream sessionloop.Stream) error {
	for {
		event, err := stream.Next(ctx)
		if err != nil {
			return fmt.Errorf("realtime: session stream: %w", err)
		}
		if event.RunID == "" {
			continue
		}
		b.mu.Lock()
		state := b.run(event.RunID)
		switch event.Kind {
		case sessionloop.EventEntryCommitted:
			if event.Entry != nil && event.Entry.Role == sessionloop.RoleAssistant {
				if text := entryText(*event.Entry); text != "" {
					state.answer = text
				}
			}
		case sessionloop.EventRunSuspended:
			if event.Suspension != nil {
				state.approval = event.Suspension.Description
			}
			state.report()
		case sessionloop.EventRunSettled:
			if event.Outcome != nil {
				state.outcome = event.Outcome.Kind
			}
			state.report()
		}
		b.mu.Unlock()
	}
}

func (s *runState) report() {
	if !s.reported {
		s.reported = true
		close(s.ready)
	}
}

func (b *bridge) handle(ctx context.Context, call ToolCall) {
	output := b.delegate(ctx, call)
	if ctx.Err() != nil {
		return
	}
	result := ToolResult{CallID: call.ID, Output: output}
	if err := b.conn.Send(ctx, Action{Kind: ActionToolResult, Result: &result}); err != nil {
		b.fail(fmt.Errorf("realtime: send tool result: %w", err))
		return
	}
	b.respond(ctx)
}

func (b *bridge) delegate(ctx context.Context, call ToolCall) string {
	if call.Name != DelegateTool {
		return fmt.Sprintf("Unknown tool %q. Only %q is available.", call.Name, DelegateTool)
	}
	var args struct {
		Request string `json:"request"`
	}
	if err := json.Unmarshal(call.Arguments, &args); err != nil || strings.TrimSpace(args.Request) == "" {
		return "The request argument is required."
	}
	input := &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: args.Request}}}

	receipt, err := b.session.Dispatch(ctx, sessionloop.Command{Kind: sessionloop.CommandStart, Input: input})
	if errors.Is(err, sessionloop.ErrSessionBusy) && b.steer {
		return b.steerActive(ctx, input)
	}
	if err != nil {
		return "The backend did not accept the request: " + err.Error()
	}

	b.mu.Lock()
	state := b.run(receipt.RunID)
	b.mu.Unlock()
	select {
	case <-state.ready:
	case <-ctx.Done():
		return ""
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.runs, receipt.RunID)
	return state.result()
}

func (b *bridge) steerActive(ctx context.Context, input *sessionloop.Input) string {
	snapshot, err := b.session.Snapshot(ctx)
	if err != nil || snapshot.ActiveRunID == "" {
		return "The backend is busy; ask again in a moment."
	}
	_, err = b.session.Dispatch(ctx, sessionloop.Command{
		Kind: sessionloop.CommandSteer, RunID: snapshot.ActiveRunID, Input: input,
	})
	if err != nil {
		return "The backend is busy; ask again in a moment."
	}
	return "Added to the task already in progress. Its answer will follow; don't repeat this request."
}

func (s *runState) result() string {
	switch s.outcome {
	case "":
		return "The task is paused awaiting approval: " + s.approval
	case sessionloop.RunCompleted:
		if s.answer == "" {
			return "Done."
		}
		return s.answer
	case sessionloop.RunInterrupted:
		return "The task was interrupted before it finished."
	default:
		return "The task failed."
	}
}

// respond asks the voice model to speak now, or after its current response
// when one is in progress. Providers reject a second concurrent response.
func (b *bridge) respond(ctx context.Context) {
	b.mu.Lock()
	if b.responding {
		b.deferred = true
		b.mu.Unlock()
		return
	}
	b.responding = true
	b.mu.Unlock()
	if err := b.conn.Send(ctx, Action{Kind: ActionRespond}); err != nil {
		b.fail(fmt.Errorf("realtime: request response: %w", err))
	}
}

func (b *bridge) responseStarted() {
	b.mu.Lock()
	b.responding = true
	b.mu.Unlock()
}

func (b *bridge) responseDone(ctx context.Context) {
	b.mu.Lock()
	b.responding = false
	deferred := b.deferred
	b.deferred = false
	b.mu.Unlock()
	if deferred {
		b.respond(ctx)
	}
}

func historyTurns(entries []sessionloop.Entry) []Turn {
	var turns []Turn
	for _, entry := range entries {
		var role Role
		switch entry.Role {
		case sessionloop.RoleUser:
			role = RoleUser
		case sessionloop.RoleAssistant:
			role = RoleAssistant
		default:
			continue
		}
		if text := entryText(entry); text != "" {
			turns = append(turns, Turn{Role: role, Text: text})
		}
	}
	return turns
}

func entryText(entry sessionloop.Entry) string {
	var parts []string
	for _, block := range entry.Blocks {
		if block.Kind == sessionloop.EntryBlockText && strings.TrimSpace(block.Text) != "" {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}
