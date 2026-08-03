// Package testhost provides a deterministic, credential-free implementation
// of the TUI port for reducer, renderer, and terminal probes.
package testhost

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	uit "github.com/regularkevvv/agentic/tui"
)

var ErrLagged = errors.New("testhost subscriber lagged")

type Script func(context.Context, *Session, uit.Input) error

type Host struct {
	mu       sync.Mutex
	next     int
	sessions map[string]*Session
	script   Script
}

func New(script Script) *Host {
	if script == nil {
		script = DefaultScript
	}
	return &Host{sessions: make(map[string]*Session), script: script}
}

func (h *Host) NewSession(_ context.Context, _ uit.SessionOptions) (uit.Session, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.next++
	id := fmt.Sprintf("offline-%03d", h.next)
	session := &Session{
		id: id, host: h, snapshot: uit.Snapshot{
			SessionID: id, State: uit.StateIdle, ProfileLabel: "offline", Workspace: "/offline", Execution: "offline test host",
		},
		subscribers: make(map[*subscription]struct{}),
	}
	h.sessions[id] = session
	return session, nil
}

func (h *Host) ResumeSession(_ context.Context, id string) (uit.Session, error) {
	h.mu.Lock()
	session := h.sessions[id]
	h.mu.Unlock()
	if session == nil {
		return nil, fmt.Errorf("session %q not found", id)
	}
	session.mu.Lock()
	session.closed = false
	if session.snapshot.State == uit.StateClosed {
		session.snapshot.State = uit.StateIdle
	}
	session.mu.Unlock()
	return session, nil
}

func (h *Host) ListSessions(context.Context) ([]uit.SessionInfo, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	result := make([]uit.SessionInfo, 0, len(h.sessions))
	for id, session := range h.sessions {
		session.mu.Lock()
		state := session.snapshot.State
		session.mu.Unlock()
		result = append(result, uit.SessionInfo{ID: id, State: state})
	}
	return result, nil
}

func (h *Host) ExportTranscript(_ context.Context, id string) (string, error) {
	h.mu.Lock()
	session := h.sessions[id]
	h.mu.Unlock()
	if session == nil {
		return "", fmt.Errorf("session %q not found", id)
	}
	snapshot, _ := session.Snapshot(context.Background())
	var output strings.Builder
	for _, entry := range snapshot.Transcript {
		fmt.Fprintf(&output, "%s\n%s\n\n", strings.ToUpper(string(entry.Role)), entry.Text)
	}
	return strings.TrimSpace(output.String()) + "\n", nil
}

type Session struct {
	mu          sync.Mutex
	id          string
	host        *Host
	snapshot    uit.Snapshot
	history     []uit.Event
	subscribers map[*subscription]struct{}
	closed      bool
}

func (s *Session) ID() string { return s.id }

func (s *Session) Snapshot(context.Context) (uit.Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneSnapshot(s.snapshot), nil
}

func (s *Session) Subscribe(options uit.SubscribeOptions) uit.Subscription {
	if options.Buffer <= 0 {
		options.Buffer = 64
	}
	sub := &subscription{
		events: make(chan uit.Event, options.Buffer), errors: make(chan error, 1),
		preview: options.Preview, owner: s,
	}
	s.mu.Lock()
	for _, event := range s.history {
		if event.Cursor <= options.AfterCursor {
			continue
		}
		select {
		case sub.events <- cloneEvent(event):
		default:
			sub.errors <- ErrLagged
			sub.closed = true
			close(sub.events)
			close(sub.errors)
			s.mu.Unlock()
			return sub
		}
	}
	s.subscribers[sub] = struct{}{}
	s.mu.Unlock()
	return sub
}

func (s *Session) Submit(ctx context.Context, input uit.Input) error {
	if err := input.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return errors.New("session lease is closed")
	}
	if s.snapshot.State != uit.StateIdle {
		s.mu.Unlock()
		return fmt.Errorf("session is %s", s.snapshot.State)
	}
	entry := uit.Entry{Role: uit.RoleUser, Text: input.Text}
	s.snapshot.Transcript = append(s.snapshot.Transcript, entry)
	s.snapshot.State = uit.StateRunning
	s.emitLocked(uit.Event{Kind: uit.EventMessagesInjected, Entries: []uit.Entry{entry}, Durable: true})
	s.emitLocked(uit.Event{Kind: uit.EventRunStarted, State: uit.StateRunning, Durable: true})
	s.mu.Unlock()
	return s.host.script(ctx, s, input)
}

func (s *Session) Steer(ctx context.Context, input uit.Input) error {
	return s.queue(ctx, uit.QueueSteer, input)
}

func (s *Session) FollowUp(ctx context.Context, input uit.Input) error {
	return s.queue(ctx, uit.QueueFollowUp, input)
}

func (s *Session) NextTurn(ctx context.Context, input uit.Input) error {
	return s.queue(ctx, uit.QueueNextTurn, input)
}

func (s *Session) queue(ctx context.Context, kind uit.QueueKind, input uit.Input) error {
	if err := input.Validate(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	queued := uit.QueuedInput{ID: fmt.Sprintf("queue-%d", len(s.snapshot.Pending)+1), Kind: kind, Text: input.Text}
	s.snapshot.Pending = append(s.snapshot.Pending, queued)
	s.emitLocked(uit.Event{Kind: uit.EventQueueAccepted, Queue: &queued, Durable: true})
	return nil
}

func (s *Session) Resolve(ctx context.Context, resolution uit.Resolution) error {
	if err := resolution.Validate(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	suspension := s.snapshot.Suspension
	if suspension == nil || suspension.ID != resolution.SuspensionID {
		return errors.New("resolution does not match current suspension")
	}
	required := make(map[string]bool, len(suspension.Approvals))
	for _, approval := range suspension.Approvals {
		required[approval.CallID] = true
	}
	for _, decision := range resolution.Decisions {
		if !required[decision.CallID] {
			return fmt.Errorf("unknown approval %q", decision.CallID)
		}
		delete(required, decision.CallID)
	}
	if len(required) != 0 {
		return errors.New("resolution is incomplete")
	}
	entry := uit.Entry{Role: uit.RoleAssistant, Text: "Permission decision recorded."}
	s.snapshot.Transcript = append(s.snapshot.Transcript, entry)
	s.snapshot.Suspension = nil
	s.snapshot.State = uit.StateIdle
	s.emitLocked(uit.Event{Kind: uit.EventAssistantCommitted, Entry: &entry, Durable: true})
	s.emitLocked(uit.Event{Kind: uit.EventRunEnded, State: uit.StateIdle, Durable: true})
	return nil
}

func (s *Session) Interrupt(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.snapshot.State != uit.StateRunning {
		return nil
	}
	s.snapshot.State = uit.StateIdle
	s.emitLocked(uit.Event{Kind: uit.EventRunInterrupted, State: uit.StateIdle, Durable: true})
	return nil
}

func (s *Session) Close(context.Context) error {
	s.mu.Lock()
	s.closed = true
	s.snapshot.State = uit.StateClosed
	for subscriber := range s.subscribers {
		subscriber.closeLocked()
		delete(s.subscribers, subscriber)
	}
	s.mu.Unlock()
	return nil
}

func (s *Session) PreviewText(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.emitLocked(uit.Event{Kind: uit.EventTextDelta, TextDelta: text})
}

func (s *Session) PreviewThinking(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	thinking := uit.Thinking{Text: text}
	s.emitLocked(uit.Event{Kind: uit.EventThinkingDelta, Thinking: &thinking})
}

func (s *Session) Tool(tool uit.Tool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	kind := uit.EventToolStarted
	if tool.State == uit.ToolDone || tool.State == uit.ToolError {
		kind = uit.EventToolResult
	}
	s.emitLocked(uit.Event{Kind: kind, Tool: &tool, Durable: tool.State != uit.ToolPreview})
}

func (s *Session) Complete(text string, usage uit.Usage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entry := uit.Entry{Role: uit.RoleAssistant, Text: text}
	s.snapshot.Transcript = append(s.snapshot.Transcript, entry)
	s.snapshot.Usage = usage
	s.snapshot.State = uit.StateIdle
	s.emitLocked(uit.Event{Kind: uit.EventAssistantCommitted, Entry: &entry, Durable: true})
	s.emitLocked(uit.Event{Kind: uit.EventRunCompleted, Usage: &usage, State: uit.StateIdle, Durable: true})
}

func (s *Session) Suspend(value uit.Suspension) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.Suspension = cloneSuspension(&value)
	s.snapshot.State = uit.StateSuspended
	s.emitLocked(uit.Event{Kind: uit.EventRunSuspended, Suspension: cloneSuspension(&value), State: uit.StateSuspended, Durable: true})
}

func (s *Session) emitLocked(event uit.Event) {
	event.SessionID = s.id
	if event.Durable {
		s.snapshot.Cursor++
		event.Cursor = s.snapshot.Cursor
		s.history = append(s.history, cloneEvent(event))
	}
	for subscriber := range s.subscribers {
		if !event.Durable && !subscriber.preview {
			continue
		}
		copyOfEvent := cloneEvent(event)
		copyOfEvent.Dropped += subscriber.dropped
		select {
		case subscriber.events <- copyOfEvent:
			subscriber.dropped = 0
		default:
			if !event.Durable {
				subscriber.dropped++
				continue
			}
			select {
			case subscriber.errors <- ErrLagged:
			default:
			}
			subscriber.closeLocked()
			delete(s.subscribers, subscriber)
		}
	}
}

type subscription struct {
	events  chan uit.Event
	errors  chan error
	preview bool
	owner   *Session
	dropped uint64
	closed  bool
}

func (s *subscription) Events() <-chan uit.Event { return s.events }
func (s *subscription) Errors() <-chan error     { return s.errors }

func (s *subscription) Close() {
	if s == nil {
		return
	}
	s.owner.mu.Lock()
	if !s.closed {
		s.closeLocked()
		delete(s.owner.subscribers, s)
	}
	s.owner.mu.Unlock()
}

func (s *subscription) closeLocked() {
	if s.closed {
		return
	}
	s.closed = true
	close(s.events)
	close(s.errors)
}

func DefaultScript(ctx context.Context, session *Session, input uit.Input) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if strings.Contains(strings.ToLower(input.Text), "permission") {
		session.PreviewThinking("checking policy")
		session.Tool(uit.Tool{CallID: "call-write", Name: "write_file", State: uit.ToolPlanned})
		session.Suspend(uit.Suspension{
			ID: "permission-1", Kind: "permission", Supported: true,
			Approvals: []uit.Approval{{CallID: "call-write", ToolName: "write_file", Capability: "filesystem", Action: "write", ResourceScheme: "file", CanonicalResource: "/offline/result.txt", ResourceDisplay: "result.txt"}},
		})
		return nil
	}
	session.PreviewThinking("planning")
	session.Tool(uit.Tool{CallID: "call-1", Name: "inspect", State: uit.ToolRunning})
	for _, chunk := range []string{"Offline ", "task ", "complete."} {
		session.PreviewText(chunk)
	}
	session.Tool(uit.Tool{CallID: "call-1", Name: "inspect", State: uit.ToolDone, Summary: "safe summary"})
	session.Complete("Offline task complete.", uit.Usage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120, CacheReadTokens: 75, Requests: 1, ToolCalls: 1})
	return nil
}

func cloneSnapshot(value uit.Snapshot) uit.Snapshot {
	value.Transcript = append([]uit.Entry(nil), value.Transcript...)
	for index := range value.Transcript {
		value.Transcript[index].Thinking = append([]uit.Thinking(nil), value.Transcript[index].Thinking...)
		value.Transcript[index].Tools = append([]uit.Tool(nil), value.Transcript[index].Tools...)
	}
	value.Pending = append([]uit.QueuedInput(nil), value.Pending...)
	value.Suspension = cloneSuspension(value.Suspension)
	return value
}

func cloneSuspension(value *uit.Suspension) *uit.Suspension {
	if value == nil {
		return nil
	}
	copyOfValue := *value
	copyOfValue.Approvals = append([]uit.Approval(nil), value.Approvals...)
	return &copyOfValue
}

func cloneEvent(value uit.Event) uit.Event {
	if value.Entry != nil {
		entry := *value.Entry
		value.Entry = &entry
	}
	value.Entries = append([]uit.Entry(nil), value.Entries...)
	value.Tools = append([]uit.Tool(nil), value.Tools...)
	if value.Tool != nil {
		tool := *value.Tool
		value.Tool = &tool
	}
	if value.Usage != nil {
		usage := *value.Usage
		value.Usage = &usage
	}
	value.Suspension = cloneSuspension(value.Suspension)
	if value.Queue != nil {
		queue := *value.Queue
		value.Queue = &queue
	}
	return value
}

var _ uit.Host = (*Host)(nil)
var _ uit.SessionLister = (*Host)(nil)
var _ uit.TranscriptExporter = (*Host)(nil)
var _ uit.Session = (*Session)(nil)
