// Package observe defines a typed, presentation-neutral projection of Harness
// session activity. It contains no terminal, ANSI, markdown, or provider code.
package observe

import (
	"sync"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/event"
)

// Kind is a stable semantic event name. Unknown capability-owned kinds remain
// visible as their durable string instead of being guessed by a client.
type Kind string

const (
	KindTextDelta          Kind = "text.delta"
	KindThinkingDelta      Kind = "thinking.delta"
	KindAssistantCommitted Kind = "assistant.committed"
	KindToolPlanned        Kind = "tool.planned"
	KindToolStarted        Kind = "tool.started"
	KindToolResult         Kind = "tool.result"
	KindMessagesInjected   Kind = "messages.injected"
	KindRunStarted         Kind = "run.started"
	KindTurnStarted        Kind = "turn.started"
	KindTurnEnded          Kind = "turn.ended"
	KindRunSuspended       Kind = "run.suspended"
	KindRunCompleted       Kind = "run.completed"
	KindRunInterrupted     Kind = "run.interrupted"
	KindRunFailed          Kind = "run.failed"
	KindRunEnded           Kind = "run.ended"
	KindOutputValidated    Kind = "output.validated"
	KindQueueAccepted      Kind = "queue.accepted"
	KindQueueDrained       Kind = "queue.drained"
	KindQueueCancelled     Kind = "queue.cancelled"
	KindSessionCreated     Kind = "session.created"
	KindSessionRecovered   Kind = "session.recovered"
	KindSessionFaulted     Kind = "session.faulted"
	KindSessionClosed      Kind = "session.closed"
	KindUsage              Kind = "usage"
	KindCompaction         Kind = "transcript.compaction"
	KindToolUpdate         Kind = "tool.update"
)

// ToolState is deliberately small; presentation of raw arguments and results
// requires an explicit capability-owned redactor outside this default view.
type ToolState string

const (
	ToolPreview ToolState = "preview"
	ToolPlanned ToolState = "planned"
	ToolRunning ToolState = "running"
	ToolDone    ToolState = "done"
	ToolError   ToolState = "error"
)

type Thinking struct {
	Text         string
	ProviderName string
	ThinkingID   string
	Redacted     bool
}

type Tool struct {
	CallID  string
	Name    string
	State   ToolState
	Attempt int
	Summary string
}

// Message is a conservative transcript projection. Tool inputs, tool results,
// provider metadata, signatures, and opaque parts are never copied into it.
type Message struct {
	Role     string
	Text     string
	Thinking []Thinking
	Tools    []Tool
}

type Usage struct {
	PromptTokens        int
	CompletionTokens    int
	TotalTokens         int
	CacheReadTokens     int
	CacheCreationTokens int
	ReasoningTokens     int
	Requests            int
	ToolCalls           int
}

func UsageFromAgentic(value agentic.Usage) Usage {
	return Usage{
		PromptTokens: value.PromptTokens, CompletionTokens: value.CompletionTokens,
		TotalTokens: value.TotalTokens, CacheReadTokens: value.CacheReadTokens,
		CacheCreationTokens: value.CacheCreationTokens, ReasoningTokens: value.ReasoningTokens,
		Requests: value.Requests, ToolCalls: value.ToolCalls,
	}
}

// CacheHitPercent is the cache-read share of the reported prompt token total.
func (u Usage) CacheHitPercent() float64 {
	if u.PromptTokens <= 0 || u.CacheReadTokens <= 0 {
		return 0
	}
	return 100 * float64(u.CacheReadTokens) / float64(u.PromptTokens)
}

type Suspension struct {
	ID   string
	Kind string
}

type Failure struct{ Message string }

type Queue struct {
	ID      string
	Kind    string
	Message *Message
}

// Event is a copy-owned typed projection. Cursor is zero only for a transient
// preview. Dropped reports lost previews and requires a snapshot redraw.
type Event struct {
	Cursor     uint64
	Ordinal    uint64
	Nature     agentic.EventNature
	SessionID  string
	ParentID   string
	Agent      string
	Depth      int
	Turn       int
	Kind       Kind
	TextDelta  string
	Message    *Message
	Messages   []Message
	Thinking   *Thinking
	Tool       *Tool
	Tools      []Tool
	Usage      *Usage
	Suspension *Suspension
	Failure    *Failure
	Queue      *Queue
	State      string
	Dropped    uint64
}

type SubscribeOptions struct {
	AfterCursor uint64
	Buffer      int
	Preview     bool
}

// Subscription separates observation from execution and mirrors the recovery
// semantics of the underlying authoritative event stream.
type Subscription interface {
	Events() <-chan Event
	Errors() <-chan error
	Close()
}

// Projector converts one copied raw record inside Harness, where the configured
// payload codec and private durable entry schemas are available.
type Projector func(event.Record) (Event, error)

type subscription struct {
	events <-chan Event
	errors <-chan error
	close  func()
	once   sync.Once
}

func (s *subscription) Events() <-chan Event { return s.events }
func (s *subscription) Errors() <-chan error { return s.errors }
func (s *subscription) Close() {
	if s == nil {
		return
	}
	s.once.Do(func() {
		if s.close != nil {
			s.close()
		}
	})
}

// ProjectSubscription adapts a raw Harness subscription without changing its
// backpressure contract. A projection error is terminal and is reported once.
func ProjectSubscription(source *event.Subscription, projector Projector) Subscription {
	events := make(chan Event)
	errors := make(chan error, 1)
	done := make(chan struct{})
	var closeOnce sync.Once
	closeFn := func() {
		closeOnce.Do(func() {
			close(done)
			if source != nil {
				source.Close()
			}
		})
	}
	result := &subscription{events: events, errors: errors, close: closeFn}
	go func() {
		defer close(events)
		defer close(errors)
		defer closeFn()
		if source == nil || projector == nil {
			return
		}
		records := source.Events
		sourceErrors := source.Err
		for records != nil || sourceErrors != nil {
			select {
			case <-done:
				return
			case record, ok := <-records:
				if !ok {
					records = nil
					continue
				}
				projected, err := projector(event.Clone(record))
				if err != nil {
					errors <- err
					return
				}
				select {
				case <-done:
					return
				case events <- projected:
				}
			case err, ok := <-sourceErrors:
				if !ok {
					sourceErrors = nil
					continue
				}
				if err != nil {
					errors <- err
				}
				return
			}
		}
	}()
	return result
}
