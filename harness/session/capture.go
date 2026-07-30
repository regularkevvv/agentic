package session

import (
	"context"
	"errors"
	"fmt"
	"sync"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

// History returns the durable parent transcript before the currently executing
// assistant tool frontier. A child receives a copy and never shares mutation
// with its parent.
func (s *Session[O]) History() []agentic.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	history := cloneMessages(s.messages)
	if s.run == nil || len(history) == 0 {
		return history
	}

	// A batch can have partial results while another handler is still
	// executing. Capture must omit the entire open assistant frontier, not just
	// an assistant message that happens to be last, so the child never receives
	// an unpaired transcript.
	for index := len(history) - 1; index >= 0; index-- {
		calls := history[index].GetToolUses()
		if len(calls) == 0 {
			continue
		}
		for _, call := range calls {
			if s.run.results == nil || !s.run.results[call.ID] {
				return history[:index]
			}
		}
		return history
	}
	return history
}

func (s *Session[O]) Toolsets() []agentic.Toolset {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]agentic.Toolset(nil), s.toolsets...)
}

func (s *Session[O]) ToolGate() agentic.ToolGate {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.toolGate
}

func (s *Session[O]) DelegationTools() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.delegation...)
}

func (s *Session[O]) Scope() harnessruntime.Scope {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.scope
}

func (s *Session[O]) AcquireBudget(ctx context.Context) (harnessruntime.BudgetLease, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-s.childBudget:
	}

	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		s.childBudget <- struct{}{}
		return nil, err
	}
	if s.state != Running {
		state := s.state
		s.mu.Unlock()
		s.childBudget <- struct{}{}
		return nil, fmt.Errorf("child budget requires a running parent session, got %s", state)
	}
	var limits *agentic.UsageLimits
	if s.budget != nil {
		remaining, err := remainingLimits(*s.budget, s.usage)
		if err != nil {
			s.mu.Unlock()
			s.childBudget <- struct{}{}
			return nil, err
		}
		limits = &remaining
	}
	s.mu.Unlock()
	return &childBudgetLease[O]{session: s, limits: cloneLimitsPointer(limits)}, nil
}

// ProjectEvent assigns a parent cursor to a tagged child event. Child preview
// remains transient; authoritative and lifecycle records are write-ahead facts
// in the parent journal and survive Snapshot/resubscribe recovery.
func (s *Session[O]) ProjectEvent(ctx context.Context, record event.Record) error {
	record = event.Clone(record)
	if record.SessionID == "" {
		return errors.New("projected child event requires a session ID")
	}
	if record.SessionID == s.id {
		return errors.New("session cannot project its own event as a child")
	}
	if record.Agent == "" {
		return errors.New("projected child event requires an agent name")
	}
	if record.ParentID == "" {
		record.ParentID = s.id
	}
	if record.ParentID == s.id && record.Depth != s.scope.Depth+1 {
		return errors.New("projected child event has an invalid direct-child depth")
	}
	if record.ParentID != s.id && record.Depth <= s.scope.Depth+1 {
		return errors.New("projected descendant event has an invalid topology depth")
	}
	if record.Nature != agentic.EventPreview &&
		record.Nature != agentic.EventAuthoritative &&
		record.Nature != agentic.EventLifecycle {
		return errors.New("projected child event has an invalid nature")
	}
	record.Cursor = 0
	if record.Nature == agentic.EventPreview {
		s.mu.Lock()
		if s.state == Faulted {
			fault := &FaultError{SessionID: s.id, Cause: s.fault}
			s.mu.Unlock()
			return fault
		}
		if s.state == Closed {
			s.mu.Unlock()
			return ErrSessionClosed
		}
		s.mu.Unlock()
		s.bus.PublishPreview(record)
		return nil
	}
	pendingEntry, err := pending(s.codec, kindChildEvent, record)
	if err != nil {
		s.mu.Lock()
		if s.state == Faulted {
			fault := &FaultError{SessionID: s.id, Cause: s.fault}
			s.mu.Unlock()
			return fault
		}
		if s.state == Closed {
			s.mu.Unlock()
			return ErrSessionClosed
		}
		cancel := s.faultLocked(err)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return &FaultError{SessionID: s.id, Cause: err}
	}
	s.mu.Lock()
	if s.state == Faulted {
		fault := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return fault
	}
	if s.state == Closed {
		s.mu.Unlock()
		return ErrSessionClosed
	}
	commit, appendErr := s.journal.Append(context.WithoutCancel(ctx), s.cursor, pendingEntry)
	if appendErr != nil {
		cancel := s.faultLocked(appendErr)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return &FaultError{SessionID: s.id, Cause: appendErr}
	}
	record.Cursor = commit.Cursor.Seq
	s.cursor = commit.Cursor
	s.mu.Unlock()
	s.bus.PublishDurable(record)
	return nil
}

func (s *Session[O]) scopedRecord(record event.Record) event.Record {
	return scopeRecord(s.scope, record)
}

func scopeRecord(scope harnessruntime.Scope, record event.Record) event.Record {
	if record.SessionID != "" {
		return record
	}
	record.SessionID = scope.SessionID
	record.ParentID = scope.ParentSessionID
	record.Agent = scope.Agent
	record.Depth = scope.Depth
	return record
}

type childBudgetLease[O any] struct {
	session *Session[O]
	limits  *agentic.UsageLimits

	mu        sync.Mutex
	committed bool
	closed    bool
}

func (l *childBudgetLease[O]) Limits() *agentic.UsageLimits {
	l.mu.Lock()
	defer l.mu.Unlock()
	return cloneLimitsPointer(l.limits)
}

func (l *childBudgetLease[O]) Commit(ctx context.Context, charge harnessruntime.UsageCharge) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return errors.New("child budget lease is closed")
	}
	if l.committed {
		return errors.New("child usage was already committed")
	}
	if err := validateUsageCharge(charge); err != nil {
		return err
	}
	l.committed = true

	s := l.session
	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return err
	}
	switch s.state {
	case Running, Closing, Interrupting:
	case Closed:
		s.mu.Unlock()
		return ErrSessionClosed
	default:
		state := s.state
		s.mu.Unlock()
		return fmt.Errorf("child usage commit requires an active parent session, got %s", state)
	}
	next := addUsage(s.usage, charge.Usage)
	entry, err := pending(s.codec, kindChildUsage, childUsagePayload{
		Charge:  cloneUsageCharge(charge),
		Session: cloneUsage(next),
	})
	if err != nil {
		cancel := s.faultLocked(err)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return &FaultError{SessionID: s.id, Cause: err}
	}
	commit, appendErr := s.journal.Append(context.WithoutCancel(ctx), s.cursor, entry)
	if appendErr != nil {
		cancel := s.faultLocked(appendErr)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return &FaultError{SessionID: s.id, Cause: appendErr}
	}
	s.usage = next
	s.cursor = commit.Cursor
	if s.run != nil {
		s.run.childUsageCharged = true
	}
	var budgetErr error
	if s.budget != nil {
		_, budgetErr = remainingLimits(*s.budget, next)
	}
	s.mu.Unlock()
	s.publishOwn(commit.Entries, agentic.EventAuthoritative)
	return budgetErr
}

func (l *childBudgetLease[O]) Close() {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return
	}
	l.closed = true
	l.mu.Unlock()
	l.session.childBudget <- struct{}{}
}

func validateUsageCharge(charge harnessruntime.UsageCharge) error {
	if charge.SessionID == "" {
		return errors.New("child usage charge requires a session ID")
	}
	usage := charge.Usage
	if usage.PromptTokens < 0 || usage.CompletionTokens < 0 || usage.TotalTokens < 0 ||
		usage.CacheReadTokens < 0 || usage.CacheCreationTokens < 0 || usage.ReasoningTokens < 0 ||
		usage.Requests < 0 || usage.ToolCalls < 0 {
		return errors.New("child usage charge cannot contain negative counters")
	}
	return nil
}

func cloneUsageCharge(charge harnessruntime.UsageCharge) harnessruntime.UsageCharge {
	charge.Usage = cloneUsage(charge.Usage)
	return charge
}

var _ harnessruntime.CaptureRuntime = (*Session[struct{}])(nil)
