package session

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/codec"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/env"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/repair"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
)

type activeRun struct {
	id                  string
	mode                agentic.DriveMode
	history             []agentic.Message
	expected            []agentic.Message
	lastUsage           agentic.Usage
	pendingInjectionIDs []string
	turn                int
	previewOrdinal      uint64
	interrupted         bool
	planned             []agentic.ToolUse
	started             map[string]bool
	results             map[string]bool
	contextMarkerCount  int
	limits              *agentic.UsageLimits
	resumeInProgress    bool
	resumeEventSeen     bool
	publicNewStart      int
	childUsageCharged   bool
}

type contextMarker struct {
	after   int
	message agentic.Message
}

type Session[O any] struct {
	id          string
	driver      agentic.Driver[O]
	journal     store.Journal
	codec       codec.Codec
	environment env.Lease
	processor   agentic.ToolResultProcessor
	clock       harnessruntime.Clock
	ids         harnessruntime.IDGenerator
	grace       time.Duration
	bus         event.Hub
	toolsets    []agentic.Toolset
	toolGate    agentic.ToolGate
	context     contextpolicy.Projector
	eventSink   agentic.EventSink
	lifecycle   []harnessruntime.LifecycleHook
	resume      harnessruntime.ResumePlanner
	scope       harnessruntime.Scope
	delegation  []string
	compaction  *contextpolicy.Compaction

	closeMu           sync.Mutex
	childBudget       chan struct{}
	previewMu         sync.Mutex
	busClosed         bool
	journalClosed     bool
	environmentClosed bool
	closingHookDone   bool
	closedHookDone    bool

	mu             sync.Mutex
	state          State
	stateChange    chan struct{}
	fault          error
	cursor         store.Cursor
	messages       []agentic.Message
	contextMarkers []contextMarker
	queue          []QueueEntry
	usage          agentic.Usage
	budget         *agentic.UsageLimits
	drainAll       bool
	suspension     *agentic.Suspension
	run            *activeRun
	runCancel      context.CancelFunc
	recoveryInputs []QueueEntry
}

func New[O any](ctx context.Context, config Config[O], opts ...Option) (*Session[O], error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	settings := options{}
	for _, option := range opts {
		if option == nil {
			return nil, errors.New("session option must not be nil")
		}
		if err := option(&settings); err != nil {
			return nil, err
		}
	}
	initialHistory, err := validateInitialHistory(settings.initialHistory)
	if err != nil {
		return nil, err
	}
	grace := config.ToolCancellationGrace
	if grace == 0 {
		grace = time.Second
	}
	contextProjector := config.Context
	if contextProjector == nil {
		contextProjector = contextpolicy.Passthrough()
	}
	resumePlanner := config.ResumePlanner
	if resumePlanner == nil {
		resumePlanner = harnessruntime.DefaultResumePlanner()
	}
	scope := config.Scope
	scope.SessionID = config.ID
	environment, err := config.Environments.Open(ctx, config.ID)
	if err != nil {
		return nil, err
	}
	if environment == nil {
		return nil, errors.New("environment factory returned nil")
	}
	cleanupEnvironment := true
	defer func() {
		if cleanupEnvironment {
			_ = environment.Close(context.Background())
		}
	}()
	processor, err := config.ResultProcessors.Open(ctx, config.ID)
	if err != nil {
		return nil, err
	}
	if processor == nil {
		return nil, errors.New("result-processor factory returned nil")
	}
	bus, err := config.Events.Open(ctx, nil)
	if err != nil {
		return nil, err
	}
	if bus == nil {
		return nil, errors.New("event factory returned nil")
	}
	cleanupBus := true
	defer func() {
		if cleanupBus {
			bus.Close()
		}
	}()
	persistedScope := scope
	payload := sessionCreatedPayload{
		Options: persistedOptions{Budget: settings.budget, DrainAll: settings.drainAll},
		Scope:   &persistedScope,
	}
	initial, err := pending(config.Codec, kindSessionCreated, payload)
	if err != nil {
		return nil, err
	}
	initialEntries := make([]store.PendingEntry, 0, 1+len(initialHistory))
	initialEntries = append(initialEntries, initial)
	for _, message := range initialHistory {
		entry, encodeErr := pending(config.Codec, kindMessage, messagePayload{
			Message: message,
			Source:  "initial_history",
		})
		if encodeErr != nil {
			return nil, encodeErr
		}
		initialEntries = append(initialEntries, entry)
	}
	journal, commit, err := config.Repository.Create(ctx, config.ID, initialEntries...)
	if err != nil {
		return nil, err
	}
	cleanupJournal := true
	defer func() {
		if cleanupJournal {
			_ = journal.Close(context.Background())
		}
	}()
	if len(commit.Entries) != len(initialEntries) {
		return nil, errors.New("session repository did not commit every creation entry")
	}
	for index, entry := range commit.Entries {
		nature := agentic.EventAuthoritative
		if index == 0 {
			nature = agentic.EventLifecycle
		}
		bus.PublishDurable(scopeRecord(scope, ownRecord(entry, nature)))
	}
	session := &Session[O]{
		id:          config.ID,
		driver:      config.Driver,
		journal:     journal,
		codec:       config.Codec,
		environment: environment,
		processor:   processor,
		clock:       config.Clock,
		ids:         config.IDs,
		grace:       grace,
		bus:         bus,
		toolsets:    append([]agentic.Toolset(nil), config.Toolsets...),
		toolGate:    config.ToolGate,
		context:     contextProjector,
		lifecycle:   append([]harnessruntime.LifecycleHook(nil), config.LifecycleHooks...),
		resume:      resumePlanner,
		scope:       scope,
		delegation:  append([]string(nil), config.DelegationTools...),
		childBudget: make(chan struct{}, 1),
		state:       Idle,
		stateChange: make(chan struct{}),
		cursor:      commit.Cursor,
		messages:    cloneMessages(initialHistory),
		budget:      settings.budget,
		drainAll:    settings.drainAll,
	}
	session.childBudget <- struct{}{}
	sink, err := event.Chain(session, config.EventMiddleware...)
	if err != nil {
		return nil, err
	}
	session.eventSink = sink
	if err := harnessruntime.RunLifecycleHooks(ctx, session.lifecycle, harnessruntime.LifecycleEvent{
		Phase:     harnessruntime.LifecycleSessionOpened,
		SessionID: session.id,
	}); err != nil {
		return nil, fmt.Errorf("session opened lifecycle: %w", err)
	}
	cleanupJournal = false
	cleanupBus = false
	cleanupEnvironment = false
	return session, nil
}

func validateInitialHistory(messages []agentic.Message) ([]agentic.Message, error) {
	history := cloneMessages(messages)
	if len(history) == 0 {
		return nil, nil
	}
	repaired, err := repair.Process(history, repair.CloseInterruptedFrontier, repair.PendingCalls{})
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidInitialHistory, err)
	}
	if !messagesEqual(history, repaired) {
		return nil, ErrInvalidInitialHistory
	}
	return history, nil
}

func (s *Session[O]) ID() string { return s.id }

func (s *Session[O]) State() State {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

func (s *Session[O]) Snapshot(ctx context.Context) (Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return Snapshot{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	pending := make([]QueueEntry, len(s.queue))
	for i, entry := range s.queue {
		pending[i] = entry
		pending[i].Message = cloneMessages([]agentic.Message{entry.Message})[0]
	}
	return Snapshot{
		Cursor:     s.cursor.Seq,
		State:      s.state,
		Messages:   cloneMessages(s.messages),
		Pending:    pending,
		Suspension: cloneSuspension(s.suspension),
		Usage:      cloneUsage(s.usage),
	}, nil
}

func (s *Session[O]) Subscribe(options SubscribeOptions) *Subscription {
	return s.bus.Subscribe(options)
}

func (s *Session[O]) WaitForIdle(ctx context.Context) error {
	for {
		s.mu.Lock()
		switch s.state {
		case Idle:
			s.mu.Unlock()
			return nil
		case Faulted:
			err := &FaultError{SessionID: s.id, Cause: s.fault}
			s.mu.Unlock()
			return err
		case Closed:
			s.mu.Unlock()
			return ErrSessionClosed
		}
		changed := s.stateChange
		s.mu.Unlock()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-changed:
		}
	}
}

func (s *Session[O]) Steer(ctx context.Context, message agentic.Message) (QueueReceipt, error) {
	return s.accept(ctx, QueueSteer, message)
}

func (s *Session[O]) FollowUp(ctx context.Context, message agentic.Message) (QueueReceipt, error) {
	return s.accept(ctx, QueueFollowUp, message)
}

func (s *Session[O]) NextTurn(ctx context.Context, message agentic.Message) (QueueReceipt, error) {
	return s.accept(ctx, QueueNextTurn, message)
}

func (s *Session[O]) accept(ctx context.Context, kind QueueKind, message agentic.Message) (QueueReceipt, error) {
	if message.Role != agentic.RoleUser {
		return QueueReceipt{}, ErrInvalidMessage
	}
	s.mu.Lock()
	if err := s.acceptanceErrorLocked(kind); err != nil {
		s.mu.Unlock()
		return QueueReceipt{}, err
	}
	s.mu.Unlock()
	id, err := s.ids.New("queue")
	if err != nil {
		return QueueReceipt{}, err
	}
	entry := QueueEntry{ID: id, Kind: kind, Message: cloneMessages([]agentic.Message{message})[0], Accepted: s.clock.Now().UTC()}
	pendingEntry, err := pending(s.codec, kindQueueAccepted, queueMutationPayload{ID: id, Entry: &entry})
	if err != nil {
		return QueueReceipt{}, err
	}

	s.mu.Lock()
	if err := s.acceptanceErrorLocked(kind); err != nil {
		s.mu.Unlock()
		return QueueReceipt{}, err
	}
	commit, appendErr := s.journal.Append(ctx, s.cursor, pendingEntry)
	if appendErr != nil {
		// Acceptance is write-ahead. No in-memory queue mutation occurred, so
		// this isolated failure does not fault the session.
		s.mu.Unlock()
		return QueueReceipt{}, appendErr
	}
	s.queue = append(s.queue, entry)
	s.cursor = commit.Cursor
	s.mu.Unlock()
	s.publishOwn(commit.Entries, agentic.EventAuthoritative)
	return QueueReceipt{ID: id, Kind: kind, Cursor: commit.Cursor.Seq}, nil
}

func (s *Session[O]) acceptanceErrorLocked(kind QueueKind) error {
	if s.state == Faulted {
		return &FaultError{SessionID: s.id, Cause: s.fault}
	}
	if s.state == Closed {
		return ErrSessionClosed
	}
	if kind == QueueNextTurn {
		return nil
	}
	switch s.state {
	case Closing:
		return ErrRunClosing
	case Running:
		return nil
	case Suspended:
		return ErrSessionSuspended
	default:
		return ErrNotRunning
	}
}

func (s *Session[O]) transitionLocked(state State) {
	if s.state == state {
		return
	}
	s.state = state
	close(s.stateChange)
	s.stateChange = make(chan struct{})
}

func (s *Session[O]) faultLocked(cause error) context.CancelFunc {
	if cause == nil {
		cause = ErrSessionFaulted
	}
	s.fault = cause
	s.transitionLocked(Faulted)
	cancel := s.runCancel
	s.runCancel = nil
	return cancel
}

func (s *Session[O]) publishOwn(entries []store.Entry, nature agentic.EventNature) {
	for _, entry := range entries {
		s.bus.PublishDurable(s.scopedRecord(ownRecord(entry, nature)))
	}
}

func ownRecord(entry store.Entry, nature agentic.EventNature) event.Record {
	return event.Record{
		Cursor:  entry.Seq,
		Nature:  nature,
		Source:  "harness",
		Name:    string(entry.Kind),
		Payload: append([]byte(nil), entry.Payload...),
	}
}

func (s *Session[O]) removeQueueLocked(id string) (QueueEntry, bool) {
	for i, entry := range s.queue {
		if entry.ID != id {
			continue
		}
		s.queue = append(s.queue[:i], s.queue[i+1:]...)
		return entry, true
	}
	return QueueEntry{}, false
}

func (s *Session[O]) queueByKindsLocked(kinds ...QueueKind) []QueueEntry {
	wanted := make(map[QueueKind]bool, len(kinds))
	for _, kind := range kinds {
		wanted[kind] = true
	}
	var result []QueueEntry
	for _, entry := range s.queue {
		if wanted[entry.Kind] {
			result = append(result, entry)
		}
	}
	return result
}
