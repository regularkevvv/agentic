package session

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/codec"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
)

func (s *Session[O]) Prompt(ctx context.Context, prompt agentic.Message) (*agentic.Execution[O], error) {
	accepted, err := s.prepareStart(ctx, prompt, ctx)
	if err != nil {
		return nil, err
	}
	return s.driveAccepted(accepted)
}

// acceptedStart is the single-use product of prepareStart: one durably
// accepted, not-yet-driven run. It is consumed by exactly one driveAccepted
// call; a second consumption attempt fails without touching the session.
type acceptedStart[O any] struct {
	runID    string
	runCtx   context.Context
	history  []agentic.Message
	prompt   agentic.Message
	commit   store.Commit
	limits   *agentic.UsageLimits
	consumed atomic.Bool
}

func (a *acceptedStart[O]) consume() error {
	if !a.consumed.CompareAndSwap(false, true) {
		return errors.New("accepted start was already driven")
	}
	return nil
}

// prepareStart is the durable acceptance half of Prompt. acceptCtx governs
// instruction resolution and the acceptance journal append; runParent is the
// parent of the run context created inside the same locked acceptance window
// (the legacy Prompt passes its caller context as both).
func (s *Session[O]) prepareStart(
	acceptCtx context.Context,
	prompt agentic.Message,
	runParent context.Context,
) (*acceptedStart[O], error) {
	if prompt.Role != agentic.RoleUser {
		return nil, ErrInvalidMessage
	}
	s.mu.Lock()
	if err := s.promptErrorLocked(); err != nil {
		s.mu.Unlock()
		return nil, err
	}
	s.mu.Unlock()
	runID, err := s.ids.New("run")
	if err != nil {
		return nil, err
	}
	instructions, err := s.resolveExchangeInstructions(acceptCtx, runID)
	if err != nil {
		return nil, fmt.Errorf("resolve exchange instructions: %w", err)
	}

	s.mu.Lock()
	if err := s.promptErrorLocked(); err != nil {
		s.mu.Unlock()
		return nil, err
	}
	var limits *agentic.UsageLimits
	if s.budget != nil {
		remaining, remainingErr := remainingLimits(*s.budget, s.usage)
		if remainingErr != nil {
			s.mu.Unlock()
			return nil, remainingErr
		}
		limits = &remaining
	}
	next := s.queueByKindsLocked(QueueNextTurn)
	batch := newEntryBatch(s.codec, 2+2*len(next))
	batch.Add(kindRunOpened, runOpenedPayload{
		ID: runID, Mode: "start", Limits: cloneLimitsPointer(limits), Instructions: instructions,
	})
	for _, entry := range next {
		batch.Add(kindQueueDrained, queueMutationPayload{ID: entry.ID})
		batch.Add(kindMessage, messagePayload{Message: entry.Message, Source: string(QueueNextTurn), QueueID: entry.ID})
	}
	promptCopy := cloneMessages([]agentic.Message{prompt})[0]
	batch.Add(kindMessage, messagePayload{Message: promptCopy, Source: "prompt"})
	pendingEntries, encodeErr := batch.Result()
	if encodeErr != nil {
		s.mu.Unlock()
		return nil, encodeErr
	}
	commit, appendErr := s.journal.Append(acceptCtx, s.cursor, pendingEntries...)
	if appendErr != nil {
		s.mu.Unlock()
		return nil, appendErr
	}
	for _, entry := range next {
		_, _ = s.removeQueueLocked(entry.ID)
		s.messages = append(s.messages, cloneMessages([]agentic.Message{entry.Message})[0])
	}
	history := providerHistory(s.messages, s.contextMarkers)
	s.messages = append(s.messages, promptCopy)
	s.cursor = commit.Cursor
	runCtx, cancel := context.WithCancel(runParent)
	s.runCancel = cancel
	s.run = &activeRun{
		id:                 runID,
		mode:               agentic.DriveStart,
		history:            history,
		expected:           []agentic.Message{promptCopy},
		contextMarkerCount: len(s.contextMarkers),
		limits:             cloneLimitsPointer(limits),
		instructions:       instructions,
	}
	s.transitionLocked(Running)
	s.mu.Unlock()
	s.publishOwn(commit.Entries, agentic.EventAuthoritative)
	return &acceptedStart[O]{
		runID:   runID,
		runCtx:  runCtx,
		history: history,
		prompt:  promptCopy,
		commit:  commit,
		limits:  limits,
	}, nil
}

// driveAccepted is the execution half of Prompt. It drives the accepted run
// exactly once under the run context created at acceptance and settles it
// through the existing finish path.
func (s *Session[O]) driveAccepted(accepted *acceptedStart[O]) (*agentic.Execution[O], error) {
	if err := accepted.consume(); err != nil {
		return nil, err
	}
	if s.State() == Closed {
		return nil, ErrSessionClosed
	}
	runCtx := s.withToolRuntime(accepted.runCtx)
	inputPrompt := cloneMessages([]agentic.Message{accepted.prompt})[0]
	execution, runErr := s.driver.Drive(runCtx, agentic.DriveInput{
		Mode:    agentic.DriveStart,
		History: accepted.history,
		Prompt:  &inputPrompt,
	}, s.runOptions(accepted.limits)...)
	return s.finishExecution(execution, runErr)
}

func (s *Session[O]) resolveExchangeInstructions(ctx context.Context, runID string) (string, error) {
	if s.instructions == nil {
		return "", nil
	}
	return s.instructions.ResolveExchangeInstructions(ctx, harnessruntime.ExchangeContext{
		SessionID: s.id,
		RunID:     runID,
		Scope:     s.scope,
	})
}

func (s *Session[O]) promptErrorLocked() error {
	switch s.state {
	case Faulted:
		return &FaultError{SessionID: s.id, Cause: s.fault}
	case Suspended:
		return ErrSessionSuspended
	case Closed:
		return ErrSessionClosed
	case Idle:
		return nil
	default:
		return ErrSessionBusy
	}
}

func (s *Session[O]) runOptions(limits *agentic.UsageLimits) []agentic.RunOption {
	options := []agentic.RunOption{
		agentic.WithRunHistoryProcessor(agentic.HistoryProcessorFunc(s.projectHistory)),
		agentic.WithRunTurnHook(s.turnHook),
		agentic.WithRunEventSink(s.eventSink),
		agentic.WithRunToolResultProcessor(s.processor),
		agentic.WithRunToolCancellationGrace(s.grace),
	}
	if s.streaming {
		options = append(options, agentic.WithRunModelStreaming(true))
	}
	if s.promptCache != agentic.PromptCacheNone {
		retention := s.promptCache
		if retention == "" {
			retention = agentic.PromptCacheShort
		}
		options = append(options, agentic.WithRunPromptCache(agentic.PromptCacheConfig{
			Key: s.id, Retention: retention,
		}))
	}
	if len(s.toolsets) > 0 {
		options = append(options, agentic.WithRunToolsets(s.toolsets...))
	}
	if s.toolGate != nil {
		options = append(options, agentic.WithRunToolGate(s.toolGate))
	}
	if limits != nil {
		options = append(options, agentic.WithRunUsageLimits(*limits))
	}
	return options
}

func (s *Session[O]) withToolRuntime(ctx context.Context) context.Context {
	return harnessruntime.WithContext(ctx, harnessruntime.ToolRuntime{
		Environment: s.environment,
		SessionID:   s.id,
		Scope:       s.scope,
		Capture:     s,
		Operations:  s,
		Emit:        s.emitToolUpdate,
	})
}

func (s *Session[O]) emitToolUpdate(update harnessruntime.ToolUpdate) {
	payload, err := codec.Encode(s.codec, update)
	if err != nil {
		return
	}
	s.previewMu.Lock()
	defer s.previewMu.Unlock()
	s.mu.Lock()
	ordinal := uint64(0)
	turn := 0
	if s.run != nil {
		turn = s.run.turn
		s.run.previewOrdinal++
		ordinal = s.run.previewOrdinal
	}
	s.mu.Unlock()
	s.bus.PublishPreview(s.scopedRecord(event.Record{
		Nature:  agentic.EventPreview,
		Turn:    turn,
		Ordinal: ordinal,
		Source:  "tool",
		Name:    update.Kind,
		Payload: payload,
	}))
}

func (s *Session[O]) turnHook(ctx context.Context, turn agentic.Turn) (agentic.TurnDecision, error) {
	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return agentic.TurnDecision{}, err
	}
	if s.state == Interrupting {
		s.mu.Unlock()
		return agentic.TurnDecision{}, context.Canceled
	}
	if s.state != Running || s.run == nil {
		s.mu.Unlock()
		return agentic.TurnDecision{}, fmt.Errorf("turn hook observed session state %s", s.state)
	}
	var childBudgetErr error
	if s.run.childUsageCharged && s.budget != nil {
		_, childBudgetErr = remainingLimits(*s.budget, s.usage)
	}

	entries := s.queueByKindsLocked(QueueSteer)
	if len(entries) == 0 && turn.Candidate.Valid() {
		entries = s.queueByKindsLocked(QueueFollowUp)
	}
	if len(entries) > 0 && !s.drainAll {
		entries = entries[:1]
	}
	if len(entries) == 0 {
		if turn.Candidate.Valid() {
			s.transitionLocked(Closing)
		} else if childBudgetErr != nil {
			s.transitionLocked(Closing)
			s.mu.Unlock()
			return agentic.TurnDecision{}, childBudgetErr
		}
		s.mu.Unlock()
		return agentic.TurnDecision{Action: agentic.TurnDefault}, nil
	}
	if childBudgetErr != nil {
		s.transitionLocked(Closing)
		s.mu.Unlock()
		return agentic.TurnDecision{}, childBudgetErr
	}

	batch := newEntryBatch(s.codec, len(entries))
	injected := make([]agentic.Message, len(entries))
	ids := make([]string, len(entries))
	for i, entry := range entries {
		batch.Add(kindQueueDrained, queueMutationPayload{ID: entry.ID})
		injected[i] = cloneMessages([]agentic.Message{entry.Message})[0]
		ids[i] = entry.ID
	}
	pendingEntries, encodeErr := batch.Result()
	if encodeErr != nil {
		cancel := s.faultLocked(encodeErr)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return agentic.TurnDecision{}, encodeErr
	}
	commit, err := s.journal.Append(context.WithoutCancel(ctx), s.cursor, pendingEntries...)
	if err != nil {
		cancel := s.faultLocked(err)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return agentic.TurnDecision{}, err
	}
	for _, entry := range entries {
		_, _ = s.removeQueueLocked(entry.ID)
	}
	s.run.pendingInjectionIDs = ids
	s.cursor = commit.Cursor
	s.mu.Unlock()
	s.publishOwn(commit.Entries, agentic.EventAuthoritative)
	return agentic.TurnDecision{Action: agentic.TurnContinue, Inject: injected}, nil
}

// Emit implements agentic.EventSink. Durable root events participate in the
// session commit and are fsynced before public publication.
func (s *Session[O]) Emit(ctx context.Context, value agentic.Event) error {
	record, err := event.FromAgentic(s.codec, value)
	if err != nil {
		if value != nil && value.Nature() != agentic.EventPreview {
			s.mu.Lock()
			var cancel context.CancelFunc
			if s.state != Faulted {
				cancel = s.faultLocked(err)
			}
			s.mu.Unlock()
			if cancel != nil {
				cancel()
			}
		}
		return err
	}
	if record.Nature == agentic.EventPreview {
		s.previewMu.Lock()
		defer s.previewMu.Unlock()
		s.mu.Lock()
		if s.run != nil {
			if s.run.turn != record.Turn {
				s.run.turn = record.Turn
				s.run.previewOrdinal = 0
			}
			s.run.previewOrdinal++
			record.Ordinal = s.run.previewOrdinal
		}
		s.mu.Unlock()
		s.bus.PublishPreview(s.scopedRecord(record))
		return nil
	}
	record = s.scopedRecord(record)
	kind, err := agenticEntryKind(record.Type)
	if err != nil {
		s.mu.Lock()
		var cancel context.CancelFunc
		if s.state != Faulted {
			cancel = s.faultLocked(err)
		}
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return err
	}

	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return err
	}
	if s.run == nil {
		s.mu.Unlock()
		return errors.New("agentic event emitted without an active session run")
	}
	if injected, ok := value.(*agentic.TurnMessagesInjectedEvent); ok {
		payload := event.MessagesPayload{Messages: injected.Messages, QueueIDs: append([]string(nil), s.run.pendingInjectionIDs...)}
		record.Payload, err = codec.Encode(s.codec, payload)
		if err != nil {
			cancel := s.faultLocked(err)
			s.mu.Unlock()
			if cancel != nil {
				cancel()
			}
			return err
		}
	}
	var nextUsage agentic.Usage
	if ended, ok := value.(*agentic.TurnEndedEvent); ok {
		delta, deltaErr := usageDelta(ended.Usage, s.run.lastUsage)
		if deltaErr != nil {
			cancel := s.faultLocked(deltaErr)
			s.mu.Unlock()
			if cancel != nil {
				cancel()
			}
			return deltaErr
		}
		nextUsage = addUsage(s.usage, delta)
		copy := cloneUsage(nextUsage)
		record.SessionUsed = &copy
	}
	pendingEntry, err := pending(s.codec, kind, record)
	if err != nil {
		cancel := s.faultLocked(err)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return err
	}
	commit, appendErr := s.journal.Append(context.WithoutCancel(ctx), s.cursor, pendingEntry)
	if appendErr != nil {
		cancel := s.faultLocked(appendErr)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return appendErr
	}
	record.Cursor = commit.Cursor.Seq
	s.cursor = commit.Cursor
	s.applyAgenticEventLocked(value)
	if _, ok := value.(*agentic.TurnEndedEvent); ok {
		s.usage = nextUsage
		s.run.lastUsage = cloneUsage(value.(*agentic.TurnEndedEvent).Usage)
	}
	s.mu.Unlock()
	s.bus.PublishDurable(record)
	return nil
}

func agenticEntryKind(eventType agentic.EventType) (string, error) {
	switch eventType {
	case agentic.EventTypeAssistantCommitted:
		return kindAssistantCommitted, nil
	case agentic.EventTypeToolBatchPlanned:
		return kindToolBatchPlanned, nil
	case agentic.EventTypeToolStarted:
		return kindToolStarted, nil
	case agentic.EventTypeToolResultCommitted:
		return kindToolResult, nil
	case agentic.EventTypeOutputValidated:
		return kindOutputValidated, nil
	case agentic.EventTypeTurnMessagesInjected:
		return kindMessagesInjected, nil
	case agentic.EventTypeRunStarted:
		return kindRunStarted, nil
	case agentic.EventTypeTurnStarted:
		return kindTurnStarted, nil
	case agentic.EventTypeTurnEnded:
		return kindTurnEnded, nil
	case agentic.EventTypeRunSuspended:
		return kindRunSuspended, nil
	case agentic.EventTypeRunCompleted:
		return kindRunCompleted, nil
	case agentic.EventTypeRunInterrupted:
		return kindRunInterrupted, nil
	case agentic.EventTypeRunError:
		return kindRunError, nil
	case agentic.EventTypeRunEnded:
		return kindRunEnded, nil
	default:
		return "", fmt.Errorf("event type %d is not durable", eventType)
	}
}

func (s *Session[O]) applyAgenticEventLocked(value agentic.Event) {
	switch current := value.(type) {
	case *agentic.AssistantCommittedEvent:
		message := cloneMessages([]agentic.Message{current.Message})[0]
		s.messages = append(s.messages, message)
		s.run.expected = append(s.run.expected, message)
	case *agentic.ToolResultCommittedEvent:
		message := agentic.NewToolResultMessageFor(
			current.Result.ToolUseID,
			current.Result.ToolName,
			agentic.FormatToolResult(current.Result.Content),
			current.Result.IsError,
		)
		s.messages = append(s.messages, message)
		s.run.expected = append(s.run.expected, message)
		if s.run.results == nil {
			s.run.results = make(map[string]bool)
		}
		s.run.results[current.Result.ToolUseID] = true
	case *agentic.ToolBatchPlannedEvent:
		s.run.planned = append(s.run.planned, current.Calls...)
	case *agentic.ToolStartedEvent:
		if s.run.started == nil {
			s.run.started = make(map[string]bool)
		}
		s.run.started[current.Call.ID] = true
	case *agentic.TurnMessagesInjectedEvent:
		messages := cloneMessages(current.Messages)
		s.messages = append(s.messages, messages...)
		s.run.expected = append(s.run.expected, messages...)
		s.run.pendingInjectionIDs = nil
	case *agentic.TurnStartedEvent:
		s.run.turn = current.TurnIndex()
		s.run.previewOrdinal = 0
	case *agentic.RunStartedEvent:
		if s.run.resumeInProgress {
			s.run.resumeEventSeen = true
		}
	case *agentic.RunSuspendedEvent:
		s.suspension = cloneSuspension(&current.Suspension)
	}
}
