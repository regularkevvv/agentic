package session

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/codec"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/repair"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
)

type recoverySuspensionPayload struct {
	Version int                  `json:"version"`
	Calls   []repair.PendingCall `json:"calls"`
}

func Recover[O any](ctx context.Context, config Config[O]) (*Session[O], error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
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
	journal, err := config.Repository.Open(ctx, config.ID)
	if err != nil {
		return nil, err
	}
	cleanupJournal := true
	defer func() {
		if cleanupJournal {
			_ = journal.Close(context.Background())
		}
	}()
	loaded, err := journal.Load(ctx)
	if err != nil {
		return nil, err
	}
	folded, history, err := fold(config.Codec, loaded.Entries)
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
	if folded.scope != nil {
		if folded.scope.SessionID != config.ID {
			return nil, fmt.Errorf(
				"persisted session scope ID %q does not match journal ID %q",
				folded.scope.SessionID,
				config.ID,
			)
		}
		scope = *folded.scope
	}
	for index := range history {
		history[index] = scopeRecord(scope, history[index])
	}
	bus, err := config.Events.Open(ctx, history)
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
	session := &Session[O]{
		id:             config.ID,
		driver:         config.Driver,
		journal:        journal,
		codec:          config.Codec,
		environment:    environment,
		processor:      processor,
		clock:          config.Clock,
		ids:            config.IDs,
		grace:          grace,
		bus:            bus,
		toolsets:       append([]agentic.Toolset(nil), config.Toolsets...),
		toolGate:       config.ToolGate,
		context:        contextProjector,
		lifecycle:      append([]harnessruntime.LifecycleHook(nil), config.LifecycleHooks...),
		resume:         resumePlanner,
		instructions:   config.Instructions,
		scope:          scope,
		delegation:     append([]string(nil), config.DelegationTools...),
		promptCache:    config.PromptCacheRetention,
		streaming:      config.ModelStreaming,
		summarize:      config.ToolSummarizer,
		childBudget:    make(chan struct{}, 1),
		compaction:     folded.compaction,
		state:          folded.state,
		stateChange:    make(chan struct{}),
		cursor:         folded.cursor,
		messages:       folded.messages,
		contextMarkers: folded.contextMarkers,
		queue:          folded.queue,
		usage:          folded.usage,
		budget:         folded.budget,
		drainAll:       folded.drainAll,
		suspension:     folded.suspension,
		run:            folded.run,
		recoveryInputs: folded.unapplied,
	}
	session.childBudget <- struct{}{}
	sink, err := event.Chain(session, config.EventMiddleware...)
	if err != nil {
		return nil, err
	}
	session.eventSink = sink
	if err := harnessruntime.RunLifecycleHooks(ctx, session.lifecycle, harnessruntime.LifecycleEvent{
		Phase:     harnessruntime.LifecycleSessionRecovered,
		SessionID: session.id,
	}); err != nil {
		return nil, fmt.Errorf("session recovered lifecycle: %w", err)
	}
	if session.run != nil && folded.pendingInterruptRunID == session.run.id {
		session.mu.Lock()
		session.run.interrupted = true
		session.transitionLocked(Interrupting)
		session.mu.Unlock()
		if err := session.finishInterrupt(&agentic.Execution[O]{Status: agentic.ExecutionInterrupted}); err != nil {
			return nil, err
		}
		cleanupJournal = false
		cleanupBus = false
		cleanupEnvironment = false
		return session, nil
	}
	if session.state == Faulted {
		session.state = Running
	}
	if session.run == nil {
		session.state = Idle
		entry, encodeErr := pending(session.codec, kindRecovered, struct{ State string }{State: Idle.String()})
		if encodeErr != nil {
			return nil, encodeErr
		}
		recovered, appendErr := session.journal.Append(ctx, session.cursor, entry)
		if appendErr != nil {
			return nil, appendErr
		}
		session.cursor = recovered.Cursor
		session.bus.PublishDurable(session.scopedRecord(ownRecord(recovered.Entries[0], agentic.EventLifecycle)))
		cleanupJournal = false
		cleanupBus = false
		cleanupEnvironment = false
		return session, nil
	}
	if session.state == Suspended && session.suspension != nil {
		cleanupJournal = false
		cleanupBus = false
		cleanupEnvironment = false
		return session, nil
	}
	if err := session.recoverOpenRun(ctx); err != nil {
		return nil, err
	}
	cleanupJournal = false
	cleanupBus = false
	cleanupEnvironment = false
	return session, nil
}

type foldedState struct {
	state                 State
	cursor                store.Cursor
	messages              []agentic.Message
	contextMarkers        []contextMarker
	queue                 []QueueEntry
	usage                 agentic.Usage
	budget                *agentic.UsageLimits
	scope                 *harnessruntime.Scope
	drainAll              bool
	suspension            *agentic.Suspension
	compaction            *contextpolicy.Compaction
	run                   *activeRun
	system                *agentic.Message
	unapplied             []QueueEntry
	pendingInterruptRunID string
}

func fold(payloadCodec codec.Codec, entries []store.Entry) (foldedState, []event.Record, error) {
	state := foldedState{state: Idle}
	queue := make(map[string]QueueEntry)
	drained := make(map[string]QueueEntry)
	var order []string
	var history []event.Record
	created := false
	for _, entry := range entries {
		state.cursor = entry.Cursor()
		if entry.Kind == kindChildEvent {
			record, err := decodePayload[event.Record](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			record.Cursor = entry.Seq
			history = append(history, record)
			continue
		}
		if isAgenticKind(entry.Kind) {
			record, err := codec.Decode[event.Record](payloadCodec, entry.Payload)
			if err != nil {
				return foldedState{}, nil, fmt.Errorf("decode event at sequence %d: %w", entry.Seq, err)
			}
			record.Cursor = entry.Seq
			history = append(history, record)
			if record.Type == agentic.EventTypeTurnMessagesInjected {
				payload, decodeErr := event.Decode[event.MessagesPayload](payloadCodec, record)
				if decodeErr != nil {
					return foldedState{}, nil, decodeErr
				}
				for _, id := range payload.QueueIDs {
					delete(drained, id)
				}
			}
			if err := applyRecoveredEvent(payloadCodec, &state, record); err != nil {
				return foldedState{}, nil, err
			}
			continue
		}
		history = append(history, ownRecord(entry, ownNature(entry.Kind)))
		switch entry.Kind {
		case kindSessionCreated:
			payload, err := decodePayload[sessionCreatedPayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			created = true
			state.drainAll = payload.Options.DrainAll
			if payload.Scope != nil {
				if err := validateScope(payload.Scope.SessionID, *payload.Scope); err != nil {
					return foldedState{}, nil, fmt.Errorf("invalid persisted session scope: %w", err)
				}
				copy := *payload.Scope
				state.scope = &copy
			}
			if payload.Options.Budget != nil {
				copy := cloneLimits(*payload.Options.Budget)
				state.budget = &copy
			}
		case kindRunOpened:
			payload, err := decodePayload[runOpenedPayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			mode := agentic.DriveStart
			if payload.Mode == "continue" {
				mode = agentic.DriveContinue
			}
			state.run = &activeRun{
				id:                 payload.ID,
				mode:               mode,
				history:            providerHistory(state.messages, state.contextMarkers),
				contextMarkerCount: len(state.contextMarkers),
				limits:             cloneLimitsPointer(payload.Limits),
				instructions:       payload.Instructions,
			}
			state.state = Running
			state.suspension = nil
		case kindRunClosed:
			if state.run != nil && state.pendingInterruptRunID == state.run.id {
				state.pendingInterruptRunID = ""
			}
			state.run = nil
			state.state = Idle
			state.suspension = nil
		case kindMessage:
			payload, err := decodePayload[messagePayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			message := cloneMessages([]agentic.Message{payload.Message})[0]
			if payload.QueueID != "" {
				delete(drained, payload.QueueID)
			}
			state.messages = append(state.messages, message)
			if state.run != nil {
				if payload.Source == string(QueueNextTurn) {
					state.run.history = append(state.run.history, message)
				} else {
					state.run.expected = append(state.run.expected, message)
				}
			}
		case kindSystemMessage:
			payload, err := decodePayload[messagePayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			message := cloneMessages([]agentic.Message{payload.Message})[0]
			state.system = &message
			if len(state.messages) == 0 || state.messages[0].Role != agentic.RoleSystem {
				shiftContextMarkers(state.contextMarkers, 1)
				state.messages = append([]agentic.Message{message}, state.messages...)
			}
			if state.run != nil && (len(state.run.history) == 0 || state.run.history[0].Role != agentic.RoleSystem) {
				state.run.history = append([]agentic.Message{message}, state.run.history...)
			}
		case kindRepair:
			payload, err := decodePayload[messagePayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			message := cloneMessages([]agentic.Message{payload.Message})[0]
			state.messages = append(state.messages, message)
			if state.run != nil {
				state.run.expected = append(state.run.expected, message)
			}
		case kindQueueAccepted:
			payload, err := decodePayload[queueMutationPayload](payloadCodec, entry)
			if err != nil || payload.Entry == nil {
				if err == nil {
					err = errors.New("accepted queue entry has no payload")
				}
				return foldedState{}, nil, err
			}
			queue[payload.ID] = *payload.Entry
			order = append(order, payload.ID)
		case kindQueueDrained:
			payload, err := decodePayload[queueMutationPayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			if accepted, ok := queue[payload.ID]; ok {
				drained[payload.ID] = accepted
			}
			delete(queue, payload.ID)
		case kindQueueCancelled:
			payload, err := decodePayload[queueMutationPayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			delete(queue, payload.ID)
			delete(drained, payload.ID)
		case kindCommandAccepted:
			payload, err := decodePayload[loopCommandAcceptedPayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			if payload.Kind == "interrupt" {
				state.pendingInterruptRunID = payload.RunID
			}
		case kindUsageCommitted:
			payload, err := decodePayload[usagePayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			state.usage = cloneUsage(payload.Session)
			if state.run != nil {
				state.run.lastUsage = cloneUsage(payload.Run)
			}
		case kindChildUsage:
			payload, err := decodePayload[childUsagePayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			state.usage = cloneUsage(payload.Session)
			if state.run != nil {
				state.run.childUsageCharged = true
			}
		case kindRecoverySuspension:
			payload, err := decodePayload[event.SuspensionPayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			state.suspension = cloneSuspension(&payload.Suspension)
			state.state = Suspended
		case kindFault:
			state.state = Faulted
		case kindInterruptMarker:
			payload, err := decodePayload[interruptMarkerPayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			state.contextMarkers = append(state.contextMarkers, contextMarker{
				after:   len(state.messages),
				message: interruptionContextMessage(payload.Message),
			})
		case kindContextMessage:
			payload, err := decodePayload[contextMessagePayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			if payload.After < 0 || payload.After > len(state.messages) {
				return foldedState{}, nil, fmt.Errorf("%w: invalid context marker position", store.ErrCorruptLog)
			}
			state.contextMarkers = append(state.contextMarkers, contextMarker{
				after:   payload.After,
				message: cloneMessages([]agentic.Message{payload.Message})[0],
			})
		case kindCompaction:
			payload, err := decodePayload[compactionPayload](payloadCodec, entry)
			if err != nil {
				return foldedState{}, nil, err
			}
			state.compaction = cloneContextCompaction(&payload.Compaction)
		case kindBranchMoved, kindRecovered:
			// Preserved in the log and event history.
		}
	}
	if !created {
		return foldedState{}, nil, fmt.Errorf("%w: missing session-created entry", store.ErrCorruptLog)
	}
	for _, id := range order {
		if entry, ok := queue[id]; ok {
			state.queue = append(state.queue, entry)
		}
		if entry, ok := drained[id]; ok {
			state.unapplied = append(state.unapplied, entry)
		}
	}
	if state.system != nil && (len(state.messages) == 0 || state.messages[0].Role != agentic.RoleSystem) {
		shiftContextMarkers(state.contextMarkers, 1)
		state.messages = append([]agentic.Message{*state.system}, state.messages...)
	}
	return state, history, nil
}

func applyRecoveredEvent(payloadCodec codec.Codec, state *foldedState, record event.Record) error {
	switch record.Type {
	case agentic.EventTypeAssistantCommitted:
		payload, err := event.Decode[event.AssistantPayload](payloadCodec, record)
		if err != nil {
			return err
		}
		message := cloneMessages([]agentic.Message{payload.Message})[0]
		state.messages = append(state.messages, message)
		if state.run != nil {
			state.run.expected = append(state.run.expected, message)
		}
	case agentic.EventTypeToolBatchPlanned:
		payload, err := event.Decode[event.ToolBatchPayload](payloadCodec, record)
		if err != nil {
			return err
		}
		if state.run != nil {
			state.run.planned = append(state.run.planned, payload.Calls...)
		}
	case agentic.EventTypeToolStarted:
		payload, err := event.Decode[event.ToolStartedPayload](payloadCodec, record)
		if err != nil {
			return err
		}
		if state.run != nil {
			if state.run.started == nil {
				state.run.started = make(map[string]bool)
			}
			state.run.started[payload.Call.ID] = true
		}
	case agentic.EventTypeToolResultCommitted:
		payload, err := event.Decode[event.ToolResultPayload](payloadCodec, record)
		if err != nil {
			return err
		}
		message := agentic.NewToolResultMessageFor(payload.ToolUseID, payload.ToolName, payload.Content, payload.IsError)
		state.messages = append(state.messages, message)
		if state.run != nil {
			state.run.expected = append(state.run.expected, message)
			if state.run.results == nil {
				state.run.results = make(map[string]bool)
			}
			state.run.results[payload.ToolUseID] = true
		}
	case agentic.EventTypeTurnMessagesInjected:
		payload, err := event.Decode[event.MessagesPayload](payloadCodec, record)
		if err != nil {
			return err
		}
		messages := cloneMessages(payload.Messages)
		state.messages = append(state.messages, messages...)
		if state.run != nil {
			state.run.expected = append(state.run.expected, messages...)
			state.run.pendingInjectionIDs = nil
		}
	case agentic.EventTypeTurnEnded:
		payload, err := event.Decode[event.TurnEndedPayload](payloadCodec, record)
		if err != nil {
			return err
		}
		if record.SessionUsed != nil {
			state.usage = cloneUsage(*record.SessionUsed)
		}
		if state.run != nil {
			state.run.lastUsage = cloneUsage(payload.RunUsage)
		}
	case agentic.EventTypeRunStarted:
		if state.run != nil && state.state == Suspended {
			state.state = Running
			state.run.resumeEventSeen = true
		}
	case agentic.EventTypeRunSuspended:
		payload, err := event.Decode[event.SuspensionPayload](payloadCodec, record)
		if err != nil {
			return err
		}
		state.suspension = cloneSuspension(&payload.Suspension)
		state.state = Suspended
	}
	return nil
}

func (s *Session[O]) recoverOpenRun(ctx context.Context) error {
	s.mu.Lock()
	if len(s.recoveryInputs) > 0 {
		messages := make([]agentic.Message, len(s.recoveryInputs))
		ids := make([]string, len(s.recoveryInputs))
		for i, entry := range s.recoveryInputs {
			messages[i] = cloneMessages([]agentic.Message{entry.Message})[0]
			ids[i] = entry.ID
		}
		payloadBytes, encodeErr := codec.Encode(s.codec, event.MessagesPayload{Messages: messages, QueueIDs: ids})
		if encodeErr != nil {
			s.mu.Unlock()
			return encodeErr
		}
		record := event.Record{
			Nature:  agentic.EventAuthoritative,
			Type:    agentic.EventTypeTurnMessagesInjected,
			Source:  "harness.recovery",
			Payload: payloadBytes,
		}
		entry, encodeErr := pending(s.codec, kindMessagesInjected, record)
		if encodeErr != nil {
			s.mu.Unlock()
			return encodeErr
		}
		commit, appendErr := s.journal.Append(ctx, s.cursor, entry)
		if appendErr != nil {
			s.mu.Unlock()
			return appendErr
		}
		s.messages = append(s.messages, messages...)
		s.run.expected = append(s.run.expected, messages...)
		s.cursor = commit.Cursor
		s.recoveryInputs = nil
		record.Cursor = commit.Cursor.Seq
		s.bus.PublishDurable(s.scopedRecord(record))
	}
	open, err := repair.InspectFrontier(s.messages)
	if err != nil {
		s.mu.Unlock()
		return err
	}
	indeterminate := make([]repair.PendingCall, 0)
	pendingCalls := repair.PendingCalls{Calls: make([]repair.PendingCall, len(open))}
	for i, call := range open {
		state := repair.PendingPlanned
		if s.run.started[call.ID] && !s.run.results[call.ID] {
			state = repair.PendingIndeterminate
			indeterminate = append(indeterminate, repair.PendingCall{ID: call.ID, Name: call.Name, State: state})
		}
		pendingCalls.Calls[i] = repair.PendingCall{ID: call.ID, Name: call.Name, State: state}
	}
	if len(indeterminate) > 0 {
		// Suspension.Payload is a JSON wire contract in the released root API.
		// It is independent from the injected codec used by the journal.
		payloadBytes, marshalErr := json.Marshal(recoverySuspensionPayload{Version: 1, Calls: indeterminate})
		if marshalErr != nil {
			s.mu.Unlock()
			return marshalErr
		}
		suspensionID, idErr := s.ids.New("recovery")
		if idErr != nil {
			s.mu.Unlock()
			return idErr
		}
		frontierBytes, frontierErr := json.Marshal(struct {
			Messages []agentic.Message
			Calls    []agentic.ToolUse
		}{s.messages, open})
		if frontierErr != nil {
			s.mu.Unlock()
			return fmt.Errorf("encode indeterminate recovery frontier: %w", frontierErr)
		}
		digest := sha256.Sum256(frontierBytes)
		suspension := agentic.Suspension{
			ID:           suspensionID,
			Kind:         "harness.recovery.indeterminate",
			FrontierHash: "h1:" + hex.EncodeToString(digest[:]),
			Payload:      payloadBytes,
		}
		entry, encodeErr := pending(s.codec, kindRecoverySuspension, event.SuspensionPayload{Suspension: suspension})
		if encodeErr != nil {
			s.mu.Unlock()
			return encodeErr
		}
		commit, appendErr := s.journal.Append(ctx, s.cursor, entry)
		if appendErr != nil {
			s.mu.Unlock()
			return appendErr
		}
		s.suspension = &suspension
		s.cursor = commit.Cursor
		s.transitionLocked(Suspended)
		s.mu.Unlock()
		s.bus.PublishDurable(s.scopedRecord(ownRecord(commit.Entries[0], agentic.EventLifecycle)))
		return nil
	}

	repaired, repairErr := repair.Process(s.messages, repair.CloseInterruptedFrontier, pendingCalls)
	if repairErr != nil {
		s.mu.Unlock()
		return repairErr
	}
	if len(repaired) < len(s.messages) || !messagesEqual(repaired[:len(s.messages)], s.messages) {
		s.mu.Unlock()
		return fmt.Errorf("%w: recovery repair rewrote committed messages", ErrCommitProjectionMismatch)
	}
	added := repaired[len(s.messages):]
	oldRunID := s.run.id
	instructions := s.run.instructions
	newRunID, idErr := s.ids.New("run")
	if idErr != nil {
		s.mu.Unlock()
		return idErr
	}
	var limits *agentic.UsageLimits
	if s.budget != nil {
		remaining, remainingErr := remainingLimits(*s.budget, s.usage)
		if remainingErr != nil {
			s.mu.Unlock()
			return remainingErr
		}
		limits = &remaining
	}
	batch := newEntryBatch(s.codec, len(added)+3)
	batch.Add(kindRecovered, struct{ State string }{State: "continue"})
	batch.Add(kindRunClosed, runClosedPayload{ID: oldRunID, Status: agentic.ExecutionInterrupted, Error: "process stopped before run termination"})
	for _, message := range added {
		batch.Add(kindRepair, messagePayload{Message: message, Source: "recovery_repair"})
	}
	batch.Add(kindRunOpened, runOpenedPayload{
		ID:           newRunID,
		Mode:         "continue",
		Recovery:     true,
		Limits:       cloneLimitsPointer(limits),
		Instructions: instructions,
	})
	pendingEntries, encodeErr := batch.Result()
	if encodeErr != nil {
		s.mu.Unlock()
		return encodeErr
	}
	commit, appendErr := s.journal.Append(ctx, s.cursor, pendingEntries...)
	if appendErr != nil {
		s.mu.Unlock()
		return appendErr
	}
	s.messages = repaired
	s.cursor = commit.Cursor
	s.run = &activeRun{
		id:                 newRunID,
		mode:               agentic.DriveContinue,
		history:            providerHistory(repaired, s.contextMarkers),
		contextMarkerCount: len(s.contextMarkers),
		limits:             cloneLimitsPointer(limits),
		instructions:       instructions,
	}
	s.suspension = nil
	s.transitionLocked(Running)
	s.mu.Unlock()
	s.publishOwnByKind(commit.Entries)
	go s.continueRecovered()
	return nil
}

func (s *Session[O]) continueRecovered() {
	s.mu.Lock()
	if s.state != Running || s.run == nil {
		s.mu.Unlock()
		return
	}
	limits := cloneLimitsPointer(s.run.limits)
	if limits == nil && s.budget != nil {
		remaining, err := remainingLimits(*s.budget, s.usage)
		if err != nil {
			s.mu.Unlock()
			_, _ = s.finishExecution(nil, err)
			return
		}
		limits = &remaining
		s.run.limits = cloneLimitsPointer(limits)
	}
	history := cloneMessages(s.run.history)
	runCtx, cancel := context.WithCancel(context.Background())
	s.runCancel = cancel
	s.mu.Unlock()
	runCtx = s.withToolRuntime(runCtx)
	execution, err := s.driver.Drive(runCtx, agentic.DriveInput{Mode: agentic.DriveContinue, History: history}, s.runOptions(limits)...)
	_, _ = s.finishExecution(execution, err)
}

func isAgenticKind(kind string) bool {
	switch kind {
	case kindAssistantCommitted, kindToolBatchPlanned, kindToolStarted,
		kindToolResult, kindMessagesInjected, kindTurnStarted,
		kindTurnEnded, kindRunStarted, kindRunSuspended,
		kindRunCompleted, kindRunInterrupted, kindRunError,
		kindRunEnded, kindOutputValidated:
		return true
	default:
		return false
	}
}

func ownNature(kind string) agentic.EventNature {
	switch kind {
	case kindSessionCreated, kindRunOpened, kindRunClosed,
		kindInterruptMarker, kindRecoverySuspension, kindRecovered,
		kindFault:
		return agentic.EventLifecycle
	default:
		return agentic.EventAuthoritative
	}
}
