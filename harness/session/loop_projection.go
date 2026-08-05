package session

// This file is the read-only projection from the session's existing durable
// facts to provider-neutral sessionloop values. It never rewrites journal
// payloads and decodes strictly through the configured codec (agentic.*
// kinds double-decode: Entry.Payload -> event.Record -> Record.Payload).
//
// Deliberate mapping of EVERY persisted journal kind (S2 exit gate). Kinds
// marked IGNORED are internal facts with the stated reason; everything else
// maps to exactly the listed protocol event.
//
//	session.created             IGNORED: session assembly options/scope are host-internal configuration.
//	run.opened                  EventRunStarted (+State running; RunID from the payload).
//	run.closed                  EventRunSettled (Completed->completed, Interrupted->interrupted, Failed/Stopped/other->failed; Failure carries finishExecution's persisted error text VERBATIM — host-supplied operator-facing text whose wording is not a contract; Output attached only live when the host captured a projected output for that run — never during replay).
//	message                     EventEntryCommitted (Source prompt -> origin start; next_turn -> origin next_turn; initial_history -> origin start with empty RunID; recovery_resolution/resume_prompt -> origin start, run attribution from the surrounding batch's run.opened; system-role content is privacy-excluded).
//	message.system              IGNORED: driver/system instructions are privacy-excluded from the default projection (plan §8.5/§13).
//	queue.accepted              EventQueueAccepted (QueuedInput{ID, Kind, Position, Content}; the fold records ID->kind for later attribution).
//	queue.drained               EventQueueDrained (queue item reconstructed through the fold index).
//	queue.cancelled             EventQueueCancelled (queue item reconstructed through the fold index).
//	agentic.assistant_committed EventEntryCommitted (assistant entry: text parts -> text blocks; ToolUse parts -> tool_call blocks with codec-independent JSON input data; Thinking parts EXCLUDED entirely).
//	agentic.tool_batch_planned  IGNORED: planning is progress detail already visible through tool_result commits.
//	agentic.tool_started        IGNORED: start marks are recovery bookkeeping, not transcript truth.
//	agentic.tool_result         EventEntryCommitted (tool entry with one tool_result block; Data set only when the content is one valid JSON value).
//	agentic.messages_injected   EventEntryCommitted per message (user role; origin steer/follow_up through the fold's queue index).
//	agentic.turn_started        IGNORED: internal turn bookkeeping without protocol meaning.
//	agentic.turn_ended          IGNORED: usage reaches the protocol through usage.committed.
//	agentic.run_started         IGNORED: run identity is authoritative in run.opened.
//	agentic.run_suspended       EventRunSuspended (+State suspended; safe Suspension via the installed suspension projector).
//	agentic.run_completed       IGNORED: settlement is authoritative in run.closed.
//	agentic.run_interrupted     IGNORED: settlement is authoritative in run.closed.
//	agentic.run_error           IGNORED: the sanitized failure is authoritative in run.closed.
//	agentic.run_ended           IGNORED: settlement is authoritative in run.closed.
//	agentic.output_validated    IGNORED: raw candidate output is application-owned; structured output requires the explicit projector.
//	usage.committed             EventUsage (session usage projected to sessionloop.Usage).
//	transcript.repair           IGNORED: synthetic frontier-closure messages are recovery bookkeeping, not authored transcript.
//	interrupt.marker            IGNORED: the marker is provider-context plumbing; interruption is visible as the interrupted settlement.
//	context.durable             IGNORED: injected context is privacy-excluded system material (plan §8.5/§13).
//	resolution.accepted         EventCommandAccepted (kind resolve).
//	recovery.suspension         EventRunSuspended (+State suspended; suspension kind harness.recovery.indeterminate).
//	session.recovered           EventSessionState (payload state idle -> idle, continue -> running).
//	session.fault               EventSessionState (faulted).
//	transcript.compaction       Namespaced extension kind "agentic.transcript.compaction" with empty standard payloads, so snapshot discontinuities stay observable (plan §7.8).
//	subagent.event              IGNORED: child-session records belong to the child session's own projection.
//	subagent.usage              EventUsage (parent session usage projected to sessionloop.Usage).
//	branch.moved                IGNORED: reserved kind with no writer; preserved in the log only.
//	runtime.operation           IGNORED: opaque capability operation facts are host-internal.
//
// Preview records (Nature == EventPreview) project to EventPreviewDelta.
// Their Ordinal is a stream-local monotonic arrival counter (never the
// session's per-turn ordinal) and their Position repeats the latest durable
// position seen by the stream (plan §7.6); unknown preview shapes are lossy
// by contract and are dropped. Authoritative events carry Position
// {Sequence: record cursor, Token: ""}; receipts (not events) carry the
// opaque store entry ID as Token.

import (
	"context"
	"encoding/json"
	"errors"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/codec"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
)

var errQueueAcceptedWithoutEntry = errors.New("accepted queue entry has no payload")

// SuspensionProjector converts one durable suspension into the safe,
// display-only protocol view. Implementations must never expose opaque
// provider payloads.
type SuspensionProjector func(agentic.Suspension) (sessionloop.Suspension, error)

// defaultSuspensionProjector is the package-level fallback: identity and a
// generic fixed description, with typed decisions only when the released
// root deferral envelope parses. Unknown suspension kinds yield empty
// decisions and no error.
func defaultSuspensionProjector(value agentic.Suspension) (sessionloop.Suspension, error) {
	result := sessionloop.Suspension{
		ID:          value.ID,
		Kind:        value.Kind,
		Description: "The run is durably paused awaiting operator resolution.",
	}
	frontier, err := harnessruntime.InspectDeferred(value)
	if err != nil {
		return result, nil
	}
	names := make(map[string]string, len(frontier.Calls))
	for _, call := range frontier.Calls {
		names[call.ID] = call.Name
	}
	for _, id := range frontier.RequiredResolutionIDs {
		result.Decisions = append(result.Decisions, sessionloop.SuspensionDecision{ID: id, Name: names[id]})
	}
	return result, nil
}

// loopProjector converts ordered event records into protocol events over an
// incremental fold. The three attribution hooks and outputFor are installed
// by the live LoopView; a pure replay leaves them nil and yields empty
// command IDs and no structured output — replayed history cannot invent
// live-host attribution.
type loopProjector struct {
	sessionID string
	codec     codec.Codec
	suspend   SuspensionProjector

	commandForRun        func(runID string) sessionloop.CommandID
	commandForQueue      func(queueID string) sessionloop.CommandID
	commandForResolution func(seq uint64) sessionloop.CommandID
	outputFor            func(ctx context.Context, runID string) json.RawMessage

	fold loopFold
}

func newLoopProjector(sessionID string, payloadCodec codec.Codec, suspend SuspensionProjector) *loopProjector {
	if suspend == nil {
		suspend = defaultSuspensionProjector
	}
	return &loopProjector{sessionID: sessionID, codec: payloadCodec, suspend: suspend, fold: newLoopFold()}
}

func (p *loopProjector) runCommand(runID string) sessionloop.CommandID {
	if p.commandForRun == nil || runID == "" {
		return ""
	}
	return p.commandForRun(runID)
}

func (p *loopProjector) queueCommand(queueID string) sessionloop.CommandID {
	if p.commandForQueue == nil || queueID == "" {
		return ""
	}
	return p.commandForQueue(queueID)
}

// apply projects one record into zero or more protocol events, advancing the
// fold. ctx bounds live-only waits (structured-output capture); replay
// callers pass a background context.
func (p *loopProjector) apply(ctx context.Context, record event.Record) ([]sessionloop.Event, error) {
	p.fold.pendingDropped += record.Dropped.Preview
	if record.SessionID != "" && record.SessionID != p.sessionID {
		// Child-session records (the live counterpart of subagent.event).
		return nil, nil
	}
	if record.Nature == agentic.EventPreview {
		return p.deliver(p.applyPreview(record))
	}
	if record.Cursor > p.fold.lastDurable {
		p.fold.lastDurable = record.Cursor
	}
	switch record.Source {
	case "agentic", "harness.recovery":
		events, err := p.applyAgentic(ctx, record)
		if err != nil {
			return nil, err
		}
		return p.deliver(events)
	case "harness":
		events, err := p.applyHarness(ctx, record)
		if err != nil {
			return nil, err
		}
		return p.deliver(events)
	default:
		return nil, nil
	}
}

// deliver stamps accumulated preview-loss counts on the first emitted event.
func (p *loopProjector) deliver(events []sessionloop.Event) ([]sessionloop.Event, error) {
	if len(events) == 0 {
		return nil, nil
	}
	if p.fold.pendingDropped > 0 {
		events[0].Dropped = p.fold.pendingDropped
		p.fold.pendingDropped = 0
	}
	return events, nil
}

func (p *loopProjector) applyPreview(record event.Record) []sessionloop.Event {
	preview, ok := p.previewPayload(record)
	if !ok {
		return nil
	}
	p.fold.previewOrdinal++
	position := p.fold.lastDurable
	if position == 0 {
		position = record.Cursor
	}
	return []sessionloop.Event{{
		Position:  sessionloop.Position{Sequence: position},
		Ordinal:   p.fold.previewOrdinal,
		Nature:    sessionloop.EventPreview,
		Kind:      sessionloop.EventPreviewDelta,
		SessionID: sessionloop.SessionID(p.sessionID),
		RunID:     sessionloop.RunID(p.fold.currentRunID),
		Preview:   &preview,
	}}
}

func (p *loopProjector) previewPayload(record event.Record) (sessionloop.Preview, bool) {
	if record.Source == "tool" {
		// Tool-update previews keep their identity: Text carries the
		// capability-owned update kind label (record.Name, bounded by the
		// harness before publication). The ToolUpdate payload itself stays
		// opaque and is never projected, and no durable call identity exists
		// on tool-update records, so ToolCallID stays empty unless a future
		// record shape carries one.
		return sessionloop.Preview{Kind: sessionloop.PreviewTool, Text: record.Name}, true
	}
	switch record.Type {
	case agentic.EventTypeTextPreview:
		payload, err := event.Decode[struct{ Delta string }](p.codec, record)
		if err != nil {
			return sessionloop.Preview{}, false
		}
		return sessionloop.Preview{Kind: sessionloop.PreviewText, Text: payload.Delta}, true
	case agentic.EventTypeThinkingPreview:
		payload, err := event.Decode[struct{ Delta string }](p.codec, record)
		if err != nil {
			return sessionloop.Preview{}, false
		}
		return sessionloop.Preview{Kind: sessionloop.PreviewThinking, Text: payload.Delta}, true
	case agentic.EventTypeToolCallPreview:
		payload, err := event.Decode[event.ToolBatchPayload](p.codec, record)
		if err != nil || len(payload.Calls) == 0 {
			return sessionloop.Preview{}, false
		}
		return sessionloop.Preview{Kind: sessionloop.PreviewTool, ToolCallID: payload.Calls[0].ID}, true
	case agentic.EventTypeToolArgumentPreview:
		payload, err := event.Decode[struct {
			ToolCallID string
			Delta      string
		}](p.codec, record)
		if err != nil {
			return sessionloop.Preview{}, false
		}
		return sessionloop.Preview{Kind: sessionloop.PreviewTool, Text: payload.Delta, ToolCallID: payload.ToolCallID}, true
	default:
		// Previews are lossy by contract; unknown shapes are dropped.
		return sessionloop.Preview{}, false
	}
}

func (p *loopProjector) applyAgentic(_ context.Context, record event.Record) ([]sessionloop.Event, error) {
	switch record.Type {
	case agentic.EventTypeAssistantCommitted:
		payload, err := event.Decode[event.AssistantPayload](p.codec, record)
		if err != nil {
			return nil, err
		}
		entry := p.entry(record.Cursor, 0, sessionloop.RoleAssistant, sessionloop.OriginAssistant,
			p.fold.currentRunID, p.runCommand(p.fold.currentRunID), loopBlocks(payload.Message))
		return []sessionloop.Event{p.entryEvent(record.Cursor, entry)}, nil
	case agentic.EventTypeToolResultCommitted:
		payload, err := event.Decode[event.ToolResultPayload](p.codec, record)
		if err != nil {
			return nil, err
		}
		block := sessionloop.Block{
			Kind: sessionloop.BlockToolResult,
			Text: payload.Content,
			ToolResult: &sessionloop.ToolResult{
				CallID:  payload.ToolUseID,
				Name:    payload.ToolName,
				IsError: payload.IsError,
			},
		}
		if content := []byte(payload.Content); len(content) > 0 && json.Valid(content) {
			block.Data = json.RawMessage(append([]byte(nil), content...))
		}
		entry := p.entry(record.Cursor, 0, sessionloop.RoleTool, sessionloop.OriginTool,
			p.fold.currentRunID, p.runCommand(p.fold.currentRunID), []sessionloop.Block{block})
		return []sessionloop.Event{p.entryEvent(record.Cursor, entry)}, nil
	case agentic.EventTypeTurnMessagesInjected:
		payload, err := event.Decode[event.MessagesPayload](p.codec, record)
		if err != nil {
			return nil, err
		}
		events := make([]sessionloop.Event, 0, len(payload.Messages))
		for index, message := range payload.Messages {
			queueID := ""
			if index < len(payload.QueueIDs) {
				queueID = payload.QueueIDs[index]
			}
			entry := p.entry(record.Cursor, index, sessionloop.RoleUser, p.fold.injectedOrigin(queueID),
				p.fold.currentRunID, p.queueCommand(queueID), loopBlocks(message))
			events = append(events, p.entryEvent(record.Cursor, entry))
		}
		return events, nil
	case agentic.EventTypeRunSuspended:
		payload, err := event.Decode[event.SuspensionPayload](p.codec, record)
		if err != nil {
			return nil, err
		}
		return p.suspendedEvent(record.Cursor, payload.Suspension)
	default:
		// Every remaining agentic.* kind is documented-ignored in the table.
		return nil, nil
	}
}

func (p *loopProjector) applyHarness(ctx context.Context, record event.Record) ([]sessionloop.Event, error) {
	switch record.Name {
	case kindRunOpened:
		payload, err := codec.Decode[runOpenedPayload](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		p.fold.currentRunID = payload.ID
		events := make([]sessionloop.Event, 0, 1+len(p.fold.pendingRunEntries))
		for _, pending := range p.fold.pendingRunEntries {
			entry := pending
			entry.RunID = sessionloop.RunID(payload.ID)
			entry.CommandID = p.runCommand(payload.ID)
			events = append(events, p.entryEvent(entry.Position.Sequence, entry))
		}
		p.fold.pendingRunEntries = nil
		events = append(events, sessionloop.Event{
			Position:  sessionloop.Position{Sequence: record.Cursor},
			Nature:    sessionloop.EventAuthoritative,
			Kind:      sessionloop.EventRunStarted,
			SessionID: sessionloop.SessionID(p.sessionID),
			RunID:     sessionloop.RunID(payload.ID),
			CommandID: p.runCommand(payload.ID),
			State:     sessionloop.StateRunning,
		})
		return events, nil
	case kindRunClosed:
		payload, err := codec.Decode[runClosedPayload](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		p.fold.currentRunID = ""
		outcome := sessionloop.RunOutcome{
			RunID:   sessionloop.RunID(payload.ID),
			Kind:    loopOutcomeKind(payload.Status),
			Failure: payload.Error,
		}
		if outcome.Kind == sessionloop.RunCompleted && p.outputFor != nil {
			outcome.Output = p.outputFor(ctx, payload.ID)
		}
		return []sessionloop.Event{{
			Position:  sessionloop.Position{Sequence: record.Cursor},
			Nature:    sessionloop.EventAuthoritative,
			Kind:      sessionloop.EventRunSettled,
			SessionID: sessionloop.SessionID(p.sessionID),
			RunID:     sessionloop.RunID(payload.ID),
			CommandID: p.runCommand(payload.ID),
			Outcome:   &outcome,
		}}, nil
	case kindMessage:
		payload, err := codec.Decode[messagePayload](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		return p.messageEvents(record.Cursor, payload)
	case kindQueueAccepted:
		payload, err := codec.Decode[queueMutationPayload](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		if payload.Entry == nil {
			return nil, errQueueAcceptedWithoutEntry
		}
		position := sessionloop.Position{Sequence: record.Cursor}
		p.fold.rememberQueue(payload.ID, payload.Entry.Kind, loopBlocks(payload.Entry.Message), position)
		return p.queueEvent(sessionloop.EventQueueAccepted, record.Cursor, payload.ID), nil
	case kindQueueDrained:
		payload, err := codec.Decode[queueMutationPayload](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		return p.queueEvent(sessionloop.EventQueueDrained, record.Cursor, payload.ID), nil
	case kindQueueCancelled:
		payload, err := codec.Decode[queueMutationPayload](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		return p.queueEvent(sessionloop.EventQueueCancelled, record.Cursor, payload.ID), nil
	case kindUsageCommitted:
		payload, err := codec.Decode[usagePayload](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		usage := loopUsage(payload.Session)
		return []sessionloop.Event{{
			Position:  sessionloop.Position{Sequence: record.Cursor},
			Nature:    sessionloop.EventAuthoritative,
			Kind:      sessionloop.EventUsage,
			SessionID: sessionloop.SessionID(p.sessionID),
			RunID:     sessionloop.RunID(p.fold.currentRunID),
			Usage:     &usage,
		}}, nil
	case kindChildUsage:
		payload, err := codec.Decode[childUsagePayload](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		usage := loopUsage(payload.Session)
		return []sessionloop.Event{{
			Position:  sessionloop.Position{Sequence: record.Cursor},
			Nature:    sessionloop.EventAuthoritative,
			Kind:      sessionloop.EventUsage,
			SessionID: sessionloop.SessionID(p.sessionID),
			RunID:     sessionloop.RunID(p.fold.currentRunID),
			Usage:     &usage,
		}}, nil
	case kindResolutionAccepted:
		if _, err := codec.Decode[resolutionAcceptedPayload](p.codec, record.Payload); err != nil {
			return nil, err
		}
		commandID := sessionloop.CommandID("")
		if p.commandForResolution != nil {
			commandID = p.commandForResolution(record.Cursor)
		}
		return []sessionloop.Event{{
			Position:  sessionloop.Position{Sequence: record.Cursor},
			Nature:    sessionloop.EventAuthoritative,
			Kind:      sessionloop.EventCommandAccepted,
			SessionID: sessionloop.SessionID(p.sessionID),
			RunID:     sessionloop.RunID(p.fold.currentRunID),
			CommandID: commandID,
		}}, nil
	case kindRecoverySuspension:
		payload, err := codec.Decode[event.SuspensionPayload](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		return p.suspendedEvent(record.Cursor, payload.Suspension)
	case kindRecovered:
		payload, err := codec.Decode[struct{ State string }](p.codec, record.Payload)
		if err != nil {
			return nil, err
		}
		state := sessionloop.StateRunning
		if payload.State == Idle.String() {
			state = sessionloop.StateIdle
		}
		return []sessionloop.Event{p.stateEvent(record.Cursor, state)}, nil
	case kindFault:
		if _, err := codec.Decode[struct{ Error string }](p.codec, record.Payload); err != nil {
			return nil, err
		}
		return []sessionloop.Event{p.stateEvent(record.Cursor, sessionloop.StateFaulted)}, nil
	case kindCompaction:
		return []sessionloop.Event{{
			Position:  sessionloop.Position{Sequence: record.Cursor},
			Nature:    sessionloop.EventAuthoritative,
			Kind:      sessionloop.EventKind("agentic.transcript.compaction"),
			SessionID: sessionloop.SessionID(p.sessionID),
			RunID:     sessionloop.RunID(p.fold.currentRunID),
		}}, nil
	default:
		// Every remaining harness kind is documented-ignored in the table.
		return nil, nil
	}
}

func (p *loopProjector) messageEvents(seq uint64, payload messagePayload) ([]sessionloop.Event, error) {
	if payload.Message.Role == agentic.RoleSystem {
		// System content stays privacy-excluded regardless of source.
		return nil, nil
	}
	role := loopRole(payload.Message.Role)
	blocks := loopBlocks(payload.Message)
	switch payload.Source {
	case "prompt":
		entry := p.entry(seq, 0, role, sessionloop.OriginStart, p.fold.currentRunID,
			p.runCommand(p.fold.currentRunID), blocks)
		return []sessionloop.Event{p.entryEvent(seq, entry)}, nil
	case string(QueueNextTurn):
		entry := p.entry(seq, 0, role, sessionloop.OriginNextTurn, p.fold.currentRunID,
			p.queueCommand(payload.QueueID), blocks)
		return []sessionloop.Event{p.entryEvent(seq, entry)}, nil
	case "initial_history":
		entry := p.entry(seq, 0, role, sessionloop.OriginStart, "", "", blocks)
		return []sessionloop.Event{p.entryEvent(seq, entry)}, nil
	case "recovery_resolution", "resume_prompt":
		// Run attribution comes from the surrounding batch's run.opened.
		p.fold.pendingRunEntries = append(p.fold.pendingRunEntries,
			p.entry(seq, 0, role, sessionloop.OriginStart, "", "", blocks))
		return nil, nil
	default:
		entry := p.entry(seq, 0, role, sessionloop.OriginStart, p.fold.currentRunID,
			p.runCommand(p.fold.currentRunID), blocks)
		return []sessionloop.Event{p.entryEvent(seq, entry)}, nil
	}
}

func (p *loopProjector) suspendedEvent(seq uint64, suspension agentic.Suspension) ([]sessionloop.Event, error) {
	safe, err := p.suspend(suspension)
	if err != nil {
		return nil, err
	}
	return []sessionloop.Event{{
		Position:   sessionloop.Position{Sequence: seq},
		Nature:     sessionloop.EventAuthoritative,
		Kind:       sessionloop.EventRunSuspended,
		SessionID:  sessionloop.SessionID(p.sessionID),
		RunID:      sessionloop.RunID(p.fold.currentRunID),
		CommandID:  p.runCommand(p.fold.currentRunID),
		State:      sessionloop.StateSuspended,
		Suspension: &safe,
	}}, nil
}

func (p *loopProjector) queueEvent(kind sessionloop.EventKind, seq uint64, queueID string) []sessionloop.Event {
	queued := p.fold.queuedInput(queueID)
	queued.CommandID = p.queueCommand(queueID)
	return []sessionloop.Event{{
		Position:  sessionloop.Position{Sequence: seq},
		Nature:    sessionloop.EventAuthoritative,
		Kind:      kind,
		SessionID: sessionloop.SessionID(p.sessionID),
		RunID:     sessionloop.RunID(p.fold.currentRunID),
		CommandID: queued.CommandID,
		Queue:     &queued,
	}}
}

func (p *loopProjector) stateEvent(seq uint64, state sessionloop.State) sessionloop.Event {
	return sessionloop.Event{
		Position:  sessionloop.Position{Sequence: seq},
		Nature:    sessionloop.EventAuthoritative,
		Kind:      sessionloop.EventSessionState,
		SessionID: sessionloop.SessionID(p.sessionID),
		State:     state,
	}
}

func (p *loopProjector) entry(
	seq uint64,
	index int,
	role sessionloop.Role,
	origin sessionloop.EntryOrigin,
	runID string,
	commandID sessionloop.CommandID,
	blocks []sessionloop.Block,
) sessionloop.Entry {
	return sessionloop.Entry{
		ID:        loopEntryID(seq, index),
		SessionID: sessionloop.SessionID(p.sessionID),
		RunID:     sessionloop.RunID(runID),
		CommandID: commandID,
		Position:  sessionloop.Position{Sequence: seq},
		Role:      role,
		Origin:    origin,
		Content:   blocks,
	}
}

func (p *loopProjector) entryEvent(seq uint64, entry sessionloop.Entry) sessionloop.Event {
	return sessionloop.Event{
		Position:  sessionloop.Position{Sequence: seq},
		Nature:    sessionloop.EventAuthoritative,
		Kind:      sessionloop.EventEntryCommitted,
		SessionID: sessionloop.SessionID(p.sessionID),
		RunID:     entry.RunID,
		CommandID: entry.CommandID,
		Entry:     &entry,
	}
}

// loopRecords converts loaded journal entries into the same ordered record
// shapes the live event hub publishes, so replay and live streams share one
// projection. agentic.* kinds double-decode through the configured codec;
// child entries are skipped because the projection ignores them.
func loopRecords(payloadCodec codec.Codec, entries []store.Entry) ([]event.Record, error) {
	records := make([]event.Record, 0, len(entries))
	for _, entry := range entries {
		if entry.Kind == kindChildEvent {
			continue
		}
		if isAgenticKind(entry.Kind) {
			record, err := codec.Decode[event.Record](payloadCodec, entry.Payload)
			if err != nil {
				return nil, err
			}
			record.Cursor = entry.Seq
			records = append(records, record)
			continue
		}
		records = append(records, ownRecord(entry, ownNature(entry.Kind)))
	}
	return records, nil
}

func loopRole(role agentic.MessageRole) sessionloop.Role {
	switch role {
	case agentic.RoleAssistant:
		return sessionloop.RoleAssistant
	case agentic.RoleTool:
		return sessionloop.RoleTool
	default:
		return sessionloop.RoleUser
	}
}

// loopBlocks projects one message into protocol blocks: text parts become
// text blocks, tool uses become tool_call blocks with codec-independent JSON
// input data (a protocol projection, not a journal rewrite), tool results
// become tool_result blocks. Thinking and media parts are excluded from the
// default projection.
func loopBlocks(message agentic.Message) []sessionloop.Block {
	var blocks []sessionloop.Block
	for _, part := range message.Content {
		switch part.Type {
		case agentic.ContentText:
			if part.Text != "" {
				blocks = append(blocks, sessionloop.Block{Kind: sessionloop.BlockText, Text: part.Text})
			}
		case agentic.ContentToolUse:
			if part.ToolUse == nil {
				continue
			}
			block := sessionloop.Block{
				Kind: sessionloop.BlockToolCall,
				ToolCall: &sessionloop.ToolCall{
					CallID: part.ToolUse.ID,
					Name:   part.ToolUse.Name,
				},
			}
			if data, err := json.Marshal(part.ToolUse.Input); err == nil {
				block.ToolCall.Data = data
			}
			blocks = append(blocks, block)
		case agentic.ContentToolResult:
			if part.ToolResult == nil {
				continue
			}
			block := sessionloop.Block{
				Kind: sessionloop.BlockToolResult,
				Text: part.ToolResult.Content,
				ToolResult: &sessionloop.ToolResult{
					CallID:  part.ToolResult.ToolUseID,
					Name:    part.ToolResult.Name,
					IsError: part.ToolResult.IsError,
				},
			}
			if content := []byte(part.ToolResult.Content); len(content) > 0 && json.Valid(content) {
				block.Data = json.RawMessage(append([]byte(nil), content...))
			}
			blocks = append(blocks, block)
		}
	}
	return blocks
}

func loopOutcomeKind(status agentic.ExecutionStatus) sessionloop.RunOutcomeKind {
	switch status {
	case agentic.ExecutionCompleted:
		return sessionloop.RunCompleted
	case agentic.ExecutionInterrupted:
		return sessionloop.RunInterrupted
	default:
		// Failed, Stopped, and anything unknown settle as failed.
		return sessionloop.RunFailed
	}
}

func loopUsage(usage agentic.Usage) sessionloop.Usage {
	return sessionloop.Usage{
		PromptTokens:        int64(usage.PromptTokens),
		CompletionTokens:    int64(usage.CompletionTokens),
		TotalTokens:         int64(usage.TotalTokens),
		CacheReadTokens:     int64(usage.CacheReadTokens),
		CacheCreationTokens: int64(usage.CacheCreationTokens),
		ReasoningTokens:     int64(usage.ReasoningTokens),
		Requests:            int64(usage.Requests),
		ToolCalls:           int64(usage.ToolCalls),
	}
}
