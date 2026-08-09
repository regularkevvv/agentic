package testkit

import (
	"context"
	"fmt"
	"sync"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// sessionState is the durable half of a session. It outlives handles: Close
// releases the handle while the log, entries, queue, and usage survive for a
// later OpenSession. All fields behind mu form the single-flight actor state.
type sessionState struct {
	host *Host
	id   sessionloop.SessionID
	meta map[string]string

	mu         sync.Mutex
	handleOpen bool
	closing    bool
	seq        uint64
	log        []sessionloop.Event
	entries    []sessionloop.Entry
	pending    []sessionloop.QueuedInput
	state      sessionloop.State
	usage      sessionloop.Usage
	run        *activeRun
	subs       map[*stream]struct{}
	// keys records idempotency-key -> original receipt when the host opted
	// into WithIdempotentDispatch. It lives on the durable session state, so
	// recorded keys survive handle close and reopen.
	keys map[string]sessionloop.Receipt
}

// session is one exclusive handle onto a sessionState.
type session struct {
	state  *sessionState
	closed bool // guarded by state.mu
}

func (s *session) ID() sessionloop.SessionID { return s.state.id }

func (s *session) Capabilities() sessionloop.Capabilities { return s.state.host.capabilities() }

// Dispatch validates, accepts, and durably records one command. The context
// governs acceptance only: once the receipt is returned the run belongs to
// the session (law L4). Under WithIdempotentDispatch, a repeated key returns
// the original receipt without re-executing the command.
func (s *session) Dispatch(ctx context.Context, command sessionloop.Command) (sessionloop.Receipt, error) {
	if err := ctx.Err(); err != nil {
		return sessionloop.Receipt{}, err
	}
	if err := sessionloop.ValidateCommand(command, s.state.host.capabilities()); err != nil {
		return sessionloop.Receipt{}, err
	}
	if command.Input != nil {
		if err := sessionloop.ValidateInput(*command.Input); err != nil {
			return sessionloop.Receipt{}, err
		}
	}
	state := s.state
	state.mu.Lock()
	defer state.mu.Unlock()
	if s.closed {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: dispatch on a closed handle: %w", sessionloop.ErrSessionClosed)
	}
	if command.IdempotencyKey != "" && state.host.idempotent {
		if receipt, ok := state.keys[command.IdempotencyKey]; ok {
			return receipt, nil
		}
	}
	command = command.Clone()
	if command.ID == "" {
		command.ID = sessionloop.CommandID(state.host.nextID("cmd"))
	}
	var receipt sessionloop.Receipt
	var err error
	switch command.Kind {
	case sessionloop.CommandStart:
		receipt, err = state.acceptStartLocked(command)
	case sessionloop.CommandNextTurn:
		receipt, err = state.acceptNextTurnLocked(command)
	case sessionloop.CommandSteer, sessionloop.CommandFollowUp:
		receipt, err = state.acceptDeliveryLocked(command)
	case sessionloop.CommandResolve:
		receipt, err = state.acceptResolveLocked(command)
	case sessionloop.CommandInterrupt:
		receipt, err = state.acceptInterruptLocked(command)
	default:
		// Unreachable: ValidateCommand rejects unknown kinds.
		return sessionloop.Receipt{}, fmt.Errorf("testkit: unhandled command kind %q: %w", command.Kind, sessionloop.ErrInvalidCommand)
	}
	if err != nil {
		return sessionloop.Receipt{}, err
	}
	if command.IdempotencyKey != "" && state.host.idempotent {
		if state.keys == nil {
			state.keys = make(map[string]sessionloop.Receipt)
		}
		state.keys[command.IdempotencyKey] = receipt
	}
	return receipt, nil
}

func (st *sessionState) acceptStartLocked(command sessionloop.Command) (sessionloop.Receipt, error) {
	switch st.state {
	case sessionloop.StateIdle:
	case sessionloop.StateRunning, sessionloop.StateSuspended, sessionloop.StateInterrupting:
		return sessionloop.Receipt{}, fmt.Errorf("testkit: run %q is still active: %w", st.run.id, sessionloop.ErrSessionBusy)
	case sessionloop.StateFaulted:
		return sessionloop.Receipt{}, fmt.Errorf("testkit: session faulted: %w", sessionloop.ErrSessionFaulted)
	default:
		return sessionloop.Receipt{}, fmt.Errorf("testkit: session is %s: %w", st.state, sessionloop.ErrSessionClosed)
	}

	runID := sessionloop.RunID(st.host.nextID("run"))
	st.appendLocked(sessionloop.Event{Kind: sessionloop.EventCommandAccepted, RunID: runID, CommandID: command.ID})
	st.state = sessionloop.StateRunning
	st.appendLocked(sessionloop.Event{Kind: sessionloop.EventRunStarted, RunID: runID, CommandID: command.ID})
	for _, queued := range st.pending {
		queuedCopy := queued.Clone()
		st.appendLocked(sessionloop.Event{Kind: sessionloop.EventQueueDrained, RunID: runID, CommandID: queued.CommandID, Queue: &queuedCopy})
		st.commitEntryLocked(sessionloop.RoleUser, sessionloop.OriginNextTurn, runID, queued.CommandID, inputBlocksToEntryBlocks(queued.Blocks))
	}
	st.pending = nil
	st.commitEntryLocked(sessionloop.RoleUser, sessionloop.OriginStart, runID, command.ID, inputBlocksToEntryBlocks(command.Input.Blocks))
	position := st.appendLocked(sessionloop.Event{Kind: sessionloop.EventSessionState, RunID: runID, CommandID: command.ID})

	run := newActiveRun(runID, command.ID, *command.Input, st.host.takeHold())
	st.run = run
	go st.drive(run)

	return sessionloop.Receipt{
		CommandID: command.ID,
		SessionID: st.id,
		RunID:     runID,
		Position:  position,
		Guarantee: sessionloop.AcceptanceDurable,
	}, nil
}

func (st *sessionState) acceptNextTurnLocked(command sessionloop.Command) (sessionloop.Receipt, error) {
	if st.state == sessionloop.StateFaulted {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: session faulted: %w", sessionloop.ErrSessionFaulted)
	}
	st.appendLocked(sessionloop.Event{Kind: sessionloop.EventCommandAccepted, CommandID: command.ID})
	queued := sessionloop.QueuedInput{
		ID:        sessionloop.QueueID(st.host.nextID("queue")),
		Kind:      sessionloop.CommandNextTurn,
		CommandID: command.ID,
		Position:  st.nextPositionLocked(),
		Blocks:    command.Input.Blocks,
	}
	queuedCopy := queued.Clone()
	position := st.appendLocked(sessionloop.Event{Kind: sessionloop.EventQueueAccepted, CommandID: command.ID, Queue: &queuedCopy})
	st.pending = append(st.pending, queued)
	return sessionloop.Receipt{
		CommandID: command.ID,
		SessionID: st.id,
		QueueID:   queued.ID,
		Position:  position,
		Guarantee: sessionloop.AcceptanceDurable,
	}, nil
}

func (st *sessionState) acceptDeliveryLocked(command sessionloop.Command) (sessionloop.Receipt, error) {
	run := st.run
	if run == nil {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: %s requires an active run: %w", command.Kind, sessionloop.ErrNotRunning)
	}
	if command.RunID != run.id {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: %s targets run %q but run %q is active: %w", command.Kind, command.RunID, run.id, sessionloop.ErrStaleRun)
	}
	switch st.state {
	case sessionloop.StateSuspended:
		return sessionloop.Receipt{}, fmt.Errorf("testkit: run %q is suspended: %w", run.id, sessionloop.ErrSuspended)
	case sessionloop.StateRunning:
	default:
		return sessionloop.Receipt{}, fmt.Errorf("testkit: run %q is %s: %w", run.id, st.state, sessionloop.ErrCommandConflict)
	}

	origin := sessionloop.OriginSteer
	if command.Kind == sessionloop.CommandFollowUp {
		origin = sessionloop.OriginFollowUp
	}
	st.appendLocked(sessionloop.Event{Kind: sessionloop.EventCommandAccepted, RunID: run.id, CommandID: command.ID})
	position := st.commitEntryLocked(sessionloop.RoleUser, origin, run.id, command.ID, inputBlocksToEntryBlocks(command.Input.Blocks))
	run.inputs.push(command.Input.Clone())
	return sessionloop.Receipt{
		CommandID: command.ID,
		SessionID: st.id,
		RunID:     run.id,
		Position:  position,
		Guarantee: sessionloop.AcceptanceDurable,
	}, nil
}

func (st *sessionState) acceptResolveLocked(command sessionloop.Command) (sessionloop.Receipt, error) {
	run := st.run
	if run == nil {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: resolve requires an active run: %w", sessionloop.ErrNotRunning)
	}
	if command.RunID != run.id {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: resolve targets run %q but run %q is active: %w", command.RunID, run.id, sessionloop.ErrStaleRun)
	}
	if st.state != sessionloop.StateSuspended || run.suspension == nil {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: run %q is not suspended: %w", run.id, sessionloop.ErrCommandConflict)
	}
	if command.Resolution.SuspensionID != run.suspension.ID {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: resolution targets suspension %q but %q is pending: %w", command.Resolution.SuspensionID, run.suspension.ID, sessionloop.ErrInvalidCommand)
	}

	st.appendLocked(sessionloop.Event{Kind: sessionloop.EventCommandAccepted, RunID: run.id, CommandID: command.ID})
	if command.Input != nil {
		st.commitEntryLocked(sessionloop.RoleUser, sessionloop.OriginFollowUp, run.id, command.ID, inputBlocksToEntryBlocks(command.Input.Blocks))
	}
	run.suspension = nil
	st.state = sessionloop.StateRunning
	position := st.appendLocked(sessionloop.Event{Kind: sessionloop.EventSessionState, RunID: run.id, CommandID: command.ID})
	run.resolutions <- command.Resolution.Clone()
	return sessionloop.Receipt{
		CommandID: command.ID,
		SessionID: st.id,
		RunID:     run.id,
		Position:  position,
		Guarantee: sessionloop.AcceptanceDurable,
	}, nil
}

func (st *sessionState) acceptInterruptLocked(command sessionloop.Command) (sessionloop.Receipt, error) {
	run := st.run
	if run == nil {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: interrupt requires an active run: %w", sessionloop.ErrNotRunning)
	}
	if command.RunID != run.id {
		return sessionloop.Receipt{}, fmt.Errorf("testkit: interrupt targets run %q but run %q is active: %w", command.RunID, run.id, sessionloop.ErrStaleRun)
	}
	position := st.appendLocked(sessionloop.Event{Kind: sessionloop.EventCommandAccepted, RunID: run.id, CommandID: command.ID})
	if !run.interruptRequested {
		st.requestInterruptLocked(run)
		st.state = sessionloop.StateInterrupting
		position = st.appendLocked(sessionloop.Event{Kind: sessionloop.EventSessionState, RunID: run.id, CommandID: command.ID})
	}
	return sessionloop.Receipt{
		CommandID: command.ID,
		SessionID: st.id,
		RunID:     run.id,
		Position:  position,
		Guarantee: sessionloop.AcceptanceDurable,
	}, nil
}

func (st *sessionState) requestInterruptLocked(run *activeRun) {
	run.interruptRequested = true
	close(run.interrupted)
	run.cancel()
}

// Snapshot returns a copy-owned authoritative view (law L7).
func (s *session) Snapshot(ctx context.Context) (sessionloop.Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return sessionloop.Snapshot{}, err
	}
	state := s.state
	state.mu.Lock()
	defer state.mu.Unlock()
	if s.closed {
		return sessionloop.Snapshot{}, fmt.Errorf("testkit: snapshot on a closed handle: %w", sessionloop.ErrSessionClosed)
	}
	snapshot := sessionloop.Snapshot{
		SessionID:    state.id,
		Position:     state.currentPositionLocked(),
		State:        state.state,
		Usage:        state.usage,
		Capabilities: state.host.capabilities(),
	}
	if len(state.entries) > 0 {
		snapshot.Entries = make([]sessionloop.Entry, len(state.entries))
		for index, entry := range state.entries {
			snapshot.Entries[index] = entry.Clone()
		}
	}
	if len(state.pending) > 0 {
		snapshot.Pending = make([]sessionloop.QueuedInput, len(state.pending))
		for index, queued := range state.pending {
			snapshot.Pending[index] = queued.Clone()
		}
	}
	if state.run != nil {
		snapshot.ActiveRunID = state.run.id
		if state.run.suspension != nil {
			suspension := state.run.suspension.Clone()
			snapshot.Suspension = &suspension
		}
	}
	return snapshot, nil
}

// Subscribe replays authoritative events strictly after options.After and
// then follows the live log.
func (s *session) Subscribe(ctx context.Context, options sessionloop.SubscribeOptions) (sessionloop.Stream, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	state := s.state
	state.mu.Lock()
	defer state.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("testkit: subscribe on a closed handle: %w", sessionloop.ErrSessionClosed)
	}
	after := options.After
	if after.Sequence > state.seq {
		return nil, fmt.Errorf("testkit: position %d is beyond this session's history: %w", after.Sequence, sessionloop.ErrUnknownPosition)
	}
	if after.Token != "" && after.Token != positionToken(after.Sequence) {
		return nil, fmt.Errorf("testkit: token %q does not belong to this session's history: %w", after.Token, sessionloop.ErrUnknownPosition)
	}
	buffer := options.Buffer
	if buffer <= 0 {
		buffer = defaultStreamBuffer
	}
	subscriber := newStream(state, buffer, options.Preview)
	for _, event := range state.log {
		if event.Position.Sequence > after.Sequence {
			subscriber.queue = append(subscriber.queue, event.Clone())
		}
	}
	state.subs[subscriber] = struct{}{}
	return subscriber, nil
}

// Close is idempotent. It interrupts an actively RUNNING run, waits for its
// singular settlement, releases the handle, and terminates streams with
// io.EOF. A SUSPENDED run is a durable pause, not activity: Close does NOT
// settle it — the suspension survives together with the durable log, and
// OpenSession restores the Suspended state so the same suspension can still
// be resolved (law L11: closing releases the handle, never the session).
func (s *session) Close(context.Context) error {
	state := s.state
	state.mu.Lock()
	if s.closed {
		state.mu.Unlock()
		return nil
	}
	s.closed = true
	run := state.run
	suspended := run != nil && run.suspension != nil && state.state == sessionloop.StateSuspended
	if run != nil && !suspended {
		state.closing = true
		state.state = sessionloop.StateClosing
		state.appendLocked(sessionloop.Event{Kind: sessionloop.EventSessionState, RunID: run.id})
		if !run.interruptRequested {
			state.requestInterruptLocked(run)
		}
	}
	state.mu.Unlock()

	if run != nil && !suspended {
		<-run.done
	}

	state.mu.Lock()
	state.closing = false
	state.state = sessionloop.StateClosed
	state.appendLocked(sessionloop.Event{Kind: sessionloop.EventSessionState})
	state.handleOpen = false
	for subscriber := range state.subs {
		subscriber.end()
	}
	state.subs = make(map[*stream]struct{})
	state.mu.Unlock()
	return nil
}

func (st *sessionState) nextPositionLocked() sessionloop.Position {
	return sessionloop.Position{Sequence: st.seq + 1, Token: positionToken(st.seq + 1)}
}

func (st *sessionState) currentPositionLocked() sessionloop.Position {
	if st.seq == 0 {
		return sessionloop.Position{}
	}
	return sessionloop.Position{Sequence: st.seq, Token: positionToken(st.seq)}
}

func positionToken(sequence uint64) string {
	return fmt.Sprintf("tk-%d", sequence)
}

// appendLocked assigns the next durable position, records the event, and
// fans it out to every subscriber in log order.
func (st *sessionState) appendLocked(event sessionloop.Event) sessionloop.Position {
	position := st.nextPositionLocked()
	st.seq = position.Sequence
	event.Position = position
	event.Ordinal = position.Sequence
	event.Nature = sessionloop.EventAuthoritative
	event.SessionID = st.id
	event.State = st.state
	stored := event.Clone()
	st.log = append(st.log, stored)
	for subscriber := range st.subs {
		subscriber.deliver(stored.Clone(), true)
	}
	return position
}

// commitEntryLocked durably records one transcript entry and its
// entry.committed event, taking ownership of content.
func (st *sessionState) commitEntryLocked(role sessionloop.Role, origin sessionloop.EntryOrigin, runID sessionloop.RunID, commandID sessionloop.CommandID, blocks []sessionloop.EntryBlock) sessionloop.Position {
	entry := sessionloop.Entry{
		ID:        sessionloop.EntryID(st.host.nextID("entry")),
		SessionID: st.id,
		RunID:     runID,
		CommandID: commandID,
		Position:  st.nextPositionLocked(),
		Role:      role,
		Origin:    origin,
		Blocks:    blocks,
	}
	st.entries = append(st.entries, entry.Clone())
	return st.appendLocked(sessionloop.Event{Kind: sessionloop.EventEntryCommitted, RunID: runID, CommandID: commandID, Entry: &entry})
}

// inputBlocksToEntryBlocks is the reference host's explicit boundary from
// accepted caller input to committed conversation history. Input and Entry use
// different types because they carry different authority even when their text
// or structured-data payloads happen to match.
func inputBlocksToEntryBlocks(blocks []sessionloop.InputBlock) []sessionloop.EntryBlock {
	entries := make([]sessionloop.EntryBlock, 0, len(blocks))
	for _, block := range blocks {
		switch block.Kind {
		case sessionloop.InputBlockText:
			entries = append(entries, sessionloop.EntryBlock{Kind: sessionloop.EntryBlockText, Text: block.Text})
		case sessionloop.InputBlockData:
			entries = append(entries, sessionloop.EntryBlock{
				Kind:      sessionloop.EntryBlockData,
				MediaType: block.MediaType,
				Data:      append([]byte(nil), block.Data...),
			})
		}
	}
	return entries
}
