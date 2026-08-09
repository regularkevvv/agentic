package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
)

// LoopConfig configures one LoopView over an inner session.
type LoopConfig[O any] struct {
	// CloseRoot closes the OWNING root wrapper of the inner session so the
	// in-process ownership registry is released exactly as today (plan 8.6).
	// Required.
	CloseRoot func(context.Context) error

	// OutputProjector converts a completed run's typed output into one JSON
	// value. Nil disables the output.structured capability; the generic
	// layer never serializes O by reflection without it.
	OutputProjector func(O) (json.RawMessage, error)

	// SuspensionProjector overrides the package default projection of
	// durable suspensions into safe display values.
	SuspensionProjector SuspensionProjector

	// capabilities overrides the advertised set. Tests only.
	capabilities sessionloop.Capabilities
}

// LoopView is the sessionloop.Session view over one inner durable session.
// It owns only command dispatch and goroutine lifecycle; every durable fact,
// queue, state, and settlement stays owned by the inner session (plan 8.1).
//
// Dispatch-context ownership follows law L4: the context passed to Dispatch
// governs validation and durable acceptance only; accepted runs execute
// under the view's lifetime context and are stopped by an interrupt command
// or Close. Command attribution (CommandID on events and entries) is stamped
// at projection time from the view's in-memory maps; it is live-host
// knowledge, deliberately absent after reopening a session elsewhere.
type LoopView[O any] struct {
	inner     *Session[O]
	closeRoot func(context.Context) error
	output    func(O) (json.RawMessage, error)
	suspend   SuspensionProjector
	caps      sessionloop.Capabilities

	lifetime       context.Context
	cancelLifetime context.CancelFunc
	wg             sync.WaitGroup

	// lifecycleMu orders run-creating dispatch acceptance against Close:
	// dispatchers hold the read side across acceptance, goroutine
	// registration, and wg.Add, so Close's exclusive section observes either
	// a fully registered run (and interrupts it) or none (and the dispatcher
	// then sees closed). Without it a resolve slipping past the closed check
	// could wg.Add against a joining wg.Wait or re-activate a session mid
	// closeRoot.
	lifecycleMu sync.RWMutex

	closeMu   sync.Mutex
	closeDone bool
	closeErr  error

	mu                 sync.Mutex
	closed             bool
	runCommands        map[string]sessionloop.CommandID
	queueCommands      map[string]sessionloop.CommandID
	resolutionCommands map[uint64]sessionloop.CommandID
	outputs            map[string]json.RawMessage
	runDone            map[string]chan struct{}
	streams            map[*loopStream]struct{}
}

var _ sessionloop.Session = (*LoopView[struct{}])(nil)

// NewLoopView wraps an inner session into the protocol view. The inner
// session must be exclusively owned by the caller for the view's lifetime.
func NewLoopView[O any](inner *Session[O], config LoopConfig[O]) (*LoopView[O], error) {
	if inner == nil {
		return nil, errors.New("loop view requires a session")
	}
	if config.CloseRoot == nil {
		return nil, errors.New("loop view requires the root close function")
	}
	suspend := config.SuspensionProjector
	if suspend == nil {
		suspend = defaultSuspensionProjector
	}
	caps := config.capabilities
	if caps == nil {
		standard := []sessionloop.Capability{
			sessionloop.CapabilityDurableAcceptance,
			sessionloop.CapabilityReplay,
			sessionloop.CapabilityPreview,
			sessionloop.CapabilitySteer,
			sessionloop.CapabilityFollowUp,
			sessionloop.CapabilityNextTurn,
			sessionloop.CapabilityInterrupt,
			sessionloop.CapabilitySuspensionResolve,
			sessionloop.CapabilityDetailedTools,
		}
		if config.OutputProjector != nil {
			standard = append(standard, sessionloop.CapabilityStructuredOutput)
		}
		// dispatch.idempotent is NEVER advertised: idempotency keys are not
		// durably recorded, and an in-memory map is not durable proof.
		caps = sessionloop.NewCapabilities(standard...)
	}
	lifetime, cancel := context.WithCancel(context.Background())
	return &LoopView[O]{
		inner:              inner,
		closeRoot:          config.CloseRoot,
		output:             config.OutputProjector,
		suspend:            suspend,
		caps:               caps,
		lifetime:           lifetime,
		cancelLifetime:     cancel,
		runCommands:        make(map[string]sessionloop.CommandID),
		queueCommands:      make(map[string]sessionloop.CommandID),
		resolutionCommands: make(map[uint64]sessionloop.CommandID),
		outputs:            make(map[string]json.RawMessage),
		runDone:            make(map[string]chan struct{}),
		streams:            make(map[*loopStream]struct{}),
	}, nil
}

func (v *LoopView[O]) ID() sessionloop.SessionID { return sessionloop.SessionID(v.inner.ID()) }

func (v *LoopView[O]) Capabilities() sessionloop.Capabilities { return v.caps.Clone() }

func (v *LoopView[O]) isClosed() bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.closed
}

func loopClosedError() error {
	return fmt.Errorf("%w: %w", sessionloop.ErrSessionClosed, ErrSessionClosed)
}

// mapLoopError projects inner sentinels onto portable protocol categories
// while preserving the concrete cause: errors.Is succeeds against both the
// sessionloop sentinel and the original harness sentinel. BudgetError passes
// through without a protocol sentinel: budget policy is application-owned
// and its errors.Is identity (ErrBudgetExceeded) is already portable for
// Agentic callers (documented).
func mapLoopError(err error) error {
	if err == nil {
		return nil
	}
	sentinelPairs := []struct {
		inner error
		outer error
	}{
		{ErrSessionBusy, sessionloop.ErrSessionBusy},
		{ErrRunClosing, sessionloop.ErrCommandConflict},
		{ErrNotRunning, sessionloop.ErrNotRunning},
		{ErrSessionSuspended, sessionloop.ErrSuspended},
		{ErrSessionFaulted, sessionloop.ErrSessionFaulted},
		{ErrSessionClosed, sessionloop.ErrSessionClosed},
		{store.ErrSessionOpen, sessionloop.ErrSessionOpen},
		{errStaleRunTarget, sessionloop.ErrStaleRun},
		{ErrInvalidMessage, sessionloop.ErrInvalidCommand},
		{ErrInvalidResumeRequest, sessionloop.ErrInvalidCommand},
	}
	for _, pair := range sentinelPairs {
		if errors.Is(err, pair.inner) {
			return fmt.Errorf("%w: %w", pair.outer, err)
		}
	}
	return err
}

func (v *LoopView[O]) Dispatch(ctx context.Context, command sessionloop.Command) (sessionloop.Receipt, error) {
	if err := sessionloop.ValidateCommand(command, v.caps); err != nil {
		return sessionloop.Receipt{}, err
	}
	v.lifecycleMu.RLock()
	defer v.lifecycleMu.RUnlock()
	if v.isClosed() {
		return sessionloop.Receipt{}, loopClosedError()
	}
	commandID := command.ID
	if commandID == "" {
		generated, err := v.inner.idsGenerator().New("cmd")
		if err != nil {
			return sessionloop.Receipt{}, err
		}
		commandID = sessionloop.CommandID(generated)
	}
	switch command.Kind {
	case sessionloop.CommandSteer, sessionloop.CommandFollowUp,
		sessionloop.CommandResolve, sessionloop.CommandInterrupt:
		// Targeted commands never cross runs (law L8). next_turn is
		// session-targeted and start names no run. This is only a fast path:
		// the target is revalidated inside the acceptance primitives' locked
		// critical sections (acceptWithCursor, requestInterrupt,
		// prepareResume's suspension-ID recheck), so a run that settles
		// after this check cannot leak the command into its successor.
		if active := v.inner.currentRunID(); active != string(command.RunID) {
			return sessionloop.Receipt{}, fmt.Errorf(
				"run %q is not the active run: %w", command.RunID, sessionloop.ErrStaleRun)
		}
	}
	switch command.Kind {
	case sessionloop.CommandStart:
		return v.dispatchStart(ctx, command, commandID)
	case sessionloop.CommandSteer:
		return v.dispatchQueue(ctx, command, commandID, QueueSteer)
	case sessionloop.CommandFollowUp:
		return v.dispatchQueue(ctx, command, commandID, QueueFollowUp)
	case sessionloop.CommandNextTurn:
		return v.dispatchQueue(ctx, command, commandID, QueueNextTurn)
	case sessionloop.CommandResolve:
		return v.dispatchResolve(ctx, command, commandID)
	default:
		return v.dispatchInterrupt(ctx, command, commandID)
	}
}

// translateInputToMessage is the explicit ingress authority boundary from one
// caller-owned protocol Input to one Agentic user Message. Only text blocks are
// accepted by this v1 host; any other block kind is explicitly unsupported
// rather than silently rewritten (law L9). Input.Meta is correlation-only and
// is never translated into model-visible content.
func translateInputToMessage(input *sessionloop.Input) (agentic.Message, error) {
	if err := sessionloop.ValidateInput(*input); err != nil {
		return agentic.Message{}, err
	}
	parts := make([]agentic.Part, 0, len(input.Blocks))
	for _, block := range input.Blocks {
		if block.Kind != sessionloop.InputBlockText {
			return agentic.Message{}, fmt.Errorf(
				"%s input blocks are not supported by this host: %w", block.Kind, sessionloop.ErrUnsupported)
		}
		parts = append(parts, agentic.Part{Type: agentic.ContentText, Text: block.Text})
	}
	return agentic.Message{Role: agentic.RoleUser, Content: parts}, nil
}

func loopPosition(cursor store.Cursor) sessionloop.Position {
	return sessionloop.Position{Sequence: cursor.Seq, Token: cursor.EntryID}
}

func (v *LoopView[O]) dispatchStart(
	ctx context.Context,
	command sessionloop.Command,
	commandID sessionloop.CommandID,
) (sessionloop.Receipt, error) {
	message, err := translateInputToMessage(command.Input)
	if err != nil {
		return sessionloop.Receipt{}, err
	}
	accepted, err := v.inner.prepareStart(ctx, message, v.lifetime)
	if err != nil {
		return sessionloop.Receipt{}, mapLoopError(err)
	}
	done := make(chan struct{})
	v.mu.Lock()
	v.runCommands[accepted.runID] = commandID
	v.runDone[accepted.runID] = done
	v.mu.Unlock()
	if ctx.Err() != nil {
		// The dispatch context was canceled after durable acceptance: finish
		// the handshake deterministically (plan 8.4). The accepted run is
		// consumed unwound — never driven — and settled as interrupted on
		// this goroutine, so no unowned run survives.
		_ = accepted.consume()
		if _, interruptErr := v.inner.requestInterrupt(v.lifetime, ""); interruptErr == nil {
			_ = v.inner.finishInterrupt(&agentic.Execution[O]{Status: agentic.ExecutionInterrupted})
		}
		close(done)
		_ = v.inner.WaitForIdle(v.lifetime)
		return sessionloop.Receipt{}, ctx.Err()
	}
	v.wg.Add(1)
	go func() {
		defer v.wg.Done()
		execution, _ := v.inner.driveAccepted(accepted)
		v.captureOutput(accepted.runID, execution, done)
	}()
	return sessionloop.Receipt{
		CommandID: commandID,
		SessionID: v.ID(),
		RunID:     sessionloop.RunID(accepted.runID),
		Position:  loopPosition(accepted.commit.Cursor),
		Guarantee: sessionloop.AcceptanceDurable,
	}, nil
}

func (v *LoopView[O]) dispatchQueue(
	ctx context.Context,
	command sessionloop.Command,
	commandID sessionloop.CommandID,
	kind QueueKind,
) (sessionloop.Receipt, error) {
	message, err := translateInputToMessage(command.Input)
	if err != nil {
		return sessionloop.Receipt{}, err
	}
	// Steer and follow-up are run-targeted (law L8) and pin their target for
	// the locked acceptance recheck; next_turn is session-targeted (the L8
	// exception) and passes no target.
	targetRunID := ""
	if kind != QueueNextTurn {
		targetRunID = string(command.RunID)
	}
	receipt, cursor, err := v.inner.acceptWithCursor(ctx, kind, message, targetRunID)
	if err != nil {
		return sessionloop.Receipt{}, mapLoopError(err)
	}
	v.mu.Lock()
	v.queueCommands[receipt.ID] = commandID
	v.mu.Unlock()
	return sessionloop.Receipt{
		CommandID: commandID,
		SessionID: v.ID(),
		QueueID:   sessionloop.QueueID(receipt.ID),
		Position:  loopPosition(cursor),
		Guarantee: sessionloop.AcceptanceDurable,
	}, nil
}

// loopResumeRequest converts a protocol resolution into the harness resume
// request shape.
func loopResumeRequest(command sessionloop.Command) (ResumeRequest, error) {
	request := ResumeRequest{SuspensionID: command.Resolution.SuspensionID}
	for _, decision := range command.Resolution.Decisions {
		resolution := ToolResolution{CallID: decision.ID, Reason: decision.Reason}
		switch decision.Action {
		case sessionloop.ResolutionApprove:
			resolution.Action = ResolutionApprove
		case sessionloop.ResolutionDeny:
			resolution.Action = ResolutionDeny
		case sessionloop.ResolutionExternalResult:
			resolution.Action = ResolutionExternalResult
			if len(decision.Data) > 0 {
				var result any
				if err := json.Unmarshal(decision.Data, &result); err != nil {
					return ResumeRequest{}, fmt.Errorf(
						"resolution decision %q data: %w", decision.ID, sessionloop.ErrInvalidCommand)
				}
				resolution.Result = result
			}
		}
		request.Resolutions = append(request.Resolutions, resolution)
	}
	if command.Input != nil {
		prompt, err := translateInputToMessage(command.Input)
		if err != nil {
			return ResumeRequest{}, err
		}
		request.Prompt = &prompt
	}
	return request, nil
}

func (v *LoopView[O]) dispatchResolve(
	ctx context.Context,
	command sessionloop.Command,
	commandID sessionloop.CommandID,
) (sessionloop.Receipt, error) {
	request, err := loopResumeRequest(command)
	if err != nil {
		return sessionloop.Receipt{}, err
	}
	accepted, err := v.inner.prepareResume(ctx, request, v.lifetime)
	if err != nil {
		return sessionloop.Receipt{}, mapLoopError(err)
	}
	done := make(chan struct{})
	v.mu.Lock()
	if _, known := v.runCommands[accepted.runID]; !known {
		v.runCommands[accepted.runID] = commandID
	}
	for _, entry := range accepted.commit.Entries {
		if entry.Kind == kindResolutionAccepted {
			v.resolutionCommands[entry.Seq] = commandID
		}
	}
	v.runDone[accepted.runID] = done
	v.mu.Unlock()
	v.wg.Add(1)
	go func() {
		defer v.wg.Done()
		execution, driveErr := v.inner.driveResumed(accepted)
		if driveErr != nil && isResumeValidationError(driveErr) && v.inner.State() == Suspended {
			// The silent resume-validation bounce: finishExecution moved
			// Running back to Suspended with no journal write and no event.
			// Surface the live-only fact so protocol consumers waiting on the
			// resolve never hang (zero position = not replayable, law L6).
			v.announceLiveState(accepted.runID, commandID, sessionloop.StateSuspended)
		}
		v.captureOutput(accepted.runID, execution, done)
	}()
	return sessionloop.Receipt{
		CommandID: commandID,
		SessionID: v.ID(),
		RunID:     sessionloop.RunID(accepted.runID),
		Position:  loopPosition(accepted.commit.Cursor),
		Guarantee: sessionloop.AcceptanceDurable,
	}, nil
}

// announceLiveState injects a zero-position, authoritative session.state
// event into every live view-owned stream. Per law L6 a zero position is not
// replayable: this is the lawful seam for live-only facts that have no
// durable journal record, so the event reaches current subscribers only and
// never appears in snapshots or replays.
func (v *LoopView[O]) announceLiveState(runID string, commandID sessionloop.CommandID, state sessionloop.State) {
	announcement := sessionloop.Event{
		Nature:    sessionloop.EventAuthoritative,
		Kind:      sessionloop.EventSessionState,
		SessionID: v.ID(),
		RunID:     sessionloop.RunID(runID),
		CommandID: commandID,
		State:     state,
	}
	v.mu.Lock()
	targets := make([]*loopStream, 0, len(v.streams))
	for stream := range v.streams {
		targets = append(targets, stream)
	}
	v.mu.Unlock()
	// Injection happens outside v.mu: each stream owns its own lock and a
	// stream Close may concurrently detach from the view.
	for _, stream := range targets {
		stream.inject(announcement)
	}
}

func (v *LoopView[O]) forgetStream(stream *loopStream) {
	v.mu.Lock()
	delete(v.streams, stream)
	v.mu.Unlock()
}

func (v *LoopView[O]) dispatchInterrupt(
	ctx context.Context,
	command sessionloop.Command,
	commandID sessionloop.CommandID,
) (sessionloop.Receipt, error) {
	// The expected run ID is revalidated inside requestInterrupt's locked
	// section (law L8): an interrupt aimed at a settled run must never
	// cancel its successor.
	if _, err := v.inner.requestInterrupt(ctx, string(command.RunID)); err != nil {
		return sessionloop.Receipt{}, mapLoopError(err)
	}
	// No durable interrupt-request fact exists today; the durable trace is
	// written at settlement. The receipt is honestly "accepted" with a zero
	// position (law L3).
	return sessionloop.Receipt{
		CommandID: commandID,
		SessionID: v.ID(),
		RunID:     command.RunID,
		Guarantee: sessionloop.AcceptanceAccepted,
	}, nil
}

// captureOutput records the structured-output projection of one settled run
// and marks the run's drive as finished so stream projection can attach the
// output deterministically.
func (v *LoopView[O]) captureOutput(runID string, execution *agentic.Execution[O], done chan struct{}) {
	if v.output != nil && execution != nil && execution.Result != nil &&
		execution.Status == agentic.ExecutionCompleted {
		if projected, err := v.output(execution.Result.Output); err == nil && len(projected) > 0 {
			v.mu.Lock()
			v.outputs[runID] = append(json.RawMessage(nil), projected...)
			v.mu.Unlock()
		}
	}
	close(done)
}

// projectedOutput waits for the run's drive goroutine to record its output
// (bounded by ctx) and returns a copy, or nil when no output was captured.
// Replay projections never install this hook, so replayed settlements carry
// no structured output.
func (v *LoopView[O]) projectedOutput(ctx context.Context, runID string) json.RawMessage {
	if err := v.awaitRunFinalized(ctx, runID); err != nil {
		return nil
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if output, ok := v.outputs[runID]; ok {
		return append(json.RawMessage(nil), output...)
	}
	return nil
}

// awaitRunFinalized joins the process-local drive for a live projection. A
// nil channel means the event is replayed or belongs to a run opened by a
// different view, so its durable record can be projected immediately.
func (v *LoopView[O]) awaitRunFinalized(ctx context.Context, runID string) error {
	v.mu.Lock()
	done := v.runDone[runID]
	v.mu.Unlock()
	if done != nil {
		select {
		case <-done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func (v *LoopView[O]) commandForRun(runID string) sessionloop.CommandID {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.runCommands[runID]
}

func (v *LoopView[O]) commandForQueue(queueID string) sessionloop.CommandID {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.queueCommands[queueID]
}

func (v *LoopView[O]) commandForResolution(seq uint64) sessionloop.CommandID {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.resolutionCommands[seq]
}

func (v *LoopView[O]) newProjector(live bool) *loopProjector {
	projector := newLoopProjector(v.inner.ID(), v.inner.codecRef(), v.suspend)
	projector.commandForRun = v.commandForRun
	projector.commandForQueue = v.commandForQueue
	projector.commandForResolution = v.commandForResolution
	if live {
		projector.awaitRunFinalized = v.awaitRunFinalized
		projector.outputFor = v.projectedOutput
	}
	return projector
}

// Snapshot loads the durable journal and folds it through the projection so
// snapshot entries and full-stream replay are the same normalized state.
// Live view state (session state, pending queue, suspension, usage) comes
// from the inner session's authoritative snapshot.
func (v *LoopView[O]) Snapshot(ctx context.Context) (sessionloop.Snapshot, error) {
	if v.isClosed() {
		return sessionloop.Snapshot{}, loopClosedError()
	}
	loaded, err := v.inner.journalRef().Load(ctx)
	if err != nil {
		return sessionloop.Snapshot{}, mapLoopError(err)
	}
	records, err := loopRecords(v.inner.codecRef(), loaded.Entries)
	if err != nil {
		return sessionloop.Snapshot{}, err
	}
	projector := v.newProjector(false)
	var entries []sessionloop.Entry
	for _, record := range records {
		events, applyErr := projector.apply(ctx, record)
		if applyErr != nil {
			return sessionloop.Snapshot{}, applyErr
		}
		for _, projected := range events {
			if projected.Kind == sessionloop.EventEntryCommitted && projected.Entry != nil {
				entries = append(entries, *projected.Entry)
			}
		}
	}
	inner, err := v.inner.Snapshot(ctx)
	if err != nil {
		return sessionloop.Snapshot{}, mapLoopError(err)
	}
	snapshot := sessionloop.Snapshot{
		SessionID:    v.ID(),
		Position:     loopPosition(loaded.Cursor),
		State:        sessionloop.State(inner.State.String()),
		ActiveRunID:  sessionloop.RunID(inner.RunID),
		Entries:      entries,
		Usage:        loopUsage(inner.Usage),
		Capabilities: v.caps.Clone(),
	}
	for _, pending := range inner.Pending {
		queued := projector.fold.queuedInput(pending.ID)
		queued.CommandID = v.commandForQueue(pending.ID)
		snapshot.Pending = append(snapshot.Pending, queued)
	}
	if inner.Suspension != nil {
		safe, suspendErr := v.suspend(*inner.Suspension)
		if suspendErr != nil {
			return sessionloop.Snapshot{}, suspendErr
		}
		snapshot.Suspension = &safe
	}
	return snapshot.Clone(), nil
}

func (v *LoopView[O]) Subscribe(ctx context.Context, options sessionloop.SubscribeOptions) (sessionloop.Stream, error) {
	if v.isClosed() {
		return nil, loopClosedError()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	current := v.inner.currentCursor().Seq
	if options.After.Sequence > current {
		return nil, fmt.Errorf("replay position %d is beyond the durable history at %d: %w",
			options.After.Sequence, current, sessionloop.ErrUnknownPosition)
	}
	projector := v.newProjector(true)
	if options.After.Sequence > 0 {
		// Seed the attribution fold with everything at or before the replay
		// position so mid-stream subscribers project with full context.
		loaded, err := v.inner.journalRef().Load(ctx)
		if err != nil {
			return nil, mapLoopError(err)
		}
		records, err := loopRecords(v.inner.codecRef(), loaded.Entries)
		if err != nil {
			return nil, err
		}
		for _, record := range records {
			if record.Cursor > options.After.Sequence {
				break
			}
			if _, err := projector.apply(ctx, record); err != nil {
				return nil, err
			}
		}
	}
	subscription := v.inner.Subscribe(SubscribeOptions{
		AfterCursor: options.After.Sequence,
		Buffer:      options.Buffer,
		Preview:     options.Preview,
	})
	stream := &loopStream{
		subscription: subscription,
		projector:    projector,
		wake:         make(chan struct{}, 1),
	}
	stream.detach = func() { v.forgetStream(stream) }
	v.mu.Lock()
	v.streams[stream] = struct{}{}
	v.mu.Unlock()
	return stream, nil
}

// loopStream projects raw records on demand from Next. It is single-consumer
// like every sessionloop stream; Close is idempotent and safe to call while
// a Next waits. Besides the projected source subscription it carries a
// view-owned injection side-channel for zero-position live-only events
// (law L6); injected events bypass the projector because they are not
// durable facts.
type loopStream struct {
	subscription *Subscription
	projector    *loopProjector
	detach       func()

	mu       sync.Mutex
	queue    []sessionloop.Event
	terminal error
	closed   bool
	wake     chan struct{}
}

// inject enqueues one view-owned live event. Lock ordering: inject is always
// called WITHOUT v.mu held beyond a snapshot of the stream set, and it takes
// only s.mu, so it can never deadlock against a concurrent Close (which
// releases s.mu before detaching from the view).
func (s *loopStream) inject(event sessionloop.Event) {
	s.mu.Lock()
	if s.closed || s.terminal != nil {
		s.mu.Unlock()
		return
	}
	s.queue = append(s.queue, event.Clone())
	s.mu.Unlock()
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *loopStream) Next(ctx context.Context) (sessionloop.Event, error) {
	for {
		s.mu.Lock()
		if len(s.queue) > 0 {
			next := s.queue[0]
			s.queue = s.queue[1:]
			s.mu.Unlock()
			return next, nil
		}
		if s.terminal != nil {
			terminal := s.terminal
			s.mu.Unlock()
			return sessionloop.Event{}, terminal
		}
		if s.closed {
			s.mu.Unlock()
			return sessionloop.Event{}, io.EOF
		}
		s.mu.Unlock()
		select {
		case <-ctx.Done():
			return sessionloop.Event{}, ctx.Err()
		case <-s.wake:
			// An injected live event (or Close) is waiting; re-check the queue.
		case record, ok := <-s.subscription.Events:
			if !ok {
				s.settle()
				continue
			}
			events, err := s.projector.apply(ctx, record)
			if err != nil {
				return sessionloop.Event{}, err
			}
			s.mu.Lock()
			s.queue = append(s.queue, events...)
			s.mu.Unlock()
		}
	}
}

// settle records the stream's terminal condition after the source channels
// closed: a lag disconnect maps to ErrLagged (law L7), everything else is a
// normal end of stream.
func (s *loopStream) settle() {
	terminal := error(io.EOF)
	if err, ok := <-s.subscription.Err; ok && err != nil {
		var lagged *event.ErrSubscriberLagged
		if errors.As(err, &lagged) {
			terminal = fmt.Errorf("%w: %w", sessionloop.ErrLagged, err)
		} else {
			terminal = err
		}
	}
	s.mu.Lock()
	s.terminal = terminal
	s.mu.Unlock()
}

func (s *loopStream) Close() error {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	select {
	case s.wake <- struct{}{}:
	default:
	}
	s.subscription.Close()
	if s.detach != nil {
		s.detach()
	}
	return nil
}

// Close settles any active run, closes the owning root wrapper (releasing
// the in-process ownership registry), and joins every view goroutine. It is
// once-guarded: repeat calls return the memoized result. A wait canceled by
// ctx returns the context error without memoizing so Close can be retried.
func (v *LoopView[O]) Close(ctx context.Context) error {
	v.closeMu.Lock()
	defer v.closeMu.Unlock()
	if v.closeDone {
		return v.closeErr
	}
	// The exclusive side of lifecycleMu excludes run-creating dispatches for
	// the whole close sequence, so no run can be accepted between the state
	// switch below and closeRoot.
	v.lifecycleMu.Lock()
	defer v.lifecycleMu.Unlock()
	v.mu.Lock()
	v.closed = true
	v.mu.Unlock()
	switch v.inner.State() {
	case Running, Closing, Interrupting:
		// Suspended is deliberately NOT in this switch: a suspension is a
		// durable pause and closing is not deletion (law L11), so a
		// suspended session closes directly through closeRoot and the
		// suspension survives close/reopen instead of being destroyed by an
		// interrupt. ErrNotRunning means the run settled concurrently.
		if _, err := v.inner.requestInterrupt(ctx, ""); err != nil && !errors.Is(err, ErrNotRunning) &&
			!errors.Is(err, ErrSessionClosed) && !errors.Is(err, ErrSessionFaulted) {
			return mapLoopError(err)
		}
		if err := v.inner.WaitForIdle(ctx); err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			// Faulted and Closed are closable states; fall through.
		}
	}
	// Join every view goroutine BEFORE releasing the root: a drive settling a
	// fault must never race the closing journal.
	v.cancelLifetime()
	v.wg.Wait()
	err := mapLoopError(v.closeRoot(ctx))
	v.closeDone = true
	v.closeErr = err
	return err
}
