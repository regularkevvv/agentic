// Package sessionloop adapts a provider-neutral sessionloop host to the
// reusable TUI port. It reproduces the observable semantics of the direct
// Harness adapter (tui/adapter/harness): the same blocking operation
// boundaries, the same conservative presentation projection, and the same
// error identities, so terminal banners stay byte-identical.
package sessionloop

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/regularkevvv/agentic/harness/session"
	sl "github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"

	uit "github.com/regularkevvv/agentic/tui"
)

// waitBuffer sizes the dedicated wait streams used by blocking operations.
// It only bounds an internal subscriber; the snapshot fallback recovers from
// any lag disconnect.
const waitBuffer = 1024

// errResumeRejected reports the silent resume-validation bounce: the resolve
// was accepted, but resume validation moved the session straight back to
// suspended before any run event, announced by the host as a zero-position
// live-only session.state event. The legacy adapter surfaced the concrete
// validation error object at this boundary; the protocol deliberately
// carries no error identity for the bounce, so the bridge reports the
// boundary itself (documented differential exclusion).
var errResumeRejected = errors.New("resume attempt was rejected and the session remains suspended")

type config struct {
	profileLabel  string
	workspace     string
	execution     string
	toolPresenter uit.ToolPresenter
}

// Option mirrors the direct Harness adapter's decoration options.
type Option func(*config)

// WithProfileLabel decorates snapshots with the active profile name.
func WithProfileLabel(value string) Option { return func(c *config) { c.profileLabel = value } }

// WithWorkspace decorates snapshots with the workspace path.
func WithWorkspace(value string) Option { return func(c *config) { c.workspace = value } }

// WithExecutionLabel decorates snapshots with the execution description.
func WithExecutionLabel(value string) Option {
	return func(c *config) { c.execution = value }
}

// WithToolPresenter installs the application-owned tool presentation.
func WithToolPresenter(value uit.ToolPresenter) Option {
	return func(c *config) { c.toolPresenter = value }
}

type host struct {
	host   sl.Host
	config config
}

// New attaches to an already assembled sessionloop host. It never constructs
// a model, provider, permission policy, capability graph, or storage.
func New(value sl.Host, options ...Option) (uit.Host, error) {
	if value == nil {
		return nil, errors.New("TUI sessionloop adapter requires a host")
	}
	settings := config{}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("TUI sessionloop adapter option must not be nil")
		}
		option(&settings)
	}
	return &host{host: value, config: settings}, nil
}

func (h *host) NewSession(ctx context.Context, _ uit.SessionOptions) (uit.Session, error) {
	created, err := h.host.NewSession(ctx, sl.SessionOptions{})
	if err != nil {
		return nil, mapError(err)
	}
	return &bridgeSession{session: created, config: h.config}, nil
}

func (h *host) ResumeSession(ctx context.Context, id string) (uit.Session, error) {
	if id == "" {
		return nil, errors.New("session ID is empty")
	}
	opened, err := h.host.OpenSession(ctx, sl.SessionID(id))
	if err != nil {
		return nil, mapError(err)
	}
	return &bridgeSession{session: opened, config: h.config}, nil
}

type bridgeSession struct {
	session sl.Session
	config  config
}

func (s *bridgeSession) ID() string { return string(s.session.ID()) }

// mapError restores banner-text fidelity: the sessionloop host wraps harness
// sentinels for portable errors.Is checks, but the legacy adapter surfaced
// the bare sentinels whose Error() text feeds terminal banners directly.
func mapError(err error) error {
	if err == nil {
		return nil
	}
	sentinels := []error{
		session.ErrSessionBusy,
		session.ErrNotRunning,
		session.ErrSessionSuspended,
		session.ErrRunClosing,
		session.ErrSessionClosed,
		session.ErrSessionFaulted,
		store.ErrSessionOpen,
	}
	for _, sentinel := range sentinels {
		if errors.Is(err, sentinel) {
			return sentinel
		}
	}
	return err
}

func (s *bridgeSession) Snapshot(ctx context.Context) (uit.Snapshot, error) {
	value, err := s.session.Snapshot(ctx)
	if err != nil {
		return uit.Snapshot{}, mapError(err)
	}
	result := uit.Snapshot{
		SessionID: string(value.SessionID), Cursor: value.Position.Sequence, State: uit.State(value.State),
		Transcript: entries(value.Entries, s.config.toolPresenter),
		Usage:      usage(value.Usage), ProfileLabel: s.config.profileLabel, Workspace: s.config.workspace,
		Execution: s.config.execution,
	}
	result.Pending = make([]uit.QueuedInput, len(value.Pending))
	for index, pending := range value.Pending {
		result.Pending[index] = queuedInput(pending)
	}
	if value.Suspension != nil {
		result.Suspension = suspension(*value.Suspension)
	}
	return result, nil
}

func (s *bridgeSession) Subscribe(options uit.SubscribeOptions) uit.Subscription {
	stream, err := s.session.Subscribe(context.Background(), sl.SubscribeOptions{
		After:   sl.Position{Sequence: options.AfterCursor},
		Preview: options.Preview,
		Buffer:  options.Buffer,
	})
	if err != nil {
		return failedSubscription(mapError(err))
	}
	return pumpSubscription(stream, s, s.config.toolPresenter)
}

func textInput(text string) *sl.Input {
	return &sl.Input{Blocks: []sl.InputBlock{{Kind: sl.InputBlockText, Text: text}}}
}

// Submit dispatches a start command and then blocks until that exact run
// reaches a durable pause or its settlement, reproducing the legacy blocking
// Prompt boundary: completed and suspended return nil, interrupted returns
// context.Canceled, failed returns the sanitized outcome failure.
func (s *bridgeSession) Submit(ctx context.Context, input uit.Input) error {
	if err := input.Validate(); err != nil {
		return err
	}
	pre, err := s.session.Snapshot(ctx)
	if err != nil {
		return mapError(err)
	}
	stream, err := s.session.Subscribe(ctx, sl.SubscribeOptions{After: pre.Position, Buffer: waitBuffer})
	if err != nil {
		return mapError(err)
	}
	receipt, err := s.session.Dispatch(ctx, sl.Command{Kind: sl.CommandStart, Input: textInput(input.Text)})
	if err != nil {
		_ = stream.Close()
		return mapError(err)
	}
	return s.awaitRun(ctx, stream, pre.Position, receipt.RunID, false, "")
}

func (s *bridgeSession) Steer(ctx context.Context, input uit.Input) error {
	return s.queue(ctx, input, sl.CommandSteer)
}

func (s *bridgeSession) FollowUp(ctx context.Context, input uit.Input) error {
	return s.queue(ctx, input, sl.CommandFollowUp)
}

func (s *bridgeSession) NextTurn(ctx context.Context, input uit.Input) error {
	return s.queue(ctx, input, sl.CommandNextTurn)
}

// queue dispatches and returns at the durable acceptance boundary, dropping
// the receipt exactly as the legacy adapter drops harness queue receipts.
func (s *bridgeSession) queue(ctx context.Context, input uit.Input, kind sl.CommandKind) error {
	if err := input.Validate(); err != nil {
		return err
	}
	command := sl.Command{Kind: kind, Input: textInput(input.Text)}
	if kind == sl.CommandSteer || kind == sl.CommandFollowUp {
		// Run-targeted commands need the active run identity. Without one the
		// legacy accept path reported its state error client-side; reproduce
		// that identity instead of dispatching a structurally invalid command.
		snapshot, err := s.session.Snapshot(ctx)
		if err != nil {
			return mapError(err)
		}
		if snapshot.ActiveRunID == "" {
			return idleStateError(snapshot.State)
		}
		command.RunID = snapshot.ActiveRunID
	}
	_, err := s.session.Dispatch(ctx, command)
	if errors.Is(err, sl.ErrStaleRun) {
		// The run settled between snapshot and dispatch; the legacy accept
		// would have observed the idle session.
		return session.ErrNotRunning
	}
	return mapError(err)
}

// idleStateError reproduces the legacy acceptance error for a session
// without an active run.
func idleStateError(state sl.State) error {
	switch state {
	case sl.StateClosed:
		return session.ErrSessionClosed
	case sl.StateFaulted:
		return session.ErrSessionFaulted
	default:
		return session.ErrNotRunning
	}
}

// Resolve replicates the legacy client-side validation against a fresh
// snapshot with the exact legacy error strings, then dispatches the resolve
// command and blocks until the resumed run settles or re-suspends.
func (s *bridgeSession) Resolve(ctx context.Context, resolution uit.Resolution) error {
	if err := resolution.Validate(); err != nil {
		return err
	}
	snapshot, err := s.session.Snapshot(ctx)
	if err != nil {
		return mapError(err)
	}
	if snapshot.Suspension == nil || snapshot.Suspension.ID != resolution.SuspensionID {
		return errors.New("resolution does not match the current suspension")
	}
	required := make(map[string]bool, len(snapshot.Suspension.Decisions))
	for _, decision := range snapshot.Suspension.Decisions {
		required[decision.ID] = true
	}
	decisions := make(map[string]uit.Decision, len(resolution.Decisions))
	for _, decision := range resolution.Decisions {
		if !required[decision.CallID] {
			return fmt.Errorf("resolution contains unknown approval %q", decision.CallID)
		}
		decisions[decision.CallID] = decision
	}
	if len(decisions) != len(required) {
		return fmt.Errorf("resolution has %d decisions for %d approvals", len(decisions), len(required))
	}
	request := sl.Resolution{SuspensionID: resolution.SuspensionID}
	for _, approval := range snapshot.Suspension.Decisions {
		decision := decisions[approval.ID]
		action := sl.ResolutionApprove
		if decision.Action == uit.DecisionDeny {
			action = sl.ResolutionDeny
		}
		request.Decisions = append(request.Decisions, sl.ResolutionDecision{
			ID: approval.ID, Action: action, Reason: decision.Reason,
		})
	}
	command := sl.Command{Kind: sl.CommandResolve, RunID: snapshot.ActiveRunID, Resolution: &request}
	if resolution.Prompt != nil {
		command.Input = textInput(resolution.Prompt.Text)
	}
	stream, err := s.session.Subscribe(ctx, sl.SubscribeOptions{After: snapshot.Position, Buffer: waitBuffer})
	if err != nil {
		return mapError(err)
	}
	receipt, err := s.session.Dispatch(ctx, command)
	if err != nil {
		_ = stream.Close()
		return mapError(err)
	}
	return s.awaitRun(ctx, stream, snapshot.Position, receipt.RunID, false, receipt.CommandID)
}

// Interrupt dispatches an interrupt for the active run and blocks until that
// run settles, reproducing the legacy Interrupt + WaitForIdle boundary. An
// idle session yields the legacy session.ErrNotRunning identity.
func (s *bridgeSession) Interrupt(ctx context.Context) error {
	snapshot, err := s.session.Snapshot(ctx)
	if err != nil {
		return mapError(err)
	}
	if snapshot.ActiveRunID == "" {
		return idleStateError(snapshot.State)
	}
	stream, err := s.session.Subscribe(ctx, sl.SubscribeOptions{After: snapshot.Position, Buffer: waitBuffer})
	if err != nil {
		return mapError(err)
	}
	receipt, err := s.session.Dispatch(ctx, sl.Command{Kind: sl.CommandInterrupt, RunID: snapshot.ActiveRunID})
	if err != nil {
		_ = stream.Close()
		if errors.Is(err, sl.ErrStaleRun) {
			return session.ErrNotRunning
		}
		return mapError(err)
	}
	return s.awaitRun(ctx, stream, snapshot.Position, receipt.RunID, true, "")
}

func (s *bridgeSession) Close(ctx context.Context) error {
	return mapError(s.session.Close(ctx))
}

// awaitRun blocks until the target run reaches its boundary: a durable
// suspension (unless settleOnly), a terminal session fault, or its
// settlement. A non-empty resolveCommand marks the wait as a Resolve wait:
// the host's zero-position live-only session.state bounce (law L6) carrying
// that command's identity is then terminal too, because a bounced resolve
// produces no run event and no settlement. The stream is the primary wait;
// any stream failure falls back to an authoritative snapshot and, when the
// boundary already passed, a replay from the pre-dispatch position.
func (s *bridgeSession) awaitRun(
	ctx context.Context,
	stream sl.Stream,
	from sl.Position,
	runID sl.RunID,
	settleOnly bool,
	resolveCommand sl.CommandID,
) error {
	current := stream
	defer func() { _ = current.Close() }()
	for {
		event, err := current.Next(ctx)
		if err == nil {
			switch event.Kind {
			case sl.EventRunSuspended:
				if !settleOnly && event.RunID == runID {
					// The suspension event is published when the run emits
					// it; the durable suspended state lands with the
					// settlement-side finalize. Confirm through the
					// authoritative snapshot so the caller returns at the
					// same boundary as the legacy blocking Prompt/Resume.
					return s.confirmSuspended(ctx, from, runID, settleOnly)
				}
			case sl.EventRunSettled:
				if event.Outcome != nil && event.Outcome.RunID == runID {
					if settleOnly {
						return nil
					}
					return outcomeError(*event.Outcome)
				}
			case sl.EventSessionState:
				if event.State == sl.StateFaulted {
					// A mid-run session fault never settles the run; the
					// fault observation is terminal for every wait, with the
					// legacy identity the app's failure banner expects.
					return session.ErrSessionFaulted
				}
				if resolveCommand != "" && event.CommandID == resolveCommand &&
					event.RunID == runID && event.State == sl.StateSuspended &&
					event.Position.Sequence == 0 {
					// The silent resume-validation bounce: the session moved
					// straight back to suspended with no durable record, so
					// no run event and no settlement will ever arrive for
					// this resolve.
					return errResumeRejected
				}
			}
			continue
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// Lag, closure, or any other stream failure: reconcile through the
		// authoritative snapshot (law L7) and resume waiting on a new stream.
		_ = current.Close()
		snapshot, snapErr := s.session.Snapshot(ctx)
		if snapErr != nil {
			return mapError(snapErr)
		}
		if snapshot.ActiveRunID == runID {
			if snapshot.State == sl.StateSuspended && !settleOnly {
				return nil
			}
			if snapshot.State == sl.StateFaulted {
				return session.ErrSessionFaulted
			}
			next, subErr := s.session.Subscribe(ctx, sl.SubscribeOptions{After: snapshot.Position, Buffer: waitBuffer})
			if subErr != nil {
				return mapError(subErr)
			}
			current = next
			continue
		}
		// The run is no longer active: its settlement is durable history.
		return s.settledOutcome(ctx, from, runID, settleOnly)
	}
}

// confirmSuspended reconciles the announced suspension against authoritative
// snapshots until the durable suspended state (or a racing settlement) is
// observable. The suspended transition is already committed or in flight
// when the suspension event has been delivered, so the first poll usually
// resolves; retries back off exponentially (1ms doubling to a 20ms cap)
// instead of busy-spinning against the snapshot API.
func (s *bridgeSession) confirmSuspended(ctx context.Context, from sl.Position, runID sl.RunID, settleOnly bool) error {
	const maxPollDelay = 20 * time.Millisecond
	delay := time.Millisecond
	for {
		snapshot, err := s.session.Snapshot(ctx)
		if err != nil {
			return mapError(err)
		}
		if snapshot.ActiveRunID != runID {
			return s.settledOutcome(ctx, from, runID, settleOnly)
		}
		switch snapshot.State {
		case sl.StateSuspended:
			return nil
		case sl.StateFaulted:
			return session.ErrSessionFaulted
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
		if delay < maxPollDelay {
			delay *= 2
			if delay > maxPollDelay {
				delay = maxPollDelay
			}
		}
	}
}

// outcomeError maps a settled outcome onto the legacy blocking-call result.
func outcomeError(outcome sl.RunOutcome) error {
	switch outcome.Kind {
	case sl.RunInterrupted:
		return context.Canceled
	case sl.RunFailed:
		if outcome.Failure != "" {
			return errors.New(outcome.Failure)
		}
		return errors.New("run failed")
	default:
		return nil
	}
}

// settledOutcome replays durable history after the pre-dispatch position to
// recover the settlement of a run that finished while the wait stream was
// down.
func (s *bridgeSession) settledOutcome(ctx context.Context, from sl.Position, runID sl.RunID, settleOnly bool) error {
	replay, err := s.session.Subscribe(ctx, sl.SubscribeOptions{After: from, Buffer: waitBuffer})
	if err != nil {
		return mapError(err)
	}
	defer func() { _ = replay.Close() }()
	for {
		event, nextErr := replay.Next(ctx)
		if nextErr != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if settleOnly {
				// The session already left its running states; the legacy
				// WaitForIdle boundary is satisfied even without the record.
				return nil
			}
			return mapError(nextErr)
		}
		if event.Kind != sl.EventRunSettled || event.Outcome == nil || event.Outcome.RunID != runID {
			continue
		}
		if settleOnly {
			return nil
		}
		return outcomeError(*event.Outcome)
	}
}

var _ uit.Host = (*host)(nil)
var _ uit.Session = (*bridgeSession)(nil)
