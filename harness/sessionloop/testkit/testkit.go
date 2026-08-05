package testkit

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// Option configures a Host.
type Option func(*Host)

// WithRunFunc replaces the host's run behavior. The default is EchoRunFunc.
func WithRunFunc(fn RunFunc) Option {
	return func(h *Host) {
		if fn != nil {
			h.runFunc = fn
		}
	}
}

// Host is the in-memory reference implementation of sessionloop.Host.
// Durable session logs are retained across Close, so OpenSession reopens a
// previously closed session with its full history.
type Host struct {
	runFunc RunFunc
	ids     atomic.Uint64

	mu       sync.Mutex
	sessions map[sessionloop.SessionID]*sessionState

	holdsMu sync.Mutex
	holds   []chan struct{}
}

// New returns a host driving every run with the configured RunFunc.
func New(options ...Option) *Host {
	host := &Host{
		runFunc:  EchoRunFunc(),
		sessions: make(map[sessionloop.SessionID]*sessionState),
	}
	for _, option := range options {
		option(host)
	}
	return host
}

func hostCapabilities() sessionloop.Capabilities {
	return sessionloop.NewCapabilities(
		sessionloop.CapabilityDurableAcceptance,
		sessionloop.CapabilityReplay,
		sessionloop.CapabilityPreview,
		sessionloop.CapabilitySteer,
		sessionloop.CapabilityFollowUp,
		sessionloop.CapabilityNextTurn,
		sessionloop.CapabilityInterrupt,
		sessionloop.CapabilitySuspensionResolve,
		sessionloop.CapabilityDetailedTools,
		sessionloop.CapabilityStructuredOutput,
	)
}

// NewSession creates a fresh session and returns its exclusive handle.
func (h *Host) NewSession(ctx context.Context, options sessionloop.SessionOptions) (sessionloop.Session, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	state := &sessionState{
		host: h,
		id:   sessionloop.SessionID(h.nextID("session")),
		meta: options.Clone().Meta,
		subs: make(map[*stream]struct{}),
	}
	h.mu.Lock()
	h.sessions[state.id] = state
	h.mu.Unlock()

	state.mu.Lock()
	state.handleOpen = true
	state.state = sessionloop.StateIdle
	state.appendLocked(sessionloop.Event{Kind: sessionloop.EventSessionState})
	state.mu.Unlock()
	return &session{state: state}, nil
}

// OpenSession reopens the durable log of an existing session. The host is an
// exclusive writer: a second concurrent open fails with ErrSessionOpen.
func (h *Host) OpenSession(ctx context.Context, id sessionloop.SessionID) (sessionloop.Session, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	h.mu.Lock()
	state, ok := h.sessions[id]
	h.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("testkit: unknown session %q", id)
	}

	state.mu.Lock()
	defer state.mu.Unlock()
	if state.handleOpen {
		return nil, fmt.Errorf("testkit: session %q already has an open handle: %w", id, sessionloop.ErrSessionOpen)
	}
	state.handleOpen = true
	state.state = sessionloop.StateIdle
	state.appendLocked(sessionloop.Event{Kind: sessionloop.EventSessionState})
	return &session{state: state}, nil
}

// HoldNextRun makes the next started run block inside the engine before its
// first step until the returned release function is called. Release is
// idempotent. Interrupting or closing the session also unblocks the run.
func (h *Host) HoldNextRun() (release func()) {
	hold := make(chan struct{})
	h.holdsMu.Lock()
	h.holds = append(h.holds, hold)
	h.holdsMu.Unlock()
	var once sync.Once
	return func() { once.Do(func() { close(hold) }) }
}

func (h *Host) takeHold() chan struct{} {
	h.holdsMu.Lock()
	defer h.holdsMu.Unlock()
	if len(h.holds) == 0 {
		return nil
	}
	hold := h.holds[0]
	h.holds = h.holds[1:]
	return hold
}

func (h *Host) nextID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, h.ids.Add(1))
}
