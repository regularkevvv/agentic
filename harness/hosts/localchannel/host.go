// Package localchannel assembles a sessionloop host from a local-channel
// mailbox, an independent actor worker, and an application-configured Harness.
// Requests enqueue; only the worker opens the fenced execution journal.
package localchannel

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	queue "github.com/regularkevvv/agentic/harness/sessionloop/actor/localchannel"
	"github.com/regularkevvv/agentic/harness/store"
)

// Config is bootstrap-only. Build must use the supplied repository for all
// journal operations, including recovery and asynchronous execution.
type Config struct {
	Journals store.Repository
	Build    func(store.Repository) (sessionloop.Host, error)
	OnError  func(actor.ActorID, error)
}

// Host implements the session protocol. Run belongs to service bootstrap,
// never to the submission path. Keep the Host alive across worker restarts.
type Host struct {
	cfg       Config
	mailbox   *queue.Store
	worker    actor.Worker
	mu        sync.Mutex
	sessions  map[sessionloop.SessionID]*state
	started   chan struct{}
	startOnce sync.Once
	runErr    error
	running   atomic.Bool
}

type state struct {
	snapshot  sessionloop.Snapshot
	history   []sessionloop.Event
	covered   sessionloop.Position
	published map[eventKey]bool
	receipts  map[actor.CommandID]sessionloop.Receipt
	streams   map[*stream]struct{}
	changed   chan struct{}
	ready     bool
	err       error
	handle    *handle
	current   *observedSession
	startID   actor.CommandID
}

func (s *state) signal() { close(s.changed); s.changed = make(chan struct{}) }

// New constructs the flavor without starting execution.
func New(cfg Config) (*Host, error) {
	if cfg.Journals == nil || cfg.Build == nil {
		return nil, errors.New("localchannel: journal repository and host builder are required")
	}
	h := &Host{cfg: cfg, mailbox: queue.New(), sessions: make(map[sessionloop.SessionID]*state), started: make(chan struct{})}
	w, err := actor.NewWorker(actor.Config{
		Owner: "local-" + rand.Text(), Adapter: &delivery{Store: h.mailbox, host: h},
		SessionOpener: actor.SessionOpenerFunc(h.openOwned), EventSink: h,
		OnError: func(id actor.ActorID, err error) {
			h.mu.Lock()
			if s := h.sessions[sessionloop.SessionID(id)]; s != nil {
				s.err = err
				s.signal()
			}
			h.mu.Unlock()
			if cfg.OnError != nil {
				cfg.OnError(id, err)
			}
		},
	})
	if err != nil {
		return nil, err
	}
	h.worker = w
	return h, nil
}

// Run executes retained work until the service context ends and joins cleanup.
func (h *Host) Run(ctx context.Context) error {
	if !h.running.CompareAndSwap(false, true) {
		return actor.ErrWorkerRunning
	}
	defer h.running.Store(false)
	h.mu.Lock()
	h.runErr = nil
	h.mu.Unlock()
	h.startOnce.Do(func() { close(h.started) })
	err := h.worker.Run(ctx)
	h.mu.Lock()
	h.runErr = err
	for _, s := range h.sessions {
		s.signal()
		for sub := range s.streams {
			sub.endLocked(err)
		}
	}
	h.mu.Unlock()
	return err
}

// NewSession creates and closes a fresh journal before publishing its identity.
// No input executes here. Subsequent opening and recovery belong to the worker.
func (h *Host) NewSession(ctx context.Context, options sessionloop.SessionOptions) (sessionloop.Session, error) {
	native, err := h.cfg.Build(h.cfg.Journals)
	if err != nil {
		return nil, err
	}
	s, err := native.NewSession(ctx, options)
	if err != nil {
		return nil, err
	}
	id := s.ID()
	if err := s.Close(context.WithoutCancel(ctx)); err != nil {
		return nil, err
	}
	return h.OpenSession(ctx, id)
}

// OpenSession attaches a client view, not an unguarded execution handle.
func (h *Host) OpenSession(ctx context.Context, id sessionloop.SessionID) (sessionloop.Session, error) {
	if err := store.ValidateSessionID(string(id)); err != nil {
		return nil, err
	}
	h.mu.Lock()
	s := h.sessions[id]
	if s == nil {
		s = &state{receipts: make(map[actor.CommandID]sessionloop.Receipt), published: make(map[eventKey]bool), streams: make(map[*stream]struct{}), changed: make(chan struct{})}
		h.sessions[id] = s
	}
	if s.handle != nil {
		h.mu.Unlock()
		return nil, sessionloop.ErrSessionOpen
	}
	v := &handle{host: h, state: s, id: id}
	s.handle, s.err = v, nil
	h.mu.Unlock()
	if err := h.mailbox.Recover(ctx, actor.ActorID(id)); err != nil {
		_ = v.Close(ctx)
		return nil, err
	}
	select {
	case <-ctx.Done():
		_ = v.Close(ctx)
		return nil, ctx.Err()
	case <-h.started:
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.runErr != nil {
		s.handle = nil
		return nil, h.runErr
	}
	for !s.ready {
		if err := h.wait(ctx, s); err != nil {
			s.handle = nil
			return nil, err
		}
	}
	return v, nil
}

// wait returns with h.mu held; state and wait-channel capture are atomic.
func (h *Host) wait(ctx context.Context, s *state) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if h.runErr != nil {
		return h.runErr
	}
	if s.err != nil {
		return s.err
	}
	changed := s.changed
	h.mu.Unlock()
	select {
	case <-ctx.Done():
	case <-changed:
	}
	h.mu.Lock()
	return ctx.Err()
}

type journalSession interface {
	actor.Session
	Replay(context.Context, sessionloop.Position, sessionloop.Position) ([]sessionloop.Event, error)
}

func (h *Host) openOwned(ctx context.Context, lease actor.Lease) (actor.Session, error) {
	native, err := h.cfg.Build(store.WithAuthority(h.cfg.Journals, h.mailbox.Authority(lease)))
	if err != nil {
		return nil, err
	}
	s, err := native.OpenSession(ctx, sessionloop.SessionID(lease.ActorID))
	if err != nil {
		return nil, err
	}
	journal, ok := s.(journalSession)
	if !ok {
		return nil, errors.Join(errors.New("localchannel: host must provide journal acceptance and finite replay"), s.Close(context.WithoutCancel(ctx)))
	}
	opened := &observedSession{journalSession: journal, host: h, lease: lease}
	h.mu.Lock()
	h.sessions[s.ID()].current = opened
	h.mu.Unlock()
	return opened, nil
}

// observedSession repairs the ephemeral UI projection from existing journal
// facts. This cache is observation only, never mailbox or recovery authority.
type observedSession struct {
	journalSession
	host      *Host
	lease     actor.Lease
	mu        sync.Mutex
	closed    bool
	streaming bool
}

func (s *observedSession) Snapshot(ctx context.Context) (sessionloop.Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return sessionloop.Snapshot{}, sessionloop.ErrSessionClosed
	}
	return s.refresh(ctx, !s.streaming)
}

func (s *observedSession) refresh(ctx context.Context, reconcile bool) (sessionloop.Snapshot, error) {
	snapshot, err := s.journalSession.Snapshot(ctx)
	if err != nil {
		return snapshot, err
	}
	h := s.host
	h.mu.Lock()
	state := h.sessions[s.ID()]
	after := state.covered
	h.mu.Unlock()
	var events []sessionloop.Event
	if reconcile {
		events, err = s.Replay(ctx, after, snapshot.Position)
		if err != nil {
			return snapshot, err
		}
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if state.current != s {
		return snapshot, actor.ErrLeaseLost
	}
	state.snapshot, state.ready, state.err = snapshot.Clone(), true, nil
	if receipt, found := state.receipts[state.startID]; found && snapshot.Position.Sequence >= receipt.Position.Sequence {
		state.startID = ""
	}
	for _, event := range events {
		state.publish(event)
	}
	if reconcile {
		state.covered = snapshot.Position
	}
	state.signal()
	return snapshot, nil
}

func (s *observedSession) Close(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	_, refreshErr := s.refresh(ctx, true)
	err := s.journalSession.Close(ctx)
	if err != nil {
		return errors.Join(refreshErr, err)
	}
	s.closed = true
	s.host.mu.Lock()
	if state := s.host.sessions[s.ID()]; state.current == s {
		state.current = nil
		state.signal()
	}
	s.host.mu.Unlock()
	return refreshErr
}

// Failed ownership cannot perform a final projection read. The next owner
// repairs this observation cache from the unchanged journal.
func (s *observedSession) Abandon(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if err := s.journalSession.Abandon(ctx); err != nil {
		return err
	}
	s.closed = true
	s.host.mu.Lock()
	if state := s.host.sessions[s.ID()]; state.current == s {
		state.current = nil
		state.signal()
	}
	s.host.mu.Unlock()
	return nil
}

func (s *observedSession) Subscribe(ctx context.Context, options sessionloop.SubscribeOptions) (sessionloop.Stream, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	stream, err := s.journalSession.Subscribe(ctx, options)
	if err == nil {
		s.streaming = true
	}
	return stream, err
}

// A journal position may project several entries. Deduplication includes the
// entry/queue identity, never only the numeric cursor.
type eventKey struct {
	sequence uint64
	kind     sessionloop.EventKind
	entry    sessionloop.EntryID
	queue    sessionloop.QueueID
}

func (s *state) publish(event sessionloop.Event) {
	if event.Nature == sessionloop.EventAuthoritative && !event.Position.IsZero() {
		key := eventKey{sequence: event.Position.Sequence, kind: event.Kind}
		if event.Entry != nil {
			key.entry = event.Entry.ID
		}
		if event.Queue != nil {
			key.queue = event.Queue.ID
		}
		if s.published[key] {
			return
		}
		s.published[key] = true
		s.history = append(s.history, event.Clone())
	}
	for sub := range s.streams {
		sub.deliverLocked(event)
	}
	s.signal()
}

// Observe preserves stream order, including previews before settlement.
// Replay repairs history only at opening/retirement, not ahead of a live stream.
func (h *Host) Observe(_ context.Context, lease actor.Lease, event sessionloop.Event) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if s := h.sessions[sessionloop.SessionID(lease.ActorID)]; s != nil {
		if s.current == nil || s.current.lease.Fence != lease.Fence {
			return actor.ErrLeaseLost
		}
		s.publish(event)
	}
	return nil
}

type delivery struct {
	*queue.Store
	host *Host
}

func (d *delivery) Acknowledge(ctx context.Context, lease actor.Lease, id actor.CommandID, receipt sessionloop.Receipt) error {
	if err := d.Store.Acknowledge(ctx, lease, id, receipt); err != nil {
		return err
	}
	d.host.mu.Lock()
	defer d.host.mu.Unlock()
	s := d.host.sessions[sessionloop.SessionID(lease.ActorID)]
	s.receipts[id] = receipt
	if s.startID == id && s.snapshot.Position.Sequence >= receipt.Position.Sequence {
		s.startID = ""
	}
	s.signal()
	return nil
}

// handle writes only through Mailbox and reads the journal-backed projection.
// It never holds a writable native session handle.
type handle struct {
	host   *Host
	state  *state
	id     sessionloop.SessionID
	closed bool
}

func (s *handle) ID() sessionloop.SessionID { return s.id }
func (s *handle) Capabilities() sessionloop.Capabilities {
	s.host.mu.Lock()
	defer s.host.mu.Unlock()
	return s.state.snapshot.Capabilities.Clone()
}

func (s *handle) Snapshot(ctx context.Context) (sessionloop.Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return sessionloop.Snapshot{}, err
	}
	s.host.mu.Lock()
	if s.closed {
		s.host.mu.Unlock()
		return sessionloop.Snapshot{}, sessionloop.ErrSessionClosed
	}
	current := s.state.current
	s.host.mu.Unlock()
	if current != nil {
		if _, err := current.Snapshot(ctx); err != nil && !errors.Is(err, sessionloop.ErrSessionClosed) {
			return sessionloop.Snapshot{}, err
		}
	}
	s.host.mu.Lock()
	defer s.host.mu.Unlock()
	return s.state.snapshot.Clone(), nil
}

// Dispatch waits for a verified journal receipt, not execution completion.
// Canceling this wait never removes work already accepted by the mailbox.
func (s *handle) Dispatch(ctx context.Context, command sessionloop.Command) (sessionloop.Receipt, error) {
	if err := sessionloop.ValidateCommand(command, s.Capabilities()); err != nil {
		return sessionloop.Receipt{}, err
	}
	// A settlement may reach the client before the worker's next snapshot.
	// Refresh from its owned journal before making a Start's busy decision;
	// startID below still serializes concurrently submitted, unaccepted Starts.
	if command.Kind == sessionloop.CommandStart {
		if _, err := s.Snapshot(ctx); err != nil {
			return sessionloop.Receipt{}, err
		}
	}
	if command.ID == "" {
		command.ID = sessionloop.CommandID(command.IdempotencyKey)
	}
	if command.ID == "" {
		command.ID = sessionloop.CommandID("cmd_" + rand.Text())
	}
	if command.IdempotencyKey == "" {
		command.IdempotencyKey = string(command.ID)
	}
	h := s.host
	h.mu.Lock()
	if s.closed {
		h.mu.Unlock()
		return sessionloop.Receipt{}, sessionloop.ErrSessionClosed
	}
	id := actor.CommandID(command.ID)
	_, accepted := s.state.receipts[id]
	if command.Kind == sessionloop.CommandStart && !accepted && s.state.startID != id &&
		(s.state.startID != "" || s.state.snapshot.State != sessionloop.StateIdle) {
		h.mu.Unlock()
		return sessionloop.Receipt{}, sessionloop.ErrSessionBusy
	}
	_, err := h.mailbox.Submit(ctx, actor.Command{ActorID: actor.ActorID(s.id), ID: actor.CommandID(command.ID), Command: command})
	if err != nil {
		h.mu.Unlock()
		return sessionloop.Receipt{}, err
	}
	if command.Kind == sessionloop.CommandStart && !accepted {
		s.state.startID = id
	}
	defer h.mu.Unlock()
	for {
		if receipt, found := s.state.receipts[actor.CommandID(command.ID)]; found {
			return receipt, rejectionError(receipt.Rejection)
		}
		if s.closed {
			return sessionloop.Receipt{}, sessionloop.ErrSessionClosed
		}
		if err := h.wait(ctx, s.state); err != nil {
			return sessionloop.Receipt{}, err
		}
	}
}

func rejectionError(reason sessionloop.Rejection) error {
	switch reason {
	case "":
		return nil
	case sessionloop.RejectionInvalidCommand:
		return sessionloop.ErrInvalidCommand
	case sessionloop.RejectionNotRunning:
		return sessionloop.ErrNotRunning
	case sessionloop.RejectionStaleRun:
		return sessionloop.ErrStaleRun
	case sessionloop.RejectionUnsupported:
		return sessionloop.ErrUnsupported
	default:
		return fmt.Errorf("localchannel: unknown rejection %q", reason)
	}
}

// Close detaches the client. Accepted work belongs to the independent worker;
// service shutdown cancels Run and joins writable-handle cleanup.
func (s *handle) Close(context.Context) error {
	s.host.mu.Lock()
	defer s.host.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.state.handle == s {
		s.state.handle = nil
	}
	for sub := range s.state.streams {
		if sub.owner == s {
			sub.endLocked(nil)
			delete(s.state.streams, sub)
		}
	}
	s.state.signal()
	return nil
}

var _ sessionloop.Host = (*Host)(nil)
var _ sessionloop.Session = (*handle)(nil)
