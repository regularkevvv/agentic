// Boundary tests keep failures conservative: no failed handoff may acknowledge
// input or retire unfinished execution, even when adapters return bad replies.
package actor

import (
	"context"
	"errors"
	"io"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

var errInjected = errors.New("errInjected boundary failure")

func TestOldSuspensionCannotSettleResumedRun(t *testing.T) {
	w := &worker{}
	waiting := map[sessionloop.RunID]executionWait{"run": {after: 20, command: "new-resolve"}}
	old := sessionloop.Event{Nature: sessionloop.EventAuthoritative, Kind: sessionloop.EventRunSuspended, RunID: "run", Position: sessionloop.Position{Sequence: 10}}
	if err := w.observe(t.Context(), Lease{}, streamResult{event: old}, waiting); err != nil {
		t.Fatal(err)
	}
	if len(waiting) != 1 {
		t.Fatal("old suspension released resumed execution")
	}
	bounce := sessionloop.Event{Nature: sessionloop.EventAuthoritative, Kind: sessionloop.EventSessionState, State: sessionloop.StateSuspended, RunID: "run", CommandID: "old-resolve"}
	if err := w.observe(t.Context(), Lease{}, streamResult{event: bounce}, waiting); err != nil || len(waiting) != 1 {
		t.Fatal("old live-only bounce released resumed execution", err)
	}
	old.Position.Sequence = 21
	if err := w.observe(t.Context(), Lease{}, streamResult{event: old}, waiting); err != nil {
		t.Fatal(err)
	}
	if len(waiting) != 0 {
		t.Fatal("new suspension did not settle the wait")
	}
	waiting["run"] = executionWait{after: 20, command: "new-resolve"}
	bounce.CommandID = "new-resolve"
	if err := w.observe(t.Context(), Lease{}, streamResult{event: bounce}, waiting); err != nil || len(waiting) != 0 {
		t.Fatal("current live-only bounce did not settle the wait", err)
	}
}

type adapterStub struct {
	Adapter
	guarantee                                sessionloop.AcceptanceGuarantee
	acquireErr, pendingErr, ackErr, renewErr error
	inputs                                   []Command
	releases, acks                           atomic.Int32
	released                                 func() error
	read                                     func(context.Context) (ActorID, error)
}

func (a *adapterStub) Guarantee() sessionloop.AcceptanceGuarantee {
	if a.guarantee != "" {
		return a.guarantee
	}
	return sessionloop.AcceptanceDurable
}
func (a *adapterStub) Receive(ctx context.Context) (ActorID, error) {
	if a.read != nil {
		return a.read(ctx)
	}
	<-ctx.Done()
	return "", ctx.Err()
}
func (a *adapterStub) Acquire(context.Context, ActorID, string, time.Duration) (Lease, error) {
	return Lease{ActorID: "a", Owner: "worker", Fence: 1}, a.acquireErr
}
func (a *adapterStub) Renew(_ context.Context, l Lease, _ time.Duration) (Lease, error) {
	return l, a.renewErr
}
func (a *adapterStub) Release(context.Context, Lease) error {
	a.releases.Add(1)
	if a.released != nil {
		return a.released()
	}
	return nil
}
func (a *adapterStub) Pending(context.Context, Lease, uint64, int) ([]Command, error) {
	return a.inputs, a.pendingErr
}
func (a *adapterStub) Acknowledge(context.Context, Lease, CommandID, sessionloop.Receipt) error {
	a.acks.Add(1)
	return a.ackErr
}

type sessionStub struct {
	caps                   sessionloop.Capabilities
	snapshot               func() (sessionloop.Snapshot, error)
	lookup                 func(sessionloop.Command) (sessionloop.Receipt, bool, error)
	dispatch               func(sessionloop.Command) (sessionloop.Receipt, error)
	subscribeErr, closeErr error
	stream                 sessionloop.Stream
	closed                 bool
	abandoned              bool
}

func (s *sessionStub) Reject(_ context.Context, _ sessionloop.Command, reason sessionloop.Rejection) (sessionloop.Receipt, error) {
	r := goodReceipt()
	r.Rejection = reason
	r.RunID = ""
	s.lookup = foundReceipt(r)
	return r, nil
}

func (s *sessionStub) ID() sessionloop.SessionID { return "journal" }
func (s *sessionStub) Capabilities() sessionloop.Capabilities {
	if s.caps != nil {
		return s.caps
	}
	return sessionloop.NewCapabilities(sessionloop.CapabilityDurableAcceptance, sessionloop.CapabilityIdempotentDispatch)
}
func (s *sessionStub) Dispatch(_ context.Context, c sessionloop.Command) (sessionloop.Receipt, error) {
	if s.dispatch != nil {
		return s.dispatch(c)
	}
	return goodReceipt(), nil
}
func (s *sessionStub) Acceptance(_ context.Context, c sessionloop.Command) (sessionloop.Receipt, bool, error) {
	if s.lookup != nil {
		return s.lookup(c)
	}
	return goodReceipt(), true, nil
}
func (s *sessionStub) Snapshot(context.Context) (sessionloop.Snapshot, error) {
	if s.snapshot != nil {
		return s.snapshot()
	}
	return sessionloop.Snapshot{SessionID: "journal", State: sessionloop.StateIdle}, nil
}
func (s *sessionStub) Subscribe(context.Context, sessionloop.SubscribeOptions) (sessionloop.Stream, error) {
	if s.stream != nil {
		return s.stream, s.subscribeErr
	}
	return &blockedStream{}, s.subscribeErr
}
func (s *sessionStub) Close(context.Context) error { s.closed = true; return s.closeErr }
func (s *sessionStub) Abandon(context.Context) error {
	s.closed, s.abandoned = true, true
	return s.closeErr
}

type blockedStream struct {
	err   error
	event *sessionloop.Event
}

func (s *blockedStream) Next(ctx context.Context) (sessionloop.Event, error) {
	if s.err != nil {
		return sessionloop.Event{}, s.err
	}
	if s.event != nil {
		return *s.event, nil
	}
	<-ctx.Done()
	return sessionloop.Event{}, ctx.Err()
}
func (*blockedStream) Close() error { return nil }

func goodInput() Command {
	return Command{ID: "one", ActorID: "a", Sequence: 1, Command: sessionloop.Command{Kind: sessionloop.CommandStart, Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "hello"}}}}}
}
func goodReceipt() sessionloop.Receipt {
	return sessionloop.Receipt{CommandID: "one", SessionID: "journal", RunID: "run", Position: sessionloop.Position{Sequence: 1, Token: "entry"}, Guarantee: sessionloop.AcceptanceDurable}
}
func fixture(a *adapterStub, s Session) *worker {
	return &worker{cfg: Config{Owner: "worker", Adapter: a, SessionOpener: SessionOpenerFunc(func(context.Context, Lease) (Session, error) { return s, nil }), LeaseTTL: 30 * time.Millisecond, PollInterval: time.Millisecond, CleanupTimeout: time.Second, BatchSize: 4, EventBuffer: 128}}
}

func TestNormalizeAndWorkerConfiguration(t *testing.T) {
	for _, edit := range []func(*Command){func(c *Command) { c.ID = "" }, func(c *Command) { c.ActorID = "" }, func(c *Command) { c.Command.ID = "different" }, func(c *Command) { c.Command.Kind = "invalid" }, func(c *Command) { c.Command.Input.Blocks[0].Kind = "invalid" }} {
		c := goodInput()
		edit(&c)
		if _, err := c.Normalize(); err == nil {
			t.Fatal("invalid command normalized")
		}
	}
	c := goodInput()
	normalized, err := c.Normalize()
	if err != nil {
		t.Fatal(err)
	}
	normalized.Command.Input.Blocks[0].Text = "copy"
	if c.Command.Input.Blocks[0].Text != "hello" {
		t.Fatal("normalization aliases input")
	}
	c.Command.IdempotencyKey = "explicit-key"
	normalized, err = c.Normalize()
	if err != nil || normalized.Command.IdempotencyKey != "explicit-key" {
		t.Fatal("normalization replaced explicit immutable key", normalized, err)
	}
	if _, err := NewWorker(Config{}); err == nil {
		t.Fatal("empty config accepted")
	}
	a := &adapterStub{guarantee: "false guarantee"}
	cfg := fixture(a, &sessionStub{}).cfg
	if _, err := NewWorker(cfg); err == nil {
		t.Fatal("invalid guarantee accepted")
	}
	a.guarantee = ""
	cfg.LeaseTTL = time.Nanosecond
	if _, err := NewWorker(cfg); err == nil {
		t.Fatal("TTL too short accepted")
	}
	got, err := NewWorker(Config{Owner: "w", Adapter: a, SessionOpener: cfg.SessionOpener})
	if err != nil {
		t.Fatal(err)
	}
	w := got.(*worker)
	if w.cfg.LeaseTTL != 30*time.Second || w.cfg.MaxActors != 8 || w.cfg.BatchSize != 64 || w.cfg.EventBuffer != 256 {
		t.Fatal(w.cfg)
	}
	w.running.Store(true)
	if err := w.Run(t.Context()); !errors.Is(err, ErrWorkerRunning) {
		t.Fatal(err)
	}
}

func TestDeliveryFailuresNeverRemoveInput(t *testing.T) {
	cases := []struct {
		name string
		edit func(*sessionStub, *Command)
		want error
	}{
		{"invalid envelope", func(_ *sessionStub, c *Command) { c.ID = "" }, sessionloop.ErrInvalidCommand},
		{"lookup failure", func(s *sessionStub, _ *Command) {
			s.lookup = func(sessionloop.Command) (sessionloop.Receipt, bool, error) {
				return sessionloop.Receipt{}, false, errInjected
			}
		}, errInjected},
		{"dispatch failure", func(s *sessionStub, _ *Command) {
			s.dispatch = func(sessionloop.Command) (sessionloop.Receipt, error) { return sessionloop.Receipt{}, errInjected }
		}, errInjected},
		{"post-commit lookup failure", func(s *sessionStub, _ *Command) {
			n := 0
			s.lookup = func(sessionloop.Command) (sessionloop.Receipt, bool, error) {
				n++
				if n == 1 {
					return sessionloop.Receipt{}, false, nil
				}
				return sessionloop.Receipt{}, false, errInjected
			}
		}, errInjected},
		{"no journal receipt", func(_ *sessionStub, _ *Command) {}, ErrInvalidReceipt},
		{"different journal receipt", func(s *sessionStub, _ *Command) {
			n := 0
			s.lookup = func(sessionloop.Command) (sessionloop.Receipt, bool, error) {
				n++
				r := goodReceipt()
				r.RunID = "another"
				return r, n > 1, nil
			}
		}, ErrInvalidReceipt},
		{"wrong identity", func(s *sessionStub, _ *Command) {
			r := goodReceipt()
			r.CommandID = "wrong"
			s.lookup = foundReceipt(r)
		}, ErrInvalidReceipt},
		{"wrong journal", func(s *sessionStub, _ *Command) {
			r := goodReceipt()
			r.SessionID = "other"
			s.lookup = foundReceipt(r)
		}, ErrInvalidReceipt},
		{"missing journal", func(s *sessionStub, _ *Command) { r := goodReceipt(); r.SessionID = ""; s.lookup = foundReceipt(r) }, ErrInvalidReceipt},
		{"unknown guarantee", func(s *sessionStub, _ *Command) {
			r := goodReceipt()
			r.Guarantee = "unknown"
			s.lookup = foundReceipt(r)
		}, ErrInvalidReceipt},
		{"volatile receipt", func(s *sessionStub, _ *Command) {
			r := goodReceipt()
			r.Guarantee = sessionloop.AcceptanceAccepted
			s.lookup = foundReceipt(r)
		}, ErrInvalidReceipt},
		{"missing durable position", func(s *sessionStub, _ *Command) {
			r := goodReceipt()
			r.Position = sessionloop.Position{}
			s.lookup = foundReceipt(r)
		}, ErrInvalidReceipt},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			a := &adapterStub{}
			s := &sessionStub{lookup: func(sessionloop.Command) (sessionloop.Receipt, bool, error) { return sessionloop.Receipt{}, false, nil }}
			c := goodInput()
			test.edit(s, &c)
			_, err := fixture(a, s).deliver(t.Context(), Lease{}, s, sessionloop.Snapshot{State: sessionloop.StateIdle}, c)
			if !errors.Is(err, test.want) || a.acks.Load() != 0 {
				t.Fatalf("err=%v acknowledgments=%d", err, a.acks.Load())
			}
		})
	}
	for _, busy := range []bool{true, false} {
		a := &adapterStub{}
		s := &sessionStub{lookup: func(sessionloop.Command) (sessionloop.Receipt, bool, error) { return sessionloop.Receipt{}, false, nil }, dispatch: func(sessionloop.Command) (sessionloop.Receipt, error) {
			return sessionloop.Receipt{}, sessionloop.ErrSessionBusy
		}}
		state := sessionloop.StateIdle
		if busy {
			state = sessionloop.StateRunning
		}
		if _, err := fixture(a, s).deliver(t.Context(), Lease{}, s, sessionloop.Snapshot{State: state}, goodInput()); err != nil || a.acks.Load() != 0 {
			t.Fatalf("busy=%v %v", busy, err)
		}
	}
	a := &adapterStub{ackErr: errInjected}
	s := &sessionStub{}
	if _, err := fixture(a, s).deliver(t.Context(), Lease{}, s, sessionloop.Snapshot{}, goodInput()); !errors.Is(err, errInjected) {
		t.Fatal(err)
	}
}

func TestOnlyExplicitTerminalRejectionsPermitAcknowledgment(t *testing.T) {
	for _, cause := range []error{sessionloop.ErrStaleRun, sessionloop.ErrNotRunning, sessionloop.ErrUnsupported, sessionloop.ErrInvalidCommand} {
		a := &adapterStub{}
		s := &sessionStub{
			lookup:   func(sessionloop.Command) (sessionloop.Receipt, bool, error) { return sessionloop.Receipt{}, false, nil },
			dispatch: func(sessionloop.Command) (sessionloop.Receipt, error) { return sessionloop.Receipt{}, cause },
		}
		if _, err := fixture(a, s).deliver(t.Context(), Lease{}, s, sessionloop.Snapshot{State: sessionloop.StateIdle}, goodInput()); err != nil || a.acks.Load() != 1 {
			t.Fatalf("rejection=%v ack=%d error=%v", cause, a.acks.Load(), err)
		}
		r, found, err := s.Acceptance(t.Context(), goodInput().Command)
		if err != nil || !found || r.Rejection != terminalRejection(cause) {
			t.Fatalf("unrecorded rejection=%+v %v", r, err)
		}
	}
	if terminalRejection(errInjected) != "" || terminalRejection(sessionloop.ErrCommandConflict) != "" {
		t.Fatal("ambiguous error classified as terminal")
	}
}

func foundReceipt(r sessionloop.Receipt) func(sessionloop.Command) (sessionloop.Receipt, bool, error) {
	return func(sessionloop.Command) (sessionloop.Receipt, bool, error) { return r, true, nil }
}

type snapshotSink struct{ err error }

func (s snapshotSink) Observe(context.Context, Lease, sessionloop.Event) error { return s.err }
func (s snapshotSink) ObserveSnapshot(context.Context, Lease, sessionloop.Snapshot) error {
	return s.err
}

func TestFailedActivationNeverRetiresExecution(t *testing.T) {
	for _, point := range []string{"acquire", "open", "nil session", "capability", "durability", "snapshot", "snapshot sink", "subscribe", "pending", "wrong actor", "wrong order", "delivery", "post-delivery snapshot", "faulted", "closed", "close", "stream", "event sink"} {
		t.Run(point, func(t *testing.T) {
			a := &adapterStub{}
			s := &sessionStub{}
			w := fixture(a, s)
			switch point {
			case "acquire":
				a.acquireErr = errInjected
			case "open":
				w.cfg.SessionOpener = SessionOpenerFunc(func(context.Context, Lease) (Session, error) { return nil, errInjected })
			case "nil session":
				w.cfg.SessionOpener = SessionOpenerFunc(func(context.Context, Lease) (Session, error) { return nil, nil })
			case "capability":
				s.caps = sessionloop.Capabilities{}
			case "durability":
				s.caps = sessionloop.NewCapabilities(sessionloop.CapabilityIdempotentDispatch)
			case "snapshot":
				s.snapshot = func() (sessionloop.Snapshot, error) { return sessionloop.Snapshot{}, errInjected }
			case "snapshot sink":
				w.cfg.EventSink = snapshotSink{errInjected}
			case "subscribe":
				s.subscribeErr = errInjected
			case "pending":
				a.pendingErr = errInjected
			case "wrong actor":
				c := goodInput()
				c.ActorID = "other"
				a.inputs = []Command{c}
			case "wrong order":
				c := goodInput()
				c.Sequence = 0
				a.inputs = []Command{c}
			case "delivery":
				a.inputs = []Command{goodInput()}
				s.lookup = func(sessionloop.Command) (sessionloop.Receipt, bool, error) {
					return sessionloop.Receipt{}, false, errInjected
				}
			case "post-delivery snapshot":
				a.inputs = []Command{goodInput()}
				n := 0
				s.snapshot = func() (sessionloop.Snapshot, error) {
					n++
					if n == 3 {
						return sessionloop.Snapshot{}, errInjected
					}
					return sessionloop.Snapshot{State: sessionloop.StateIdle}, nil
				}
			case "faulted":
				s.snapshot = func() (sessionloop.Snapshot, error) {
					return sessionloop.Snapshot{State: sessionloop.StateFaulted}, nil
				}
			case "closed":
				s.snapshot = func() (sessionloop.Snapshot, error) { return sessionloop.Snapshot{State: sessionloop.StateClosed}, nil }
			case "close":
				s.closeErr = errInjected
			case "stream":
				s.snapshot = func() (sessionloop.Snapshot, error) {
					return sessionloop.Snapshot{State: sessionloop.StateRunning}, nil
				}
				s.stream = &blockedStream{err: io.EOF}
			case "event sink":
				s.snapshot = func() (sessionloop.Snapshot, error) {
					return sessionloop.Snapshot{State: sessionloop.StateRunning}, nil
				}
				s.stream = &blockedStream{event: &sessionloop.Event{}}
				w.cfg.EventSink = EventSinkFunc(func(context.Context, Lease, sessionloop.Event) error { return errInjected })
			}
			if err := w.runActor(t.Context(), "a"); err == nil || a.releases.Load() != 0 {
				t.Fatalf("err=%v releases=%d", err, a.releases.Load())
			}
		})
	}
}

func TestRenewalLossCancelsSessionAndRetainsRecovery(t *testing.T) {
	a := &adapterStub{renewErr: errInjected}
	s := &sessionStub{snapshot: func() (sessionloop.Snapshot, error) {
		return sessionloop.Snapshot{State: sessionloop.StateRunning}, nil
	}}
	w := fixture(a, s)
	if err := w.runActor(t.Context(), "a"); !errors.Is(err, ErrLeaseLost) || !errors.Is(err, errInjected) || !s.abandoned || a.releases.Load() != 0 {
		t.Fatalf("err=%v closed=%v releases=%d", err, s.closed, a.releases.Load())
	}
}

func TestClosePrecedesRetirementAndReceiveErrorsRetry(t *testing.T) {
	a := &adapterStub{}
	s := &sessionStub{}
	a.released = func() error {
		if !s.closed {
			t.Error("release preceded Close")
		}
		return nil
	}
	w := fixture(a, s)
	w.cfg.EventSink = snapshotSink{}
	if err := w.runActor(t.Context(), "a"); err != nil || a.releases.Load() != 1 {
		t.Fatal(err, a.releases.Load())
	}
	if quiescent(sessionloop.Snapshot{State: sessionloop.StateSuspended}) {
		t.Fatal("unbacked suspension considered quiescent")
	}
	if !quiescent(sessionloop.Snapshot{State: sessionloop.StateSuspended, Suspension: &sessionloop.Suspension{ID: "durable"}}) {
		t.Fatal("persisted suspension not parked")
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	w.cfg.MaxActors = 1
	w.cfg.RetryInterval = time.Millisecond
	a.read = func(context.Context) (ActorID, error) { return "", errInjected }
	w.cfg.OnError = func(_ ActorID, err error) {
		if !errors.Is(err, errInjected) {
			t.Error(err)
		}
		cancel()
	}
	if err := w.Run(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
