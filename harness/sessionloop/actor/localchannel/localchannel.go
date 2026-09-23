// Package localchannel is the process-local receive flavor of the actor contract.
// A mutex linearizes state changes; a condition channel wakes receivers, which
// always recheck retained state. A timer makes lease expiry discoverable without
// another submission. Process death loses this store: it never claims durability.
package localchannel

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
)

type sessionState struct {
	history        map[actor.CommandID]actor.Command // one immutable identity record
	pending        map[actor.CommandID]struct{}      // disposable IDs, never a second payload
	sequence       uint64
	generation     actor.Fence
	lease          actor.Lease
	needsExecution bool
}

// Store implements Mailbox and the runtime Adapter. Keep it alive across worker
// restarts. Its Guarantee is accepted, never durable; use a durable adapter in
// production. Acknowledged IDs remain in identity history, not in the mailbox.
type Store struct {
	mu       sync.Mutex
	sessions map[actor.ActorID]*sessionState
	order    []actor.ActorID
	next     int
	changed  chan struct{}
	now      func() time.Time
}

func New() *Store {
	return &Store{sessions: make(map[actor.ActorID]*sessionState), changed: make(chan struct{}), now: time.Now}
}

func (*Store) Guarantee() sessionloop.AcceptanceGuarantee { return sessionloop.AcceptanceAccepted }

// Recover schedules opening an already-bound journal even when its mailbox is
// empty. Bootstrap uses this to discover unfinished execution after restoring a
// session. It never creates a journal, grants ownership, or clears pending work.
func (s *Store) Recover(ctx context.Context, id actor.ActorID) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if id == "" {
		return errors.New("localchannel: session identity required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state := s.sessions[id]
	if state == nil {
		state = &sessionState{history: make(map[actor.CommandID]actor.Command), pending: make(map[actor.CommandID]struct{})}
		s.sessions[id] = state
		s.order = append(s.order, id)
	}
	state.needsExecution = true
	s.signal()
	return nil
}

func (s *Store) signal() { close(s.changed); s.changed = make(chan struct{}) }

func (s *Store) Submit(ctx context.Context, command actor.Command) (actor.Submission, error) {
	if err := ctx.Err(); err != nil {
		return actor.Submission{}, err
	}
	command, err := command.Normalize()
	if err != nil {
		return actor.Submission{}, err
	}
	// Validate JSON-bearing inputs before comparing their immutable semantics.
	encoded, err := json.Marshal(command.Command)
	if err != nil {
		return actor.Submission{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state := s.sessions[command.ActorID]
	if state == nil {
		state = &sessionState{history: make(map[actor.CommandID]actor.Command), pending: make(map[actor.CommandID]struct{})}
		s.sessions[command.ActorID] = state
		s.order = append(s.order, command.ActorID)
	}
	duplicate := false
	if previous, ok := state.history[command.ID]; ok {
		original, _ := json.Marshal(previous.Command)
		if string(original) != string(encoded) {
			return actor.Submission{}, actor.ErrCommandConflict
		}
		command, duplicate = previous, true
	} else {
		if state.sequence == math.MaxUint64 {
			return actor.Submission{}, actor.ErrGenerationExhausted
		}
		state.sequence++
		command.Sequence = state.sequence
		state.history[command.ID] = command
		state.pending[command.ID] = struct{}{}
		s.signal()
	}
	return actor.Submission{ID: command.ID, ActorID: command.ActorID, Sequence: command.Sequence,
		Duplicate: duplicate, Guarantee: s.Guarantee()}, nil
}

func (s *Store) Receive(ctx context.Context) (actor.ActorID, error) {
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		s.mu.Lock()
		now := s.now()
		var expiry time.Time
		for i := range len(s.order) {
			index := (s.next + i) % len(s.order)
			id := s.order[index]
			state := s.sessions[id]
			if len(state.pending) == 0 && !state.needsExecution {
				continue
			}
			if state.lease.Owner == "" || !state.lease.Expires.After(now) {
				s.next = (index + 1) % len(s.order)
				s.mu.Unlock()
				return id, nil
			}
			if expiry.IsZero() || state.lease.Expires.Before(expiry) {
				expiry = state.lease.Expires
			}
		}
		changed := s.changed // subscribe and check under the SAME mutex
		s.mu.Unlock()
		var timer *time.Timer
		var expired <-chan time.Time
		if !expiry.IsZero() {
			timer = time.NewTimer(expiry.Sub(now))
			expired = timer.C
		}
		select {
		case <-ctx.Done():
		case <-changed:
		case <-expired:
		}
		if timer != nil {
			timer.Stop()
		}
	}
}

func (s *Store) Acquire(ctx context.Context, id actor.ActorID, owner string, ttl time.Duration) (actor.Lease, error) {
	if err := ctx.Err(); err != nil {
		return actor.Lease{}, err
	}
	if id == "" || owner == "" || ttl <= 0 {
		return actor.Lease{}, errors.New("localchannel: identity and positive TTL required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state := s.sessions[id]
	if state == nil || (len(state.pending) == 0 && !state.needsExecution) {
		return actor.Lease{}, actor.ErrNoWork
	}
	now := s.now()
	if state.lease.Owner != "" && state.lease.Expires.After(now) {
		return actor.Lease{}, actor.ErrLeaseHeld
	}
	if state.generation == actor.Fence(math.MaxUint64) {
		return actor.Lease{}, actor.ErrGenerationExhausted
	}
	state.generation++
	state.lease = actor.Lease{ActorID: id, Owner: owner, Fence: state.generation, Expires: now.Add(ttl)}
	state.needsExecution = true
	s.signal()
	return state.lease, nil
}

func (s *Store) owned(lease actor.Lease) (*sessionState, error) {
	state := s.sessions[lease.ActorID]
	if state == nil || state.lease.Owner == "" || state.lease.Owner != lease.Owner ||
		state.lease.Fence != lease.Fence || !state.lease.Expires.After(s.now()) {
		return nil, actor.ErrLeaseLost
	}
	return state, nil
}

func (s *Store) Renew(ctx context.Context, lease actor.Lease, ttl time.Duration) (actor.Lease, error) {
	if err := ctx.Err(); err != nil {
		return actor.Lease{}, err
	}
	if ttl <= 0 {
		return actor.Lease{}, errors.New("localchannel: positive TTL required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.owned(lease)
	if err != nil {
		return actor.Lease{}, err
	}
	state.lease.Expires = s.now().Add(ttl)
	s.signal()
	return state.lease, nil
}

func (s *Store) Release(ctx context.Context, lease actor.Lease) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.owned(lease)
	if err != nil {
		return err
	}
	// Caller establishes durable quiescence and closes the handle first. This
	// operation NEVER clears pending input, including a concurrent submission.
	state.lease = actor.Lease{}
	state.needsExecution = false
	s.signal()
	return nil
}

func (s *Store) Pending(ctx context.Context, lease actor.Lease, after uint64, limit int) ([]actor.Command, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.owned(lease)
	if err != nil {
		return nil, err
	}
	var result []actor.Command
	for id := range state.pending {
		command := state.history[id]
		if command.Sequence <= after {
			continue
		}
		command.Command = command.Command.Clone()
		result = append(result, command)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Sequence < result[j].Sequence })
	if limit > 0 && len(result) > limit {
		result = result[:limit]
	}
	return result, nil
}

func (s *Store) Acknowledge(ctx context.Context, lease actor.Lease, id actor.CommandID, receipt sessionloop.Receipt) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.owned(lease)
	if err != nil {
		return err
	}
	if _, ok := state.history[id]; !ok {
		return actor.ErrCommandNotFound
	}
	if receipt.CommandID != sessionloop.CommandID(id) || receipt.SessionID == "" ||
		(receipt.Guarantee != sessionloop.AcceptanceAccepted && receipt.Guarantee != sessionloop.AcceptanceDurable) {
		return actor.ErrInvalidReceipt
	}
	delete(state.pending, id)
	// needsExecution remains true even when the last pending ID disappears.
	return nil
}

// Authority binds journal primitives to the SAME mutex and generation as the
// mailbox. It structurally implements harness/store.Authority without importing
// Harness into this standard-library-only module.
func (s *Store) Authority(lease actor.Lease) *Authority { return &Authority{store: s, lease: lease} }

type Authority struct {
	store *Store
	lease actor.Lease
}

// Commit validates and executes one bounded storage primitive under the grant.
// The callback must not reenter this Store or run a model/tool. The mutex is
// held across the mutation, not merely across a preflight ownership check.
func (a *Authority) Commit(ctx context.Context, mutate func(context.Context) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	a.store.mu.Lock()
	defer a.store.mu.Unlock()
	if _, err := a.store.owned(a.lease); err != nil {
		return err
	}
	return mutate(ctx)
}

var _ actor.Adapter = (*Store)(nil)
