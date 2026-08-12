// Package memory provides deterministic in-process actor infrastructure for
// tests, CLIs, and single-process demos. It deliberately does not claim crash
// durability; production applications should replace it with durable stores.
package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
)

type Store struct {
	mu        sync.Mutex
	commands  map[actor.ActorID][]actor.Command
	byID      map[actor.CommandID]actor.Command
	leases    map[actor.ActorID]actor.Lease
	nextSeq   map[actor.ActorID]uint64
	nextFence map[actor.ActorID]actor.Fence
	now       func() time.Time
}

func NewStore() *Store {
	return &Store{
		commands:  make(map[actor.ActorID][]actor.Command),
		byID:      make(map[actor.CommandID]actor.Command),
		leases:    make(map[actor.ActorID]actor.Lease),
		nextSeq:   make(map[actor.ActorID]uint64),
		nextFence: make(map[actor.ActorID]actor.Fence),
		now:       time.Now,
	}
}

func (s *Store) Enqueue(ctx context.Context, command actor.Command) (actor.Submission, error) {
	if err := ctx.Err(); err != nil {
		return actor.Submission{}, err
	}
	if command.ID == "" || command.ActorID == "" {
		return actor.Submission{}, errors.New("memory actor store: command ID and actor ID are required")
	}
	if err := command.Command.Validate(); err != nil {
		return actor.Submission{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if previous, exists := s.byID[command.ID]; exists {
		if previous.ActorID != command.ActorID || !sameCommand(previous.Command, command.Command) {
			return actor.Submission{}, actor.ErrCommandConflict
		}
		return actor.Submission{ID: previous.ID, ActorID: previous.ActorID, Sequence: previous.Sequence, Duplicate: true}, nil
	}
	s.nextSeq[command.ActorID]++
	command.Sequence = s.nextSeq[command.ActorID]
	command.State = actor.CommandPending
	if command.Created.IsZero() {
		command.Created = s.now().UTC()
	}
	command.Command = command.Command.Clone()
	s.commands[command.ActorID] = append(s.commands[command.ActorID], cloneCommand(command))
	s.byID[command.ID] = cloneCommand(command)
	return actor.Submission{ID: command.ID, ActorID: command.ActorID, Sequence: command.Sequence}, nil
}

func sameCommand(left, right sessionloop.Command) bool {
	l, _ := json.Marshal(left)
	r, _ := json.Marshal(right)
	return string(l) == string(r)
}

func (s *Store) Pending(ctx context.Context, lease actor.Lease, limit int) ([]actor.Command, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.validateLeaseLocked(lease); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = len(s.commands[lease.ActorID])
	}
	result := make([]actor.Command, 0, limit)
	for _, command := range s.commands[lease.ActorID] {
		if command.State != actor.CommandPending && command.State != actor.CommandDispatched {
			continue
		}
		result = append(result, cloneCommand(command))
		if len(result) == limit {
			break
		}
	}
	return result, nil
}

func (s *Store) MarkDispatched(ctx context.Context, lease actor.Lease, id actor.CommandID, receipt sessionloop.Receipt) error {
	return s.mutate(ctx, lease, id, func(command *actor.Command) error {
		if command.State == actor.CommandDispatched && command.Receipt != nil && *command.Receipt == receipt {
			return nil
		}
		if command.State != actor.CommandPending {
			return actor.ErrCommandConflict
		}
		copy := receipt
		command.Receipt = &copy
		command.State = actor.CommandDispatched
		return nil
	})
}

func (s *Store) MarkSettled(ctx context.Context, lease actor.Lease, id actor.CommandID, outcome *sessionloop.RunOutcome) error {
	return s.mutate(ctx, lease, id, func(command *actor.Command) error {
		if command.State == actor.CommandSettled {
			return nil
		}
		if command.State != actor.CommandDispatched {
			return actor.ErrCommandConflict
		}
		command.State = actor.CommandSettled
		if outcome != nil {
			copy := outcome.Clone()
			command.Outcome = &copy
		}
		return nil
	})
}

func (s *Store) MarkFailed(ctx context.Context, lease actor.Lease, id actor.CommandID, cause error) error {
	return s.mutate(ctx, lease, id, func(command *actor.Command) error {
		command.State = actor.CommandFailed
		if cause != nil {
			command.Error = cause.Error()
		}
		return nil
	})
}

func (s *Store) mutate(ctx context.Context, lease actor.Lease, id actor.CommandID, apply func(*actor.Command) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.validateLeaseLocked(lease); err != nil {
		return err
	}
	commands := s.commands[lease.ActorID]
	for index := range commands {
		if commands[index].ID != id {
			continue
		}
		if err := apply(&commands[index]); err != nil {
			return err
		}
		s.commands[lease.ActorID] = commands
		s.byID[id] = cloneCommand(commands[index])
		return nil
	}
	return actor.ErrCommandNotFound
}

func (s *Store) ReadyActors(ctx context.Context, limit int) ([]actor.ActorID, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var result []actor.ActorID
	for id, commands := range s.commands {
		for _, command := range commands {
			if command.State == actor.CommandPending || command.State == actor.CommandDispatched {
				result = append(result, id)
				break
			}
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	if limit > 0 && len(result) > limit {
		result = result[:limit]
	}
	return result, nil
}

func (s *Store) Acquire(ctx context.Context, id actor.ActorID, owner string, ttl time.Duration) (actor.Lease, error) {
	if err := ctx.Err(); err != nil {
		return actor.Lease{}, err
	}
	if id == "" || owner == "" || ttl <= 0 {
		return actor.Lease{}, errors.New("memory actor store: actor, owner, and positive TTL are required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now().UTC()
	if current, exists := s.leases[id]; exists && current.Expires.After(now) {
		return actor.Lease{}, actor.ErrLeaseHeld
	}
	s.nextFence[id]++
	lease := actor.Lease{ActorID: id, Owner: owner, Fence: s.nextFence[id], Expires: now.Add(ttl)}
	s.leases[id] = lease
	return lease, nil
}

func (s *Store) Renew(ctx context.Context, lease actor.Lease, ttl time.Duration) (actor.Lease, error) {
	if err := ctx.Err(); err != nil {
		return actor.Lease{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.validateLeaseLocked(lease); err != nil {
		return actor.Lease{}, err
	}
	lease.Expires = s.now().UTC().Add(ttl)
	s.leases[lease.ActorID] = lease
	return lease, nil
}

func (s *Store) Release(ctx context.Context, lease actor.Lease) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	current, exists := s.leases[lease.ActorID]
	if !exists {
		return nil
	}
	if current.Owner != lease.Owner || current.Fence != lease.Fence {
		return actor.ErrLeaseLost
	}
	delete(s.leases, lease.ActorID)
	return nil
}

func (s *Store) validateLeaseLocked(lease actor.Lease) error {
	current, exists := s.leases[lease.ActorID]
	if !exists || current.Owner != lease.Owner || current.Fence != lease.Fence || !current.Expires.After(s.now().UTC()) {
		return actor.ErrLeaseLost
	}
	return nil
}

func cloneCommand(command actor.Command) actor.Command {
	command.Command = command.Command.Clone()
	if command.Receipt != nil {
		copy := *command.Receipt
		command.Receipt = &copy
	}
	if command.Outcome != nil {
		copy := command.Outcome.Clone()
		command.Outcome = &copy
	}
	return command
}

type Notifier struct {
	mu     sync.Mutex
	nextID uint64
	subs   map[uint64]chan actor.ActorID
	closed bool
}

func NewNotifier() *Notifier { return &Notifier{subs: make(map[uint64]chan actor.ActorID)} }

func (n *Notifier) Publish(ctx context.Context, id actor.ActorID) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.closed {
		return io.EOF
	}
	for _, target := range n.subs {
		select {
		case target <- id:
		default:
		}
	}
	return nil
}

func (n *Notifier) Subscribe(ctx context.Context) (actor.Subscription, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.closed {
		return nil, io.EOF
	}
	n.nextID++
	sub := &subscription{owner: n, id: n.nextID, events: make(chan actor.ActorID, 64), done: make(chan struct{})}
	n.subs[sub.id] = sub.events
	return sub, nil
}

func (n *Notifier) Close() {
	n.mu.Lock()
	n.closed = true
	for id, events := range n.subs {
		close(events)
		delete(n.subs, id)
	}
	n.mu.Unlock()
}

type subscription struct {
	owner  *Notifier
	id     uint64
	events chan actor.ActorID
	done   chan struct{}
	once   sync.Once
}

func (s *subscription) Next(ctx context.Context) (actor.ActorID, error) {
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-s.done:
		return "", io.EOF
	case id, ok := <-s.events:
		if !ok {
			return "", io.EOF
		}
		return id, nil
	}
}

func (s *subscription) Close() error {
	s.once.Do(func() {
		s.owner.mu.Lock()
		delete(s.owner.subs, s.id)
		s.owner.mu.Unlock()
		close(s.done)
	})
	return nil
}

var (
	_ actor.CommandStore = (*Store)(nil)
	_ actor.LeaseStore   = (*Store)(nil)
	_ actor.Notifier     = (*Notifier)(nil)
)

func (s *Store) Command(id actor.CommandID) (actor.Command, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	command, exists := s.byID[id]
	if !exists {
		return actor.Command{}, fmt.Errorf("%w: %s", actor.ErrCommandNotFound, id)
	}
	return cloneCommand(command), nil
}
