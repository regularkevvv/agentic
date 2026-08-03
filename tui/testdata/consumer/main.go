package main

import (
	"context"
	"errors"
	"fmt"

	uit "github.com/regularkevvv/agentic/tui"
)

type host struct{ session *session }

func (h *host) NewSession(context.Context, uit.SessionOptions) (uit.Session, error) {
	h.session = &session{id: "consumer-session", snapshot: uit.Snapshot{SessionID: "consumer-session", State: uit.StateIdle}}
	return h.session, nil
}

func (h *host) ResumeSession(_ context.Context, id string) (uit.Session, error) {
	if h.session == nil || h.session.id != id {
		return nil, errors.New("session not found")
	}
	return h.session, nil
}

type session struct {
	id       string
	snapshot uit.Snapshot
}

func (s *session) ID() string { return s.id }

func (s *session) Snapshot(context.Context) (uit.Snapshot, error) { return s.snapshot, nil }

func (*session) Subscribe(uit.SubscribeOptions) uit.Subscription { return closedSubscription() }

func (s *session) Submit(_ context.Context, input uit.Input) error {
	if err := input.Validate(); err != nil {
		return err
	}
	s.snapshot.Cursor++
	s.snapshot.Transcript = append(s.snapshot.Transcript,
		uit.Entry{Role: uit.RoleUser, Text: input.Text},
		uit.Entry{Role: uit.RoleAssistant, Text: "compatible host"},
	)
	return nil
}

func (*session) Steer(context.Context, uit.Input) error        { return nil }
func (*session) FollowUp(context.Context, uit.Input) error     { return nil }
func (*session) NextTurn(context.Context, uit.Input) error     { return nil }
func (*session) Resolve(context.Context, uit.Resolution) error { return nil }
func (*session) Interrupt(context.Context) error               { return nil }
func (*session) Close(context.Context) error                   { return nil }

type subscription struct {
	events chan uit.Event
	errors chan error
}

func closedSubscription() uit.Subscription {
	events := make(chan uit.Event)
	errorsChannel := make(chan error)
	close(events)
	close(errorsChannel)
	return &subscription{events: events, errors: errorsChannel}
}

func (s *subscription) Events() <-chan uit.Event { return s.events }
func (s *subscription) Errors() <-chan error     { return s.errors }
func (*subscription) Close()                     {}

func main() {
	ctx := context.Background()
	compatible := &host{}
	current, err := compatible.NewSession(ctx, uit.SessionOptions{})
	if err != nil {
		panic(err)
	}
	if err := current.Submit(ctx, uit.Input{Text: "consumer proof"}); err != nil {
		panic(err)
	}
	snapshot, err := current.Snapshot(ctx)
	if err != nil || snapshot.SessionID != current.ID() || snapshot.Cursor != 1 || len(snapshot.Transcript) != 2 {
		panic(fmt.Sprintf("incompatible snapshot: %#v, %v", snapshot, err))
	}
	fmt.Printf("compatible_host=%s state=%s cursor=%d transcript=%d\n", snapshot.SessionID, snapshot.State, snapshot.Cursor, len(snapshot.Transcript))
}

var _ uit.Host = (*host)(nil)
var _ uit.Session = (*session)(nil)
