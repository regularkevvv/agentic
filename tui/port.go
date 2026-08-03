// Package tui defines a presentation-neutral interactive-session client and a
// Bubble Tea application that can operate any compatible durable harness.
package tui

import "context"

type Host interface {
	NewSession(context.Context, SessionOptions) (Session, error)
	ResumeSession(context.Context, string) (Session, error)
}

type Session interface {
	ID() string
	Snapshot(context.Context) (Snapshot, error)
	Subscribe(SubscribeOptions) Subscription

	Submit(context.Context, Input) error
	Steer(context.Context, Input) error
	FollowUp(context.Context, Input) error
	NextTurn(context.Context, Input) error
	Resolve(context.Context, Resolution) error
	Interrupt(context.Context) error
	Close(context.Context) error
}

type Subscription interface {
	Events() <-chan Event
	Errors() <-chan error
	Close()
}

type SessionOptions struct{}

type SubscribeOptions struct {
	AfterCursor uint64
	Buffer      int
	Preview     bool
}

// Optional host extensions keep the minimum runtime port small.
type SessionInfo struct {
	ID      string
	State   State
	Updated string
}

type SessionLister interface {
	ListSessions(context.Context) ([]SessionInfo, error)
}

type TranscriptExporter interface {
	ExportTranscript(context.Context, string) (string, error)
}
