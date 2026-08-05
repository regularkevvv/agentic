package harness

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestNewSessionLoopHostValidatesInputs(t *testing.T) {
	if _, err := NewSessionLoopHost[string](nil); err == nil {
		t.Fatal("nil runtime accepted")
	}
	runtime, err := NewRuntime[string](&facadeDriver{}, runtimeConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewSessionLoopHost(runtime, nil); err == nil {
		t.Fatal("nil option accepted")
	}
	if _, err := NewSessionLoopHost(runtime, WithSessionLoopOutputProjector[string](nil)); err == nil {
		t.Fatal("nil output projector accepted")
	}
	if _, err := NewSessionLoopHost(runtime, WithSessionLoopSuspensionProjector[string](nil)); err == nil {
		t.Fatal("nil suspension projector accepted")
	}
	if _, err := NewSessionLoopHost(runtime, WithSessionLoopSessionOptions[string](nil)); err == nil {
		t.Fatal("nil session option accepted")
	}
}

func TestSessionLoopHostOpenSessionValidation(t *testing.T) {
	runtime, err := NewRuntime[string](&facadeDriver{}, runtimeConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	host, err := NewSessionLoopHost(runtime)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := host.OpenSession(sessionLoopTestContext(t), ""); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatalf("empty session ID err = %v", err)
	}
	// A second concurrent handle maps ErrSessionOpen onto the portable
	// sentinel while preserving the harness identity.
	first, err := host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = first.Close(context.Background()) }()
	if _, err := host.OpenSession(sessionLoopTestContext(t), first.ID()); !errors.Is(err, sessionloop.ErrSessionOpen) || !errors.Is(err, ErrSessionOpen) {
		t.Fatalf("second open err = %v", err)
	}
}

func TestSessionLoopHostSessionOptionsAndMeta(t *testing.T) {
	config := runtimeConfig(t)
	config.Sessions = storememory.New()
	runtime, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	host, err := NewSessionLoopHost(runtime,
		WithSessionLoopSessionOptions[string](WithInitialHistory(agentic.NewTextMessage(agentic.RoleUser, "seeded"))),
		WithSessionLoopOutputProjector[string](func(output string) (json.RawMessage, error) {
			return json.Marshal(output)
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	// SessionOptions.Meta is correlation-only and deliberately ignored.
	session, err := host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{
		Meta: map[string]string{"trace": "abc"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(context.Background()) }()
	snapshot, err := session.Snapshot(sessionLoopTestContext(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Entries) != 1 || snapshot.Entries[0].Origin != sessionloop.OriginStart ||
		snapshot.Entries[0].RunID != "" || snapshot.Entries[0].Content[0].Text != "seeded" {
		t.Fatalf("seeded history projection = %#v", snapshot.Entries)
	}
	if !session.Capabilities().Supports(sessionloop.CapabilityStructuredOutput) {
		t.Fatalf("capabilities = %v", session.Capabilities())
	}
}

func TestProjectPermissionSuspension(t *testing.T) {
	// Unsupported deferral kinds project a generic description, no
	// decisions, and no error (they stay resolvable elsewhere).
	unsupported, err := ProjectPermissionSuspension(agentic.Suspension{
		ID: "susp-1", Kind: "custom.kind", Payload: json.RawMessage(`{}`),
	})
	if err != nil {
		t.Fatalf("unsupported kind err = %v", err)
	}
	if unsupported.ID != "susp-1" || unsupported.Kind != "custom.kind" ||
		unsupported.Description == "" || len(unsupported.Decisions) != 0 {
		t.Fatalf("unsupported projection = %#v", unsupported)
	}

	// A malformed envelope of the supported kind is an error.
	if _, err := ProjectPermissionSuspension(agentic.Suspension{
		ID: "susp-2", Kind: harnessruntime.PermissionDeferralKind, Payload: json.RawMessage(`{`),
	}); err == nil {
		t.Fatal("malformed permission envelope accepted")
	}

	// A real permission suspension yields one typed decision per approval.
	env := newSessionLoopEnv(t, storememory.New())
	session, err := env.Host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(context.Background()) }()
	stream, err := session.Subscribe(sessionLoopTestContext(t), sessionloop.SubscribeOptions{Buffer: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	receipt, err := session.Dispatch(sessionLoopTestContext(t), sessionloop.Command{
		Kind: sessionloop.CommandStart,
		Input: &sessionloop.Input{Content: []sessionloop.Block{
			{Kind: sessionloop.BlockText, Text: "please pause"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for {
		event, nextErr := stream.Next(sessionLoopTestContext(t))
		if nextErr != nil {
			t.Fatalf("Next = %v", nextErr)
		}
		if event.Kind != sessionloop.EventRunSuspended {
			continue
		}
		if event.RunID != receipt.RunID || event.Suspension == nil ||
			len(event.Suspension.Decisions) != 1 {
			t.Fatalf("suspended event = %#v", event)
		}
		decision := event.Suspension.Decisions[0]
		if decision.Name != "danger" || decision.ID == "" ||
			decision.Capability == "" || decision.Action == "" {
			t.Fatalf("permission decision = %#v", decision)
		}
		break
	}
}

func TestSessionLoopHostErrorHelpersAndProjectorOption(t *testing.T) {
	if mapSessionLoopHostError(nil) != nil {
		t.Fatal("nil must map to nil")
	}
	plain := errors.New("plain failure")
	if mapSessionLoopHostError(plain) != plain {
		t.Fatal("unknown errors must pass through")
	}
	mapped := mapSessionLoopHostError(ErrSessionOpen)
	if !errors.Is(mapped, sessionloop.ErrSessionOpen) || !errors.Is(mapped, ErrSessionOpen) {
		t.Fatalf("mapped ErrSessionOpen = %v", mapped)
	}

	// A custom suspension projector installs and is used by the view.
	runtime, err := NewRuntime[string](&facadeDriver{}, runtimeConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	host, err := NewSessionLoopHost(runtime,
		WithSessionLoopSuspensionProjector[string](func(value agentic.Suspension) (sessionloop.Suspension, error) {
			return sessionloop.Suspension{ID: value.ID, Kind: value.Kind, Description: "custom"}, nil
		}))
	if err != nil {
		t.Fatal(err)
	}
	session, err := host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

type failingSessIDs struct{}

func (failingSessIDs) New(prefix string) (string, error) {
	return "", errors.New("id generator down: " + prefix)
}

func TestSessionLoopHostNewSessionPropagatesRuntimeErrors(t *testing.T) {
	config := runtimeConfig(t)
	config.IDs = failingSessIDs{}
	runtime, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	host, err := NewSessionLoopHost(runtime)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := host.NewSession(sessionLoopTestContext(t), sessionloop.SessionOptions{}); err == nil {
		t.Fatal("NewSession with a failing ID generator succeeded")
	}
	if _, err := host.OpenSession(sessionLoopTestContext(t), "missing-session"); err == nil {
		t.Fatal("OpenSession of a missing session succeeded")
	}
}
