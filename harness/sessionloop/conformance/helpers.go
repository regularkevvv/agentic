package conformance

import (
	"context"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// watchdog bounds every wait. Cases synchronize only through receipts,
// events, snapshots, and the Gate; the deadline exists purely so a
// non-conforming host fails instead of hanging.
const watchdog = 10 * time.Second

func watchdogContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), watchdog)
	t.Cleanup(cancel)
	return ctx
}

func probeCapabilities(t *testing.T, host sessionloop.Host) sessionloop.Capabilities {
	t.Helper()
	probe, err := host.NewSession(watchdogContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatalf("NewSession for the capability probe failed: %v", err)
	}
	capabilities := probe.Capabilities().Clone()
	if err := probe.Close(context.Background()); err != nil {
		t.Fatalf("closing the capability probe session failed: %v", err)
	}
	return capabilities
}

func newSession(t *testing.T, host sessionloop.Host) sessionloop.Session {
	t.Helper()
	session, err := host.NewSession(watchdogContext(t), sessionloop.SessionOptions{})
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}
	t.Cleanup(func() { _ = session.Close(context.Background()) })
	return session
}

func subscribe(t *testing.T, session sessionloop.Session, options sessionloop.SubscribeOptions) sessionloop.Stream {
	t.Helper()
	stream, err := session.Subscribe(watchdogContext(t), options)
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	return stream
}

func textInput(text string) *sessionloop.Input {
	return &sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: text}}}
}

func scenarioInput(text, scenario string) *sessionloop.Input {
	input := textInput(text)
	input.Meta = map[string]string{MetaScenario: scenario}
	return input
}

func startCommand(input *sessionloop.Input) sessionloop.Command {
	return sessionloop.Command{Kind: sessionloop.CommandStart, Input: input}
}

func dispatch(t *testing.T, session sessionloop.Session, command sessionloop.Command) sessionloop.Receipt {
	t.Helper()
	receipt, err := session.Dispatch(watchdogContext(t), command)
	if err != nil {
		t.Fatalf("Dispatch(%s) failed: %v", command.Kind, err)
	}
	if receipt.CommandID == "" {
		t.Fatalf("Dispatch(%s) receipt has no command ID: %#v", command.Kind, receipt)
	}
	return receipt
}

func nextEvent(t *testing.T, stream sessionloop.Stream) sessionloop.Event {
	t.Helper()
	event, err := stream.Next(watchdogContext(t))
	if err != nil {
		t.Fatalf("Stream.Next failed: %v", err)
	}
	return event
}

// awaitKind reads events until the first one of the given kind and returns
// it together with everything read so far, the match included.
func awaitKind(t *testing.T, stream sessionloop.Stream, kind sessionloop.EventKind) (sessionloop.Event, []sessionloop.Event) {
	t.Helper()
	var seen []sessionloop.Event
	for {
		event := nextEvent(t, stream)
		seen = append(seen, event)
		if event.Kind == kind {
			return event, seen
		}
	}
}

// awaitSettled reads events until the run's settlement and returns the
// settlement together with everything read so far, the settlement included.
func awaitSettled(t *testing.T, stream sessionloop.Stream, runID sessionloop.RunID) (sessionloop.Event, []sessionloop.Event) {
	t.Helper()
	var seen []sessionloop.Event
	for {
		event := nextEvent(t, stream)
		seen = append(seen, event)
		if event.Kind == sessionloop.EventRunSettled && event.RunID == runID {
			if event.Outcome == nil {
				t.Fatalf("run.settled event carries no outcome: %#v", event)
			}
			return event, seen
		}
	}
}

// committedEntries extracts the committed entries of one run in event order.
func committedEntries(events []sessionloop.Event, runID sessionloop.RunID) []sessionloop.Entry {
	var entries []sessionloop.Entry
	for _, event := range events {
		if event.Nature != sessionloop.EventAuthoritative || event.Kind != sessionloop.EventEntryCommitted {
			continue
		}
		if event.Entry == nil || (runID != "" && event.Entry.RunID != runID) {
			continue
		}
		entries = append(entries, *event.Entry)
	}
	return entries
}

func snapshotOf(t *testing.T, session sessionloop.Session) sessionloop.Snapshot {
	t.Helper()
	snapshot, err := session.Snapshot(watchdogContext(t))
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	return snapshot
}

func approveAll(suspension sessionloop.Suspension) sessionloop.Resolution {
	resolution := sessionloop.Resolution{SuspensionID: suspension.ID}
	for _, decision := range suspension.Decisions {
		resolution.Decisions = append(resolution.Decisions, sessionloop.ResolutionDecision{
			ID:     decision.ID,
			Action: sessionloop.ResolutionApprove,
		})
	}
	return resolution
}
