package conformance

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func caseDurableReplay(t *testing.T, env Env, capabilities sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	receipt := dispatch(t, session, startCommand(textInput("replay me")))
	if capabilities.Supports(sessionloop.CapabilityDurableAcceptance) && receipt.Position.IsZero() {
		t.Fatalf("a durable receipt must carry a non-zero position: %#v", receipt)
	}
	_, seen := awaitSettled(t, stream, receipt.RunID)

	var previous uint64
	for _, event := range seen {
		if event.Nature != sessionloop.EventAuthoritative {
			continue
		}
		if event.Position.IsZero() {
			t.Fatalf("authoritative event has a zero position under events.replay: %#v", event)
		}
		if event.Position.Sequence <= previous {
			t.Fatalf("authoritative positions are not strictly increasing: %d after %d", event.Position.Sequence, previous)
		}
		previous = event.Position.Sequence
	}

	middle := seen[len(seen)/2].Position
	var expected []sessionloop.Event
	for _, event := range seen {
		if event.Nature == sessionloop.EventAuthoritative && event.Position.Sequence > middle.Sequence {
			expected = append(expected, event)
		}
	}
	replayed := subscribe(t, session, sessionloop.SubscribeOptions{After: middle})
	for index, want := range expected {
		got := nextEvent(t, replayed)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("replay diverged at suffix event %d:\ngot  %#v\nwant %#v", index, got, want)
		}
	}
}

func casePreview(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{Preview: true, Buffer: 256})

	receipt := dispatch(t, session, startCommand(scenarioInput("show progress", ScenarioPreview)))
	_, seen := awaitSettled(t, stream, receipt.RunID)

	previews := 0
	for _, event := range seen {
		if event.Nature == sessionloop.EventPreview {
			previews++
			if event.Kind != sessionloop.EventPreviewDelta {
				t.Fatalf("preview event kind = %q, want preview.delta", event.Kind)
			}
		}
	}
	if previews == 0 {
		t.Fatal("the preview scenario emitted no preview events")
	}

	var committed []sessionloop.EntryID
	for _, entry := range committedEntries(seen, "") {
		committed = append(committed, entry.ID)
	}
	var snapshotIDs []sessionloop.EntryID
	for _, entry := range snapshotOf(t, session).Entries {
		snapshotIDs = append(snapshotIDs, entry.ID)
	}
	if !reflect.DeepEqual(snapshotIDs, committed) {
		t.Fatalf("snapshot entries %v diverge from committed entries %v; previews must never alter transcript truth", snapshotIDs, committed)
	}
}

func caseSteerMidRun(t *testing.T, env Env, _ sessionloop.Capabilities) {
	deliveryMidRun(t, env, sessionloop.CommandSteer, sessionloop.OriginSteer)
}

func caseFollowUpMidRun(t *testing.T, env Env, _ sessionloop.Capabilities) {
	deliveryMidRun(t, env, sessionloop.CommandFollowUp, sessionloop.OriginFollowUp)
}

func deliveryMidRun(t *testing.T, env Env, kind sessionloop.CommandKind, origin sessionloop.EntryOrigin) {
	t.Helper()
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	release := env.Gate.HoldNextRun()
	defer release()
	receipt := dispatch(t, session, startCommand(textInput("primary question")))
	started, _ := awaitKind(t, stream, sessionloop.EventRunStarted)
	if started.RunID != receipt.RunID {
		t.Fatalf("run.started run = %q, receipt run = %q", started.RunID, receipt.RunID)
	}

	dispatch(t, session, sessionloop.Command{Kind: kind, RunID: receipt.RunID, Input: textInput("mid-run addition")})
	release()

	_, seen := awaitSettled(t, stream, receipt.RunID)
	entries := committedEntries(seen, receipt.RunID)
	startIndex, deliveredIndex := -1, -1
	for index, entry := range entries {
		switch entry.Origin {
		case sessionloop.OriginStart:
			startIndex = index
		case origin:
			deliveredIndex = index
		}
	}
	if deliveredIndex < 0 {
		t.Fatalf("no committed entry with origin %q before settlement: %#v", origin, entries)
	}
	if startIndex < 0 || deliveredIndex < startIndex {
		t.Fatalf("origin %q entry at %d precedes the start entry at %d", origin, deliveredIndex, startIndex)
	}
}

func caseNextTurnSurvivesInterrupt(t *testing.T, env Env, capabilities sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	release := env.Gate.HoldNextRun()
	defer release()
	first := dispatch(t, session, startCommand(textInput("busy run")))

	queued := dispatch(t, session, sessionloop.Command{Kind: sessionloop.CommandNextTurn, Input: textInput("queued for later")})
	if queued.QueueID == "" {
		t.Fatalf("next_turn receipt carries no queue identity: %#v", queued)
	}

	if capabilities.Supports(sessionloop.CapabilityInterrupt) {
		dispatch(t, session, sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: first.RunID})
		settled, _ := awaitSettled(t, stream, first.RunID)
		if settled.Outcome.Kind != sessionloop.RunInterrupted {
			t.Fatalf("interrupted run outcome = %q, want interrupted", settled.Outcome.Kind)
		}
	} else {
		release()
		awaitSettled(t, stream, first.RunID)
	}

	pending := snapshotOf(t, session).Pending
	found := false
	for _, item := range pending {
		if item.ID == queued.QueueID {
			found = true
		}
	}
	if !found {
		t.Fatalf("queued next-turn input %q missing from pending after the first run: %#v", queued.QueueID, pending)
	}

	second := dispatch(t, session, startCommand(textInput("next run")))
	_, seen := awaitSettled(t, stream, second.RunID)
	entries := committedEntries(seen, second.RunID)
	nextTurnIndex, startIndex := -1, -1
	for index, entry := range entries {
		switch entry.Origin {
		case sessionloop.OriginNextTurn:
			nextTurnIndex = index
		case sessionloop.OriginStart:
			startIndex = index
		}
	}
	if nextTurnIndex < 0 {
		t.Fatalf("drained next-turn entry missing from the second run: %#v", entries)
	}
	if startIndex < 0 || nextTurnIndex > startIndex {
		t.Fatalf("next-turn entry at %d must precede the start entry at %d", nextTurnIndex, startIndex)
	}
	if remaining := snapshotOf(t, session).Pending; len(remaining) != 0 {
		t.Fatalf("pending queue not drained by the second run: %#v", remaining)
	}
}

func caseInterrupt(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	release := env.Gate.HoldNextRun()
	defer release()
	receipt := dispatch(t, session, startCommand(textInput("interrupt me")))
	dispatch(t, session, sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: receipt.RunID})

	settled, _ := awaitSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunInterrupted {
		t.Fatalf("outcome = %q, want interrupted", settled.Outcome.Kind)
	}
	if snapshot := snapshotOf(t, session); snapshot.State != sessionloop.StateIdle {
		t.Fatalf("state after interrupted settlement = %q, want idle", snapshot.State)
	}
}

func caseSuspensionResolve(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	receipt := dispatch(t, session, startCommand(scenarioInput("please pause", ScenarioSuspend)))
	suspended, _ := awaitKind(t, stream, sessionloop.EventRunSuspended)
	if suspended.RunID != receipt.RunID {
		t.Fatalf("run.suspended run = %q, receipt run = %q", suspended.RunID, receipt.RunID)
	}
	if suspended.Suspension == nil || suspended.Suspension.ID == "" {
		t.Fatalf("run.suspended carries no suspension identity: %#v", suspended)
	}

	snapshot := snapshotOf(t, session)
	if snapshot.State != sessionloop.StateSuspended || snapshot.Suspension == nil {
		t.Fatalf("snapshot during suspension = state %q suspension %#v", snapshot.State, snapshot.Suspension)
	}
	if snapshot.Suspension.ID != suspended.Suspension.ID {
		t.Fatalf("snapshot suspension = %q, event suspension = %q", snapshot.Suspension.ID, suspended.Suspension.ID)
	}

	wrong := approveAll(*snapshot.Suspension)
	wrong.SuspensionID += "-wrong"
	if _, err := session.Dispatch(watchdogContext(t), sessionloop.Command{
		Kind:       sessionloop.CommandResolve,
		RunID:      receipt.RunID,
		Resolution: &wrong,
	}); err == nil {
		t.Fatal("resolving a wrong suspension ID succeeded; it must fail without consuming the suspension")
	}
	after := snapshotOf(t, session)
	if after.State != sessionloop.StateSuspended || after.Suspension == nil || after.Suspension.ID != suspended.Suspension.ID {
		t.Fatalf("wrong-ID resolve consumed the suspension: %#v", after)
	}

	right := approveAll(*snapshot.Suspension)
	dispatch(t, session, sessionloop.Command{
		Kind:       sessionloop.CommandResolve,
		RunID:      receipt.RunID,
		Resolution: &right,
	})
	settled, _ := awaitSettled(t, stream, receipt.RunID)
	if settled.Outcome.RunID != receipt.RunID {
		t.Fatalf("resolution continued run %q, want the suspended run %q", settled.Outcome.RunID, receipt.RunID)
	}
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome after resolution = %q, want completed", settled.Outcome.Kind)
	}
}

// caseSuspensionSurvivesCloseAndReopen proves a suspension is a durable
// pause, not activity Close may destroy (law L11): closing a Suspended
// session must not settle its run, reopening restores the SAME suspension,
// and resolving it after reopen completes the run.
func caseSuspensionSurvivesCloseAndReopen(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	id := session.ID()
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	receipt := dispatch(t, session, startCommand(scenarioInput("please pause across close", ScenarioSuspend)))
	suspended, _ := awaitKind(t, stream, sessionloop.EventRunSuspended)
	if suspended.Suspension == nil || suspended.Suspension.ID == "" {
		t.Fatalf("run.suspended carries no suspension identity: %#v", suspended)
	}

	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("Close of a suspended session failed: %v", err)
	}

	reopened, err := env.Host.OpenSession(watchdogContext(t), id)
	if err != nil {
		t.Fatalf("OpenSession after closing a suspended session failed: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close(context.Background()) })
	snapshot := snapshotOf(t, reopened)
	if snapshot.State != sessionloop.StateSuspended {
		t.Fatalf("reopened state = %q, want suspended (Close destroyed the durable pause)", snapshot.State)
	}
	if snapshot.Suspension == nil || snapshot.Suspension.ID != suspended.Suspension.ID {
		t.Fatalf("reopened suspension = %#v, want the original %q", snapshot.Suspension, suspended.Suspension.ID)
	}
	if snapshot.ActiveRunID != receipt.RunID {
		t.Fatalf("reopened active run = %q, want the suspended run %q", snapshot.ActiveRunID, receipt.RunID)
	}

	resolution := approveAll(*snapshot.Suspension)
	// Subscribe from the beginning: replay carries no settlement for the
	// suspended run, so the first run.settled observed is the resolution's.
	reopenedStream := subscribe(t, reopened, sessionloop.SubscribeOptions{Buffer: 256})
	dispatch(t, reopened, sessionloop.Command{
		Kind:       sessionloop.CommandResolve,
		RunID:      snapshot.ActiveRunID,
		Resolution: &resolution,
	})
	settled, _ := awaitSettled(t, reopenedStream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome after reopen and resolve = %q, want completed", settled.Outcome.Kind)
	}
}

func caseDetailedTools(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	receipt := dispatch(t, session, startCommand(scenarioInput("use a tool", ScenarioTools)))
	_, seen := awaitSettled(t, stream, receipt.RunID)

	var call *sessionloop.ToolCall
	var result *sessionloop.ToolResult
	for _, entry := range committedEntries(seen, receipt.RunID) {
		for _, block := range entry.Content {
			switch block.Kind {
			case sessionloop.BlockToolCall:
				call = block.ToolCall
			case sessionloop.BlockToolResult:
				result = block.ToolResult
			}
		}
	}
	if call == nil || call.CallID == "" || call.Name == "" {
		t.Fatalf("no complete tool_call block committed: %#v", call)
	}
	if len(call.Data) == 0 || !json.Valid(call.Data) {
		t.Fatalf("tool call data is not one complete JSON value: %q", call.Data)
	}
	if result == nil || result.CallID != call.CallID {
		t.Fatalf("no tool_result block matching call %q: %#v", call.CallID, result)
	}
	if len(result.Data) > 0 && !json.Valid(result.Data) {
		t.Fatalf("tool result data is not one complete JSON value: %q", result.Data)
	}
}

func caseStructuredOutput(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	receipt := dispatch(t, session, startCommand(scenarioInput("produce output", ScenarioOutput)))
	settled, _ := awaitSettled(t, stream, receipt.RunID)
	if settled.Outcome.Kind != sessionloop.RunCompleted {
		t.Fatalf("outcome = %q, want completed", settled.Outcome.Kind)
	}
	if len(settled.Outcome.Output) == 0 || !json.Valid(settled.Outcome.Output) {
		t.Fatalf("structured output is not one complete JSON value: %q", settled.Outcome.Output)
	}
}

func caseIdempotentDispatch(t *testing.T, env Env, _ sessionloop.Capabilities) {
	session := newSession(t, env.Host)
	stream := subscribe(t, session, sessionloop.SubscribeOptions{})

	command := startCommand(textInput("exactly once"))
	command.IdempotencyKey = "conformance-idempotency-key"
	first := dispatch(t, session, command)
	second := dispatch(t, session, command)
	if first.CommandID != second.CommandID || first.RunID != second.RunID || first.QueueID != second.QueueID {
		t.Fatalf("idempotent receipts diverge:\nfirst  %#v\nsecond %#v", first, second)
	}

	_, seen := awaitSettled(t, stream, first.RunID)
	starts := 0
	for _, event := range seen {
		if event.Kind == sessionloop.EventRunStarted && event.RunID == first.RunID {
			starts++
		}
	}
	if starts != 1 {
		t.Fatalf("idempotent start produced %d run.started events, want exactly one", starts)
	}
}
