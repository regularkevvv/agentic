package sessionloop

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/session"
	sl "github.com/regularkevvv/agentic/harness/sessionloop"
	uit "github.com/regularkevvv/agentic/tui"
)

type staticOwner struct {
	snapshot uit.Snapshot
	err      error
}

func (o staticOwner) Snapshot(context.Context) (uit.Snapshot, error) { return o.snapshot, o.err }

func TestMapEventTable(t *testing.T) {
	t.Parallel()
	presenter := uit.ToolPresenterFunc(func(tool uit.Tool) uit.ToolPresentation {
		return uit.ToolPresentation{Title: "safe " + tool.Name}
	})
	owner := staticOwner{snapshot: uit.Snapshot{Suspension: &uit.Suspension{ID: "susp", Supported: true}}}
	base := sl.Position{Sequence: 11}

	cases := []struct {
		name  string
		event sl.Event
		check func(t *testing.T, mapped []uit.Event)
	}{
		{"preview-text", sl.Event{Kind: sl.EventPreviewDelta, Nature: sl.EventPreview, Position: base, Ordinal: 3, Preview: &sl.Preview{Kind: sl.PreviewText, Text: "delta"}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventTextDelta || mapped[0].TextDelta != "delta" ||
					mapped[0].Durable || mapped[0].Cursor != 11 || mapped[0].Ordinal != 3 {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"preview-thinking", sl.Event{Kind: sl.EventPreviewDelta, Nature: sl.EventPreview, Preview: &sl.Preview{Kind: sl.PreviewThinking, Text: "thought"}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventThinkingDelta || mapped[0].Thinking.Text != "thought" {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"preview-tool", sl.Event{Kind: sl.EventPreviewDelta, Nature: sl.EventPreview, Preview: &sl.Preview{Kind: sl.PreviewTool, ToolCallID: "c1"}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventToolPlanned || mapped[0].Tool.CallID != "c1" ||
					mapped[0].Tool.State != uit.ToolPreview {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"preview-unknown", sl.Event{Kind: sl.EventPreviewDelta, Nature: sl.EventPreview, Preview: &sl.Preview{Kind: "mystery"}},
			func(t *testing.T, mapped []uit.Event) {
				if mapped != nil {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"preview-missing", sl.Event{Kind: sl.EventPreviewDelta, Nature: sl.EventPreview},
			func(t *testing.T, mapped []uit.Event) {
				if mapped != nil {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"entry-assistant", sl.Event{Kind: sl.EventEntryCommitted, Position: base, Entry: &sl.Entry{
			Role: sl.RoleAssistant, Origin: sl.OriginAssistant, Content: []sl.Block{
				{Kind: sl.BlockText, Text: "answer"},
				{Kind: sl.BlockToolCall, ToolCall: &sl.ToolCall{CallID: "c1", Name: "lookup"}},
			}}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventAssistantCommitted || mapped[0].Entry.Text != "answer" ||
					len(mapped[0].Entry.Tools) != 1 || mapped[0].Entry.Tools[0].Presentation.Title != "safe lookup" {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"entry-tool", sl.Event{Kind: sl.EventEntryCommitted, Entry: &sl.Entry{
			Role: sl.RoleTool, Origin: sl.OriginTool, Content: []sl.Block{
				{Kind: sl.BlockToolResult, Text: "raw", ToolResult: &sl.ToolResult{CallID: "c1", Name: "lookup", IsError: true}},
			}}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventToolResult || mapped[0].Tool.State != uit.ToolError ||
					mapped[0].Entry != nil {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"entry-steer", sl.Event{Kind: sl.EventEntryCommitted, Entry: &sl.Entry{
			Role: sl.RoleUser, Origin: sl.OriginSteer, Content: []sl.Block{{Kind: sl.BlockText, Text: "left"}}}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventMessagesInjected || len(mapped[0].Entries) != 1 ||
					mapped[0].Entries[0].Text != "left" {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"entry-follow-up", sl.Event{Kind: sl.EventEntryCommitted, Entry: &sl.Entry{
			Role: sl.RoleUser, Origin: sl.OriginFollowUp, Content: []sl.Block{{Kind: sl.BlockText, Text: "then"}}}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventMessagesInjected {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"entry-start", sl.Event{Kind: sl.EventEntryCommitted, Entry: &sl.Entry{
			Role: sl.RoleUser, Origin: sl.OriginStart, Content: []sl.Block{{Kind: sl.BlockText, Text: "prompt"}}}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventKind("message") || mapped[0].Entry.Text != "prompt" {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"entry-missing", sl.Event{Kind: sl.EventEntryCommitted},
			func(t *testing.T, mapped []uit.Event) {
				if mapped != nil {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"queue-accepted", sl.Event{Kind: sl.EventQueueAccepted, Position: base, Queue: &sl.QueuedInput{
			ID: "q1", Kind: sl.CommandSteer, Content: []sl.Block{{Kind: sl.BlockText, Text: "queued"}}}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventQueueAccepted || mapped[0].Queue.ID != "q1" ||
					mapped[0].Queue.Kind != uit.QueueSteer || mapped[0].Queue.Text != "queued" {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"queue-drained-empty", sl.Event{Kind: sl.EventQueueDrained},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventQueueDrained || mapped[0].Queue != nil {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"run-started", sl.Event{Kind: sl.EventRunStarted, Position: base, RunID: "run-1"},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventRunStarted || mapped[0].State != uit.StateRunning {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"run-suspended", sl.Event{Kind: sl.EventRunSuspended, Position: base, RunID: "run-1",
			Suspension: &sl.Suspension{ID: "partial"}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventRunSuspended || mapped[0].Suspension == nil ||
					mapped[0].Suspension.ID != "susp" || mapped[0].State != "" {
					t.Fatalf("snapshot suspension not substituted: %#v", mapped)
				}
			}},
		{"run-settled-completed", sl.Event{Kind: sl.EventRunSettled, Position: base, Dropped: 2,
			Outcome: &sl.RunOutcome{RunID: "run-1", Kind: sl.RunCompleted}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 2 || mapped[0].Kind != uit.EventRunCompleted || mapped[1].Kind != uit.EventRunEnded ||
					mapped[1].State != uit.StateIdle || mapped[0].Cursor != 11 || mapped[1].Cursor != 11 ||
					mapped[0].Dropped != 2 || mapped[1].Dropped != 0 {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"run-settled-interrupted", sl.Event{Kind: sl.EventRunSettled,
			Outcome: &sl.RunOutcome{RunID: "run-1", Kind: sl.RunInterrupted, Failure: "context canceled"}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 2 || mapped[0].Kind != uit.EventRunInterrupted || mapped[0].Failure != "" ||
					mapped[1].Failure != "context canceled" {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"run-settled-failed", sl.Event{Kind: sl.EventRunSettled,
			Outcome: &sl.RunOutcome{RunID: "run-1", Kind: sl.RunFailed, Failure: "boom"}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 2 || mapped[0].Kind != uit.EventRunFailed || mapped[0].Failure != "boom" ||
					mapped[1].Failure != "boom" {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"run-settled-missing", sl.Event{Kind: sl.EventRunSettled},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 2 || mapped[0].Kind != uit.EventRunCompleted || mapped[1].Kind != uit.EventRunEnded {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"usage", sl.Event{Kind: sl.EventUsage, Usage: &sl.Usage{TotalTokens: 9}},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventUsage || mapped[0].Usage.TotalTokens != 9 {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"usage-missing", sl.Event{Kind: sl.EventUsage},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Usage != nil {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"session-faulted", sl.Event{Kind: sl.EventSessionState, State: sl.StateFaulted},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventSessionFaulted || mapped[0].State != uit.StateFaulted {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"session-recovered", sl.Event{Kind: sl.EventSessionState, State: sl.StateIdle},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventSessionRecovered || mapped[0].State != uit.StateIdle {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
		{"extension-passthrough", sl.Event{Kind: sl.EventKind("agentic.transcript.compaction")},
			func(t *testing.T, mapped []uit.Event) {
				if len(mapped) != 1 || mapped[0].Kind != uit.EventKind("agentic.transcript.compaction") {
					t.Fatalf("mapped = %#v", mapped)
				}
			}},
	}
	for _, current := range cases {
		mapped, err := mapEvent(current.event, owner, presenter)
		if err != nil {
			t.Fatalf("%s: %v", current.name, err)
		}
		current.check(t, mapped)
	}

	if _, err := mapEvent(sl.Event{Kind: sl.EventRunSuspended}, staticOwner{err: errors.New("snapshot down")}, nil); err == nil {
		t.Fatal("suspension substitution snapshot error ignored")
	}
}

func TestFirstTextSkipsNonTextBlocks(t *testing.T) {
	t.Parallel()
	if firstText([]sl.Block{{Kind: sl.BlockData, Data: []byte(`1`)}}) != "" {
		t.Fatal("data block produced text")
	}
	if firstText([]sl.Block{{Kind: sl.BlockData}, {Kind: sl.BlockText, Text: "found"}}) != "found" {
		t.Fatal("text block missed")
	}
}

func TestPumpSubscriptionContract(t *testing.T) {
	t.Parallel()
	owner := staticOwner{}

	// Events map and deliver in order; a settled event fans out to the pair.
	stream := newFakeStream(
		streamStep{event: sl.Event{Kind: sl.EventRunStarted, RunID: "run-1", Position: sl.Position{Sequence: 4}}},
		streamStep{event: settledEvent("run-1", sl.RunCompleted, "")},
	)
	mapped := pumpSubscription(stream, owner, nil)
	if cap(mapped.Errors()) != 1 || cap(mapped.Events()) != 0 {
		t.Fatalf("channel contract: errors cap %d, events cap %d", cap(mapped.Errors()), cap(mapped.Events()))
	}
	kinds := []uit.EventKind{uit.EventRunStarted, uit.EventRunCompleted, uit.EventRunEnded}
	for _, kind := range kinds {
		select {
		case value := <-mapped.Events():
			if value.Kind != kind {
				t.Fatalf("event kind = %s, want %s", value.Kind, kind)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("no %s event", kind)
		}
	}
	mapped.Close()
	mapped.Close()
	for range mapped.Events() {
	}
	if _, ok := <-mapped.Errors(); ok {
		t.Fatal("closed pump left an error behind")
	}

	// A clean stream end closes both channels with no error.
	clean := newFakeStream()
	cleanMapped := pumpSubscription(clean, owner, nil)
	_ = clean.Close()
	if _, ok := <-cleanMapped.Events(); ok {
		t.Fatal("clean end delivered an event")
	}
	if _, ok := <-cleanMapped.Errors(); ok {
		t.Fatal("clean end delivered an error")
	}
	cleanMapped.Close()

	// Lag propagates as exactly one terminal error, then both channels close.
	lag := fmt.Errorf("%w: too slow", sl.ErrLagged)
	lagMapped := pumpSubscription(newFakeStream(streamStep{err: lag}), owner, nil)
	if err := <-lagMapped.Errors(); !errors.Is(err, sl.ErrLagged) {
		t.Fatalf("lag error = %v", err)
	}
	if _, ok := <-lagMapped.Events(); ok {
		t.Fatal("lagged pump delivered an event")
	}
	if _, ok := <-lagMapped.Errors(); ok {
		t.Fatal("lagged pump delivered a second error")
	}

	// A suspension substitution failure is terminal for the subscription.
	projection := pumpSubscription(
		newFakeStream(streamStep{event: sl.Event{Kind: sl.EventRunSuspended}}),
		staticOwner{err: errors.New("snapshot down")}, nil)
	if err := <-projection.Errors(); err == nil {
		t.Fatal("projection error missing")
	}
	if _, ok := <-projection.Events(); ok {
		t.Fatal("projection failure delivered an event")
	}

	// Close while the pump is blocked sending unblocks and closes cleanly.
	blocked := pumpSubscription(newFakeStream(
		streamStep{event: sl.Event{Kind: sl.EventUsage}},
		streamStep{event: sl.Event{Kind: sl.EventUsage}},
	), owner, nil)
	blocked.Close()
	for range blocked.Events() {
	}

	var nilSubscription *mappedSubscription
	nilSubscription.Close()
}

func TestPortSubscribeMapsOptionsAndFailures(t *testing.T) {
	t.Parallel()
	inner := &fakeSession{subscribes: []subscribeStep{{stream: newFakeStream()}}}
	target := &bridgeSession{session: inner}
	subscription := target.Subscribe(uit.SubscribeOptions{AfterCursor: 12, Buffer: 34, Preview: true})
	options := inner.subscribed()
	if len(options) != 1 || options[0].After.Sequence != 12 || options[0].Buffer != 34 || !options[0].Preview {
		t.Fatalf("subscribe options = %#v", options)
	}
	subscription.Close()
	for range subscription.Events() {
	}

	failing := &bridgeSession{session: &fakeSession{subscribes: []subscribeStep{
		{err: fmt.Errorf("%w: %w", sl.ErrSessionClosed, session.ErrSessionClosed)},
	}}}
	failed := failing.Subscribe(uit.SubscribeOptions{})
	if err := <-failed.Errors(); err != session.ErrSessionClosed {
		t.Fatalf("failed subscribe error = %v", err)
	}
	if _, ok := <-failed.Events(); ok {
		t.Fatal("failed subscribe delivered events")
	}
	failed.Close()
}
