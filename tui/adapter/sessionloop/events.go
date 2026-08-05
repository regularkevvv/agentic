package sessionloop

import (
	"context"
	"errors"
	"io"
	"sync"

	sl "github.com/regularkevvv/agentic/harness/sessionloop"
	uit "github.com/regularkevvv/agentic/tui"
)

// entries projects authoritative protocol entries into transcript entries.
// System entries never arrive by protocol design, so nothing is skipped.
func entries(values []sl.Entry, presenter uit.ToolPresenter) []uit.Entry {
	result := make([]uit.Entry, 0, len(values))
	for _, value := range values {
		result = append(result, entry(value, presenter))
	}
	return result
}

// entry maps one committed protocol entry: Text is the concatenation of the
// text blocks; tool_call blocks become planned tools; tool_result blocks
// become done/error tools. Raw tool payloads (Block.Data) never cross.
//
// Documented lossy field: the legacy adapter fills Tool.Summary from the
// session's application-owned ToolSummary redactor; the protocol carries no
// summary, so the bridge uses the tool name instead.
func entry(value sl.Entry, presenter uit.ToolPresenter) uit.Entry {
	result := uit.Entry{Role: uit.Role(value.Role)}
	for _, block := range value.Content {
		switch block.Kind {
		case sl.BlockText:
			result.Text += block.Text
		case sl.BlockToolCall:
			if block.ToolCall != nil {
				result.Tools = append(result.Tools, presentTool(uit.Tool{
					CallID: block.ToolCall.CallID, Name: block.ToolCall.Name,
					State: uit.ToolPlanned, Summary: block.ToolCall.Name,
				}, presenter))
			}
		case sl.BlockToolResult:
			if block.ToolResult != nil {
				result.Tools = append(result.Tools, presentTool(resultTool(*block.ToolResult), presenter))
			}
		}
	}
	return result
}

func resultTool(value sl.ToolResult) uit.Tool {
	state := uit.ToolDone
	if value.IsError {
		state = uit.ToolError
	}
	return uit.Tool{CallID: value.CallID, Name: value.Name, State: state}
}

func presentTool(tool uit.Tool, presenter uit.ToolPresenter) uit.Tool {
	if presenter != nil {
		tool.Presentation = presenter.PresentTool(tool)
	}
	return tool
}

func queuedInput(value sl.QueuedInput) uit.QueuedInput {
	return uit.QueuedInput{ID: string(value.ID), Kind: uit.QueueKind(value.Kind), Text: firstText(value.Content)}
}

func firstText(blocks []sl.Block) string {
	for _, block := range blocks {
		if block.Kind == sl.BlockText {
			return block.Text
		}
	}
	return ""
}

// suspension maps the protocol's safe display suspension. Supported means
// the terminal can resolve it: the protocol expresses that as at least one
// typed decision.
func suspension(value sl.Suspension) *uit.Suspension {
	result := &uit.Suspension{
		ID: value.ID, Kind: value.Kind,
		Supported:   len(value.Decisions) > 0,
		Description: value.Description,
	}
	if len(value.Decisions) == 0 {
		return result
	}
	result.Approvals = make([]uit.Approval, len(value.Decisions))
	for index, decision := range value.Decisions {
		result.Approvals[index] = uit.Approval{
			CallID: decision.ID, ToolName: decision.Name,
			Capability: decision.Capability, Action: decision.Action,
			ResourceDisplay: decision.Resource,
		}
	}
	return result
}

func usage(value sl.Usage) uit.Usage {
	return uit.Usage{
		PromptTokens: int(value.PromptTokens), CompletionTokens: int(value.CompletionTokens),
		TotalTokens: int(value.TotalTokens), CacheReadTokens: int(value.CacheReadTokens),
		CacheCreationTokens: int(value.CacheCreationTokens), ReasoningTokens: int(value.ReasoningTokens),
		Requests: int(value.Requests), ToolCalls: int(value.ToolCalls),
	}
}

// snapshotOwner is the projection callback used to substitute suspension
// payloads with the authoritative snapshot view, mirroring the legacy
// adapter's owner contract.
type snapshotOwner interface {
	Snapshot(context.Context) (uit.Snapshot, error)
}

// mapEvent projects one protocol event into zero or more TUI events,
// preserving the legacy event vocabulary the app reduces over.
func mapEvent(value sl.Event, owner snapshotOwner, presenter uit.ToolPresenter) ([]uit.Event, error) {
	base := uit.Event{
		Cursor: value.Position.Sequence, Ordinal: value.Ordinal,
		Durable:   value.Nature != sl.EventPreview,
		SessionID: string(value.SessionID), Dropped: value.Dropped,
	}
	switch value.Kind {
	case sl.EventPreviewDelta:
		return previewEvent(base, value.Preview, presenter), nil
	case sl.EventEntryCommitted:
		return entryEvent(base, value.Entry, presenter), nil
	case sl.EventQueueAccepted, sl.EventQueueDrained, sl.EventQueueCancelled:
		base.Kind = uit.EventKind(value.Kind)
		if value.Queue != nil {
			queued := queuedInput(*value.Queue)
			base.Queue = &queued
		}
		return []uit.Event{base}, nil
	case sl.EventRunStarted:
		base.Kind, base.State = uit.EventRunStarted, uit.StateRunning
		return []uit.Event{base}, nil
	case sl.EventRunSuspended:
		// The full approval view is reconciled from Snapshot, never
		// reconstructed from a partial event payload; a snapshot error is
		// terminal for the subscription (legacy mapObservation contract).
		base.Kind = uit.EventRunSuspended
		snapshot, err := owner.Snapshot(context.Background())
		if err != nil {
			return nil, err
		}
		base.Suspension = snapshot.Suspension
		return []uit.Event{base}, nil
	case sl.EventRunSettled:
		return settledEvents(base, value.Outcome), nil
	case sl.EventUsage:
		base.Kind = uit.EventUsage
		if value.Usage != nil {
			mapped := usage(*value.Usage)
			base.Usage = &mapped
		}
		return []uit.Event{base}, nil
	case sl.EventSessionState:
		if value.State == sl.StateFaulted {
			base.Kind, base.State = uit.EventSessionFaulted, uit.StateFaulted
		} else {
			base.Kind, base.State = uit.EventSessionRecovered, uit.State(value.State)
		}
		return []uit.Event{base}, nil
	default:
		// Unknown and namespaced extension kinds pass through as strings,
		// exactly as the legacy adapter passes unknown observe kinds.
		base.Kind = uit.EventKind(value.Kind)
		return []uit.Event{base}, nil
	}
}

// previewEvent maps lossy previews: text and thinking deltas accumulate in
// the app; tool previews upsert the live tool list.
func previewEvent(base uit.Event, preview *sl.Preview, presenter uit.ToolPresenter) []uit.Event {
	if preview == nil {
		return nil
	}
	switch preview.Kind {
	case sl.PreviewText:
		base.Kind, base.TextDelta = uit.EventTextDelta, preview.Text
	case sl.PreviewThinking:
		base.Kind = uit.EventThinkingDelta
		base.Thinking = &uit.Thinking{Text: preview.Text}
	case sl.PreviewTool:
		base.Kind = uit.EventToolPlanned
		tool := presentTool(uit.Tool{CallID: preview.ToolCallID, State: uit.ToolPreview}, presenter)
		base.Tool = &tool
	default:
		// Previews are lossy by contract; unknown shapes are dropped.
		return nil
	}
	return []uit.Event{base}
}

// entryEvent maps committed entries onto the legacy event vocabulary by
// origin, mirroring how the legacy observe projection names the underlying
// journal records: assistant commits, tool results, mid-run injections, and
// plain "message" records for prompts and next-turn drains.
func entryEvent(base uit.Event, value *sl.Entry, presenter uit.ToolPresenter) []uit.Event {
	if value == nil {
		return nil
	}
	switch value.Origin {
	case sl.OriginAssistant:
		base.Kind = uit.EventAssistantCommitted
		mapped := entry(*value, presenter)
		base.Entry = &mapped
	case sl.OriginTool:
		base.Kind = uit.EventToolResult
		for _, block := range value.Content {
			if block.Kind == sl.BlockToolResult && block.ToolResult != nil {
				tool := presentTool(resultTool(*block.ToolResult), presenter)
				base.Tool = &tool
				break
			}
		}
	case sl.OriginSteer, sl.OriginFollowUp:
		base.Kind = uit.EventMessagesInjected
		base.Entries = []uit.Entry{entry(*value, presenter)}
	default:
		// Prompt, next-turn, and seeded-history commits surface as the legacy
		// pass-through "message" kind the app deliberately ignores.
		base.Kind = uit.EventKind("message")
		mapped := entry(*value, presenter)
		base.Entry = &mapped
	}
	return []uit.Event{base}
}

// settledEvents synthesizes the legacy settlement pair from the singular
// protocol settlement: the outcome kind event followed by run.ended, both at
// the settlement cursor. The preview-loss count rides the first event only.
// Like the legacy run-closed projection, run.ended carries the sanitized
// failure text whenever the settlement recorded one.
func settledEvents(base uit.Event, outcome *sl.RunOutcome) []uit.Event {
	first, second := base, base
	second.Dropped = 0
	second.Kind, second.State = uit.EventRunEnded, uit.StateIdle
	first.Kind = uit.EventRunCompleted
	if outcome != nil {
		second.Failure = outcome.Failure
		switch outcome.Kind {
		case sl.RunInterrupted:
			first.Kind = uit.EventRunInterrupted
		case sl.RunFailed:
			first.Kind = uit.EventRunFailed
			first.Failure = outcome.Failure
		}
	}
	return []uit.Event{first, second}
}

type mappedSubscription struct {
	events <-chan uit.Event
	errors <-chan error
	close  func()
	once   sync.Once
}

func (s *mappedSubscription) Events() <-chan uit.Event { return s.events }
func (s *mappedSubscription) Errors() <-chan error     { return s.errors }
func (s *mappedSubscription) Close() {
	if s == nil {
		return
	}
	s.once.Do(s.close)
}

// failedSubscription reports one subscription-establishment error and closes
// both channels, so the app's lag-recovery path treats it like any terminal
// stream failure.
func failedSubscription(err error) uit.Subscription {
	events := make(chan uit.Event)
	errs := make(chan error, 1)
	errs <- err
	close(events)
	close(errs)
	return &mappedSubscription{events: events, errors: errs, close: func() {}}
}

// pumpSubscription adapts a protocol stream to the TUI subscription
// contract: unbuffered events, an error channel of capacity one, exactly one
// terminal error, both channels closed on exit, and an idempotent Close. A
// clean stream end (io.EOF) closes both channels without an error, exactly
// like the legacy mapped subscription.
func pumpSubscription(stream sl.Stream, owner snapshotOwner, presenter uit.ToolPresenter) uit.Subscription {
	events := make(chan uit.Event)
	errs := make(chan error, 1)
	done := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	var closeOnce sync.Once
	closeFn := func() {
		closeOnce.Do(func() {
			close(done)
			cancel()
			_ = stream.Close()
		})
	}
	result := &mappedSubscription{events: events, errors: errs, close: closeFn}
	go func() {
		defer close(events)
		defer close(errs)
		defer closeFn()
		for {
			value, err := stream.Next(ctx)
			if err != nil {
				if errors.Is(err, io.EOF) || ctx.Err() != nil {
					return
				}
				errs <- err
				return
			}
			mapped, mapErr := mapEvent(value, owner, presenter)
			if mapErr != nil {
				errs <- mapErr
				return
			}
			for _, event := range mapped {
				select {
				case <-done:
					return
				case events <- event:
				}
			}
		}
	}()
	return result
}

var _ uit.Subscription = (*mappedSubscription)(nil)
