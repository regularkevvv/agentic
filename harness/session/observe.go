package session

import (
	"fmt"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/observe"
)

// Observe returns the typed presentation projection. Codec-specific decoding
// remains inside the owning session; clients never parse durable wire bytes.
func (s *Session[O]) Observe(options observe.SubscribeOptions) observe.Subscription {
	raw := s.Subscribe(event.SubscribeOptions{
		AfterCursor: options.AfterCursor,
		Buffer:      options.Buffer,
		Preview:     options.Preview,
	})
	return observe.ProjectSubscription(raw, s.projectObservation)
}

func (s *Session[O]) projectObservation(record event.Record) (observe.Event, error) {
	result := observe.Event{
		Cursor: record.Cursor, Ordinal: record.Ordinal, Nature: record.Nature,
		SessionID: record.SessionID, ParentID: record.ParentID, Agent: record.Agent,
		Depth: record.Depth, Turn: record.Turn, Dropped: record.Dropped.Preview,
	}
	if record.SessionUsed != nil {
		usage := observe.UsageFromAgentic(*record.SessionUsed)
		result.Usage = &usage
	}
	switch record.Source {
	case "agentic":
		return s.projectAgenticObservation(result, record)
	case "tool":
		result.Kind = observe.KindToolUpdate
		result.Tool = &observe.Tool{State: observe.ToolRunning, Summary: boundedToolSummary(record.Name)}
		return result, nil
	case "harness":
		return s.projectHarnessObservation(result, record)
	default:
		result.Kind = observe.Kind(record.Name)
		if result.Kind == "" {
			result.Kind = observe.Kind("event")
		}
		return result, nil
	}
}

const maxToolSummaryRunes = 256

func boundedToolSummary(value string) string {
	runes := []rune(value)
	if len(runes) <= maxToolSummaryRunes {
		return value
	}
	return string(runes[:maxToolSummaryRunes]) + "…"
}

func (s *Session[O]) projectAgenticObservation(result observe.Event, record event.Record) (observe.Event, error) {
	switch record.Type {
	case agentic.EventTypeTextPreview:
		payload, err := event.Decode[struct{ Delta string }](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind, result.TextDelta = observe.KindTextDelta, payload.Delta
	case agentic.EventTypeThinkingPreview:
		payload, err := event.Decode[struct {
			Delta        string
			Signature    string
			ProviderName string
			ThinkingID   string
		}](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind = observe.KindThinkingDelta
		result.Thinking = &observe.Thinking{
			Text: payload.Delta, ProviderName: payload.ProviderName, ThinkingID: payload.ThinkingID,
		}
	case agentic.EventTypeToolCallPreview:
		payload, err := event.Decode[event.ToolBatchPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind = observe.KindToolPlanned
		result.Tools = projectTools(payload.Calls, observe.ToolPreview, s.summarize)
	case agentic.EventTypeToolArgumentPreview:
		payload, err := event.Decode[struct {
			ToolCallID string
			Delta      string
		}](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind = observe.KindToolPlanned
		result.Tool = &observe.Tool{CallID: payload.ToolCallID, State: observe.ToolPreview}
	case agentic.EventTypeAssistantCommitted:
		payload, err := event.Decode[event.AssistantPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		message := projectMessage(payload.Message, s.summarize)
		result.Kind, result.Message = observe.KindAssistantCommitted, &message
	case agentic.EventTypeToolBatchPlanned:
		payload, err := event.Decode[event.ToolBatchPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind = observe.KindToolPlanned
		result.Tools = projectTools(payload.Calls, observe.ToolPlanned, s.summarize)
	case agentic.EventTypeToolStarted:
		payload, err := event.Decode[event.ToolStartedPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind = observe.KindToolStarted
		result.Tool = &observe.Tool{
			CallID: payload.Call.ID, Name: payload.Call.Name, State: observe.ToolRunning,
			Attempt: payload.Attempt, Summary: s.ToolSummary(payload.Call),
		}
	case agentic.EventTypeToolResultCommitted:
		payload, err := event.Decode[event.ToolResultPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		state := observe.ToolDone
		if payload.IsError {
			state = observe.ToolError
		}
		result.Kind = observe.KindToolResult
		result.Tool = &observe.Tool{CallID: payload.ToolUseID, Name: payload.ToolName, State: state}
	case agentic.EventTypeOutputValidated:
		result.Kind = observe.KindOutputValidated
	case agentic.EventTypeTurnMessagesInjected:
		payload, err := event.Decode[event.MessagesPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind, result.Messages = observe.KindMessagesInjected, projectMessages(payload.Messages, s.summarize)
	case agentic.EventTypeRunStarted:
		result.Kind = observe.KindRunStarted
	case agentic.EventTypeTurnStarted:
		result.Kind = observe.KindTurnStarted
	case agentic.EventTypeTurnEnded:
		payload, err := event.Decode[event.TurnEndedPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind = observe.KindTurnEnded
		if result.Usage == nil {
			usage := observe.UsageFromAgentic(payload.RunUsage)
			result.Usage = &usage
		}
	case agentic.EventTypeRunSuspended:
		payload, err := event.Decode[event.SuspensionPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind = observe.KindRunSuspended
		result.Suspension = &observe.Suspension{ID: payload.Suspension.ID, Kind: payload.Suspension.Kind}
	case agentic.EventTypeRunCompleted:
		payload, err := event.Decode[event.RunCompletedPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		usage := observe.UsageFromAgentic(payload.Usage)
		result.Kind, result.Usage = observe.KindRunCompleted, &usage
	case agentic.EventTypeRunInterrupted:
		result.Kind = observe.KindRunInterrupted
	case agentic.EventTypeRunError:
		payload, err := event.Decode[event.RunErrorPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind, result.Failure = observe.KindRunFailed, &observe.Failure{Message: payload.Error}
	case agentic.EventTypeRunEnded:
		result.Kind = observe.KindRunEnded
	default:
		return observe.Event{}, fmt.Errorf("observe unsupported agentic event type %d", record.Type)
	}
	return result, nil
}

func (s *Session[O]) projectHarnessObservation(result observe.Event, record event.Record) (observe.Event, error) {
	result.Kind = observe.Kind(record.Name)
	switch record.Name {
	case kindSessionCreated:
		result.Kind, result.State = observe.KindSessionCreated, Idle.String()
	case kindRunOpened:
		result.Kind, result.State = observe.KindRunStarted, Running.String()
	case kindRunClosed:
		payload, err := event.Decode[runClosedPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind, result.State = observe.KindRunEnded, Idle.String()
		if payload.Error != "" {
			result.Failure = &observe.Failure{Message: payload.Error}
		}
	case kindMessage, kindSystemMessage:
		payload, err := event.Decode[messagePayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		message := projectMessage(payload.Message, s.summarize)
		result.Message = &message
	case kindContextMessage:
		payload, err := event.Decode[contextMessagePayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		message := projectMessage(payload.Message, s.summarize)
		result.Message = &message
	case kindQueueAccepted, kindQueueDrained, kindQueueCancelled:
		payload, err := event.Decode[queueMutationPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		switch record.Name {
		case kindQueueAccepted:
			result.Kind = observe.KindQueueAccepted
		case kindQueueDrained:
			result.Kind = observe.KindQueueDrained
		case kindQueueCancelled:
			result.Kind = observe.KindQueueCancelled
		}
		result.Queue = &observe.Queue{ID: payload.ID}
		if payload.Entry != nil {
			message := projectMessage(payload.Entry.Message, s.summarize)
			result.Queue.Kind, result.Queue.Message = string(payload.Entry.Kind), &message
		}
	case kindUsageCommitted:
		payload, err := event.Decode[usagePayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		usage := observe.UsageFromAgentic(payload.Session)
		result.Kind, result.Usage = observe.KindUsage, &usage
	case kindChildUsage:
		payload, err := event.Decode[childUsagePayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		usage := observe.UsageFromAgentic(payload.Session)
		result.Kind, result.Usage = observe.KindUsage, &usage
	case kindRecoverySuspension:
		payload, err := event.Decode[event.SuspensionPayload](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind = observe.KindRunSuspended
		result.Suspension = &observe.Suspension{ID: payload.Suspension.ID, Kind: payload.Suspension.Kind}
	case kindRecovered:
		payload, err := event.Decode[struct{ State string }](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind, result.State = observe.KindSessionRecovered, payload.State
	case kindFault:
		payload, err := event.Decode[struct{ Error string }](s.codec, record)
		if err != nil {
			return observe.Event{}, err
		}
		result.Kind, result.State = observe.KindSessionFaulted, Faulted.String()
		result.Failure = &observe.Failure{Message: payload.Error}
	case kindCompaction:
		result.Kind = observe.KindCompaction
	case kindInterruptMarker:
		result.Kind, result.State = observe.KindRunInterrupted, Interrupting.String()
	}
	return result, nil
}

func projectMessages(messages []agentic.Message, summarizers ...observe.ToolSummarizer) []observe.Message {
	result := make([]observe.Message, len(messages))
	for index, message := range messages {
		result[index] = projectMessage(message, summarizers...)
	}
	return result
}

func projectMessage(message agentic.Message, summarizers ...observe.ToolSummarizer) observe.Message {
	result := observe.Message{Role: string(message.Role)}
	for _, part := range message.Content {
		switch part.Type {
		case agentic.ContentText:
			result.Text += part.Text
		case agentic.ContentThinking:
			if part.Thinking == nil {
				continue
			}
			result.Thinking = append(result.Thinking, observe.Thinking{
				Text: part.Thinking.Text, ProviderName: part.Thinking.ProviderName,
				ThinkingID: part.Thinking.ID, Redacted: part.Thinking.IsRedacted(),
			})
		case agentic.ContentToolUse:
			if part.ToolUse != nil {
				result.Tools = append(result.Tools, observe.Tool{
					CallID: part.ToolUse.ID, Name: part.ToolUse.Name, State: observe.ToolPlanned,
					Summary: summarizeTool(*part.ToolUse, summarizers),
				})
			}
		case agentic.ContentToolResult:
			if part.ToolResult != nil {
				state := observe.ToolDone
				if part.ToolResult.IsError {
					state = observe.ToolError
				}
				result.Tools = append(result.Tools, observe.Tool{CallID: part.ToolResult.ToolUseID, Name: part.ToolResult.Name, State: state})
			}
		}
	}
	return result
}

func projectTools(calls []agentic.ToolUse, state observe.ToolState, summarizers ...observe.ToolSummarizer) []observe.Tool {
	result := make([]observe.Tool, len(calls))
	for index, call := range calls {
		result[index] = observe.Tool{CallID: call.ID, Name: call.Name, State: state, Summary: summarizeTool(call, summarizers)}
	}
	return result
}

func summarizeTool(call agentic.ToolUse, summarizers []observe.ToolSummarizer) string {
	if len(summarizers) == 0 || summarizers[0] == nil {
		return ""
	}
	return boundedToolSummary(summarizers[0](call))
}

// ToolSummary invokes the application-owned redactor without exposing raw
// arguments through the observation or TUI contracts.
func (s *Session[O]) ToolSummary(call agentic.ToolUse) string {
	if s == nil || s.summarize == nil {
		return ""
	}
	return boundedToolSummary(s.summarize(call))
}
