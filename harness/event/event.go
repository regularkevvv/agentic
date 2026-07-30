// Package event defines nonblocking, cursor-based public event ports.
package event

import (
	"context"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/codec"
)

// Record is the harness projection of an Agentic or tool-runtime event.
// Authoritative and lifecycle records have a durable Cursor. Preview records
// carry the most recent durable cursor and a transient per-turn Ordinal.
type Record struct {
	Cursor      uint64
	Nature      agentic.EventNature
	Type        agentic.EventType
	Turn        int
	Ordinal     uint64
	SessionID   string
	ParentID    string
	Agent       string
	Depth       int
	Source      string
	Name        string
	Payload     []byte
	Dropped     EventsDropped
	SessionUsed *agentic.Usage
}

type EventsDropped struct {
	Preview uint64
}

func (d EventsDropped) Empty() bool { return d.Preview == 0 }

type AssistantPayload struct {
	Message agentic.Message
}

type ToolBatchPayload struct {
	Calls []agentic.ToolUse
}

type ToolStartedPayload struct {
	Call    agentic.ToolUse
	Attempt int
}

// ToolResultPayload stores error text because Agentic tool errors may contain
// arbitrary Go error values that no durable codec can reconstruct faithfully.
type ToolResultPayload struct {
	ToolUseID string
	ToolName  string
	Content   string
	IsError   bool
	Error     string
}

type MessagesPayload struct {
	Messages []agentic.Message
	QueueIDs []string
}

type TurnEndedPayload struct {
	Candidate agentic.CompletionCandidate
	RunUsage  agentic.Usage
}

type SuspensionPayload struct {
	Suspension agentic.Suspension
}

type RunCompletedPayload struct {
	Usage        agentic.Usage
	FinishReason agentic.FinishReason
}

type RunErrorPayload struct {
	Error string
}

type RunEndedPayload struct {
	Status agentic.ExecutionStatus
}

type OutputPayload struct {
	Candidate agentic.CompletionCandidate
}

// FromAgentic converts the closed root event taxonomy without relying on its
// unexported event-base representation or a particular payload encoding.
func FromAgentic(payloadCodec codec.Codec, value agentic.Event) (Record, error) {
	if value == nil {
		return Record{}, errors.New("agentic event is nil")
	}
	record := Record{
		Nature: value.Nature(),
		Type:   value.Type(),
		Turn:   value.TurnIndex(),
		Source: "agentic",
	}
	var payload any
	switch current := value.(type) {
	case *agentic.TextPreviewEvent:
		payload = struct{ Delta string }{current.Delta}
	case *agentic.ThinkingPreviewEvent:
		payload = struct {
			Delta        string
			Signature    string
			ProviderName string
			ThinkingID   string
		}{current.Delta, current.Signature, current.ProviderName, current.ThinkingID}
	case *agentic.ToolCallPreviewEvent:
		payload = ToolBatchPayload{Calls: []agentic.ToolUse{current.Call}}
	case *agentic.ToolArgumentPreviewEvent:
		payload = struct {
			ToolCallID string
			Delta      string
		}{current.ToolCallID, current.Delta}
	case *agentic.AssistantCommittedEvent:
		payload = AssistantPayload{Message: current.Message}
	case *agentic.ToolBatchPlannedEvent:
		payload = ToolBatchPayload{Calls: current.Calls}
	case *agentic.ToolStartedEvent:
		payload = ToolStartedPayload{Call: current.Call, Attempt: current.Attempt}
	case *agentic.ToolResultCommittedEvent:
		errorText := ""
		if current.Result.Error != nil {
			errorText = current.Result.Error.Error()
		}
		payload = ToolResultPayload{
			ToolUseID: current.Result.ToolUseID,
			ToolName:  current.Result.ToolName,
			Content:   agentic.FormatToolResult(current.Result.Content),
			IsError:   current.Result.IsError,
			Error:     errorText,
		}
	case *agentic.OutputValidatedEvent:
		payload = OutputPayload{Candidate: current.Candidate}
	case *agentic.TurnMessagesInjectedEvent:
		payload = MessagesPayload{Messages: current.Messages}
	case *agentic.TurnEndedEvent:
		payload = TurnEndedPayload{Candidate: current.Candidate, RunUsage: current.Usage}
	case *agentic.RunSuspendedEvent:
		payload = SuspensionPayload{Suspension: current.Suspension}
	case *agentic.RunCompletedEvent:
		payload = RunCompletedPayload{Usage: current.Usage, FinishReason: current.FinishReason}
	case *agentic.RunErrorEvent:
		message := ""
		if current.Error != nil {
			message = current.Error.Error()
		}
		payload = RunErrorPayload{Error: message}
	case *agentic.RunEndedEvent:
		payload = RunEndedPayload{Status: current.Status}
	case *agentic.RunStartedEvent, *agentic.TurnStartedEvent, *agentic.RunInterruptedEvent:
		payload = struct{}{}
	default:
		payload = value
	}
	encoded, err := codec.Encode(payloadCodec, payload)
	if err != nil {
		return Record{}, fmt.Errorf("encode agentic event %d: %w", value.Type(), err)
	}
	record.Payload = encoded
	return record, nil
}

func Decode[T any](payloadCodec codec.Codec, record Record) (T, error) {
	value, err := codec.Decode[T](payloadCodec, record.Payload)
	if err != nil {
		return value, fmt.Errorf("decode %s event payload: %w", record.Name, err)
	}
	return value, nil
}

func Clone(record Record) Record {
	record.Payload = append([]byte(nil), record.Payload...)
	if record.SessionUsed != nil {
		usage := *record.SessionUsed
		usage.RequestUsages = append([]agentic.RequestUsage(nil), record.SessionUsed.RequestUsages...)
		record.SessionUsed = &usage
	}
	return record
}

// Hub is the session event-distribution port. Implementations must never make
// PublishPreview or PublishDurable wait for a public subscriber.
type Hub interface {
	Cursor() uint64
	PublishDurable(Record)
	PublishPreview(Record)
	Subscribe(SubscribeOptions) *Subscription
	// Close disconnects subscribers and must be idempotent.
	Close()
}

// Factory creates one event hub for a session. History contains only durable
// authoritative/lifecycle records and is already ordered by cursor.
type Factory interface {
	Open(context.Context, []Record) (Hub, error)
}

type FactoryFunc func(context.Context, []Record) (Hub, error)

func (f FactoryFunc) Open(ctx context.Context, history []Record) (Hub, error) {
	return f(ctx, history)
}
