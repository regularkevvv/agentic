package event

import (
	"encoding/json"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
)

type customEvent struct {
	nature agentic.EventNature
	typ    agentic.EventType
	turn   int
	Value  string `json:"value"`
}

func (e customEvent) Nature() agentic.EventNature { return e.nature }
func (e customEvent) Type() agentic.EventType     { return e.typ }
func (e customEvent) TurnIndex() int              { return e.turn }

func TestFromAgenticNormalizesConcretePayloads(t *testing.T) {
	t.Parallel()
	payloadCodec := jsoncodec.New()
	call := agentic.ToolUse{ID: "call", Name: "tool"}
	message := agentic.NewTextMessage(agentic.RoleAssistant, "answer")
	suspension := agentic.Suspension{ID: "s", Kind: "test", FrontierHash: "hash", Payload: json.RawMessage(`{"v":1}`)}
	cases := []agentic.Event{
		&agentic.TextPreviewEvent{Delta: "text"},
		&agentic.ThinkingPreviewEvent{Delta: "thought", Signature: "sig", ProviderName: "provider", ThinkingID: "id"},
		&agentic.ToolCallPreviewEvent{Call: call},
		&agentic.ToolArgumentPreviewEvent{ToolCallID: call.ID, Delta: "{}"},
		&agentic.AssistantCommittedEvent{Message: message},
		&agentic.ToolBatchPlannedEvent{Calls: []agentic.ToolUse{call}},
		&agentic.ToolStartedEvent{Call: call, Attempt: 2},
		&agentic.ToolResultCommittedEvent{Result: agentic.ToolExecutionResult{ToolUseID: call.ID, ToolName: call.Name, Content: map[string]int{"n": 1}, IsError: true, Error: errors.New("failed")}},
		&agentic.OutputValidatedEvent{Candidate: agentic.CompletionCandidate{Source: agentic.CompletionText}},
		&agentic.TurnMessagesInjectedEvent{Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "more")}},
		&agentic.TurnEndedEvent{Usage: agentic.Usage{TotalTokens: 3}},
		&agentic.RunSuspendedEvent{Suspension: suspension},
		&agentic.RunCompletedEvent{Usage: agentic.Usage{TotalTokens: 3}, FinishReason: agentic.FinishReasonStop},
		&agentic.RunErrorEvent{Error: errors.New("run failed")},
		&agentic.RunEndedEvent{Status: agentic.ExecutionFailed},
		&agentic.RunStartedEvent{},
		&agentic.TurnStartedEvent{},
		&agentic.RunInterruptedEvent{},
		customEvent{nature: agentic.EventLifecycle, typ: agentic.EventTypeRunEnded, turn: 9, Value: "fallback"},
	}
	for _, value := range cases {
		record, err := FromAgentic(payloadCodec, value)
		if err != nil {
			t.Fatalf("FromAgentic(%T): %v", value, err)
		}
		if record.Source != "agentic" || !json.Valid(record.Payload) {
			t.Fatalf("record for %T = %#v", value, record)
		}
	}
	if _, err := FromAgentic(payloadCodec, nil); err == nil {
		t.Fatal("nil event succeeded")
	}
	record, err := FromAgentic(payloadCodec, cases[7])
	if err != nil {
		t.Fatal(err)
	}
	payload, err := Decode[ToolResultPayload](payloadCodec, record)
	if err != nil || payload.Content != `{"n":1}` || payload.Error != "failed" || !payload.IsError {
		t.Fatalf("tool payload = %#v, %v", payload, err)
	}
}

func TestDecodeRejectsInvalidPayload(t *testing.T) {
	t.Parallel()
	if _, err := Decode[ToolBatchPayload](jsoncodec.New(), Record{Name: "bad", Payload: []byte(`{`)}); err == nil {
		t.Fatal("invalid payload decoded")
	}
}

func TestSubscriptionCloseDeliversNilTerminal(t *testing.T) {
	t.Parallel()
	events := make(chan Record)
	terminals := make(chan error, 1)
	sub := NewSubscription(events, terminals, func() {
		terminals <- nil
		close(events)
		close(terminals)
	})
	sub.Close()
	if terminal, ok := <-sub.Err; !ok || terminal != nil {
		t.Fatalf("terminal = %v, open=%v", terminal, ok)
	}
	if _, ok := <-sub.Err; ok {
		t.Fatal("Err remained open")
	}
	if _, ok := <-sub.Events; ok {
		t.Fatal("Events remained open")
	}
	// Close is idempotent even after channel termination.
	sub.Close()
}
