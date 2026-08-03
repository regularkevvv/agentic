package session

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/observe"
)

func observationSession() *Session[string] { return &Session[string]{codec: jsoncodec.New()} }

func observationRecord(t *testing.T, payloadCodec codec.Codec, source string, typ agentic.EventType, name string, payload any) event.Record {
	t.Helper()
	encoded, err := codec.Encode(payloadCodec, payload)
	if err != nil {
		t.Fatal(err)
	}
	return event.Record{
		Cursor: 3, Nature: agentic.EventAuthoritative, Type: typ, Turn: 2, Ordinal: 4,
		SessionID: "session", ParentID: "parent", Agent: "child", Depth: 1,
		Source: source, Name: name, Payload: encoded, Dropped: event.EventsDropped{Preview: 2},
	}
}

func TestProjectEveryAgenticObservationAndRedaction(t *testing.T) {
	t.Parallel()
	session := observationSession()
	secretCall := agentic.ToolUse{ID: "call", Name: "tool", Input: map[string]any{"token": "secret-input"}}
	secretMessage := agentic.Message{Role: agentic.RoleAssistant, Content: []agentic.Part{
		{Type: agentic.ContentText, Text: "safe"},
		{Type: agentic.ContentThinking, Thinking: &agentic.ThinkingBlock{Text: "thought", ID: "redacted_thinking", Signature: "secret-signature", ProviderDetails: map[string]any{"token": "secret"}}},
		{Type: agentic.ContentToolUse, ToolUse: &secretCall},
		{Type: agentic.ContentToolResult, ToolResult: &agentic.ToolResult{ToolUseID: "call", Name: "tool", Content: "secret-result", IsError: true}},
		{Type: agentic.ContentThinking}, {Type: agentic.ContentToolUse}, {Type: agentic.ContentToolResult},
	}}
	tests := []struct {
		typ     agentic.EventType
		payload any
		kind    observe.Kind
	}{
		{agentic.EventTypeTextPreview, struct{ Delta string }{"hello"}, observe.KindTextDelta},
		{agentic.EventTypeThinkingPreview, struct{ Delta, Signature, ProviderName, ThinkingID string }{"thinking", "secret", "provider", "id"}, observe.KindThinkingDelta},
		{agentic.EventTypeToolCallPreview, event.ToolBatchPayload{Calls: []agentic.ToolUse{secretCall}}, observe.KindToolPlanned},
		{agentic.EventTypeToolArgumentPreview, struct{ ToolCallID, Delta string }{"call", "secret-argument"}, observe.KindToolPlanned},
		{agentic.EventTypeAssistantCommitted, event.AssistantPayload{Message: secretMessage}, observe.KindAssistantCommitted},
		{agentic.EventTypeToolBatchPlanned, event.ToolBatchPayload{Calls: []agentic.ToolUse{secretCall}}, observe.KindToolPlanned},
		{agentic.EventTypeToolStarted, event.ToolStartedPayload{Call: secretCall, Attempt: 2}, observe.KindToolStarted},
		{agentic.EventTypeToolResultCommitted, event.ToolResultPayload{ToolUseID: "call", ToolName: "tool", Content: "secret-result", IsError: true, Error: "secret-error"}, observe.KindToolResult},
		{agentic.EventTypeOutputValidated, struct{}{}, observe.KindOutputValidated},
		{agentic.EventTypeTurnMessagesInjected, event.MessagesPayload{Messages: []agentic.Message{secretMessage}}, observe.KindMessagesInjected},
		{agentic.EventTypeRunStarted, struct{}{}, observe.KindRunStarted},
		{agentic.EventTypeTurnStarted, struct{}{}, observe.KindTurnStarted},
		{agentic.EventTypeTurnEnded, event.TurnEndedPayload{RunUsage: agentic.Usage{TotalTokens: 10}}, observe.KindTurnEnded},
		{agentic.EventTypeRunSuspended, event.SuspensionPayload{Suspension: agentic.Suspension{ID: "s", Kind: "permission"}}, observe.KindRunSuspended},
		{agentic.EventTypeRunCompleted, event.RunCompletedPayload{Usage: agentic.Usage{TotalTokens: 11}}, observe.KindRunCompleted},
		{agentic.EventTypeRunInterrupted, struct{}{}, observe.KindRunInterrupted},
		{agentic.EventTypeRunError, event.RunErrorPayload{Error: "failed"}, observe.KindRunFailed},
		{agentic.EventTypeRunEnded, event.RunEndedPayload{}, observe.KindRunEnded},
	}
	for _, test := range tests {
		record := observationRecord(t, session.codec, "agentic", test.typ, "", test.payload)
		if test.typ == agentic.EventTypeTurnEnded {
			record.SessionUsed = &agentic.Usage{TotalTokens: 20}
		}
		projected, err := session.projectObservation(record)
		if err != nil {
			t.Fatalf("type %d: %v", test.typ, err)
		}
		if projected.Kind != test.kind || projected.Cursor != 3 || projected.ParentID != "parent" || projected.Dropped != 2 {
			t.Fatalf("type %d = %#v", test.typ, projected)
		}
		encoded, _ := json.Marshal(projected)
		for _, secret := range []string{"secret-input", "secret-result", "secret-signature", "secret-argument", "secret-error"} {
			if strings.Contains(string(encoded), secret) {
				t.Fatalf("type %d leaked %q: %s", test.typ, secret, encoded)
			}
		}
	}
	malformed := observationRecord(t, session.codec, "agentic", agentic.EventTypeTextPreview, "", struct{}{})
	malformed.Payload = []byte("bad")
	if _, err := session.projectObservation(malformed); err == nil {
		t.Fatal("malformed payload succeeded")
	}
	unsupported := observationRecord(t, session.codec, "agentic", agentic.EventType(255), "", struct{}{})
	if _, err := session.projectObservation(unsupported); err == nil {
		t.Fatal("unsupported event type succeeded")
	}
}

func TestAgenticObservationRejectsEveryMalformedTypedPayload(t *testing.T) {
	t.Parallel()
	session := observationSession()
	types := []agentic.EventType{
		agentic.EventTypeTextPreview,
		agentic.EventTypeThinkingPreview,
		agentic.EventTypeToolCallPreview,
		agentic.EventTypeToolArgumentPreview,
		agentic.EventTypeAssistantCommitted,
		agentic.EventTypeToolBatchPlanned,
		agentic.EventTypeToolStarted,
		agentic.EventTypeToolResultCommitted,
		agentic.EventTypeTurnMessagesInjected,
		agentic.EventTypeTurnEnded,
		agentic.EventTypeRunSuspended,
		agentic.EventTypeRunCompleted,
		agentic.EventTypeRunError,
	}
	for _, typ := range types {
		record := event.Record{Source: "agentic", Type: typ, Payload: []byte("not-json")}
		if _, err := session.projectObservation(record); err == nil {
			t.Fatalf("type %d accepted malformed payload", typ)
		}
	}
}

func TestObservationProjectsUsageFallbackAndSuccessfulToolResult(t *testing.T) {
	t.Parallel()
	session := observationSession()
	turn, err := session.projectObservation(observationRecord(
		t, session.codec, "agentic", agentic.EventTypeTurnEnded, "",
		event.TurnEndedPayload{RunUsage: agentic.Usage{TotalTokens: 17}},
	))
	if err != nil || turn.Usage == nil || turn.Usage.TotalTokens != 17 {
		t.Fatalf("turn = %#v, %v", turn, err)
	}
	message := agentic.Message{Role: agentic.RoleTool, Content: []agentic.Part{{
		Type:       agentic.ContentToolResult,
		ToolResult: &agentic.ToolResult{ToolUseID: "call", Name: "tool"},
	}}}
	projected := projectMessage(message)
	if len(projected.Tools) != 1 || projected.Tools[0].State != observe.ToolDone {
		t.Fatalf("message = %#v", projected)
	}
}

func TestProjectHarnessToolAndUnknownObservations(t *testing.T) {
	t.Parallel()
	session := observationSession()
	message := agentic.NewTextMessage(agentic.RoleUser, "hello")
	tests := []struct {
		name    string
		payload any
		kind    observe.Kind
		state   string
	}{
		{kindSessionCreated, sessionCreatedPayload{}, observe.KindSessionCreated, Idle.String()},
		{kindRunOpened, runOpenedPayload{}, observe.KindRunStarted, Running.String()},
		{kindRunClosed, runClosedPayload{Error: "failed"}, observe.KindRunEnded, Idle.String()},
		{kindMessage, messagePayload{Message: message}, observe.Kind(kindMessage), ""},
		{kindSystemMessage, messagePayload{Message: message}, observe.Kind(kindSystemMessage), ""},
		{kindContextMessage, contextMessagePayload{Message: message}, observe.Kind(kindContextMessage), ""},
		{kindQueueAccepted, queueMutationPayload{ID: "q", Entry: &QueueEntry{ID: "q", Kind: QueueSteer, Message: message}}, observe.KindQueueAccepted, ""},
		{kindQueueDrained, queueMutationPayload{ID: "q"}, observe.KindQueueDrained, ""},
		{kindQueueCancelled, queueMutationPayload{ID: "q"}, observe.KindQueueCancelled, ""},
		{kindUsageCommitted, usagePayload{Session: agentic.Usage{TotalTokens: 2}}, observe.KindUsage, ""},
		{kindChildUsage, childUsagePayload{Session: agentic.Usage{TotalTokens: 3}}, observe.KindUsage, ""},
		{kindRecoverySuspension, event.SuspensionPayload{Suspension: agentic.Suspension{ID: "s", Kind: "custom"}}, observe.KindRunSuspended, ""},
		{kindRecovered, struct{ State string }{Suspended.String()}, observe.KindSessionRecovered, Suspended.String()},
		{kindFault, struct{ Error string }{"fault"}, observe.KindSessionFaulted, Faulted.String()},
		{kindCompaction, struct{}{}, observe.KindCompaction, ""},
		{kindInterruptMarker, struct{}{}, observe.KindRunInterrupted, Interrupting.String()},
		{"custom.event", struct{}{}, observe.Kind("custom.event"), ""},
	}
	for _, test := range tests {
		projected, err := session.projectObservation(observationRecord(t, session.codec, "harness", 0, test.name, test.payload))
		if err != nil || projected.Kind != test.kind || projected.State != test.state {
			t.Fatalf("%s = %#v, %v", test.name, projected, err)
		}
	}
	tool, err := session.projectObservation(event.Record{Source: "tool", Name: "safe summary"})
	if err != nil || tool.Kind != observe.KindToolUpdate || tool.Tool == nil || tool.Tool.Summary != "safe summary" {
		t.Fatalf("tool = %#v, %v", tool, err)
	}
	bounded, err := session.projectObservation(event.Record{Source: "tool", Name: strings.Repeat("x", maxToolSummaryRunes+10)})
	if err != nil || bounded.Tool == nil || len([]rune(bounded.Tool.Summary)) != maxToolSummaryRunes+1 || !strings.HasSuffix(bounded.Tool.Summary, "…") {
		t.Fatalf("bounded tool summary = %#v, %v", bounded.Tool, err)
	}
	unknown, err := session.projectObservation(event.Record{Source: "custom"})
	if err != nil || unknown.Kind != observe.Kind("event") {
		t.Fatalf("unknown = %#v, %v", unknown, err)
	}
	malformed := event.Record{Source: "harness", Name: kindFault, Payload: []byte("bad")}
	if _, err := session.projectObservation(malformed); err == nil {
		t.Fatal("malformed harness event succeeded")
	}
}

func TestHarnessObservationRejectsEveryMalformedTypedPayload(t *testing.T) {
	t.Parallel()
	session := observationSession()
	names := []string{
		kindRunClosed,
		kindMessage,
		kindSystemMessage,
		kindContextMessage,
		kindQueueAccepted,
		kindQueueDrained,
		kindQueueCancelled,
		kindUsageCommitted,
		kindChildUsage,
		kindRecoverySuspension,
		kindRecovered,
		kindFault,
	}
	for _, name := range names {
		record := event.Record{Source: "harness", Name: name, Payload: []byte("not-json")}
		if _, err := session.projectObservation(record); err == nil {
			t.Fatalf("%s accepted malformed payload", name)
		}
	}
}

func TestObserveProjectsLiveHub(t *testing.T) {
	t.Parallel()
	factory := inproc.NewFactory()
	hub, err := factory.Open(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	session := observationSession()
	session.bus = hub
	subscription := session.Observe(observe.SubscribeOptions{Buffer: 4, Preview: true})
	record := observationRecord(t, session.codec, "agentic", agentic.EventTypeTextPreview, "", struct{ Delta string }{"live"})
	record.Nature = agentic.EventPreview
	hub.PublishPreview(record)
	projected := <-subscription.Events()
	if projected.TextDelta != "live" {
		t.Fatalf("projected = %#v", projected)
	}
	subscription.Close()
	hub.Close()
}
