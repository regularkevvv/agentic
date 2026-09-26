package session

// Every projection that attaches a command ID must propagate invalidation,
// including a fault that races after the initial availability check.

import (
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/store"
)

func TestProjectionPropagatesAttributionFailure(t *testing.T) {
	c := jsoncodec.New()
	user := agentic.NewTextMessage(agentic.RoleUser, "input")
	entries := []store.Entry{
		loopTestEntry(t, c, 1, kindRunOpened, runOpenedPayload{ID: "run"}),
		loopTestEntry(t, c, 1, kindRunClosed, runClosedPayload{ID: "run"}),
		loopTestEntry(t, c, 1, kindResolutionAccepted, resolutionAcceptedPayload{SuspensionID: "suspension"}),
		loopTestEntry(t, c, 1, kindMessage, messagePayload{Message: user, Source: "prompt"}),
		loopTestEntry(t, c, 1, kindMessage, messagePayload{Message: user, Source: string(QueueNextTurn), QueueID: "queue"}),
		loopTestEntry(t, c, 1, kindMessage, messagePayload{Message: user, Source: "other"}),
		loopTestEntry(t, c, 1, kindQueueAccepted, queueMutationPayload{ID: "queue", Entry: &QueueEntry{ID: "queue", Kind: QueueSteer, Message: user}}),
		loopTestEntry(t, c, 1, kindQueueDrained, queueMutationPayload{ID: "queue"}),
		loopTestEntry(t, c, 1, kindQueueCancelled, queueMutationPayload{ID: "queue"}),
		loopAgenticEntry(t, c, 1, kindAssistantCommitted, agentic.EventTypeAssistantCommitted, event.AssistantPayload{Message: user}),
		loopAgenticEntry(t, c, 1, kindToolResult, agentic.EventTypeToolResultCommitted, event.ToolResultPayload{}),
		loopAgenticEntry(t, c, 1, kindMessagesInjected, agentic.EventTypeTurnMessagesInjected, event.MessagesPayload{Messages: []agentic.Message{user}, QueueIDs: []string{"queue"}}),
		loopAgenticEntry(t, c, 1, kindRunSuspended, agentic.EventTypeRunSuspended, event.SuspensionPayload{}),
	}
	boom := errors.New("attribution invalidated after availability check")
	for _, entry := range entries {
		t.Run(entry.Kind, func(t *testing.T) {
			p := newLoopProjector("session", c, nil)
			p.fold.currentRunID = "run"
			p.commandForRun = func(string) (sessionloop.CommandID, error) { return "", boom }
			p.commandForQueue = p.commandForRun
			p.commandForResolution = func(uint64) (sessionloop.CommandID, error) { return "", boom }
			records, err := loopRecords(c, []store.Entry{entry})
			if err != nil {
				t.Fatal(err)
			}
			got, err := p.apply(t.Context(), records[0])
			if !errors.Is(err, boom) || len(got) != 0 {
				t.Fatalf("projection exposed events on attribution failure: %+v, %v", got, err)
			}
		})
	}
}
