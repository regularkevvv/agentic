package session

import (
	"fmt"

	"github.com/regularkevvv/agentic/harness/codec"
	"github.com/regularkevvv/agentic/harness/store"
)

const (
	kindSessionCreated     = "session.created"
	kindRunOpened          = "run.opened"
	kindRunClosed          = "run.closed"
	kindMessage            = "message"
	kindSystemMessage      = "message.system"
	kindQueueAccepted      = "queue.accepted"
	kindQueueDrained       = "queue.drained"
	kindQueueCancelled     = "queue.cancelled"
	kindAssistantCommitted = "agentic.assistant_committed"
	kindToolBatchPlanned   = "agentic.tool_batch_planned"
	kindToolStarted        = "agentic.tool_started"
	kindToolResult         = "agentic.tool_result"
	kindMessagesInjected   = "agentic.messages_injected"
	kindTurnStarted        = "agentic.turn_started"
	kindTurnEnded          = "agentic.turn_ended"
	kindRunStarted         = "agentic.run_started"
	kindRunSuspended       = "agentic.run_suspended"
	kindRunCompleted       = "agentic.run_completed"
	kindRunInterrupted     = "agentic.run_interrupted"
	kindRunError           = "agentic.run_error"
	kindRunEnded           = "agentic.run_ended"
	kindOutputValidated    = "agentic.output_validated"
	kindUsageCommitted     = "usage.committed"
	kindRepair             = "transcript.repair"
	kindInterruptMarker    = "interrupt.marker"
	kindContextMessage     = "context.durable"
	kindResolutionAccepted = "resolution.accepted"
	kindRecoverySuspension = "recovery.suspension"
	kindRecovered          = "session.recovered"
	kindFault              = "session.fault"
	kindCompaction         = "transcript.compaction"
	kindChildEvent         = "subagent.event"
	kindChildUsage         = "subagent.usage"
	kindBranchMoved        = "branch.moved"
)

func pending(payloadCodec codec.Codec, kind string, value any) (store.PendingEntry, error) {
	payload, err := codec.Encode(payloadCodec, value)
	if err != nil {
		return store.PendingEntry{}, fmt.Errorf("encode %s entry: %w", kind, err)
	}
	return store.PendingEntry{
		Kind:       kind,
		Payload:    payload,
		Durability: store.DurabilitySync,
	}, nil
}

func decodePayload[T any](payloadCodec codec.Codec, entry store.Entry) (T, error) {
	value, err := codec.Decode[T](payloadCodec, entry.Payload)
	if err != nil {
		return value, fmt.Errorf("decode %s entry at sequence %d: %w", entry.Kind, entry.Seq, err)
	}
	return value, nil
}

type entryBatch struct {
	codec   codec.Codec
	entries []store.PendingEntry
	err     error
}

func newEntryBatch(payloadCodec codec.Codec, capacity int) *entryBatch {
	return &entryBatch{codec: payloadCodec, entries: make([]store.PendingEntry, 0, capacity)}
}

func (b *entryBatch) Add(kind string, value any) {
	if b.err != nil {
		return
	}
	entry, err := pending(b.codec, kind, value)
	if err != nil {
		b.err = err
		return
	}
	b.entries = append(b.entries, entry)
}

func (b *entryBatch) Result() ([]store.PendingEntry, error) {
	if b.err != nil {
		return nil, b.err
	}
	return store.ClonePending(b.entries), nil
}
