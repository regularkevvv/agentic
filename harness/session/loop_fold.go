package session

import (
	"fmt"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// loopFold is the incremental attribution state of the sessionloop
// projection. It is consumed record by record, in durable order, and works
// identically for a live event stream and a journal.Load replay: everything
// it tracks is derivable from the log alone, so it is never a second source
// of truth beside the session.
type loopFold struct {
	// queue indexes durable queue acceptance facts by queue ID so later
	// drain/cancel records and injected messages can be attributed to their
	// original kind, content, and acceptance position.
	queue map[string]loopQueueFact

	// currentRunID is the open run identity: set by run.opened, cleared by
	// run.closed. Suspension does not clear it (a suspension pauses the same
	// run).
	currentRunID string

	// pendingRunEntries buffers recovery_resolution/resume_prompt message
	// entries committed inside an indeterminate-resume batch BEFORE that
	// batch's run.opened: their run attribution comes from the surrounding
	// batch's run.opened, so they are emitted, in position order, when it
	// arrives. A torn batch that never reaches its run.opened keeps them
	// buffered; recovery rewrites such logs before they are replayed.
	pendingRunEntries []sessionloop.Entry

	// lastDurable is the highest authoritative position seen by this fold.
	// Preview events repeat it instead of carrying a zero position.
	lastDurable uint64

	// previewOrdinal is the STREAM-LOCAL monotonic arrival counter stamped on
	// preview events. It is never the session's per-turn preview ordinal.
	previewOrdinal uint64

	// pendingDropped accumulates preview-loss counts from records that
	// project to no event, so the loss is reported on the next emitted event
	// instead of disappearing.
	pendingDropped uint64
}

// loopQueueFact is the remembered acceptance of one durable queue item.
type loopQueueFact struct {
	kind     sessionloop.CommandKind
	content  []sessionloop.Block
	position sessionloop.Position
}

func newLoopFold() loopFold {
	return loopFold{queue: make(map[string]loopQueueFact)}
}

// rememberQueue records one queue acceptance for later attribution.
func (f *loopFold) rememberQueue(id string, kind QueueKind, content []sessionloop.Block, position sessionloop.Position) {
	f.queue[id] = loopQueueFact{kind: loopCommandKind(kind), content: content, position: position}
}

// queuedInput reconstructs the protocol view of one queue item. Unknown IDs
// (possible only when a caller replays from a mid-stream position without
// seeding the fold) degrade to the bare ID.
func (f *loopFold) queuedInput(id string) sessionloop.QueuedInput {
	fact, ok := f.queue[id]
	if !ok {
		return sessionloop.QueuedInput{ID: sessionloop.QueueID(id)}
	}
	return sessionloop.QueuedInput{
		ID:       sessionloop.QueueID(id),
		Kind:     fact.kind,
		Position: fact.position,
		Content:  cloneLoopBlocks(fact.content),
	}
}

// injectedOrigin resolves the entry origin of one injected message through
// the queue index. Steer is the documented default when the queue ID is
// unknown to this fold.
func (f *loopFold) injectedOrigin(queueID string) sessionloop.EntryOrigin {
	if fact, ok := f.queue[queueID]; ok && fact.kind == sessionloop.CommandFollowUp {
		return sessionloop.OriginFollowUp
	}
	return sessionloop.OriginSteer
}

func loopCommandKind(kind QueueKind) sessionloop.CommandKind {
	switch kind {
	case QueueSteer:
		return sessionloop.CommandSteer
	case QueueFollowUp:
		return sessionloop.CommandFollowUp
	case QueueNextTurn:
		return sessionloop.CommandNextTurn
	default:
		return sessionloop.CommandKind(kind)
	}
}

func loopEntryID(seq uint64, index int) sessionloop.EntryID {
	if index == 0 {
		return sessionloop.EntryID(fmt.Sprintf("e%d", seq))
	}
	return sessionloop.EntryID(fmt.Sprintf("e%d.%d", seq, index))
}

func cloneLoopBlocks(blocks []sessionloop.Block) []sessionloop.Block {
	if blocks == nil {
		return nil
	}
	clones := make([]sessionloop.Block, len(blocks))
	for index, block := range blocks {
		clones[index] = block.Clone()
	}
	return clones
}
