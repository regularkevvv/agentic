package session

import (
	"context"
	"fmt"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/contextpolicy"
	"github.com/regularkevvv/agentic/harness/repair"
	"github.com/regularkevvv/agentic/harness/store"
)

// projectHistory is the terminal session-owned HistoryProcessor. Capability
// policy runs first, durable additions/compaction are committed, and Repair is
// always the last transformation before a provider request.
func (s *Session[O]) projectHistory(ctx context.Context, rootMessages []agentic.Message) ([]agentic.Message, error) {
	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return nil, err
	}
	if s.run == nil {
		s.mu.Unlock()
		return nil, fmt.Errorf("%w: context policy observed no active run", ErrCommitProjectionMismatch)
	}
	expectedRoot, fullDurable := s.contextViewsLocked(rootMessages)
	if !messagesEqual(expectedRoot, rootMessages) {
		cause := fmt.Errorf("%w: driver history differs before context projection", ErrCommitProjectionMismatch)
		cancel := s.faultLocked(cause)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return nil, cause
	}
	currentCompaction := cloneContextCompaction(s.compaction)
	instructions := s.run.instructions
	s.mu.Unlock()

	projection, err := s.context.Project(ctx, contextpolicy.ProjectionRequest{
		Messages:   fullDurable,
		Compaction: currentCompaction,
	})
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	if s.state == Faulted {
		err := &FaultError{SessionID: s.id, Cause: s.fault}
		s.mu.Unlock()
		return nil, err
	}
	if s.state != Running || s.run == nil {
		s.mu.Unlock()
		return nil, context.Canceled
	}
	batch := newEntryBatch(s.codec, len(projection.DurableAdditions)+1)
	after := len(s.messages)
	for _, message := range projection.DurableAdditions {
		batch.Add(kindContextMessage, contextMessagePayload{After: after, Message: message})
	}
	if projection.CompactionChanged && projection.Compaction != nil {
		batch.Add(kindCompaction, compactionPayload{Compaction: *projection.Compaction})
	}
	pendingEntries, encodeErr := batch.Result()
	if encodeErr != nil {
		cancel := s.faultLocked(encodeErr)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return nil, encodeErr
	}
	var committed []store.Entry
	if len(pendingEntries) > 0 {
		commit, appendErr := s.journal.Append(context.WithoutCancel(ctx), s.cursor, pendingEntries...)
		if appendErr != nil {
			cancel := s.faultLocked(appendErr)
			s.mu.Unlock()
			if cancel != nil {
				cancel()
			}
			return nil, appendErr
		}
		for _, message := range projection.DurableAdditions {
			s.contextMarkers = append(s.contextMarkers, contextMarker{
				after:   after,
				message: cloneMessages([]agentic.Message{message})[0],
			})
		}
		if projection.CompactionChanged {
			s.compaction = cloneContextCompaction(projection.Compaction)
		}
		s.cursor = commit.Cursor
		committed = commit.Entries
	}
	s.mu.Unlock()
	s.publishOwn(committed, agentic.EventAuthoritative)

	projectedMessages := appendExchangeInstructions(projection.Messages, instructions)
	repaired, err := repair.Process(projectedMessages, repair.CloseInterruptedFrontier, repair.PendingCalls{})
	if err != nil {
		return nil, err
	}
	return repaired, nil
}

func appendExchangeInstructions(messages []agentic.Message, instructions string) []agentic.Message {
	result := cloneMessages(messages)
	if instructions == "" {
		return result
	}
	for index := range result {
		if result[index].Role != agentic.RoleSystem {
			continue
		}
		result[index].Content = append(result[index].Content, agentic.Part{
			Type: agentic.ContentText,
			Text: "\n\n" + instructions,
		})
		return result
	}
	return append([]agentic.Message{agentic.NewTextMessage(agentic.RoleSystem, instructions)}, result...)
}

func (s *Session[O]) contextViewsLocked(rootMessages []agentic.Message) ([]agentic.Message, []agentic.Message) {
	base := cloneMessages(s.messages)
	markers := cloneContextMarkers(s.contextMarkers)
	if len(rootMessages) > 0 && rootMessages[0].Role == agentic.RoleSystem &&
		(len(base) == 0 || base[0].Role != agentic.RoleSystem) {
		base = append([]agentic.Message{cloneMessages(rootMessages[:1])[0]}, base...)
		for index := range markers {
			markers[index].after++
		}
	}
	included := s.run.contextMarkerCount
	if included > len(markers) {
		included = len(markers)
	}
	expected := providerHistory(base, markers[:included])
	full := providerHistory(base, markers)
	return expected, full
}

func cloneContextMarkers(markers []contextMarker) []contextMarker {
	result := make([]contextMarker, len(markers))
	for index, marker := range markers {
		result[index] = contextMarker{
			after:   marker.after,
			message: cloneMessages([]agentic.Message{marker.message})[0],
		}
	}
	return result
}

func cloneContextCompaction(value *contextpolicy.Compaction) *contextpolicy.Compaction {
	if value == nil {
		return nil
	}
	copy := *value
	copy.Summary = cloneMessages([]agentic.Message{value.Summary})[0]
	return &copy
}
