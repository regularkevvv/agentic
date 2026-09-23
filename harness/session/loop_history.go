// Package session provides finite journal projection to reconcile observation across
// worker ownership changes without opening another writable journal handle.
package session

import (
	"context"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// Replay returns copy-owned authoritative events in (after, through]. It reads
// the existing journal; it neither dispatches commands nor changes ownership.
// The caller obtains through from Snapshot. Live-only observations are excluded.
func (v *LoopView[O]) Replay(ctx context.Context, after, through sessionloop.Position) ([]sessionloop.Event, error) {
	if v.isClosed() {
		return nil, loopClosedError()
	}
	loaded, err := v.inner.journalRef().Load(ctx)
	if err != nil {
		return nil, mapLoopError(err)
	}
	if after.Sequence > through.Sequence || through.Sequence > loaded.Cursor.Seq {
		return nil, sessionloop.ErrUnknownPosition
	}
	records, err := loopRecords(v.inner.codecRef(), loaded.Entries)
	if err != nil {
		return nil, err
	}
	projector := v.newProjector(true)
	var result []sessionloop.Event
	for _, record := range records {
		if record.Cursor > through.Sequence {
			break
		}
		projected, err := projector.apply(ctx, record)
		if err != nil {
			return nil, err
		}
		if record.Cursor <= after.Sequence {
			continue
		}
		for _, value := range projected {
			if value.Nature == sessionloop.EventAuthoritative && !value.Position.IsZero() {
				result = append(result, value.Clone())
			}
		}
	}
	return result, nil
}
