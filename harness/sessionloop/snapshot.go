package sessionloop

// Snapshot is a copy-owned authoritative view of one session (law L7).
// Entries are in authoritative conversation order and retain run and command
// attribution. When a stream reports lag, an unknown position, or an event
// gap, the consumer discards speculative state, loads a snapshot, and
// subscribes after its position.
type Snapshot struct {
	SessionID    SessionID
	Position     Position
	State        State
	ActiveRunID  RunID
	Entries      []Entry
	Pending      []QueuedInput
	Suspension   *Suspension
	Usage        Usage
	Capabilities Capabilities
}

// Clone returns a deep, copy-owned copy of the snapshot.
func (s Snapshot) Clone() Snapshot {
	clone := s
	if s.Entries != nil {
		clone.Entries = make([]Entry, len(s.Entries))
		for index, entry := range s.Entries {
			clone.Entries[index] = entry.Clone()
		}
	}
	if s.Pending != nil {
		clone.Pending = make([]QueuedInput, len(s.Pending))
		for index, queued := range s.Pending {
			clone.Pending[index] = queued.Clone()
		}
	}
	if s.Suspension != nil {
		suspension := s.Suspension.Clone()
		clone.Suspension = &suspension
	}
	clone.Capabilities = s.Capabilities.Clone()
	return clone
}
