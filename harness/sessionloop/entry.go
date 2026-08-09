package sessionloop

import (
	"encoding/json"
	"slices"
)

// Role names the conversational author of an entry.
type Role string

// Standard roles.
const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// EntryOrigin records which command or run activity committed an entry, so a
// run containing multiple user messages stays distinguishable.
type EntryOrigin string

// Standard entry origins.
const (
	OriginStart     EntryOrigin = "start"
	OriginSteer     EntryOrigin = "steer"
	OriginFollowUp  EntryOrigin = "follow_up"
	OriginNextTurn  EntryOrigin = "next_turn"
	OriginAssistant EntryOrigin = "assistant"
	OriginTool      EntryOrigin = "tool"
)

// Entry is one complete, authoritative conversation item committed by the
// session. An Entry is not what Stream.Next returns: the stream returns Event,
// and an EventEntryCommitted event carries an Entry. Entries also appear in
// Snapshot and are therefore independent of live delivery.
//
// The default authoritative projection excludes system/developer instructions,
// hidden reasoning, provider signatures, and provider-private metadata (law
// L12).
type Entry struct {
	ID        EntryID
	SessionID SessionID
	RunID     RunID
	CommandID CommandID
	Position  Position
	Role      Role
	Origin    EntryOrigin
	Blocks    []EntryBlock
}

// Clone returns a deep, copy-owned copy of the entry.
func (e Entry) Clone() Entry {
	clone := e
	clone.Blocks = cloneEntryBlocks(e.Blocks)
	return clone
}

// EntryBlockKind names one complete session-to-consumer content shape.
type EntryBlockKind string

// Standard entry block kinds.
const (
	EntryBlockText       EntryBlockKind = "text"
	EntryBlockData       EntryBlockKind = "data"
	EntryBlockToolCall   EntryBlockKind = "tool_call"
	EntryBlockToolResult EntryBlockKind = "tool_result"
)

// EntryBlock is one complete content part of a committed Entry. It is durable
// observation data, not caller Input and not an incomplete streaming Preview.
// Exactly the fields belonging to Kind are populated.
type EntryBlock struct {
	Kind       EntryBlockKind
	Text       string
	MediaType  string
	Data       json.RawMessage
	ToolCall   *EntryToolCall
	ToolResult *EntryToolResult
}

// Clone returns a deep, copy-owned copy of the block.
func (b EntryBlock) Clone() EntryBlock {
	clone := b
	clone.Data = cloneRaw(b.Data)
	if b.ToolCall != nil {
		call := b.ToolCall.Clone()
		clone.ToolCall = &call
	}
	if b.ToolResult != nil {
		result := b.ToolResult.Clone()
		clone.ToolResult = &result
	}
	return clone
}

// EntryToolCall describes one observed tool invocation with a stable call ID.
type EntryToolCall struct {
	CallID string
	Name   string
	Data   json.RawMessage
}

// Clone returns a deep, copy-owned copy of the tool call.
func (t EntryToolCall) Clone() EntryToolCall {
	clone := t
	clone.Data = cloneRaw(t.Data)
	return clone
}

// EntryToolResult describes one observed tool outcome with a stable call ID and
// error state. Textual result content lives in the Text of the surrounding
// EntryBlock; Data carries optional structured detail.
type EntryToolResult struct {
	CallID  string
	Name    string
	IsError bool
	Data    json.RawMessage
}

// Clone returns a deep, copy-owned copy of the tool result.
func (t EntryToolResult) Clone() EntryToolResult {
	clone := t
	clone.Data = cloneRaw(t.Data)
	return clone
}

func cloneEntryBlocks(blocks []EntryBlock) []EntryBlock {
	if blocks == nil {
		return nil
	}
	clones := make([]EntryBlock, len(blocks))
	for index, block := range blocks {
		clones[index] = block.Clone()
	}
	return clones
}

func cloneRaw(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return nil
	}
	return slices.Clone(raw)
}
