package sessionloop

import (
	"encoding/json"
	"fmt"
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

// Entry is one authoritative committed conversation item with run and
// command attribution. The default authoritative projection excludes
// system/developer instructions, hidden reasoning, provider signatures, and
// provider-private metadata (law L12).
type Entry struct {
	ID        EntryID
	SessionID SessionID
	RunID     RunID
	CommandID CommandID
	Position  Position
	Role      Role
	Origin    EntryOrigin
	Content   []Block
}

// Clone returns a deep, copy-owned copy of the entry.
func (e Entry) Clone() Entry {
	clone := e
	clone.Content = cloneBlocks(e.Content)
	return clone
}

// BlockKind names one standard content block shape.
type BlockKind string

// Standard block kinds.
const (
	BlockText       BlockKind = "text"
	BlockData       BlockKind = "data"
	BlockToolCall   BlockKind = "tool_call"
	BlockToolResult BlockKind = "tool_result"
)

// Block is one content block inside an entry or input. Exactly the fields
// belonging to its kind are populated: Text for text blocks, MediaType and
// Data for data blocks, ToolCall for tool calls, and ToolResult (with the
// textual result content in Text) for tool results.
type Block struct {
	Kind       BlockKind
	Text       string
	MediaType  string
	Data       json.RawMessage
	ToolCall   *ToolCall
	ToolResult *ToolResult
}

// Clone returns a deep, copy-owned copy of the block.
func (b Block) Clone() Block {
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

// ToolCall describes one tool invocation with a stable call ID.
type ToolCall struct {
	CallID string
	Name   string
	Data   json.RawMessage
}

// Clone returns a deep, copy-owned copy of the tool call.
func (t ToolCall) Clone() ToolCall {
	clone := t
	clone.Data = cloneRaw(t.Data)
	return clone
}

// ToolResult describes one tool outcome with a stable call ID and error
// state. The textual result content lives in the Text of the surrounding
// tool_result block; Data carries optional structured detail.
type ToolResult struct {
	CallID  string
	Name    string
	IsError bool
	Data    json.RawMessage
}

// Clone returns a deep, copy-owned copy of the tool result.
func (t ToolResult) Clone() ToolResult {
	clone := t
	clone.Data = cloneRaw(t.Data)
	return clone
}

// Input is copy-owned content with optional application metadata. Metadata
// is for correlation, not instruction smuggling: implementations must not
// translate unknown metadata into model-visible text.
type Input struct {
	Content []Block
	Meta    map[string]string
}

// Clone returns a deep, copy-owned copy of the input.
func (i Input) Clone() Input {
	return Input{Content: cloneBlocks(i.Content), Meta: cloneMeta(i.Meta)}
}

// ValidateInput enforces the structural rules of input content: at least one
// block, the fields required by each block kind, and that any present Data
// is exactly one complete JSON value. Every failure wraps ErrInvalidCommand
// because invalid content makes the carrying command invalid.
func ValidateInput(input Input) error {
	if len(input.Content) == 0 {
		return fmt.Errorf("input requires at least one content block: %w", ErrInvalidCommand)
	}
	for index, block := range input.Content {
		if err := validateBlock(block); err != nil {
			return fmt.Errorf("input block %d: %w", index, err)
		}
	}
	return nil
}

func validateBlock(block Block) error {
	if len(block.Data) > 0 && !json.Valid(block.Data) {
		return fmt.Errorf("data must be exactly one complete JSON value: %w", ErrInvalidCommand)
	}
	switch block.Kind {
	case BlockText:
		if block.Text == "" {
			return fmt.Errorf("text block requires text: %w", ErrInvalidCommand)
		}
		if block.ToolCall != nil || block.ToolResult != nil {
			return fmt.Errorf("text block must not carry tool payloads: %w", ErrInvalidCommand)
		}
	case BlockData:
		if len(block.Data) == 0 {
			return fmt.Errorf("data block requires data: %w", ErrInvalidCommand)
		}
		if block.ToolCall != nil || block.ToolResult != nil {
			return fmt.Errorf("data block must not carry tool payloads: %w", ErrInvalidCommand)
		}
	case BlockToolCall:
		if block.ToolCall == nil {
			return fmt.Errorf("tool_call block requires a tool call: %w", ErrInvalidCommand)
		}
		if block.ToolCall.CallID == "" || block.ToolCall.Name == "" {
			return fmt.Errorf("tool call requires a call ID and name: %w", ErrInvalidCommand)
		}
		if len(block.ToolCall.Data) > 0 && !json.Valid(block.ToolCall.Data) {
			return fmt.Errorf("tool call data must be exactly one complete JSON value: %w", ErrInvalidCommand)
		}
		if block.ToolResult != nil {
			return fmt.Errorf("tool_call block must not carry a tool result: %w", ErrInvalidCommand)
		}
	case BlockToolResult:
		if block.ToolResult == nil {
			return fmt.Errorf("tool_result block requires a tool result: %w", ErrInvalidCommand)
		}
		if block.ToolResult.CallID == "" {
			return fmt.Errorf("tool result requires a call ID: %w", ErrInvalidCommand)
		}
		if len(block.ToolResult.Data) > 0 && !json.Valid(block.ToolResult.Data) {
			return fmt.Errorf("tool result data must be exactly one complete JSON value: %w", ErrInvalidCommand)
		}
		if block.ToolCall != nil {
			return fmt.Errorf("tool_result block must not carry a tool call: %w", ErrInvalidCommand)
		}
	default:
		return fmt.Errorf("unknown block kind %q: %w", block.Kind, ErrInvalidCommand)
	}
	return nil
}

func cloneBlocks(blocks []Block) []Block {
	if blocks == nil {
		return nil
	}
	clones := make([]Block, len(blocks))
	for index, block := range blocks {
		clones[index] = block.Clone()
	}
	return clones
}

func cloneMeta(meta map[string]string) map[string]string {
	if meta == nil {
		return nil
	}
	clone := make(map[string]string, len(meta))
	for key, value := range meta {
		clone[key] = value
	}
	return clone
}

func cloneRaw(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return nil
	}
	return slices.Clone(raw)
}
