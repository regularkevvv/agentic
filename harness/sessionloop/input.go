package sessionloop

import (
	"encoding/json"
	"fmt"
)

// Input is one logical caller submission carried into a session by a Command.
// Blocks are ordered parts of that single submission, not separate turns.
// Input is translated by a concrete host into its own model-facing Message
// representation; it is not itself model or conversation-history data.
//
// Meta is for correlation, not instruction smuggling: implementations must not
// translate unknown metadata into model-visible text.
type Input struct {
	Blocks []InputBlock
	Meta   map[string]string
}

// Clone returns a deep, copy-owned copy of the input.
func (i Input) Clone() Input {
	return Input{Blocks: cloneInputBlocks(i.Blocks), Meta: cloneMeta(i.Meta)}
}

// InputBlockKind names one caller-to-session input shape. Text is the baseline
// shape. Data is a structured input that concrete hosts may reject with
// ErrUnsupported when they cannot translate it safely.
type InputBlockKind string

// Standard input block kinds.
const (
	InputBlockText InputBlockKind = "text"
	InputBlockData InputBlockKind = "data"
)

// InputBlock is one ordered part of a caller submission. It deliberately has
// no tool-call or tool-result fields: those describe observed execution facts
// and belong to EntryBlock. Exactly the fields belonging to Kind are populated.
type InputBlock struct {
	Kind      InputBlockKind
	Text      string
	MediaType string
	Data      json.RawMessage
}

// Clone returns a deep, copy-owned copy of the block.
func (b InputBlock) Clone() InputBlock {
	clone := b
	clone.Data = cloneRaw(b.Data)
	return clone
}

// ValidateInput enforces the structural rules of caller input: at least one
// block, the fields required by each block kind, and exactly one complete JSON
// value whenever Data is present. Every failure wraps ErrInvalidCommand because
// invalid input makes the carrying command invalid.
func ValidateInput(input Input) error {
	if len(input.Blocks) == 0 {
		return fmt.Errorf("input requires at least one input block: %w", ErrInvalidCommand)
	}
	for index, block := range input.Blocks {
		if err := validateInputBlock(block); err != nil {
			return fmt.Errorf("input block %d: %w", index, err)
		}
	}
	return nil
}

func validateInputBlock(block InputBlock) error {
	if len(block.Data) > 0 && !json.Valid(block.Data) {
		return fmt.Errorf("data must be exactly one complete JSON value: %w", ErrInvalidCommand)
	}
	switch block.Kind {
	case InputBlockText:
		if block.Text == "" {
			return fmt.Errorf("text input block requires text: %w", ErrInvalidCommand)
		}
		if len(block.Data) > 0 || block.MediaType != "" {
			return fmt.Errorf("text input block must not carry structured data: %w", ErrInvalidCommand)
		}
	case InputBlockData:
		if len(block.Data) == 0 {
			return fmt.Errorf("data input block requires data: %w", ErrInvalidCommand)
		}
		if block.Text != "" {
			return fmt.Errorf("data input block must not carry text: %w", ErrInvalidCommand)
		}
	default:
		return fmt.Errorf("unknown input block kind %q: %w", block.Kind, ErrInvalidCommand)
	}
	return nil
}

func cloneInputBlocks(blocks []InputBlock) []InputBlock {
	if blocks == nil {
		return nil
	}
	clones := make([]InputBlock, len(blocks))
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
