package agentic

import (
	"context"
	"encoding/json"
	"fmt"
)

// OutputSpec defines how to extract structured output from the LLM response.
// The default (nil) extracts plain text.
type OutputSpec[O any] interface {
	// Tools returns any additional tools needed for output extraction.
	// For example, ToolOutputSpec adds a hidden "final_result" tool.
	Tools() []Tool

	// Parse extracts the typed output from the agent response.
	// It receives the final assistant message and returns the parsed output.
	Parse(msg Message) (O, error)
}

// TextOutputSpec is the default output spec — extracts text content.
type TextOutputSpec struct{}

func (s *TextOutputSpec) Tools() []Tool { return nil }

func (s *TextOutputSpec) Parse(msg Message) (string, error) {
	return msg.GetTextContent(), nil
}

// ToolOutputSpec uses a designated tool call for structured output.
// The LLM is instructed to call a tool whose input IS the structured output.
// This is the most reliable way to get structured output across all providers.
type ToolOutputSpec[T any] struct {
	toolName    string
	description string
}

// NewToolOutput creates a ToolOutputSpec that extracts structured output via a tool call.
// The type T defines the output schema. The LLM will be given a tool with this schema
// and instructed to call it with the final result.
//
// Use `validate` struct tags to enforce constraints on the output. If the LLM produces
// invalid output, the validation error is sent back to the model for automatic retry.
// See https://github.com/go-playground/validator for all available tags.
//
// Example:
//
//	type MovieReview struct {
//	    Title   string  `json:"title"   validate:"required,min=1"                  description:"Movie title"`
//	    Rating  float64 `json:"rating"  validate:"required,min=1,max=10"           description:"Rating from 1-10"`
//	    Genre   string  `json:"genre"   validate:"required,oneof=action comedy drama" description:"Movie genre"`
//	    Summary string  `json:"summary" validate:"required,min=10"                 description:"Brief review"`
//	}
//
//	output := agentic.NewToolOutput[MovieReview]("Review a movie and provide structured feedback")
func NewToolOutput[T any](description string) *ToolOutputSpec[T] {
	return &ToolOutputSpec[T]{
		toolName:    "__output__",
		description: description,
	}
}

func (s *ToolOutputSpec[T]) Tools() []Tool {
	var zero T
	tool, err := NewToolFromStruct(s.toolName, s.description, zero)
	if err != nil {
		// This should not happen with valid structs
		panic(fmt.Sprintf("failed to create output tool: %v", err))
	}
	return []Tool{tool}
}

func (s *ToolOutputSpec[T]) Parse(msg Message) (T, error) {
	var zero T
	toolUses := msg.GetToolUses()
	for _, tu := range toolUses {
		if tu.Name == s.toolName {
			// Marshal the tool input back to JSON, then unmarshal to the target type
			data, err := json.Marshal(tu.Input)
			if err != nil {
				return zero, fmt.Errorf("marshal output tool input: %w", err)
			}
			var result T
			if err := json.Unmarshal(data, &result); err != nil {
				return zero, fmt.Errorf("unmarshal to %T: %w", result, err)
			}
			// Automatically validate using struct tags if present
			if err := ValidateStruct(result); err != nil {
				return zero, err
			}
			return result, nil
		}
	}
	// If no matching tool call found, try to parse text as JSON
	text := msg.GetTextContent()
	if text != "" {
		var result T
		if err := json.Unmarshal([]byte(text), &result); err != nil {
			return zero, fmt.Errorf("no %q tool call found, and text is not valid JSON: %w", s.toolName, err)
		}
		// Automatically validate using struct tags if present
		if err := ValidateStruct(result); err != nil {
			return zero, err
		}
		return result, nil
	}
	return zero, fmt.Errorf("no output found: expected %q tool call or JSON text", s.toolName)
}

// noopToolHandler is a handler for output tools that should never actually execute.
// If the model calls the output tool, we extract the input as the output —
// the tool itself doesn't need to run.
type noopToolHandler struct {
	name string
}

func (h *noopToolHandler) Execute(ctx context.Context, input map[string]interface{}, deps any) (interface{}, error) {
	// Return the input as-is — it IS the structured output
	return input, nil
}

func (h *noopToolHandler) Name() string {
	return h.name
}
