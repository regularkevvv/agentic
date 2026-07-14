package core

import (
	"context"
	"encoding/json"
	"fmt"
)

// ToolHandler is the interface that all tool handlers must implement.
// It provides a clean abstraction for executing tools with type safety.
type ToolHandler interface {
	// Execute runs the tool with the given input and returns the result.
	// deps is optional — nil if the agent has no deps.
	Execute(ctx context.Context, input map[string]interface{}, deps any) (interface{}, error)

	// Name returns the name of the tool this handler is for.
	Name() string
}

// ToolExecutionResult represents the result of executing a tool.
type ToolExecutionResult struct {
	ToolUseID string
	ToolName  string
	Content   interface{}
	IsError   bool
	Error     error
}

// FormatToolResult converts a tool result to a string for sending back to the LLM.
func FormatToolResult(result interface{}) string {
	if result == nil {
		return ""
	}
	if str, ok := result.(string); ok {
		return str
	}
	data, err := json.Marshal(result)
	if err != nil {
		return fmt.Sprintf("Error formatting result: %v", err)
	}
	return string(data)
}
