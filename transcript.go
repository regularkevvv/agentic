package agentic

import (
	"encoding/json"
	"fmt"
)

// cloneMessages makes hook and caller boundaries independent from mutation of
// the slices passed into a run. Message payloads are JSON-shaped by contract;
// fall back to a structural slice copy only for deliberately non-serializable
// application metadata.
func cloneMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return nil
	}
	encoded, err := json.Marshal(messages)
	if err == nil {
		var cloned []Message
		if json.Unmarshal(encoded, &cloned) == nil {
			return cloned
		}
	}
	return append([]Message(nil), messages...)
}

func cloneToolUses(calls []ToolUse) []ToolUse {
	if len(calls) == 0 {
		return nil
	}
	encoded, err := json.Marshal(calls)
	if err == nil {
		var cloned []ToolUse
		if json.Unmarshal(encoded, &cloned) == nil {
			return cloned
		}
	}
	return append([]ToolUse(nil), calls...)
}

func cloneToolResults(results []ToolExecutionResult) []ToolExecutionResult {
	return append([]ToolExecutionResult(nil), results...)
}

// inspectTranscript validates tool pairing one assistant frontier at a time.
// Call IDs may be reused by a provider on a later, fully closed frontier, but
// cannot be ambiguous inside one frontier.
func inspectTranscript(messages []Message) ([]ToolUse, error) {
	var frontier []ToolUse
	resolved := make(map[string]bool)

	for messageIndex, message := range messages {
		uses := message.GetToolUses()
		results := message.GetToolResults()

		if len(uses) > 0 {
			if message.Role != RoleAssistant {
				return nil, fmt.Errorf("%w: message %d contains tool calls outside an assistant message", ErrTranscriptInvalid, messageIndex)
			}
			if len(frontier) > 0 {
				return nil, fmt.Errorf("%w: message %d starts a new tool frontier before the previous one is paired", ErrTranscriptInvalid, messageIndex)
			}
			seen := make(map[string]struct{}, len(uses))
			for _, call := range uses {
				if call.ID == "" {
					return nil, fmt.Errorf("%w: tool call at message %d has an empty ID", ErrTranscriptInvalid, messageIndex)
				}
				if _, exists := seen[call.ID]; exists {
					return nil, fmt.Errorf("%w: duplicate tool call ID %q in one frontier", ErrTranscriptInvalid, call.ID)
				}
				seen[call.ID] = struct{}{}
			}
			frontier = cloneToolUses(uses)
			resolved = make(map[string]bool, len(frontier))
		}

		if len(results) > 0 {
			if message.Role != RoleTool {
				return nil, fmt.Errorf("%w: message %d contains tool results outside a tool message", ErrTranscriptInvalid, messageIndex)
			}
			if len(frontier) == 0 {
				return nil, fmt.Errorf("%w: tool result at message %d has no open call frontier", ErrTranscriptInvalid, messageIndex)
			}
			callsByID := make(map[string]ToolUse, len(frontier))
			for _, call := range frontier {
				callsByID[call.ID] = call
			}
			for _, result := range results {
				call, exists := callsByID[result.ToolUseID]
				if !exists {
					return nil, fmt.Errorf("%w: result for unknown tool call %q", ErrTranscriptInvalid, result.ToolUseID)
				}
				if resolved[result.ToolUseID] {
					return nil, fmt.Errorf("%w: duplicate result for tool call %q", ErrTranscriptInvalid, result.ToolUseID)
				}
				if result.Name != "" && result.Name != call.Name {
					return nil, fmt.Errorf("%w: result for %q names %q instead of %q", ErrTranscriptInvalid, result.ToolUseID, result.Name, call.Name)
				}
				resolved[result.ToolUseID] = true
			}
			complete := true
			for _, call := range frontier {
				if !resolved[call.ID] {
					complete = false
					break
				}
			}
			if complete {
				frontier = nil
				resolved = make(map[string]bool)
			}
			continue
		}

		if len(frontier) > 0 && message.Role != RoleTool && len(uses) == 0 {
			return nil, fmt.Errorf("%w: message %d appears before the open tool frontier is paired", ErrTranscriptInvalid, messageIndex)
		}
	}

	return cloneToolUses(frontier), nil
}

func transcriptToolState(messages []Message) ([]ToolUse, []ToolExecutionResult) {
	var calls []ToolUse
	var results []ToolExecutionResult
	for _, message := range messages {
		calls = append(calls, message.GetToolUses()...)
		for _, result := range message.GetToolResults() {
			results = append(results, ToolExecutionResult{
				ToolUseID: result.ToolUseID,
				ToolName:  result.Name,
				Content:   result.Content,
				IsError:   result.IsError,
			})
		}
	}
	return calls, results
}
