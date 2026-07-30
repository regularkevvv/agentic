// Package repair builds deterministic provider projections from durable
// transcripts. It never mutates the durable message slice.
package repair

import (
	"context"
	"encoding/json"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
)

type FrontierMode uint8

const (
	CloseInterruptedFrontier FrontierMode = iota
	PreserveDeferredFrontier
)

type PendingState uint8

const (
	PendingUnknown PendingState = iota
	PendingPlanned
	PendingStarted
	PendingIndeterminate
)

type PendingCall struct {
	ID    string       `json:"id"`
	Name  string       `json:"name,omitempty"`
	State PendingState `json:"state,omitempty"`
}

type PendingCalls struct {
	Calls  []PendingCall `json:"calls,omitempty"`
	Reason string        `json:"reason,omitempty"`
}

// Repair is terminal: it should be the final HistoryProcessor after any
// compaction or injection transforms.
func Repair(mode FrontierMode, pending PendingCalls) agentic.HistoryProcessor {
	captured := PendingCalls{Reason: pending.Reason, Calls: append([]PendingCall(nil), pending.Calls...)}
	return agentic.HistoryProcessorFunc(func(_ context.Context, messages []agentic.Message) ([]agentic.Message, error) {
		return Process(messages, mode, captured)
	})
}

type frontier struct {
	calls       []agentic.ToolUse
	resolved    map[string]bool
	outputIndex int
}

// Process exposes Repair's pure transform for recovery code and property tests.
func Process(messages []agentic.Message, mode FrontierMode, pending PendingCalls) ([]agentic.Message, error) {
	input, err := cloneMessages(messages)
	if err != nil {
		return nil, err
	}
	output := make([]agentic.Message, 0, len(input)+len(pending.Calls))
	var open *frontier
	var recentlyResolved map[string]bool

	closeOpen := func() {
		for _, call := range open.calls {
			if open.resolved[call.ID] {
				continue
			}
			output = append(output, syntheticResult(call, pending))
		}
		open = nil
	}

	for index, message := range input {
		uses := message.GetToolUses()
		results := message.GetToolResults()
		if len(uses) > 0 {
			recentlyResolved = nil
			if message.Role != agentic.RoleAssistant || len(results) > 0 {
				return nil, invalid("message %d contains an invalid tool-call frontier", index)
			}
			if open != nil {
				if mode == PreserveDeferredFrontier {
					return nil, invalid("message %d starts a new frontier before the deferred frontier is resolved", index)
				}
				closeOpen()
			}
			seen := make(map[string]struct{}, len(uses))
			for _, call := range uses {
				if call.ID == "" {
					return nil, invalid("tool call at message %d has an empty ID", index)
				}
				if _, exists := seen[call.ID]; exists {
					return nil, invalid("duplicate tool call ID %q in one frontier", call.ID)
				}
				seen[call.ID] = struct{}{}
			}
			output = append(output, message)
			open = &frontier{calls: uses, resolved: make(map[string]bool, len(uses)), outputIndex: len(output) - 1}
			continue
		}

		if len(results) > 0 {
			if message.Role != agentic.RoleTool {
				return nil, invalid("message %d contains tool results outside a tool message", index)
			}
			kept := message
			kept.Content = kept.Content[:0]
			calls := make(map[string]agentic.ToolUse)
			if open != nil {
				for _, call := range open.calls {
					calls[call.ID] = call
				}
			}
			for _, part := range message.Content {
				if part.Type != agentic.ContentToolResult || part.ToolResult == nil {
					kept.Content = append(kept.Content, part)
					continue
				}
				result := part.ToolResult
				call, exists := calls[result.ToolUseID]
				if !exists {
					if recentlyResolved[result.ToolUseID] {
						return nil, invalid("duplicate result for tool call %q", result.ToolUseID)
					}
					continue // orphan results stay durable but are omitted from projection
				}
				if open.resolved[result.ToolUseID] {
					return nil, invalid("duplicate result for tool call %q", result.ToolUseID)
				}
				if result.Name != "" && result.Name != call.Name {
					return nil, invalid("result for %q names %q instead of %q", result.ToolUseID, result.Name, call.Name)
				}
				open.resolved[result.ToolUseID] = true
				kept.Content = append(kept.Content, part)
			}
			if len(kept.Content) > 0 {
				output = append(output, kept)
			}
			if open != nil && frontierComplete(open) {
				recentlyResolved = make(map[string]bool, len(open.calls))
				for _, call := range open.calls {
					recentlyResolved[call.ID] = true
				}
				open = nil
			}
			continue
		}

		if open != nil {
			if mode == PreserveDeferredFrontier {
				return nil, invalid("message %d follows an unresolved deferred frontier", index)
			}
			closeOpen()
		}
		recentlyResolved = nil
		output = append(output, message)
	}

	if open == nil {
		return output, nil
	}
	if mode == PreserveDeferredFrontier {
		unresolved := unresolvedCalls(open)
		if !samePending(unresolved, pending.Calls) {
			return nil, invalid("deferred frontier does not match pending call IDs")
		}
		// An open frontier is durable state, not a valid provider request. Omit
		// the assistant frontier and any partial result messages after it.
		return output[:open.outputIndex], nil
	}
	closeOpen()
	return output, nil
}

// InspectFrontier validates pair identity and returns the final unresolved
// calls. Orphan results are ignored exactly as Repair ignores them.
func InspectFrontier(messages []agentic.Message) ([]agentic.ToolUse, error) {
	projected, err := Process(messages, PreserveDeferredFrontier, pendingFromFinalFrontier(messages))
	if err != nil {
		return nil, err
	}
	_ = projected
	return rawFinalFrontier(messages)
}

func rawFinalFrontier(messages []agentic.Message) ([]agentic.ToolUse, error) {
	var calls []agentic.ToolUse
	resolved := make(map[string]bool)
	for index, message := range messages {
		uses := message.GetToolUses()
		results := message.GetToolResults()
		if len(uses) > 0 {
			if len(calls) > 0 {
				return nil, invalid("message %d starts a new frontier before the previous one is resolved", index)
			}
			seen := make(map[string]bool)
			for _, call := range uses {
				if call.ID == "" || seen[call.ID] {
					return nil, invalid("invalid duplicate or empty call ID %q", call.ID)
				}
				seen[call.ID] = true
			}
			calls = uses
			resolved = make(map[string]bool)
		}
		if len(results) > 0 && len(calls) > 0 {
			byID := make(map[string]agentic.ToolUse, len(calls))
			for _, call := range calls {
				byID[call.ID] = call
			}
			for _, result := range results {
				call, ok := byID[result.ToolUseID]
				if !ok {
					continue
				}
				if resolved[result.ToolUseID] {
					return nil, invalid("duplicate result for tool call %q", result.ToolUseID)
				}
				if result.Name != "" && result.Name != call.Name {
					return nil, invalid("result name mismatch for %q", result.ToolUseID)
				}
				resolved[result.ToolUseID] = true
			}
			complete := true
			for _, call := range calls {
				complete = complete && resolved[call.ID]
			}
			if complete {
				calls = nil
				resolved = make(map[string]bool)
			}
		}
	}
	if len(calls) == 0 {
		return nil, nil
	}
	var unresolved []agentic.ToolUse
	for _, call := range calls {
		if !resolved[call.ID] {
			unresolved = append(unresolved, call)
		}
	}
	return unresolved, nil
}

func pendingFromFinalFrontier(messages []agentic.Message) PendingCalls {
	calls, _ := rawFinalFrontier(messages)
	pending := PendingCalls{Calls: make([]PendingCall, len(calls))}
	for i, call := range calls {
		pending.Calls[i] = PendingCall{ID: call.ID, Name: call.Name}
	}
	return pending
}

func syntheticResult(call agentic.ToolUse, pending PendingCalls) agentic.Message {
	reason := pending.Reason
	state := PendingUnknown
	for _, candidate := range pending.Calls {
		if candidate.ID == call.ID {
			state = candidate.State
			break
		}
	}
	if reason == "" {
		switch state {
		case PendingIndeterminate, PendingStarted:
			reason = "the tool started but no durable result was observed; its effect is indeterminate"
		case PendingPlanned:
			reason = "the tool was abandoned before it started"
		default:
			reason = "the tool call was interrupted before a durable result was committed"
		}
	}
	content := "Harness synthetic result: " + reason + "; the call was not retried."
	return agentic.NewToolResultMessageFor(call.ID, call.Name, content, true)
}

func frontierComplete(value *frontier) bool {
	for _, call := range value.calls {
		if !value.resolved[call.ID] {
			return false
		}
	}
	return true
}

func unresolvedCalls(value *frontier) []agentic.ToolUse {
	var calls []agentic.ToolUse
	for _, call := range value.calls {
		if !value.resolved[call.ID] {
			calls = append(calls, call)
		}
	}
	return calls
}

func samePending(calls []agentic.ToolUse, pending []PendingCall) bool {
	if len(calls) != len(pending) {
		return false
	}
	for i, call := range calls {
		if call.ID != pending[i].ID || (pending[i].Name != "" && call.Name != pending[i].Name) {
			return false
		}
	}
	return true
}

func cloneMessages(messages []agentic.Message) ([]agentic.Message, error) {
	if len(messages) == 0 {
		return nil, nil
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		return nil, fmt.Errorf("clone transcript: %w", err)
	}
	var cloned []agentic.Message
	if err := json.Unmarshal(encoded, &cloned); err != nil {
		return nil, fmt.Errorf("clone transcript: %w", err)
	}
	return cloned, nil
}

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", agentic.ErrTranscriptInvalid, fmt.Sprintf(format, args...))
}
