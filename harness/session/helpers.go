package session

import (
	"encoding/json"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
)

func cloneMessages(messages []agentic.Message) []agentic.Message {
	if len(messages) == 0 {
		return nil
	}
	// This is a defensive deep copy of the released root Message shape, not a
	// durable representation choice. Journal payloads always use codec.Codec.
	encoded, err := json.Marshal(messages)
	if err == nil {
		var cloned []agentic.Message
		if json.Unmarshal(encoded, &cloned) == nil {
			return cloned
		}
	}
	return append([]agentic.Message(nil), messages...)
}

func cloneUsage(usage agentic.Usage) agentic.Usage {
	usage.RequestUsages = append([]agentic.RequestUsage(nil), usage.RequestUsages...)
	return usage
}

func cloneSuspension(value *agentic.Suspension) *agentic.Suspension {
	if value == nil {
		return nil
	}
	copy := *value
	copy.Payload = append([]byte(nil), value.Payload...)
	return &copy
}

func cloneLimits(limits agentic.UsageLimits) agentic.UsageLimits {
	return agentic.UsageLimits{
		MaxRequestTokens:  cloneInt(limits.MaxRequestTokens),
		MaxResponseTokens: cloneInt(limits.MaxResponseTokens),
		MaxTotalTokens:    cloneInt(limits.MaxTotalTokens),
		MaxRequests:       cloneInt(limits.MaxRequests),
		MaxToolCalls:      cloneInt(limits.MaxToolCalls),
	}
}

func cloneInt(value *int) *int {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func usageDelta(current, previous agentic.Usage) (agentic.Usage, error) {
	if current.PromptTokens < previous.PromptTokens || current.CompletionTokens < previous.CompletionTokens ||
		current.TotalTokens < previous.TotalTokens || current.CacheReadTokens < previous.CacheReadTokens ||
		current.CacheCreationTokens < previous.CacheCreationTokens || current.ReasoningTokens < previous.ReasoningTokens ||
		current.Requests < previous.Requests || current.ToolCalls < previous.ToolCalls ||
		len(current.RequestUsages) < len(previous.RequestUsages) {
		return agentic.Usage{}, errors.New("driver usage counters moved backwards")
	}
	return agentic.Usage{
		PromptTokens:        current.PromptTokens - previous.PromptTokens,
		CompletionTokens:    current.CompletionTokens - previous.CompletionTokens,
		TotalTokens:         current.TotalTokens - previous.TotalTokens,
		CacheReadTokens:     current.CacheReadTokens - previous.CacheReadTokens,
		CacheCreationTokens: current.CacheCreationTokens - previous.CacheCreationTokens,
		ReasoningTokens:     current.ReasoningTokens - previous.ReasoningTokens,
		Requests:            current.Requests - previous.Requests,
		ToolCalls:           current.ToolCalls - previous.ToolCalls,
		RequestUsages:       append([]agentic.RequestUsage(nil), current.RequestUsages[len(previous.RequestUsages):]...),
	}, nil
}

func addUsage(total, delta agentic.Usage) agentic.Usage {
	total.PromptTokens += delta.PromptTokens
	total.CompletionTokens += delta.CompletionTokens
	total.TotalTokens += delta.TotalTokens
	total.CacheReadTokens += delta.CacheReadTokens
	total.CacheCreationTokens += delta.CacheCreationTokens
	total.ReasoningTokens += delta.ReasoningTokens
	total.Requests += delta.Requests
	total.ToolCalls += delta.ToolCalls
	total.RequestUsages = append(total.RequestUsages, delta.RequestUsages...)
	return total
}

func usageEmpty(usage agentic.Usage) bool {
	return usage.PromptTokens == 0 && usage.CompletionTokens == 0 && usage.TotalTokens == 0 &&
		usage.CacheReadTokens == 0 && usage.CacheCreationTokens == 0 && usage.ReasoningTokens == 0 &&
		usage.Requests == 0 && usage.ToolCalls == 0 && len(usage.RequestUsages) == 0
}

func remainingLimits(budget agentic.UsageLimits, used agentic.Usage) (agentic.UsageLimits, error) {
	remaining := cloneLimits(budget)
	checks := []struct {
		value **int
		used  int
		name  string
	}{
		{&remaining.MaxRequestTokens, used.PromptTokens, "request_tokens"},
		{&remaining.MaxResponseTokens, used.CompletionTokens, "response_tokens"},
		{&remaining.MaxTotalTokens, used.TotalTokens, "total_tokens"},
		{&remaining.MaxRequests, used.Requests, "requests"},
		{&remaining.MaxToolCalls, used.ToolCalls, "tool_calls"},
	}
	for _, check := range checks {
		if *check.value == nil {
			continue
		}
		value := **check.value - check.used
		if value <= 0 {
			return agentic.UsageLimits{}, &BudgetError{Cause: fmt.Errorf("%s exhausted", check.name)}
		}
		*check.value = &value
	}
	return remaining, nil
}

func messagesEqual(left, right []agentic.Message) bool {
	if len(left) != len(right) {
		return false
	}
	if len(left) == 0 {
		return true
	}
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
}

func interruptionContextMessage(text string) agentic.Message {
	return agentic.NewTextMessage(
		agentic.RoleUser,
		`<harness_context type="interruption">`+text+`</harness_context>`,
	)
}

func providerHistory(messages []agentic.Message, markers []contextMarker) []agentic.Message {
	if len(markers) == 0 {
		return cloneMessages(messages)
	}
	projected := make([]agentic.Message, 0, len(messages)+len(markers))
	markerIndex := 0
	for position := 0; position <= len(messages); position++ {
		for markerIndex < len(markers) && markers[markerIndex].after <= position {
			projected = append(projected, cloneMessages([]agentic.Message{markers[markerIndex].message})[0])
			markerIndex++
		}
		if position < len(messages) {
			projected = append(projected, cloneMessages([]agentic.Message{messages[position]})[0])
		}
	}
	for markerIndex < len(markers) {
		projected = append(projected, cloneMessages([]agentic.Message{markers[markerIndex].message})[0])
		markerIndex++
	}
	return projected
}

func shiftContextMarkers(markers []contextMarker, amount int) {
	for i := range markers {
		markers[i].after += amount
	}
}
