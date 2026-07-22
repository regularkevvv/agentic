package agentic

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const toolSuspensionPayloadVersion = 1

type toolSuspensionPayload struct {
	Version           int
	SuspensionID      string
	Calls             []ToolUse
	ExecutableCallIDs []string
	Iteration         int
	Usage             Usage
	RetryCounts       map[string]int
	TotalRetries      int
	ValidationRetries int
	EndStrategy       EndStrategy
	ConfigFingerprint string
	LastFinishReason  FinishReason
	Deferral          ToolDeferral
}

func frontierHash(messages []Message, calls []ToolUse) string {
	payload := struct {
		Version  int
		Messages []Message
		Calls    []ToolUse
	}{
		Version:  1,
		Messages: messages,
		Calls:    calls,
	}
	encoded, _ := json.Marshal(payload)
	hash := sha256.Sum256(encoded)
	return "v1:" + hex.EncodeToString(hash[:])
}

func (c *agentCore) executionFingerprint(ls *loopState) string {
	payload := struct {
		Version              int
		Model                string
		Tools                []Tool
		OutputToolNames      map[string]bool
		ResponseFormat       *ResponseFormat
		EndStrategy          EndStrategy
		MaxIterations        int
		UsageLimits          *UsageLimits
		RetryConfig          RetryConfig
		MaxValidationRetries int
		ToolChoice           *ToolChoice
	}{
		Version:              1,
		Model:                c.model.Name(),
		Tools:                ls.registry.Tools(),
		OutputToolNames:      copyNameSet(c.outputToolNames),
		ResponseFormat:       c.responseFormat,
		EndStrategy:          ls.endStrategy,
		MaxIterations:        ls.maxIterations,
		UsageLimits:          ls.usageLimits,
		RetryConfig:          c.config.retryConfig,
		MaxValidationRetries: c.config.maxValidationRetries,
		ToolChoice:           firstNonNil(ls.options.toolChoice, c.config.toolChoice),
	}
	encoded, _ := json.Marshal(payload)
	hash := sha256.Sum256(encoded)
	return hex.EncodeToString(hash[:])
}

func (c *agentCore) newToolSuspension(ls *loopState, calls, executable []ToolUse, deferral ToolDeferral) (*Suspension, error) {
	id := newSuspensionID()
	ids := make([]string, len(executable))
	for index, call := range executable {
		ids[index] = call.ID
	}
	payload := toolSuspensionPayload{
		Version:           toolSuspensionPayloadVersion,
		SuspensionID:      id,
		Calls:             cloneToolUses(calls),
		ExecutableCallIDs: ids,
		Iteration:         ls.iteration,
		Usage:             ls.totalUsage,
		RetryCounts:       copyRetryCounts(ls.retryCounts),
		TotalRetries:      ls.totalRetries,
		ValidationRetries: ls.validationRetries,
		EndStrategy:       ls.endStrategy,
		ConfigFingerprint: ls.configFingerprint,
		LastFinishReason:  ls.lastFinishReason,
		Deferral:          ToolDeferral{Kind: deferral.Kind, Payload: append([]byte(nil), deferral.Payload...)},
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal tool suspension: %w", err)
	}
	frontier, err := inspectTranscript(ls.messages)
	if err != nil {
		return nil, err
	}
	return &Suspension{
		ID:           id,
		Kind:         deferral.Kind,
		FrontierHash: frontierHash(ls.messages, frontier),
		Payload:      encoded,
	}, nil
}

func decodeToolSuspension(suspension Suspension) (toolSuspensionPayload, error) {
	var payload toolSuspensionPayload
	if err := json.Unmarshal(suspension.Payload, &payload); err != nil {
		return toolSuspensionPayload{}, fmt.Errorf("%w: decode payload: %v", ErrSuspensionVersion, err)
	}
	if payload.Version != toolSuspensionPayloadVersion {
		return toolSuspensionPayload{}, fmt.Errorf("%w: got %d", ErrSuspensionVersion, payload.Version)
	}
	if payload.SuspensionID == "" || len(payload.Calls) == 0 || len(payload.ExecutableCallIDs) == 0 {
		return toolSuspensionPayload{}, fmt.Errorf("%w: missing suspended tool calls", ErrSuspensionMismatch)
	}
	return payload, nil
}

func copyRetryCounts(counts map[string]int) map[string]int {
	if len(counts) == 0 {
		return nil
	}
	copy := make(map[string]int, len(counts))
	for name, count := range counts {
		copy[name] = count
	}
	return copy
}
