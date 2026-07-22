package core

import (
	"encoding/json"
	"sync"
)

// StreamEventType represents the type of streaming event.
type StreamEventType int

const (
	StreamEventTextDelta StreamEventType = iota
	StreamEventToolCallStart
	StreamEventToolCallDelta
	StreamEventToolResult
	StreamEventDone
	StreamEventError
	StreamEventThinkingDelta
)

// StreamEvent represents a single event during streaming.
type StreamEvent struct {
	Type       StreamEventType
	Delta      string   // Text delta for StreamEventTextDelta, argument delta for StreamEventToolCallDelta
	ToolUse    *ToolUse // For StreamEventToolCallStart
	ToolCallID string   // For StreamEventToolCallDelta — identifies which tool call
	Usage      *Usage   // Set on StreamEventDone
	Error      error    // Set on StreamEventError

	// Signature is the provider-issued signature for a thinking block,
	// delivered as a terminal event on that block (Anthropic signature_delta,
	// Bedrock reasoningContent.signature, Gemini thoughtSignature, OpenAI
	// Responses encrypted_content).
	//
	// Providers that issue a signature reject a thinking block replayed
	// without it, so a streamed reasoning turn that is fed back into a
	// subsequent request must carry this through. Set on
	// StreamEventThinkingDelta.
	Signature string

	// ProviderName identifies the provider that produced a thinking block, so
	// it is not replayed to a different provider that would reject it. Set on
	// StreamEventThinkingDelta.
	ProviderName string

	// ThinkingID is the provider's reasoning-item identifier (OpenAI
	// Responses "rs_..."). Empty for providers that do not issue one.
	ThinkingID string

	// FinishReason is why generation ended. Set on StreamEventDone, so a
	// caller can tell a complete answer from one truncated at max tokens or
	// halted by a content filter.
	FinishReason FinishReason
}

// StreamResult provides a channel-based streaming interface.
//
// You can either:
//  1. Range over Events to process events as they arrive, OR
//  2. Call Text() or Wait() to block until complete
//
// Do NOT do both — choose one consumption pattern.
type StreamResult struct {
	// Events yields streaming events until closed.
	Events <-chan StreamEvent

	once sync.Once
	text string
	err  error
	done chan struct{}

	snapshotMu sync.RWMutex
	snapshot   *ExecutionSnapshot
}

// NewStreamResult creates a StreamResult backed by the given channel.
// Provider implementations use this to construct streaming results.
func NewStreamResult(ch <-chan StreamEvent) *StreamResult {
	return &StreamResult{
		Events: ch,
		done:   make(chan struct{}),
	}
}

// consume drains the Events channel and accumulates text.
// Called exactly once by Text() or Wait().
func (sr *StreamResult) consume() {
	sr.once.Do(func() {
		defer close(sr.done)
		for event := range sr.Events {
			switch event.Type {
			case StreamEventTextDelta:
				sr.text += event.Delta
			case StreamEventError:
				sr.err = event.Error
			}
		}
	})
}

// Text blocks until streaming is complete and returns the full accumulated text.
// This consumes the Events channel — do not also range over Events.
func (sr *StreamResult) Text() (string, error) {
	sr.consume()
	return sr.text, sr.err
}

// Wait blocks until streaming is complete.
// This consumes the Events channel — do not also range over Events.
func (sr *StreamResult) Wait() error {
	sr.consume()
	return sr.err
}

// SetSnapshot records the final or partial execution state for an
// agent-owned stream. It is exported so the root agent package can attach the
// result of its execution fold without making provider streams depend on that
// package. Provider implementations normally do not call it.
func (sr *StreamResult) SetSnapshot(snapshot ExecutionSnapshot) {
	sr.snapshotMu.Lock()
	defer sr.snapshotMu.Unlock()
	copy := cloneExecutionSnapshot(snapshot)
	sr.snapshot = &copy
}

// Snapshot returns the final or partial execution state when this stream was
// produced by an Agentic agent run. It never consumes Events and is therefore
// safe to call after ranging over the event channel. Provider-level streams
// return false because they do not own an agent execution.
func (sr *StreamResult) Snapshot() (ExecutionSnapshot, bool) {
	sr.snapshotMu.RLock()
	defer sr.snapshotMu.RUnlock()
	if sr.snapshot == nil {
		return ExecutionSnapshot{}, false
	}
	return cloneExecutionSnapshot(*sr.snapshot), true
}

func cloneExecutionSnapshot(snapshot ExecutionSnapshot) ExecutionSnapshot {
	copy := snapshot
	copy.Messages = cloneSnapshotJSON(snapshot.Messages)
	copy.ToolCalls = cloneSnapshotJSON(snapshot.ToolCalls)
	copy.ToolResults = append([]ToolExecutionResult(nil), snapshot.ToolResults...)
	if snapshot.Suspension != nil {
		suspension := *snapshot.Suspension
		suspension.Payload = append([]byte(nil), snapshot.Suspension.Payload...)
		copy.Suspension = &suspension
	}
	return copy
}

func cloneSnapshotJSON[T any](value []T) []T {
	if len(value) == 0 {
		return nil
	}
	encoded, err := json.Marshal(value)
	if err == nil {
		var copy []T
		if json.Unmarshal(encoded, &copy) == nil {
			return copy
		}
	}
	return append([]T(nil), value...)
}
