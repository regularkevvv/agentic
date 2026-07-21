//go:build e2e
// +build e2e

// This file verifies the v0.3.0 wire fixes against the live APIs.
//
// Every test here maps to a specific defect that unit and transport tests
// could not have caught: those prove our code sends what we BELIEVE the API
// wants, while these prove the API actually accepts it. Several of the bugs
// below failed on the ordinary path, so a regression is a shipping outage
// rather than an edge case.
//
// Run with: go test -v ./e2e/ -tags=e2e -timeout=300s
package e2e

import (
	"context"
	"strings"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/anthropic"
	"github.com/regularkevvv/agentic/provider/grok"
	"github.com/regularkevvv/agentic/provider/openai"
	"github.com/regularkevvv/agentic/provider/openrouter"
	"github.com/regularkevvv/agentic/provider/together"
)

// simpleRequest is a minimal one-turn request for wire-level assertions that
// do not care about the content of the answer.
func simpleRequest(model string) *core.ChatRequest {
	return &core.ChatRequest{
		Model:    model,
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "Reply with exactly: ok")},
	}
}

// drainStream consumes a stream to completion and returns the terminal Done
// event, which is where usage and the finish reason arrive.
func drainStream(t *testing.T, sr *core.StreamResult) core.StreamEvent {
	t.Helper()
	var done core.StreamEvent
	var sawDone bool
	for event := range sr.Events {
		switch event.Type {
		case core.StreamEventDone:
			done = event
			sawDone = true
		case core.StreamEventError:
			t.Fatalf("stream error: %v", event.Error)
		}
	}
	if !sawDone {
		t.Fatal("stream ended without a Done event")
	}
	return done
}

// ============================================================================
// CRITICAL: streaming usage was never requested
//
// stream_options.include_usage was never sent, so chunk.Usage was always zero
// and every streamed response reported no token usage at all. This degraded
// OpenAI, OpenRouter, Together, Grok, Azure and Ollama identically. The
// pre-existing transport test could not catch it: its fake server emitted a
// usage chunk unconditionally, so the suite passed while the live API returned
// nothing.
// ============================================================================

func TestE2E_V030_StreamingReportsUsage(t *testing.T) {
	tests := []struct {
		name  string
		skip  func(*testing.T)
		model func(*testing.T) core.StreamModel
	}{
		{
			name: "openai",
			skip: skipIfNoOpenAIKey,
			model: func(t *testing.T) core.StreamModel {
				m, err := openai.New("gpt-4o-mini")
				if err != nil {
					t.Fatalf("openai.New: %v", err)
				}
				return m
			},
		},
		{
			name: "together",
			skip: skipIfNoTogetherKey,
			model: func(t *testing.T) core.StreamModel {
				m, err := together.New("meta-llama/Llama-3.3-70B-Instruct-Turbo")
				if err != nil {
					t.Fatalf("together.New: %v", err)
				}
				return m
			},
		},
		{
			name: "grok",
			skip: skipIfNoGrokKey,
			model: func(t *testing.T) core.StreamModel {
				m, err := grok.New("grok-4.5")
				if err != nil {
					t.Fatalf("grok.New: %v", err)
				}
				return m
			},
		},
		{
			name: "openrouter",
			skip: skipIfNoOpenRouterKey,
			model: func(t *testing.T) core.StreamModel {
				m, err := openrouter.New("openai/gpt-4o-mini")
				if err != nil {
					t.Fatalf("openrouter.New: %v", err)
				}
				return m
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.skip(t)
			ctx := ctxWithTimeout(t)

			model := tt.model(t)
			sr, err := model.RequestStream(ctx, simpleRequest(model.Name()))
			if err != nil {
				t.Fatalf("RequestStream: %v", err)
			}

			done := drainStream(t, sr)
			if done.Usage == nil {
				t.Fatal("Done event carried no usage; stream_options.include_usage is not reaching the API")
			}
			if done.Usage.TotalTokens == 0 {
				t.Errorf("usage.TotalTokens = 0, want non-zero — the API reported no usage for the stream")
			}
			t.Logf("%s streamed usage: prompt=%d completion=%d total=%d",
				tt.name, done.Usage.PromptTokens, done.Usage.CompletionTokens, done.Usage.TotalTokens)
		})
	}
}

// TestE2E_V030_StreamingReportsFinishReason proves a streaming caller can now
// tell a complete answer from a truncated one. StreamEvent carried no finish
// reason at all before v0.3.0.
func TestE2E_V030_StreamingReportsFinishReason(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx := ctxWithTimeout(t)

	model, err := openai.New("gpt-4o-mini")
	if err != nil {
		t.Fatalf("openai.New: %v", err)
	}

	// Cap output hard so the model is cut off mid-answer: the finish reason
	// must report length, not a clean stop.
	maxTokens := 8
	req := &core.ChatRequest{
		Model:     model.Name(),
		Messages:  []core.Message{core.NewTextMessage(core.RoleUser, "Write a detailed 500-word essay about the ocean.")},
		MaxTokens: &maxTokens,
	}

	sr, err := model.RequestStream(ctx, req)
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	done := drainStream(t, sr)
	if done.FinishReason != core.FinishReasonLength {
		t.Errorf("FinishReason = %q, want %q — a truncated stream must not look like a clean stop",
			done.FinishReason, core.FinishReasonLength)
	}
	t.Logf("truncated stream finish reason: %q", done.FinishReason)
}

// ============================================================================
// CRITICAL: Anthropic thinking was a deterministic 400
//
// The default max_tokens of 1024 sat below the default thinking budget of
// 10000, which the API rejects outright. Enabling thinking without explicitly
// raising max tokens could never succeed.
// ============================================================================

func TestE2E_V030_AnthropicThinkingWithoutExplicitMaxTokens(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	model, err := anthropic.New("claude-sonnet-4-6")
	if err != nil {
		t.Fatalf("anthropic.New: %v", err)
	}

	// Deliberately set no MaxTokens: this is the exact shape that used to 400.
	req := &core.ChatRequest{
		Model:    model.Name(),
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "What is 17 * 23? Think it through.")},
		Thinking: &core.ThinkingConfig{Enabled: true},
	}

	resp, err := model.Request(ctx, req)
	if err != nil {
		t.Fatalf("thinking request failed — max_tokens must exceed the thinking budget: %v", err)
	}
	if len(resp.Message.Content) == 0 {
		t.Fatal("response carried no content")
	}
	t.Logf("thinking response finish=%q raw=%q text=%.80q",
		resp.FinishReason, resp.RawFinishReason, resp.Message.GetTextContent())
}

// TestE2E_V030_AnthropicThinkingRoundTripsThroughStreaming is the highest-value
// test in this file.
//
// A streamed thinking block used to be reconstructed without its provider
// signature, so replaying it in a second turn was rejected — making streaming
// and multi-turn reasoning mutually exclusive. This streams a reasoning turn,
// feeds the reconstructed assistant message straight back, and requires the
// API to accept it.
func TestE2E_V030_AnthropicThinkingRoundTripsThroughStreaming(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	model, err := anthropic.New("claude-sonnet-4-6")
	if err != nil {
		t.Fatalf("anthropic.New: %v", err)
	}

	maxTokens := 4000
	first := &core.ChatRequest{
		Model:     model.Name(),
		Messages:  []core.Message{core.NewTextMessage(core.RoleUser, "What is 12 * 12? Think briefly.")},
		MaxTokens: &maxTokens,
		Thinking:  &core.ThinkingConfig{Enabled: true, BudgetTokens: 1024},
	}

	sr, err := model.RequestStream(ctx, first)
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	// Rebuild the assistant turn from the stream the way the run loop does.
	var (
		thinkingText strings.Builder
		answerText   strings.Builder
		signature    string
		providerName string
		thinkingID   string
	)
	for event := range sr.Events {
		switch event.Type {
		case core.StreamEventThinkingDelta:
			thinkingText.WriteString(event.Delta)
			if event.Signature != "" {
				signature = event.Signature
			}
			if event.ProviderName != "" {
				providerName = event.ProviderName
			}
			if event.ThinkingID != "" {
				thinkingID = event.ThinkingID
			}
		case core.StreamEventTextDelta:
			answerText.WriteString(event.Delta)
		case core.StreamEventError:
			t.Fatalf("stream error: %v", event.Error)
		}
	}

	if thinkingText.Len() == 0 {
		t.Skip("model returned no thinking content; nothing to round-trip")
	}
	if signature == "" {
		t.Fatal("no thinking signature captured from the stream — replaying this block will be rejected")
	}
	t.Logf("captured signature (%d bytes), provider=%q id=%q", len(signature), providerName, thinkingID)

	assistant := core.Message{
		Role: core.RoleAssistant,
		Content: []core.Part{
			{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{
				Text:         thinkingText.String(),
				ID:           thinkingID,
				Signature:    signature,
				ProviderName: providerName,
			}},
			{Type: core.ContentText, Text: answerText.String()},
		},
	}

	// Turn two: replay the reasoning turn verbatim. This is what used to 400.
	second := &core.ChatRequest{
		Model: model.Name(),
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "What is 12 * 12? Think briefly."),
			assistant,
			core.NewTextMessage(core.RoleUser, "Now multiply that result by 2."),
		},
		MaxTokens: &maxTokens,
		Thinking:  &core.ThinkingConfig{Enabled: true, BudgetTokens: 1024},
	}

	resp, err := model.Request(ctx, second)
	if err != nil {
		t.Fatalf("replaying a streamed thinking block was rejected: %v", err)
	}
	t.Logf("second turn accepted: %.120q", resp.Message.GetTextContent())
}

// TestE2E_V030_AnthropicToolChoiceRequiredWithThinking covers the combination
// that used to 400. Typed agents hit it implicitly through the output tool.
func TestE2E_V030_AnthropicToolChoiceRequiredWithThinking(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	model, err := anthropic.New("claude-sonnet-4-6")
	if err != nil {
		t.Fatalf("anthropic.New: %v", err)
	}

	maxTokens := 4000
	required := core.ToolChoiceRequired
	req := &core.ChatRequest{
		Model:     model.Name(),
		Messages:  []core.Message{core.NewTextMessage(core.RoleUser, "What is the weather in Lima?")},
		MaxTokens: &maxTokens,
		Thinking:  &core.ThinkingConfig{Enabled: true, BudgetTokens: 1024},
		Tools: []core.Tool{{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "get_weather",
				Description: "Get the current weather for a city.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{"city": map[string]any{"type": "string"}},
					"required":   []string{"city"},
				},
			},
		}},
		ToolChoice: &required,
	}

	if _, err := model.Request(ctx, req); err != nil {
		t.Fatalf("forced tool choice combined with thinking was rejected: %v", err)
	}
}

// TestE2E_V030_AnthropicNestedToolSchema covers the tool-schema rebuild that
// dropped $defs while still forwarding $ref, so any struct-derived nested
// schema reached the API as a dangling reference.
func TestE2E_V030_AnthropicNestedToolSchema(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	model, err := anthropic.New("claude-sonnet-4-6")
	if err != nil {
		t.Fatalf("anthropic.New: %v", err)
	}

	maxTokens := 1024
	req := &core.ChatRequest{
		Model:     model.Name(),
		Messages:  []core.Message{core.NewTextMessage(core.RoleUser, "Book a trip to Lima for Ana.")},
		MaxTokens: &maxTokens,
		Tools: []core.Tool{{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "book_trip",
				Description: "Book a trip for a traveler.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"traveler":    map[string]any{"$ref": "#/$defs/Person"},
						"destination": map[string]any{"type": "string"},
					},
					"required": []string{"traveler", "destination"},
					"$defs": map[string]any{
						"Person": map[string]any{
							"type":       "object",
							"properties": map[string]any{"name": map[string]any{"type": "string"}},
							"required":   []string{"name"},
						},
					},
				},
			},
		}},
	}

	if _, err := model.Request(ctx, req); err != nil {
		t.Fatalf("a tool schema using $defs/$ref was rejected — the definitions are being dropped: %v", err)
	}
}

// TestE2E_V030_AnthropicZeroArgToolSchema covers the else-branch that turned a
// zero-argument tool into one with a phantom argument named "type".
func TestE2E_V030_AnthropicZeroArgToolSchema(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	model, err := anthropic.New("claude-sonnet-4-6")
	if err != nil {
		t.Fatalf("anthropic.New: %v", err)
	}

	maxTokens := 1024
	req := &core.ChatRequest{
		Model:     model.Name(),
		Messages:  []core.Message{core.NewTextMessage(core.RoleUser, "What time is it? Use the tool.")},
		MaxTokens: &maxTokens,
		Tools: []core.Tool{{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "current_time",
				Description: "Get the current time. Takes no arguments.",
				Parameters:  map[string]any{"type": "object"},
			},
		}},
	}

	resp, err := model.Request(ctx, req)
	if err != nil {
		t.Fatalf("a zero-argument tool schema was rejected: %v", err)
	}
	for _, use := range resp.Message.GetToolUses() {
		if _, phantom := use.Input["type"]; phantom {
			t.Errorf("model supplied a phantom %q argument: %v — the schema leaked its own type keyword", "type", use.Input)
		}
	}
}

// ============================================================================
// Finish reasons and in-band failures
// ============================================================================

// TestE2E_V030_RawFinishReasonIsPopulated proves the provider's own stop
// reason now reaches the caller losslessly rather than being flattened into a
// four-value enum.
func TestE2E_V030_RawFinishReasonIsPopulated(t *testing.T) {
	tests := []struct {
		name  string
		skip  func(*testing.T)
		model func(*testing.T) core.Model
	}{
		{"anthropic", skipIfNoAnthropicKey, func(t *testing.T) core.Model {
			m, err := anthropic.New("claude-sonnet-4-6")
			if err != nil {
				t.Fatalf("anthropic.New: %v", err)
			}
			return m
		}},
		{"openai", skipIfNoOpenAIKey, func(t *testing.T) core.Model {
			m, err := openai.New("gpt-4o-mini")
			if err != nil {
				t.Fatalf("openai.New: %v", err)
			}
			return m
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.skip(t)
			ctx := ctxWithTimeout(t)

			model := tt.model(t)
			resp, err := model.Request(ctx, simpleRequest(model.Name()))
			if err != nil {
				t.Fatalf("Request: %v", err)
			}

			if resp.RawFinishReason == "" {
				t.Error("RawFinishReason is empty; the provider's original stop reason is being dropped")
			}
			if resp.FinishReason == core.FinishReasonUnknown {
				t.Errorf("FinishReason = unknown for an ordinary completion (raw %q); the mapping is missing a common value",
					resp.RawFinishReason)
			}
			t.Logf("%s: finish=%q raw=%q", tt.name, resp.FinishReason, resp.RawFinishReason)
		})
	}
}

// TestE2E_V030_OpenRouterRespectsMaxTokens covers the field-name fix.
// OpenRouter expects the legacy max_tokens; we were sending
// max_completion_tokens, which it silently ignored, so the cap never applied.
func TestE2E_V030_OpenRouterRespectsMaxTokens(t *testing.T) {
	skipIfNoOpenRouterKey(t)
	ctx := ctxWithTimeout(t)

	model, err := openrouter.New("openai/gpt-4o-mini")
	if err != nil {
		t.Fatalf("openrouter.New: %v", err)
	}

	maxTokens := 8
	req := &core.ChatRequest{
		Model:     model.Name(),
		Messages:  []core.Message{core.NewTextMessage(core.RoleUser, "Write a detailed 500-word essay about the ocean.")},
		MaxTokens: &maxTokens,
	}

	resp, err := model.Request(ctx, req)
	if err != nil {
		t.Fatalf("Request: %v", err)
	}

	// An ignored cap produces a full essay and a clean stop.
	if resp.FinishReason != core.FinishReasonLength {
		t.Errorf("FinishReason = %q (raw %q), want %q — max_tokens appears to have been ignored; got %d chars",
			resp.FinishReason, resp.RawFinishReason, core.FinishReasonLength, len(resp.Message.GetTextContent()))
	}
	t.Logf("capped output (%d chars) finish=%q", len(resp.Message.GetTextContent()), resp.FinishReason)
}

// ============================================================================
// OpenAI structured output
// ============================================================================

// TestE2E_V030_OpenAIStrictStructuredOutput covers our own default path:
// NewNativeOutput defaults strict:true and reflects a schema that carries
// neither additionalProperties nor a complete required array, which the API
// rejects unless the schema is normalized first.
func TestE2E_V030_OpenAIStrictStructuredOutput(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx := ctxWithTimeout(t)

	type Address struct {
		City    string `json:"city"`
		Country string `json:"country"`
	}
	type Person struct {
		Name    string  `json:"name"`
		Age     int     `json:"age"`
		Address Address `json:"address"`
	}

	model, err := openai.New("gpt-4o-mini")
	if err != nil {
		t.Fatalf("openai.New: %v", err)
	}

	// NewNativeOutput defaults strict:true and reflects the schema, which is
	// precisely the combination the API rejected before the schema was
	// normalized.
	agent := agentic.NewTypedAgentWithMode[Person]("", model,
		agentic.NewNativeOutput[Person]("person", "A person with a nested address."),
	)

	result, err := agent.Run(ctx, "Ana is 34 and lives in Lima, Peru.")
	if err != nil {
		t.Fatalf("strict native structured output was rejected: %v", err)
	}
	if result.Output.Name == "" || result.Output.Address.City == "" {
		t.Errorf("output did not populate the nested schema: %+v", result.Output)
	}
	t.Logf("structured output: %+v", result.Output)
}

// ============================================================================
// Stop sequences — previously unreachable on every provider
// ============================================================================

func TestE2E_V030_StopSequences(t *testing.T) {
	tests := []struct {
		name  string
		skip  func(*testing.T)
		model func(*testing.T) core.Model
	}{
		{"anthropic", skipIfNoAnthropicKey, func(t *testing.T) core.Model {
			m, err := anthropic.New("claude-sonnet-4-6")
			if err != nil {
				t.Fatalf("anthropic.New: %v", err)
			}
			return m
		}},
		{"openai", skipIfNoOpenAIKey, func(t *testing.T) core.Model {
			m, err := openai.New("gpt-4o-mini")
			if err != nil {
				t.Fatalf("openai.New: %v", err)
			}
			return m
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.skip(t)
			ctx := ctxWithTimeout(t)

			model := tt.model(t)
			maxTokens := 200
			req := &core.ChatRequest{
				Model:         model.Name(),
				Messages:      []core.Message{core.NewTextMessage(core.RoleUser, "Count from 1 to 10, one number per line.")},
				MaxTokens:     &maxTokens,
				StopSequences: []string{"5"},
			}

			resp, err := model.Request(ctx, req)
			if err != nil {
				t.Fatalf("Request: %v", err)
			}

			text := resp.Message.GetTextContent()
			if strings.Contains(text, "6") {
				t.Errorf("output ran past the stop sequence; StopSequences is not reaching the API. Got: %q", text)
			}
			t.Logf("%s stopped at %q: %q", tt.name, "5", text)
		})
	}
}
