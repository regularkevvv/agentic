package core

import (
	"errors"
	"time"
)

// ToolType represents the type of tool.
type ToolType string

const (
	ToolTypeFunction ToolType = "function"
)

// Tool represents a tool that can be called by the LLM.
type Tool struct {
	Type     ToolType `json:"type"`
	Function Function `json:"function"`
}

// Function represents a function definition for tool calling.
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"` // JSON Schema
}

// ToolChoice controls tool calling behavior.
type ToolChoice string

const (
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceRequired ToolChoice = "required"
)

// OutputMode controls how structured output is extracted from the model.
type OutputMode string

const (
	// OutputModeTool uses tool/function calls for structured output (default).
	OutputModeTool OutputMode = "tool"
	// OutputModeNative uses the model's native JSON schema output (response_format).
	OutputModeNative OutputMode = "native"
	// OutputModePrompted injects the JSON schema into the system prompt.
	OutputModePrompted OutputMode = "prompted"
	// OutputModeText returns raw text, optionally processed by a function.
	OutputModeText OutputMode = "text"
)

// ResponseFormat specifies the desired response format for the model.
type ResponseFormat struct {
	// Type is the response format type: "json_object" or "json_schema".
	Type string `json:"type"`
	// JSONSchema is the JSON schema to enforce (for "json_schema" type).
	JSONSchema *JSONSchemaFormat `json:"json_schema,omitempty"`
}

// JSONSchemaFormat wraps a JSON schema with a name for the response format.
type JSONSchemaFormat struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Schema      map[string]interface{} `json:"schema"`
	Strict      *bool                  `json:"strict,omitempty"`
}

// ThinkingConfig controls thinking/reasoning token behavior.
type ThinkingConfig struct {
	// Enabled turns on thinking/reasoning mode.
	Enabled bool
	// BudgetTokens sets the token budget for thinking. 0 means provider default.
	BudgetTokens int
}

// ChatRequest represents a standardized chat completion request.
type ChatRequest struct {
	Model          string
	Messages       []Message
	Temperature    *float64
	MaxTokens      *int
	TopP           *float64
	Tools          []Tool
	ToolChoice     *ToolChoice
	Stream         bool
	ResponseFormat *ResponseFormat
	Thinking       *ThinkingConfig

	// StopSequences halts generation when any of these strings is produced.
	// Supported by every provider in this library. The generated text does
	// not include the matched sequence.
	StopSequences []string

	// ProviderOptions carries provider-specific settings keyed by provider
	// name ("anthropic", "openai", "gemini", "bedrock", "openrouter").
	//
	// Each provider reads only its own key and ignores every other, so one
	// request may carry settings for several providers at once — useful when
	// the same request is retried against a fallback. Values are the typed
	// Options struct exported by that provider's package, so the producing
	// side keeps compile-time checking:
	//
	//	req.ProviderOptions = map[string]any{
	//		"anthropic": anthropic.Options{TopK: agentic.Int(40)},
	//	}
	//
	// A provider that receives a value of an unexpected type ignores it
	// rather than failing the request.
	ProviderOptions map[string]any
}

// Validate ensures the ChatRequest is valid.
func (r *ChatRequest) Validate() error {
	if r.Model == "" {
		return errors.New("model cannot be empty")
	}
	if len(r.Messages) == 0 {
		return errors.New("messages cannot be empty")
	}
	if r.Temperature != nil && (*r.Temperature < 0 || *r.Temperature > 2) {
		return errors.New("temperature must be between 0 and 2")
	}
	if r.MaxTokens != nil && *r.MaxTokens < 0 {
		return errors.New("max tokens must be non-negative")
	}
	return nil
}

// ChatResponse represents a standardized chat completion response.
type ChatResponse struct {
	ID      string
	Model   string
	Message Message
	Usage   Usage
	Created time.Time

	// FinishReason is why generation ended, normalized across providers.
	// Values a provider reports that this library does not recognize map to
	// FinishReasonUnknown rather than being reported as a clean stop.
	FinishReason FinishReason

	// RawFinishReason is the provider's original stop reason, passed through
	// losslessly so callers can act on provider-specific values without this
	// library having to enumerate them. Empty when the provider reported none.
	RawFinishReason string
}

// FinishReason represents why the response ended.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonToolCalls     FinishReason = "tool_calls"
	FinishReasonContentFilter FinishReason = "content_filter"

	// FinishReasonError indicates generation aborted before producing a
	// complete answer — malformed model output, or an upstream failure
	// reported in-band with a success status. Partial content may be
	// present in the message but must not be treated as a whole answer.
	FinishReasonError FinishReason = "error"

	// FinishReasonUnknown indicates the provider reported a stop reason this
	// library does not recognize. Inspect ChatResponse.RawFinishReason for
	// the original value.
	FinishReasonUnknown FinishReason = "unknown"
)

// Usage represents token usage statistics for an entire agent run.
type Usage struct {
	// Core token counts (accumulated across all requests).
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int

	// Provider-specific token details.
	CacheReadTokens     int // Anthropic: cache_read_input_tokens
	CacheCreationTokens int // Anthropic: cache_creation_input_tokens
	ReasoningTokens     int // OpenAI: reasoning_tokens (o1/o3 models)

	// Run-level counters.
	Requests  int // Number of LLM API requests made during the run.
	ToolCalls int // Number of successful tool calls executed.

	// Per-request breakdown (one entry per LLM API call).
	RequestUsages []RequestUsage
}

// RequestUsage holds token usage for a single LLM API request.
type RequestUsage struct {
	PromptTokens        int
	CompletionTokens    int
	TotalTokens         int
	CacheReadTokens     int
	CacheCreationTokens int
	ReasoningTokens     int
}

// Add adds usage from another Usage value (typically a single-request Usage).
func (u *Usage) Add(other Usage) {
	u.PromptTokens += other.PromptTokens
	u.CompletionTokens += other.CompletionTokens
	u.TotalTokens += other.TotalTokens
	u.CacheReadTokens += other.CacheReadTokens
	u.CacheCreationTokens += other.CacheCreationTokens
	u.ReasoningTokens += other.ReasoningTokens

	// Record per-request breakdown.
	u.RequestUsages = append(u.RequestUsages, RequestUsage{
		PromptTokens:        other.PromptTokens,
		CompletionTokens:    other.CompletionTokens,
		TotalTokens:         other.TotalTokens,
		CacheReadTokens:     other.CacheReadTokens,
		CacheCreationTokens: other.CacheCreationTokens,
		ReasoningTokens:     other.ReasoningTokens,
	})

	// Increment request counter.
	u.Requests++
}
