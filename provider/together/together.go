// Package together provides a Together AI Model implementation for Agentic.
// Together AI offers an OpenAI-compatible API with access to many open-source
// models (Llama, Mistral, Qwen, DeepSeek, etc.).
//
// The provider wraps the OpenAI provider with Together-specific defaults and
// inherits both streaming and non-streaming capabilities. Together's wire
// format is Chat Completions with no documented divergence from OpenAI's, so
// unlike the OpenRouter gateway this package delegates rather than speaking to
// the API itself. In particular Together takes the modern
// max_completion_tokens field: pydantic-ai's TogetherProvider
// (pydantic_ai/providers/together.py) leaves
// openai_chat_supports_max_completion_tokens at its default of true, which
// OpenRouter's provider must override and Together's must not.
//
// Delegating means the normalized finish-reason mapping, StopSequences and the
// rest of the shared request handling come from the OpenAI provider unchanged.
// Two consequences are worth knowing:
//
//   - Provider-specific settings are read from ChatRequest.ProviderOptions
//     under the "openai" key, not "together"; there is no Together-only option
//     struct.
//   - Thinking text from the reasoning models Together serves (DeepSeek-R1,
//     Qwen, GLM) is not surfaced. Those models return it in a non-standard
//     "reasoning" or "reasoning_content" field that the shared OpenAI response
//     conversion does not read.
package together

import (
	"errors"
	"fmt"
	"os"

	"github.com/regularkevvv/agentic/internal/core"
	oaiProvider "github.com/regularkevvv/agentic/provider/openai"
)

// DefaultBaseURL is the Together AI API endpoint, matching the base_url
// property of pydantic-ai's TogetherProvider
// (pydantic_ai/providers/together.py).
const DefaultBaseURL = "https://api.together.xyz/v1"

// Model wraps the OpenAI provider for Together AI.
// It inherits all methods including Request and RequestStream.
type Model struct {
	*oaiProvider.Model
}

// Option configures the Together AI Model.
type Option func(*config)

type config struct {
	apiKey  string
	baseURL string
}

// WithAPIKey sets the API key. If not set, the TOGETHER_API_KEY env var is
// used, falling back to TOGETHER_AI_API_KEY.
//
// TOGETHER_API_KEY is the name Together AI's own SDKs and pydantic-ai's
// TogetherProvider read (pydantic_ai/providers/together.py); TOGETHER_AI_API_KEY
// is accepted as a secondary alias and is only consulted when the primary is
// unset.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithBaseURL overrides the default Together AI base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// New creates a new Together AI Model.
//
// Examples:
//
//	model, err := together.New("meta-llama/Llama-3.3-70B-Instruct-Turbo")
//	model, err := together.New("deepseek-ai/DeepSeek-R1", together.WithAPIKey("..."))
func New(model string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("TOGETHER_API_KEY")
	}
	if apiKey == "" {
		apiKey = os.Getenv("TOGETHER_AI_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New("together: API key not set (use WithAPIKey or set TOGETHER_API_KEY)")
	}

	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	oaiModel, err := oaiProvider.New(model,
		oaiProvider.WithAPIKey(apiKey),
		oaiProvider.WithBaseURL(baseURL),
	)
	if err != nil {
		return nil, fmt.Errorf("together: %w", err)
	}

	return &Model{Model: oaiModel}, nil
}

// MustNew is like New but panics on error.
func MustNew(model string, opts ...Option) *Model {
	m, err := New(model, opts...)
	if err != nil {
		panic(err)
	}
	return m
}

// Compile-time checks that Model implements both Model and StreamModel
// (inherited from the embedded OpenAI Model).
var (
	_ core.Model       = (*Model)(nil)
	_ core.StreamModel = (*Model)(nil)
)
