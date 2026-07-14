// Package together provides a Together AI Model implementation for Agentic.
// Together AI offers an OpenAI-compatible API with access to many open-source
// models (Llama, Mistral, Qwen, DeepSeek, etc.).
//
// The provider wraps the OpenAI provider with Together-specific defaults and
// inherits both streaming and non-streaming capabilities.
package together

import (
	"fmt"
	"os"

	"github.com/regularkevvv/agentic/internal/core"
	oaiProvider "github.com/regularkevvv/agentic/provider/openai"
)

// DefaultBaseURL is the Together AI API endpoint.
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

// WithAPIKey sets the API key. If not set, the TOGETHER_API_KEY env var is used.
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
		return nil, fmt.Errorf("together: API key not set (use WithAPIKey or set TOGETHER_API_KEY)")
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
