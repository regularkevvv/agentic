// Package grok provides a Grok (xAI) Model implementation for Agentic.
// Grok models are accessed via the xAI API, which exposes an OpenAI-compatible
// endpoint at https://api.x.ai/v1.
//
// The provider wraps the OpenAI provider with Grok-specific defaults and
// inherits both streaming and non-streaming capabilities.
package grok

import (
	"fmt"
	"os"

	"github.com/regularkevvv/agentic/internal/core"
	oaiProvider "github.com/regularkevvv/agentic/provider/openai"
)

// DefaultBaseURL is the xAI API endpoint.
const DefaultBaseURL = "https://api.x.ai/v1"

// Model wraps the OpenAI provider for Grok (xAI).
// It inherits all methods including Request and RequestStream.
type Model struct {
	*oaiProvider.Model
}

// Option configures the Grok Model.
type Option func(*config)

type config struct {
	apiKey  string
	baseURL string
}

// WithAPIKey sets the API key. If not set, the GROK_API_KEY or XAI_API_KEY
// env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithBaseURL overrides the default xAI base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// New creates a new Grok Model.
//
// Examples:
//
//	model, err := grok.New("grok-3-mini")
//	model, err := grok.New("grok-3", grok.WithAPIKey("xai-..."))
func New(model string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("GROK_API_KEY")
	}
	if apiKey == "" {
		apiKey = os.Getenv("XAI_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("grok: API key not set (use WithAPIKey or set GROK_API_KEY / XAI_API_KEY)")
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
		return nil, fmt.Errorf("grok: %w", err)
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
