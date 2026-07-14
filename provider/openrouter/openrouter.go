// Package openrouter provides an OpenRouter Model implementation for Agentic.
// OpenRouter is an API gateway that routes to many LLM providers (OpenAI, Anthropic,
// Google, Meta, etc.) through a unified OpenAI-compatible API.
//
// The provider wraps the OpenAI provider with OpenRouter-specific defaults and
// inherits both streaming and non-streaming capabilities.
package openrouter

import (
	"fmt"
	"os"

	"github.com/regularkevvv/agentic/internal/core"
	oaiProvider "github.com/regularkevvv/agentic/provider/openai"

	"github.com/openai/openai-go/option"
)

// DefaultBaseURL is the OpenRouter API endpoint.
const DefaultBaseURL = "https://openrouter.ai/api/v1"

// Model wraps the OpenAI provider for OpenRouter.
// It inherits all methods including Request and RequestStream.
type Model struct {
	*oaiProvider.Model
}

// Option configures the OpenRouter Model.
type Option func(*config)

type config struct {
	apiKey      string
	baseURL     string
	httpReferer string
	appTitle    string
}

// WithAPIKey sets the API key. If not set, the OPENROUTER_API_KEY env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithBaseURL overrides the default OpenRouter base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// WithHTTPReferer sets the HTTP-Referer header for OpenRouter rankings.
// This helps OpenRouter attribute traffic to your site.
func WithHTTPReferer(referer string) Option {
	return func(c *config) { c.httpReferer = referer }
}

// WithAppTitle sets the X-Title header shown in OpenRouter dashboards.
func WithAppTitle(title string) Option {
	return func(c *config) { c.appTitle = title }
}

// New creates a new OpenRouter Model.
//
// Example:
//
//	model, err := openrouter.New("anthropic/claude-sonnet-4", openrouter.WithAPIKey("sk-or-..."))
//	model, err := openrouter.New("openai/gpt-4o")  // uses OPENROUTER_API_KEY env var
func New(model string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENROUTER_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("openrouter: API key not set (use WithAPIKey or set OPENROUTER_API_KEY)")
	}

	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	// Build extra headers for OpenRouter
	var extraOpts []option.RequestOption
	if cfg.httpReferer != "" {
		extraOpts = append(extraOpts, option.WithHeader("HTTP-Referer", cfg.httpReferer))
	}
	if cfg.appTitle != "" {
		extraOpts = append(extraOpts, option.WithHeader("X-Title", cfg.appTitle))
	}

	oaiOpts := []oaiProvider.Option{
		oaiProvider.WithAPIKey(apiKey),
		oaiProvider.WithBaseURL(baseURL),
	}
	if len(extraOpts) > 0 {
		oaiOpts = append(oaiOpts, oaiProvider.WithRequestOptions(extraOpts...))
	}

	oaiModel, err := oaiProvider.New(model, oaiOpts...)
	if err != nil {
		return nil, fmt.Errorf("openrouter: %w", err)
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
