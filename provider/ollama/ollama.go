// Package ollama provides an Ollama Model implementation for Agentic.
// Ollama runs LLMs locally and exposes an OpenAI-compatible API.
//
// The provider wraps the OpenAI provider with Ollama-specific defaults and
// inherits both streaming and non-streaming capabilities.
package ollama

import (
	"fmt"
	"os"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"
	oaiProvider "github.com/regularkevvv/agentic/provider/openai"
)

// DefaultBaseURL is the default Ollama API endpoint.
const DefaultBaseURL = "http://localhost:11434/v1"

// Model wraps the OpenAI provider for Ollama.
// It inherits all methods including Request and RequestStream.
type Model struct {
	*oaiProvider.Model
}

// Option configures the Ollama Model.
type Option func(*config)

type config struct {
	host   string
	apiKey string
}

// WithHost sets the Ollama host (e.g. "http://192.168.1.100:11434").
// If not set, the OLLAMA_HOST env var is used, falling back to "http://localhost:11434".
func WithHost(host string) Option {
	return func(c *config) { c.host = host }
}

// WithAPIKey sets an optional API key. Ollama doesn't require one by default,
// but some deployments may use authentication.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// New creates a new Ollama Model.
//
// Examples:
//
//	model, err := ollama.New("llama3.2")
//	model, err := ollama.New("qwen2.5:72b", ollama.WithHost("http://gpu-server:11434"))
func New(model string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	baseURL := buildBaseURL(cfg.host)

	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("OLLAMA_API_KEY")
	}

	oaiOpts := []oaiProvider.Option{
		oaiProvider.WithBaseURL(baseURL),
	}
	if apiKey != "" {
		oaiOpts = append(oaiOpts, oaiProvider.WithAPIKey(apiKey))
	} else {
		// OpenAI SDK requires a key but Ollama doesn't — use a placeholder.
		oaiOpts = append(oaiOpts, oaiProvider.WithAPIKey("ollama"))
	}

	oaiModel, err := oaiProvider.New(model, oaiOpts...)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
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

// buildBaseURL constructs the Ollama base URL from the host.
func buildBaseURL(host string) string {
	if host != "" {
		return strings.TrimRight(host, "/") + "/v1"
	}
	if envHost := os.Getenv("OLLAMA_HOST"); envHost != "" {
		return strings.TrimRight(envHost, "/") + "/v1"
	}
	return DefaultBaseURL
}

// Compile-time checks that Model implements both Model and StreamModel
// (inherited from the embedded OpenAI Model).
var (
	_ core.Model       = (*Model)(nil)
	_ core.StreamModel = (*Model)(nil)
)
