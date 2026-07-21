// Package ollama provides an Ollama Model implementation for Agentic.
// Ollama runs LLMs locally and exposes an OpenAI-compatible API.
//
// The provider wraps the OpenAI provider with Ollama-specific defaults and
// inherits both streaming and non-streaming capabilities.
//
// NewEmbedder covers the same ground for Ollama's OpenAI-compatible
// /v1/embeddings endpoint. Because that route is standard rather than
// Ollama-specific, WithHost also points this package at any other server
// speaking the OpenAI embeddings API — Hugging Face's Text Embeddings
// Inference (TEI), llama.cpp's server, LocalAI, vLLM — with no other change:
//
//	embedder, err := ollama.NewEmbedder("BAAI/bge-large-en-v1.5",
//	    ollama.WithHost("http://tei-server:8080"))
package ollama

import (
	"fmt"
	"net/url"
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
//
// The scheme may be omitted, matching Ollama's own OLLAMA_HOST format
// ("127.0.0.1:11434"); "http://" is assumed in that case. A host that cannot be
// parsed as an http or https URL is reported as an error by New.
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
//	model, err := ollama.New("llama3.2", ollama.WithHost("127.0.0.1:11434")) // scheme optional
func New(model string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	baseURL, err := buildBaseURL(cfg.host)
	if err != nil {
		return nil, err
	}

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

// buildBaseURL constructs the Ollama base URL from the host, normalizing
// schemeless hosts and validating the result.
//
// Ollama's own documented OLLAMA_HOST format is schemeless — "127.0.0.1:11434".
// Appending "/v1" to such a value produces a string that url.Parse either
// rejects outright ("127.0.0.1:11434/v1" → "first path segment in URL cannot
// contain colon") or, worse, silently misreads: "localhost:11434/v1" parses as
// an opaque URL with scheme "localhost", dropping the port entirely. Hosts with
// no scheme are therefore prefixed with "http://" before parsing.
//
// The parsed result is validated here so New reports a malformed host up front
// instead of failing opaquely on the first request.
func buildBaseURL(host string) (string, error) {
	source := host
	if source == "" {
		source = os.Getenv("OLLAMA_HOST")
	}
	if source == "" {
		return DefaultBaseURL, nil
	}

	normalized := source
	if !strings.Contains(normalized, "://") {
		normalized = "http://" + normalized
	}
	normalized = strings.TrimRight(normalized, "/")

	u, err := url.Parse(normalized)
	if err != nil {
		return "", fmt.Errorf("ollama: invalid host %q: %w", source, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("ollama: invalid host %q: scheme must be http or https, got %q", source, u.Scheme)
	}
	if u.Host == "" {
		return "", fmt.Errorf("ollama: invalid host %q: missing host", source)
	}

	return normalized + "/v1", nil
}

// Compile-time checks that Model implements both Model and StreamModel
// (inherited from the embedded OpenAI Model).
var (
	_ core.Model       = (*Model)(nil)
	_ core.StreamModel = (*Model)(nil)
)
