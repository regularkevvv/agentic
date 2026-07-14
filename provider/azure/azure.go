// Package azure provides an Azure OpenAI Model implementation for Agentic.
// It wraps the OpenAI provider with Azure-specific authentication and URL patterns.
//
// Azure OpenAI uses a different URL structure and authentication scheme than
// the standard OpenAI API. This provider handles the mapping automatically.
package azure

import (
	"fmt"
	"os"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"
	oaiProvider "github.com/regularkevvv/agentic/provider/openai"

	"github.com/openai/openai-go/option"
)

// DefaultAPIVersion is the default Azure OpenAI API version.
const DefaultAPIVersion = "2025-01-01-preview"

// Model wraps the OpenAI provider for Azure OpenAI.
// It inherits all methods including Request and RequestStream.
type Model struct {
	*oaiProvider.Model
}

// Option configures the Azure OpenAI Model.
type Option func(*config)

type config struct {
	endpoint   string
	deployment string
	apiKey     string
	apiVersion string
}

// WithEndpoint sets the Azure OpenAI endpoint URL.
// If not set, the AZURE_OPENAI_ENDPOINT env var is used.
//
// Example: "https://my-resource.openai.azure.com"
func WithEndpoint(endpoint string) Option {
	return func(c *config) { c.endpoint = endpoint }
}

// WithDeployment sets the Azure deployment name.
// If not set, the model parameter passed to New is used as both model and deployment name.
func WithDeployment(deployment string) Option {
	return func(c *config) { c.deployment = deployment }
}

// WithAPIKey sets the Azure API key. If not set, the AZURE_OPENAI_API_KEY env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithAPIVersion sets the Azure OpenAI API version.
// Defaults to DefaultAPIVersion if not set.
func WithAPIVersion(version string) Option {
	return func(c *config) { c.apiVersion = version }
}

// New creates a new Azure OpenAI Model.
//
// The model parameter specifies the model name (e.g., "gpt-4", "gpt-4o").
// Use WithDeployment to override the deployment name if it differs from the model name.
//
// Examples:
//
//	model, err := azure.New("gpt-4o",
//	    azure.WithEndpoint("https://my-resource.openai.azure.com"),
//	    azure.WithAPIKey("..."),
//	)
//
//	model, err := azure.New("gpt-4o",
//	    azure.WithEndpoint("https://my-resource.openai.azure.com"),
//	    azure.WithDeployment("my-gpt4o-deployment"),
//	)
func New(model string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	endpoint := cfg.endpoint
	if endpoint == "" {
		endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}
	if endpoint == "" {
		return nil, fmt.Errorf("azure: endpoint not set (use WithEndpoint or set AZURE_OPENAI_ENDPOINT)")
	}

	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("AZURE_OPENAI_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("azure: API key not set (use WithAPIKey or set AZURE_OPENAI_API_KEY)")
	}

	apiVersion := cfg.apiVersion
	if apiVersion == "" {
		apiVersion = os.Getenv("OPENAI_API_VERSION")
	}
	if apiVersion == "" {
		apiVersion = DefaultAPIVersion
	}

	deployment := cfg.deployment
	if deployment == "" {
		deployment = model
	}

	// Build the Azure-specific base URL:
	// https://{endpoint}/openai/deployments/{deployment}
	baseURL := buildBaseURL(endpoint, deployment, apiVersion)

	oaiModel, err := oaiProvider.New(model,
		oaiProvider.WithAPIKey(apiKey),
		oaiProvider.WithBaseURL(baseURL),
		oaiProvider.WithRequestOptions(
			option.WithHeader("api-key", apiKey),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("azure: %w", err)
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

// buildBaseURL constructs the Azure OpenAI base URL.
// Azure format: {endpoint}/openai/deployments/{deployment}?api-version={version}
func buildBaseURL(endpoint, deployment, apiVersion string) string {
	endpoint = strings.TrimRight(endpoint, "/")
	return fmt.Sprintf("%s/openai/deployments/%s?api-version=%s", endpoint, deployment, apiVersion)
}

// Compile-time checks that Model implements both Model and StreamModel
// (inherited from the embedded OpenAI Model).
var (
	_ core.Model       = (*Model)(nil)
	_ core.StreamModel = (*Model)(nil)
)
