// Package azure provides an Azure OpenAI Model implementation for Agentic.
// It wraps the OpenAI provider with Azure-specific authentication and URLs.
//
// This package speaks the Azure OpenAI v1 API, which is OpenAI-compatible: the
// model is addressed by name and no api-version parameter is involved. The
// older deployment-path API (/openai/deployments/{deployment}?api-version=...)
// is not supported.
//
// That distinction is about the URL, not the resource. The v1 API is served by
// the same Azure resource, so a caller previously using
// "https://my-resource.openai.azure.com" needs no change — this package
// resolves it to the v1 path automatically. See [New].
package azure

import (
	"fmt"
	"net/url"
	"os"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"
	oaiProvider "github.com/regularkevvv/agentic/provider/openai"

	"github.com/openai/openai-go/option"
)

// Model wraps the OpenAI provider for Azure OpenAI.
// It inherits all methods including Request and RequestStream.
type Model struct {
	*oaiProvider.Model
}

// Option configures the Azure OpenAI Model.
type Option func(*config)

type config struct {
	endpoint  string
	apiKey    string
	extraOpts []option.RequestOption
}

// WithEndpoint sets the Azure OpenAI endpoint URL.
// If not set, the AZURE_OPENAI_ENDPOINT env var is used.
//
// Any of these forms is accepted; all resolve to the same v1 API:
//
//	https://my-resource.openai.azure.com
//	https://my-resource.openai.azure.com/openai/v1
//	https://my-resource.services.ai.azure.com/openai/v1
//	https://my-model.models.ai.azure.com          (AI Foundry serverless)
func WithEndpoint(endpoint string) Option {
	return func(c *config) { c.endpoint = endpoint }
}

// WithAPIKey sets the Azure API key. If not set, the AZURE_OPENAI_API_KEY env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithRequestOptions adds raw SDK request options that are applied to every
// request after the Azure defaults, so they can also override them.
//
// This is the extension point for authentication schemes other than the
// api-key header — for example an AAD/Entra bearer token obtained from
// azidentity:
//
//	model, err := azure.New("gpt-4o",
//	    azure.WithEndpoint("https://my-resource.openai.azure.com"),
//	    azure.WithAPIKey("..."),
//	    azure.WithRequestOptions(
//	        option.WithHeader("Authorization", "Bearer "+token.Token),
//	    ),
//	)
func WithRequestOptions(opts ...option.RequestOption) Option {
	return func(c *config) { c.extraOpts = append(c.extraOpts, opts...) }
}

// New creates a new Azure OpenAI Model.
//
// The model parameter is the name the Azure resource serves the model under —
// for Azure OpenAI that is the deployment name, which is frequently but not
// always the same as the underlying model name.
//
// Example:
//
//	model, err := azure.New("my-gpt4o-deployment",
//	    azure.WithEndpoint("https://my-resource.openai.azure.com"),
//	    azure.WithAPIKey("..."),
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

	baseURL, err := v1BaseURL(endpoint)
	if err != nil {
		return nil, err
	}

	reqOpts := []option.RequestOption{
		// Azure authenticates with the api-key header. Delete any Authorization
		// header the SDK may have defaulted in from OPENAI_API_KEY so we never
		// send two competing credentials.
		option.WithHeaderDel("Authorization"),
		option.WithHeader("api-key", apiKey),
	}
	// Caller options last so AAD/Entra credentials can override the defaults.
	reqOpts = append(reqOpts, cfg.extraOpts...)

	oaiModel, err := oaiProvider.New(model,
		oaiProvider.WithBaseURL(baseURL),
		oaiProvider.WithRequestOptions(reqOpts...),
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

// v1BaseURL resolves any accepted endpoint form to the v1 API base URL.
//
// Three shapes are recognized:
//
//   - A path already ending in /v1 is used as given.
//   - An AI Foundry serverless host (*.models.ai.azure.com) serves /v1 at the
//     root, so /v1 is appended.
//   - Anything else is treated as a bare Azure OpenAI resource, whose v1 API
//     lives at /openai/v1.
//
// A deployment-path endpoint is rejected rather than rewritten: silently
// redirecting it would send requests somewhere the caller did not name.
func v1BaseURL(endpoint string) (string, error) {
	stripped := strings.TrimRight(endpoint, "/")

	parsed, err := url.Parse(stripped)
	if err != nil {
		return "", fmt.Errorf("azure: invalid endpoint %q: %w", endpoint, err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("azure: endpoint %q must be an absolute URL, e.g. https://my-resource.openai.azure.com", endpoint)
	}

	if strings.Contains(parsed.Path, "/deployments/") {
		return "", fmt.Errorf("azure: endpoint %q targets the deployment-path API, which this package no longer supports; "+
			"use the resource root instead, e.g. https://my-resource.openai.azure.com", endpoint)
	}

	switch {
	case strings.HasSuffix(stripped, "/v1"):
		return stripped, nil
	case strings.HasSuffix(parsed.Hostname(), ".models.ai.azure.com"):
		return stripped + "/v1", nil
	default:
		return stripped + "/openai/v1", nil
	}
}

// Compile-time checks that Model implements both Model and StreamModel
// (inherited from the embedded OpenAI Model).
var (
	_ core.Model       = (*Model)(nil)
	_ core.StreamModel = (*Model)(nil)
)
