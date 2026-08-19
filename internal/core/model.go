package core

import (
	"context"
	"net/url"
	"strconv"
)

// Model is the core abstraction for LLM providers.
// Implement this interface to add support for a new provider.
type Model interface {
	// Request sends a chat completion request and returns the response.
	Request(ctx context.Context, req *ChatRequest) (*ChatResponse, error)

	// Name returns the model identifier (e.g., "openai:gpt-4").
	Name() string
}

// ModelMetadata describes the stable semantic identity and transport of a
// model implementation. Provider packages implement ModelMetadataProvider so
// optional integrations can describe requests without importing those
// packages or parsing Model.Name.
type ModelMetadata struct {
	Provider      string
	Operation     string
	ServerAddress string
	ServerPort    int
	InProcess     bool
}

// ModelMetadataProvider is an optional capability implemented by models that
// can report their provider and transport identity.
type ModelMetadataProvider interface {
	ModelMetadata() ModelMetadata
}

// ModelMetadataForEndpoint constructs transport metadata from an HTTP(S)
// endpoint. Invalid or empty endpoints retain provider and operation identity
// while leaving server fields unknown.
func ModelMetadataForEndpoint(provider, operation, endpoint string) ModelMetadata {
	metadata := ModelMetadata{Provider: provider, Operation: operation}
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Hostname() == "" {
		return metadata
	}
	metadata.ServerAddress = parsed.Hostname()
	if port := parsed.Port(); port != "" {
		if value, err := strconv.Atoi(port); err == nil {
			metadata.ServerPort = value
		}
	} else {
		switch parsed.Scheme {
		case "https":
			metadata.ServerPort = 443
		case "http":
			metadata.ServerPort = 80
		}
	}
	return metadata
}

// StreamModel extends Model with streaming support.
// Providers that support streaming should implement this interface.
type StreamModel interface {
	Model

	// RequestStream sends a streaming request.
	RequestStream(ctx context.Context, req *ChatRequest) (*StreamResult, error)
}
