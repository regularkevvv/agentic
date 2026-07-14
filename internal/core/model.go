package core

import "context"

// Model is the core abstraction for LLM providers.
// Implement this interface to add support for a new provider.
type Model interface {
	// Request sends a chat completion request and returns the response.
	Request(ctx context.Context, req *ChatRequest) (*ChatResponse, error)

	// Name returns the model identifier (e.g., "openai:gpt-4").
	Name() string
}

// StreamModel extends Model with streaming support.
// Providers that support streaming should implement this interface.
type StreamModel interface {
	Model

	// RequestStream sends a streaming request.
	RequestStream(ctx context.Context, req *ChatRequest) (*StreamResult, error)
}
