package agentic

import (
	"errors"
	"fmt"

	"github.com/regularkevvv/agentic/internal/core"
)

// ErrNilDeps is returned before any external effect when a dependency-aware
// run receives a nil or typed-nil dependency value.
var ErrNilDeps = errors.New("agent dependencies must not be nil")

// ErrDriverRequired is returned when a Runner does not expose the explicit
// Driver capability required for session-controlled execution.
var ErrDriverRequired = errors.New("runner must implement agentic.Driver")

// Input and transcript errors are returned before any model request or tool
// handler starts.
var (
	ErrDriveInput           = errors.New("invalid driver input")
	ErrTranscriptInvalid    = errors.New("invalid transcript")
	ErrSuspensionVersion    = errors.New("unsupported suspension version")
	ErrSuspensionMismatch   = errors.New("suspension does not match execution")
	ErrResumeDecision       = errors.New("invalid tool resume decision")
	ErrTurnDecision         = errors.New("invalid turn decision")
	ErrExecutionSuspended   = errors.New("agent execution suspended")
	ErrExecutionStopped     = errors.New("agent execution stopped")
	ErrExecutionInterrupted = errors.New("agent execution interrupted")
	ErrExecutionFailed      = errors.New("agent execution failed")
)

// Representation encoding errors. Each is matched by errors.Is from the typed
// errors the encoders return, so callers can branch on the class of failure
// without depending on a concrete error type.
var (
	ErrUnsupportedRepresentation     = core.ErrUnsupportedRepresentation
	ErrInvalidRepresentationRequest  = core.ErrInvalidRepresentationRequest
	ErrInvalidRepresentationResponse = core.ErrInvalidRepresentationResponse
)

// ModelRetry is a sentinel error that tools can return to request a retry.
// When a tool returns this error, the error message is sent back to the LLM
// as a tool error, and the agent re-enters the loop (up to MaxRetries).
type ModelRetry = core.ModelRetry

// Retry creates a ModelRetry error. Use this in tool handlers to tell the
// agent to send the error back to the model and try again.
//
// Example:
//
//	func(input SearchInput) (SearchOutput, error) {
//	    results := search(input.Query)
//	    if len(results) == 0 {
//	        return SearchOutput{}, agentic.Retry("No results found, try a different query")
//	    }
//	    return SearchOutput{Results: results}, nil
//	}
func Retry(msg string) *ModelRetry {
	return &ModelRetry{Message: msg}
}

// Retryf creates a ModelRetry error with a formatted message.
func Retryf(format string, args ...interface{}) *ModelRetry {
	return &ModelRetry{Message: fmt.Sprintf(format, args...)}
}

// IsModelRetry checks if an error is a ModelRetry.
func IsModelRetry(err error) bool {
	var mr *ModelRetry
	return errors.As(err, &mr)
}

// MaxIterationsError is returned when the agent hits the maximum iteration limit.
type MaxIterationsError struct {
	MaxIterations int
}

func (e *MaxIterationsError) Error() string {
	return fmt.Sprintf("agent reached maximum iterations (%d)", e.MaxIterations)
}

// ProviderError is returned when a provider completed the transport
// successfully but did not produce a usable turn — an in-band failure reported
// alongside a success status, or a response with no content at all.
type ProviderError struct {
	// Reason is the provider's own description of the failure when it gave
	// one, otherwise a description of what this library observed.
	Reason string
}

func (e *ProviderError) Error() string {
	if e.Reason == "" {
		return "provider returned no usable response"
	}
	return fmt.Sprintf("provider returned no usable response: %s", e.Reason)
}

// IsProviderError checks if an error is a ProviderError.
func IsProviderError(err error) bool {
	var pe *ProviderError
	return errors.As(err, &pe)
}

// IsUsageLimitExceeded checks if an error is a UsageLimitExceededError.
func IsUsageLimitExceeded(err error) bool {
	var ule *UsageLimitExceededError
	return errors.As(err, &ule)
}
