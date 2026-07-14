package agentic

import (
	"errors"
	"fmt"

	"github.com/regularkevvv/agentic/internal/core"
)

// ErrNilDeps is returned before any external effect when a dependency-aware
// run receives a nil or typed-nil dependency value.
var ErrNilDeps = errors.New("agent dependencies must not be nil")

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

// IsUsageLimitExceeded checks if an error is a UsageLimitExceededError.
func IsUsageLimitExceeded(err error) bool {
	var ule *UsageLimitExceededError
	return errors.As(err, &ule)
}
