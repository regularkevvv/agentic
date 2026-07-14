package core

// ModelRetry tells the agent loop that a tool failure is actionable by the
// model and may be retried within that tool's retry budget.
//
// It lives in core so built-in handlers can return the same sentinel without
// creating an import cycle with the root agentic package.
type ModelRetry struct {
	Message string
}

func (e *ModelRetry) Error() string {
	return e.Message
}
