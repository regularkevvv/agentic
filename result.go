package agentic

// Result contains the result of an agent run.
type Result[O any] struct {
	// Output is the final typed output from the agent.
	Output O

	// Messages is the complete conversation history including tool interactions.
	Messages []Message

	// ToolCalls contains all tool calls made during the run.
	ToolCalls []ToolUse

	// ToolResults contains all tool results from executed tools.
	ToolResults []ToolExecutionResult

	// FinishReason indicates why the agent stopped.
	FinishReason FinishReason

	// Usage contains cumulative token usage for the entire run.
	Usage Usage

	// Retries is the number of retries that occurred.
	Retries int

	historyLen int
}

// RunResult is the text-output result retained as a familiar name.
type RunResult = Result[string]

// TypedRunResult is the typed-output result retained as a familiar name.
type TypedRunResult[O any] = Result[O]

// AllMessages returns the complete conversation history including any history
// supplied to the run.
func (r *Result[O]) AllMessages() []Message {
	return r.Messages
}

// NewMessages returns only messages generated during this run.
func (r *Result[O]) NewMessages() []Message {
	if r.historyLen >= len(r.Messages) {
		return nil
	}
	return r.Messages[r.historyLen:]
}

// History returns a serializable history containing all run messages.
func (r *Result[O]) History() *MessageHistory {
	return NewHistory(r.Messages...)
}
