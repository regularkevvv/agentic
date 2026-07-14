package agentic

import "fmt"

// UsageLimits defines caps on token and request usage for an agent run.
// All fields are pointers — nil means "no limit".
type UsageLimits struct {
	// MaxRequestTokens caps the total prompt (input) tokens across all requests.
	MaxRequestTokens *int
	// MaxResponseTokens caps the total completion (output) tokens across all requests.
	MaxResponseTokens *int
	// MaxTotalTokens caps the combined (prompt + completion) tokens across all requests.
	MaxTotalTokens *int
	// MaxRequests caps the number of LLM API requests in a single run.
	// Default is 50 when UsageLimits is set (via DefaultUsageLimits).
	MaxRequests *int
	// MaxToolCalls caps the number of successful tool call executions in a single run.
	MaxToolCalls *int
}

// DefaultUsageLimits returns UsageLimits with only the request limit set (50),
// matching pydantic-ai's default behavior.
func DefaultUsageLimits() UsageLimits {
	maxReq := 50
	return UsageLimits{MaxRequests: &maxReq}
}

// UsageLimitExceededError is returned when a usage limit is hit during an agent run.
type UsageLimitExceededError struct {
	// LimitName describes which limit was exceeded (e.g., "request_tokens", "requests").
	LimitName string
	// Current is the current value at the time the limit was exceeded.
	Current int
	// Max is the configured limit that was exceeded.
	Max int
}

func (e *UsageLimitExceededError) Error() string {
	return fmt.Sprintf("usage limit exceeded: %s (%d > %d)", e.LimitName, e.Current, e.Max)
}

// checkBeforeRequest checks limits that should be enforced before making the next
// LLM request. Returns a UsageLimitExceededError if any limit would be exceeded.
func (l *UsageLimits) checkBeforeRequest(usage Usage) error {
	if l.MaxRequests != nil && usage.Requests >= *l.MaxRequests {
		return &UsageLimitExceededError{
			LimitName: "requests",
			Current:   usage.Requests,
			Max:       *l.MaxRequests,
		}
	}
	if l.MaxRequestTokens != nil && usage.PromptTokens > *l.MaxRequestTokens {
		return &UsageLimitExceededError{
			LimitName: "request_tokens",
			Current:   usage.PromptTokens,
			Max:       *l.MaxRequestTokens,
		}
	}
	if l.MaxTotalTokens != nil && usage.TotalTokens > *l.MaxTotalTokens {
		return &UsageLimitExceededError{
			LimitName: "total_tokens",
			Current:   usage.TotalTokens,
			Max:       *l.MaxTotalTokens,
		}
	}
	return nil
}

// checkAfterResponse checks token limits after receiving a response.
// Returns a UsageLimitExceededError if any limit was exceeded.
func (l *UsageLimits) checkAfterResponse(usage Usage) error {
	if l.MaxRequestTokens != nil && usage.PromptTokens > *l.MaxRequestTokens {
		return &UsageLimitExceededError{
			LimitName: "request_tokens",
			Current:   usage.PromptTokens,
			Max:       *l.MaxRequestTokens,
		}
	}
	if l.MaxResponseTokens != nil && usage.CompletionTokens > *l.MaxResponseTokens {
		return &UsageLimitExceededError{
			LimitName: "response_tokens",
			Current:   usage.CompletionTokens,
			Max:       *l.MaxResponseTokens,
		}
	}
	if l.MaxTotalTokens != nil && usage.TotalTokens > *l.MaxTotalTokens {
		return &UsageLimitExceededError{
			LimitName: "total_tokens",
			Current:   usage.TotalTokens,
			Max:       *l.MaxTotalTokens,
		}
	}
	return nil
}

// checkBeforeToolCalls checks if the projected tool call count would exceed the limit.
func (l *UsageLimits) checkBeforeToolCalls(currentToolCalls int, pendingCalls int) error {
	if l.MaxToolCalls != nil {
		projected := currentToolCalls + pendingCalls
		if projected > *l.MaxToolCalls {
			return &UsageLimitExceededError{
				LimitName: "tool_calls",
				Current:   projected,
				Max:       *l.MaxToolCalls,
			}
		}
	}
	return nil
}

// IntPtr is a helper to create *int values for UsageLimits fields.
//
// Example:
//
//	limits := agentic.UsageLimits{
//	    MaxRequests:    agentic.IntPtr(10),
//	    MaxTotalTokens: agentic.IntPtr(50000),
//	}
func IntPtr(v int) *int {
	return &v
}
