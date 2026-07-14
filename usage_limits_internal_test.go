package agentic

import (
	"errors"
	"testing"
)

func TestUsageLimitsCheckBeforeRequestBranches(t *testing.T) {
	t.Run("requests", func(t *testing.T) {
		limits := UsageLimits{MaxRequests: IntPtr(1)}
		err := limits.checkBeforeRequest(Usage{Requests: 1})
		var exceeded *UsageLimitExceededError
		if !errors.As(err, &exceeded) || exceeded.LimitName != "requests" {
			t.Fatalf("expected request limit error, got %v", err)
		}
	})

	t.Run("request tokens", func(t *testing.T) {
		limits := UsageLimits{MaxRequestTokens: IntPtr(5)}
		err := limits.checkBeforeRequest(Usage{PromptTokens: 6})
		var exceeded *UsageLimitExceededError
		if !errors.As(err, &exceeded) || exceeded.LimitName != "request_tokens" {
			t.Fatalf("expected request token limit error, got %v", err)
		}
	})

	t.Run("total tokens", func(t *testing.T) {
		limits := UsageLimits{MaxTotalTokens: IntPtr(5)}
		err := limits.checkBeforeRequest(Usage{TotalTokens: 6})
		var exceeded *UsageLimitExceededError
		if !errors.As(err, &exceeded) || exceeded.LimitName != "total_tokens" {
			t.Fatalf("expected total token limit error, got %v", err)
		}
	})

	t.Run("success", func(t *testing.T) {
		limits := UsageLimits{MaxRequests: IntPtr(2), MaxRequestTokens: IntPtr(6), MaxTotalTokens: IntPtr(10)}
		if err := limits.checkBeforeRequest(Usage{Requests: 1, PromptTokens: 6, TotalTokens: 10}); err != nil {
			t.Fatalf("expected limits to pass, got %v", err)
		}
	})
}
