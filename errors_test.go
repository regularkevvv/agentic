package agentic

import (
	"errors"
	"fmt"
	"testing"
)

func TestModelRetryError(t *testing.T) {
	mr := &ModelRetry{Message: "try again"}
	if mr.Error() != "try again" {
		t.Errorf("expected %q, got %q", "try again", mr.Error())
	}
}

func TestRetry(t *testing.T) {
	mr := Retry("bad input")
	if mr.Message != "bad input" {
		t.Errorf("expected %q, got %q", "bad input", mr.Message)
	}
}

func TestRetryf(t *testing.T) {
	mr := Retryf("error %d: %s", 42, "not found")
	expected := "error 42: not found"
	if mr.Message != expected {
		t.Errorf("expected %q, got %q", expected, mr.Message)
	}
}

func TestIsModelRetry(t *testing.T) {
	mr := Retry("retry me")
	if !IsModelRetry(mr) {
		t.Error("expected IsModelRetry to return true")
	}

	wrapped := fmt.Errorf("wrapped: %w", mr)
	if !IsModelRetry(wrapped) {
		t.Error("expected IsModelRetry to return true for wrapped error")
	}

	if IsModelRetry(errors.New("not a retry")) {
		t.Error("expected IsModelRetry to return false for non-ModelRetry error")
	}
}

func TestMaxIterationsError(t *testing.T) {
	err := &MaxIterationsError{MaxIterations: 5}
	expected := "agent reached maximum iterations (5)"
	if err.Error() != expected {
		t.Errorf("expected %q, got %q", expected, err.Error())
	}
}

func TestProviderErrorWithoutReason(t *testing.T) {
	err := &ProviderError{}
	if got := err.Error(); got != "provider returned no usable response" {
		t.Fatalf("Error() = %q", got)
	}
}
