package core

import (
	"errors"
	"testing"
)

func TestStreamResultTextAndWait(t *testing.T) {
	ch := make(chan StreamEvent, 3)
	ch <- StreamEvent{Type: StreamEventTextDelta, Delta: "hello "}
	ch <- StreamEvent{Type: StreamEventTextDelta, Delta: "world"}
	close(ch)

	result := NewStreamResult(ch)
	text, err := result.Text()
	if err != nil {
		t.Fatalf("Text: %v", err)
	}
	if text != "hello world" {
		t.Fatalf("expected accumulated text, got %q", text)
	}
	if err := result.Wait(); err != nil {
		t.Fatalf("Wait after Text should be nil, got %v", err)
	}
}

func TestStreamResultWaitReturnsError(t *testing.T) {
	expected := errors.New("stream failed")
	ch := make(chan StreamEvent, 2)
	ch <- StreamEvent{Type: StreamEventTextDelta, Delta: "partial"}
	ch <- StreamEvent{Type: StreamEventError, Error: expected}
	close(ch)

	result := NewStreamResult(ch)
	if err := result.Wait(); !errors.Is(err, expected) {
		t.Fatalf("expected %v, got %v", expected, err)
	}
	text, err := result.Text()
	if !errors.Is(err, expected) {
		t.Fatalf("expected repeated Text call to return %v, got %v", expected, err)
	}
	if text != "partial" {
		t.Fatalf("expected accumulated partial text, got %q", text)
	}
}
