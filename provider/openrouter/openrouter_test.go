package openrouter

import (
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNew(t *testing.T) {
	model, err := New("anthropic/claude-sonnet-4", WithAPIKey("sk-or-test"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "anthropic/claude-sonnet-4" {
		t.Errorf("expected name %q, got %q", "anthropic/claude-sonnet-4", model.Name())
	}
}

func TestNewWithOptions(t *testing.T) {
	model, err := New("openai/gpt-4o",
		WithAPIKey("sk-or-test"),
		WithHTTPReferer("https://myapp.com"),
		WithAppTitle("My App"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "openai/gpt-4o" {
		t.Errorf("expected name %q, got %q", "openai/gpt-4o", model.Name())
	}
}

func TestNewWithCustomBaseURL(t *testing.T) {
	model, err := New("meta-llama/llama-3.1-405b",
		WithAPIKey("sk-or-test"),
		WithBaseURL("https://custom.openrouter.ai/api/v1"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "meta-llama/llama-3.1-405b" {
		t.Errorf("expected name %q, got %q", "meta-llama/llama-3.1-405b", model.Name())
	}
}

func TestNewNoAPIKey(t *testing.T) {
	// Unset env var for this test
	t.Setenv("OPENROUTER_API_KEY", "")

	_, err := New("test-model")
	if err == nil {
		t.Error("expected error when no API key is set")
	}
}

func TestNewFromEnvVar(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "sk-or-from-env")

	model, err := New("anthropic/claude-sonnet-4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "anthropic/claude-sonnet-4" {
		t.Errorf("expected name %q, got %q", "anthropic/claude-sonnet-4", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := MustNew("openai/gpt-4o", WithAPIKey("sk-or-test"))
	if model.Name() != "openai/gpt-4o" {
		t.Errorf("expected name %q, got %q", "openai/gpt-4o", model.Name())
	}
}

func TestMustNewPanics(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "")

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when no API key is set")
		}
	}()
	MustNew("test-model")
}

func TestImplementsInterfaces(t *testing.T) {
	model, _ := New("test", WithAPIKey("sk-or-test"))

	// Verify Model interface
	var _ core.Model = model

	// Verify StreamModel interface
	var _ core.StreamModel = model
}
