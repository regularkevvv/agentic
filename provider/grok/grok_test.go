package grok

import (
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNew(t *testing.T) {
	model, err := New("grok-3-mini", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "grok-3-mini" {
		t.Errorf("expected name %q, got %q", "grok-3-mini", model.Name())
	}
}

func TestNewWithCustomBaseURL(t *testing.T) {
	model, err := New("grok-3",
		WithAPIKey("test-key"),
		WithBaseURL("https://custom.x.ai/v1"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "grok-3" {
		t.Errorf("expected name %q, got %q", "grok-3", model.Name())
	}
}

func TestNewNoAPIKey(t *testing.T) {
	t.Setenv("GROK_API_KEY", "")
	t.Setenv("XAI_API_KEY", "")

	_, err := New("grok-3-mini")
	if err == nil {
		t.Error("expected error when no API key is set")
	}
}

func TestNewFromGrokEnvVar(t *testing.T) {
	t.Setenv("GROK_API_KEY", "from-grok-env")
	t.Setenv("XAI_API_KEY", "")

	model, err := New("grok-3-mini")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "grok-3-mini" {
		t.Errorf("expected name %q, got %q", "grok-3-mini", model.Name())
	}
}

func TestNewFromXAIEnvVar(t *testing.T) {
	t.Setenv("GROK_API_KEY", "")
	t.Setenv("XAI_API_KEY", "from-xai-env")

	model, err := New("grok-3-mini")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "grok-3-mini" {
		t.Errorf("expected name %q, got %q", "grok-3-mini", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := MustNew("grok-3-mini", WithAPIKey("test-key"))
	if model.Name() != "grok-3-mini" {
		t.Errorf("expected name %q, got %q", "grok-3-mini", model.Name())
	}
}

func TestMustNewPanics(t *testing.T) {
	t.Setenv("GROK_API_KEY", "")
	t.Setenv("XAI_API_KEY", "")

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when no API key is set")
		}
	}()
	MustNew("grok-3-mini")
}

func TestImplementsInterfaces(t *testing.T) {
	model, _ := New("test", WithAPIKey("test-key"))

	var _ core.Model = model
	var _ core.StreamModel = model
}
