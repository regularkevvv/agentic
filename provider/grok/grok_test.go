package grok_test

import (
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/grok"
)

func TestNew(t *testing.T) {
	model, err := grok.New("grok-4.3", grok.WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "grok-4.3" {
		t.Errorf("expected name %q, got %q", "grok-4.3", model.Name())
	}
}

func TestNewWithCustomBaseURL(t *testing.T) {
	model, err := grok.New("grok-4.5",
		grok.WithAPIKey("test-key"),
		grok.WithBaseURL("https://custom.x.ai/v1"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "grok-4.5" {
		t.Errorf("expected name %q, got %q", "grok-4.5", model.Name())
	}
}

func TestNewNoAPIKey(t *testing.T) {
	t.Setenv("GROK_API_KEY", "")
	t.Setenv("XAI_API_KEY", "")

	_, err := grok.New("grok-4.3")
	if err == nil {
		t.Error("expected error when no API key is set")
	}
}

func TestNewFromGrokEnvVar(t *testing.T) {
	t.Setenv("GROK_API_KEY", "from-grok-env")
	t.Setenv("XAI_API_KEY", "")

	model, err := grok.New("grok-4.3")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "grok-4.3" {
		t.Errorf("expected name %q, got %q", "grok-4.3", model.Name())
	}
}

func TestNewFromXAIEnvVar(t *testing.T) {
	t.Setenv("GROK_API_KEY", "")
	t.Setenv("XAI_API_KEY", "from-xai-env")

	model, err := grok.New("grok-4.3")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "grok-4.3" {
		t.Errorf("expected name %q, got %q", "grok-4.3", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := grok.MustNew("grok-4.3", grok.WithAPIKey("test-key"))
	if model.Name() != "grok-4.3" {
		t.Errorf("expected name %q, got %q", "grok-4.3", model.Name())
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
	grok.MustNew("grok-4.3")
}

func TestImplementsInterfaces(t *testing.T) {
	model, err := grok.New("grok-4.3", grok.WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var _ core.Model = model
	var _ core.StreamModel = model
}

func TestRequestValidatesRequest(t *testing.T) {
	model := grok.MustNew("grok-4.3", grok.WithAPIKey("test-key"))

	tests := []struct {
		name string
		req  *core.ChatRequest
	}{
		{
			name: "empty model",
			req:  &core.ChatRequest{Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")}},
		},
		{
			name: "no messages",
			req:  &core.ChatRequest{Model: "grok-4.3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := model.Request(t.Context(), tt.req); err == nil {
				t.Error("expected validation error, got nil")
			}
			if _, err := model.RequestStream(t.Context(), tt.req); err == nil {
				t.Error("expected validation error from RequestStream, got nil")
			}
		})
	}
}
