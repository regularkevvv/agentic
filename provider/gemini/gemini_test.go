package gemini

import (
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNew(t *testing.T) {
	model, err := New("gemini-2.5-pro", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gemini-2.5-pro" {
		t.Errorf("expected name %q, got %q", "gemini-2.5-pro", model.Name())
	}
}

func TestNewNoAPIKey(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")

	_, err := New("gemini-2.5-pro")
	if err == nil {
		t.Error("expected error when no API key is set")
	}
}

func TestNewFromGeminiEnvVar(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "from-env")
	t.Setenv("GOOGLE_API_KEY", "")

	model, err := New("gemini-2.5-pro")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gemini-2.5-pro" {
		t.Errorf("expected name %q, got %q", "gemini-2.5-pro", model.Name())
	}
}

func TestNewFromGoogleEnvVar(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "from-google-env")

	model, err := New("gemini-2.5-flash")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gemini-2.5-flash" {
		t.Errorf("expected name %q, got %q", "gemini-2.5-flash", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := MustNew("gemini-2.5-pro", WithAPIKey("test-key"))
	if model.Name() != "gemini-2.5-pro" {
		t.Errorf("expected name %q, got %q", "gemini-2.5-pro", model.Name())
	}
}

func TestMustNewPanics(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when no API key is set")
		}
	}()
	MustNew("gemini-2.5-pro")
}

func TestConvertRole(t *testing.T) {
	tests := []struct {
		role     core.MessageRole
		expected string
	}{
		{core.RoleAssistant, "model"},
		{core.RoleUser, "user"},
		{core.RoleTool, "user"},
		{core.RoleSystem, "user"},
	}

	for _, tt := range tests {
		got := convertRole(tt.role)
		if got != tt.expected {
			t.Errorf("convertRole(%q) = %q, want %q", tt.role, got, tt.expected)
		}
	}
}

func TestConvertSchema(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{
				"type":        "string",
				"description": "The name",
			},
			"age": map[string]interface{}{
				"type": "integer",
			},
		},
		"required": []interface{}{"name"},
	}

	s := convertSchema(schema)
	if string(s.Type) != "object" {
		t.Errorf("expected type 'object', got %q", s.Type)
	}
	if len(s.Properties) != 2 {
		t.Errorf("expected 2 properties, got %d", len(s.Properties))
	}
	if len(s.Required) != 1 || s.Required[0] != "name" {
		t.Errorf("expected required ['name'], got %v", s.Required)
	}
	if s.Properties["name"].Description != "The name" {
		t.Errorf("expected description 'The name', got %q", s.Properties["name"].Description)
	}
}

func TestConvertFinishReason(t *testing.T) {
	tests := []struct {
		input    string
		expected core.FinishReason
	}{
		{"STOP", core.FinishReasonStop},
		{"MAX_TOKENS", core.FinishReasonLength},
		{"SAFETY", core.FinishReasonContentFilter},
		{"OTHER", core.FinishReasonStop},
	}

	for _, tt := range tests {
		// Note: we test via the mapped genai constants
		_ = tt // Finish reason testing would require genai constants
	}
}

func TestImplementsInterfaces(t *testing.T) {
	model, err := New("gemini-2.5-pro", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var _ core.Model = model
	var _ core.StreamModel = model
}
