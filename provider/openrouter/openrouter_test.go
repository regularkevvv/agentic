package openrouter_test

import (
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/openrouter"
)

func TestNew(t *testing.T) {
	model, err := openrouter.New("anthropic/claude-sonnet-4", openrouter.WithAPIKey("sk-or-test"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "anthropic/claude-sonnet-4" {
		t.Errorf("expected name %q, got %q", "anthropic/claude-sonnet-4", model.Name())
	}
}

func TestNewWithOptions(t *testing.T) {
	model, err := openrouter.New("openai/gpt-4o",
		openrouter.WithAPIKey("sk-or-test"),
		openrouter.WithHTTPReferer("https://myapp.com"),
		openrouter.WithAppTitle("My App"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "openai/gpt-4o" {
		t.Errorf("expected name %q, got %q", "openai/gpt-4o", model.Name())
	}
}

func TestNewWithCustomBaseURL(t *testing.T) {
	model, err := openrouter.New("meta-llama/llama-3.1-405b",
		openrouter.WithAPIKey("sk-or-test"),
		openrouter.WithBaseURL("https://custom.openrouter.ai/api/v1"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "meta-llama/llama-3.1-405b" {
		t.Errorf("expected name %q, got %q", "meta-llama/llama-3.1-405b", model.Name())
	}
}

func TestNewNoAPIKey(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "")

	_, err := openrouter.New("test-model")
	if err == nil {
		t.Error("expected error when no API key is set")
	}
}

func TestNewFromEnvVar(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "sk-or-from-env")

	model, err := openrouter.New("anthropic/claude-sonnet-4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "anthropic/claude-sonnet-4" {
		t.Errorf("expected name %q, got %q", "anthropic/claude-sonnet-4", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := openrouter.MustNew("openai/gpt-4o", openrouter.WithAPIKey("sk-or-test"))
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
	openrouter.MustNew("test-model")
}

func TestImplementsInterfaces(t *testing.T) {
	model, err := openrouter.New("test", openrouter.WithAPIKey("sk-or-test"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var _ core.Model = model
	var _ core.StreamModel = model
}

func TestAPIErrorMessage(t *testing.T) {
	tests := []struct {
		name string
		err  openrouter.APIError
		want string
	}{
		{
			name: "with code",
			err:  openrouter.APIError{Code: 502, Message: "upstream down"},
			want: "openrouter: upstream error 502: upstream down",
		},
		{
			name: "without code",
			err:  openrouter.APIError{Message: "upstream down"},
			want: "openrouter: upstream error: upstream down",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error() = %q, want %q", got, tt.want)
			}
		})
	}
}
