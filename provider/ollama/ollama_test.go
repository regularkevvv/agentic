package ollama

import (
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNew(t *testing.T) {
	model, err := New("llama3.2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "llama3.2" {
		t.Errorf("expected name %q, got %q", "llama3.2", model.Name())
	}
}

func TestNewWithHost(t *testing.T) {
	model, err := New("qwen2.5:72b", WithHost("http://gpu-server:11434"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "qwen2.5:72b" {
		t.Errorf("expected name %q, got %q", "qwen2.5:72b", model.Name())
	}
}

func TestNewWithAPIKey(t *testing.T) {
	model, err := New("llama3.2", WithAPIKey("secret"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "llama3.2" {
		t.Errorf("expected name %q, got %q", "llama3.2", model.Name())
	}
}

func TestNewFromEnvHost(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "http://remote:11434")

	model, err := New("llama3.2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "llama3.2" {
		t.Errorf("expected name %q, got %q", "llama3.2", model.Name())
	}
}

func TestNewFromEnvAPIKey(t *testing.T) {
	t.Setenv("OLLAMA_API_KEY", "env-secret")

	model, err := New("llama3.2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "llama3.2" {
		t.Errorf("expected name %q, got %q", "llama3.2", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := MustNew("llama3.2")
	if model.Name() != "llama3.2" {
		t.Errorf("expected name %q, got %q", "llama3.2", model.Name())
	}
}

func TestBuildBaseURL(t *testing.T) {
	tests := []struct {
		name     string
		host     string
		envHost  string
		expected string
	}{
		{"default", "", "", DefaultBaseURL},
		{"custom host", "http://gpu:11434", "", "http://gpu:11434/v1"},
		{"custom host trailing slash", "http://gpu:11434/", "", "http://gpu:11434/v1"},
		{"env host", "", "http://env:11434", "http://env:11434/v1"},
		{"explicit overrides env", "http://explicit:11434", "http://env:11434", "http://explicit:11434/v1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envHost != "" {
				t.Setenv("OLLAMA_HOST", tt.envHost)
			} else {
				t.Setenv("OLLAMA_HOST", "")
			}
			got := buildBaseURL(tt.host)
			if got != tt.expected {
				t.Errorf("buildBaseURL(%q) = %q, want %q", tt.host, got, tt.expected)
			}
		})
	}
}

func TestImplementsInterfaces(t *testing.T) {
	model, _ := New("test")

	var _ core.Model = model
	var _ core.StreamModel = model
}
