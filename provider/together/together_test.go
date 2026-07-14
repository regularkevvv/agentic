package together

import (
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNew(t *testing.T) {
	model, err := New("meta-llama/Llama-3.3-70B-Instruct-Turbo", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "meta-llama/Llama-3.3-70B-Instruct-Turbo" {
		t.Errorf("expected name %q, got %q", "meta-llama/Llama-3.3-70B-Instruct-Turbo", model.Name())
	}
}

func TestNewWithCustomBaseURL(t *testing.T) {
	model, err := New("deepseek-ai/DeepSeek-R1",
		WithAPIKey("test-key"),
		WithBaseURL("https://custom.together.xyz/v1"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "deepseek-ai/DeepSeek-R1" {
		t.Errorf("expected name %q, got %q", "deepseek-ai/DeepSeek-R1", model.Name())
	}
}

func TestNewNoAPIKey(t *testing.T) {
	t.Setenv("TOGETHER_API_KEY", "")

	_, err := New("test-model")
	if err == nil {
		t.Error("expected error when no API key is set")
	}
}

func TestNewFromEnvVar(t *testing.T) {
	t.Setenv("TOGETHER_API_KEY", "from-env")

	model, err := New("meta-llama/Llama-3.3-70B-Instruct-Turbo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "meta-llama/Llama-3.3-70B-Instruct-Turbo" {
		t.Errorf("expected name %q, got %q", "meta-llama/Llama-3.3-70B-Instruct-Turbo", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := MustNew("test-model", WithAPIKey("test-key"))
	if model.Name() != "test-model" {
		t.Errorf("expected name %q, got %q", "test-model", model.Name())
	}
}

func TestMustNewPanics(t *testing.T) {
	t.Setenv("TOGETHER_API_KEY", "")

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when no API key is set")
		}
	}()
	MustNew("test-model")
}

func TestImplementsInterfaces(t *testing.T) {
	model, _ := New("test", WithAPIKey("test-key"))

	var _ core.Model = model
	var _ core.StreamModel = model
}
