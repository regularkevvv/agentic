package azure

import (
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNew(t *testing.T) {
	model, err := New("gpt-4o",
		WithEndpoint("https://my-resource.openai.azure.com"),
		WithAPIKey("test-key"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gpt-4o" {
		t.Errorf("expected name %q, got %q", "gpt-4o", model.Name())
	}
}

func TestNewWithDeployment(t *testing.T) {
	model, err := New("gpt-4o",
		WithEndpoint("https://my-resource.openai.azure.com"),
		WithAPIKey("test-key"),
		WithDeployment("my-gpt4o-deployment"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gpt-4o" {
		t.Errorf("expected name %q, got %q", "gpt-4o", model.Name())
	}
}

func TestNewWithAPIVersion(t *testing.T) {
	model, err := New("gpt-4o",
		WithEndpoint("https://my-resource.openai.azure.com"),
		WithAPIKey("test-key"),
		WithAPIVersion("2024-06-01"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gpt-4o" {
		t.Errorf("expected name %q, got %q", "gpt-4o", model.Name())
	}
}

func TestNewNoEndpoint(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")

	_, err := New("gpt-4o", WithAPIKey("test-key"))
	if err == nil {
		t.Error("expected error when no endpoint is set")
	}
}

func TestNewNoAPIKey(t *testing.T) {
	t.Setenv("AZURE_OPENAI_API_KEY", "")

	_, err := New("gpt-4o", WithEndpoint("https://test.openai.azure.com"))
	if err == nil {
		t.Error("expected error when no API key is set")
	}
}

func TestNewFromEnvVars(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "https://env-resource.openai.azure.com")
	t.Setenv("AZURE_OPENAI_API_KEY", "from-env")

	model, err := New("gpt-4o")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gpt-4o" {
		t.Errorf("expected name %q, got %q", "gpt-4o", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := MustNew("gpt-4o",
		WithEndpoint("https://test.openai.azure.com"),
		WithAPIKey("test-key"),
	)
	if model.Name() != "gpt-4o" {
		t.Errorf("expected name %q, got %q", "gpt-4o", model.Name())
	}
}

func TestMustNewPanics(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	t.Setenv("AZURE_OPENAI_API_KEY", "")

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when required config is missing")
		}
	}()
	MustNew("gpt-4o")
}

func TestBuildBaseURL(t *testing.T) {
	tests := []struct {
		name       string
		endpoint   string
		deployment string
		apiVersion string
		expected   string
	}{
		{
			"basic",
			"https://my-resource.openai.azure.com",
			"gpt-4o",
			"2025-01-01-preview",
			"https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2025-01-01-preview",
		},
		{
			"trailing slash",
			"https://my-resource.openai.azure.com/",
			"my-deploy",
			"2024-06-01",
			"https://my-resource.openai.azure.com/openai/deployments/my-deploy?api-version=2024-06-01",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildBaseURL(tt.endpoint, tt.deployment, tt.apiVersion)
			if got != tt.expected {
				t.Errorf("got %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestImplementsInterfaces(t *testing.T) {
	model, _ := New("test",
		WithEndpoint("https://test.openai.azure.com"),
		WithAPIKey("test-key"),
	)

	var _ core.Model = model
	var _ core.StreamModel = model
}
