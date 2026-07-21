package azure

import (
	"strings"
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

// TestV1BaseURL covers every accepted endpoint form. All must resolve to a v1
// API base URL carrying no query string: this package speaks only the
// OpenAI-compatible v1 API, which rejects the api-version parameter that the
// older deployment-path API required.
func TestV1BaseURL(t *testing.T) {
	tests := []struct {
		name     string
		endpoint string
		want     string
	}{
		{
			name:     "bare resource resolves to the v1 path",
			endpoint: "https://my-resource.openai.azure.com",
			want:     "https://my-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "trailing slash",
			endpoint: "https://my-resource.openai.azure.com/",
			want:     "https://my-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "explicit v1 path is used as given",
			endpoint: "https://my-resource.openai.azure.com/openai/v1",
			want:     "https://my-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "explicit v1 path with trailing slash",
			endpoint: "https://my-resource.openai.azure.com/openai/v1/",
			want:     "https://my-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "services.ai.azure.com v1 path",
			endpoint: "https://my-resource.services.ai.azure.com/openai/v1",
			want:     "https://my-resource.services.ai.azure.com/openai/v1",
		},
		{
			name:     "foundry serverless host serves v1 at the root",
			endpoint: "https://my-model.models.ai.azure.com",
			want:     "https://my-model.models.ai.azure.com/v1",
		},
		{
			name:     "foundry serverless host with trailing slash",
			endpoint: "https://my-model.models.ai.azure.com/",
			want:     "https://my-model.models.ai.azure.com/v1",
		},
		{
			// A host that merely contains the Foundry suffix belongs to
			// someone else. Matching on substring rather than suffix would
			// misidentify it.
			name:     "host containing the foundry suffix as a substring is not foundry",
			endpoint: "https://models.ai.azure.com.evil.test",
			want:     "https://models.ai.azure.com.evil.test/openai/v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := v1BaseURL(tt.endpoint)
			if err != nil {
				t.Fatalf("v1BaseURL(%q) = %v, want nil error", tt.endpoint, err)
			}
			if got != tt.want {
				t.Errorf("v1BaseURL(%q) = %q, want %q", tt.endpoint, got, tt.want)
			}
			if strings.Contains(got, "?") {
				t.Errorf("base URL must not contain a query string, got %q", got)
			}
		})
	}
}

// TestV1BaseURLRejectsDeploymentPath proves a caller on the retired
// deployment-path API is told so, rather than having their endpoint silently
// rewritten to point somewhere they did not name.
func TestV1BaseURLRejectsDeploymentPath(t *testing.T) {
	_, err := v1BaseURL("https://my-resource.openai.azure.com/openai/deployments/gpt-4o")
	if err == nil {
		t.Fatal("v1BaseURL() = nil error, want a rejection for a deployment-path endpoint")
	}
	if !strings.Contains(err.Error(), "deployment-path") {
		t.Errorf("error = %q, want it to explain the deployment-path API is unsupported", err)
	}
}

func TestV1BaseURLRejectsMalformedEndpoint(t *testing.T) {
	for _, endpoint := range []string{"my-resource.openai.azure.com", "/openai/v1", "://nope"} {
		t.Run(endpoint, func(t *testing.T) {
			if _, err := v1BaseURL(endpoint); err == nil {
				t.Errorf("v1BaseURL(%q) = nil error, want a rejection", endpoint)
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
