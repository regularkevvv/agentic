package together_test

import (
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/together"
)

// clearAPIKeyEnv unsets every environment variable New consults for a key, so
// a test asserting the no-key path cannot be rescued by the developer's own
// shell. Clearing only TOGETHER_API_KEY leaves TOGETHER_AI_API_KEY live and
// New then succeeds where the test expects failure.
func clearAPIKeyEnv(t *testing.T) {
	t.Helper()
	t.Setenv("TOGETHER_API_KEY", "")
	t.Setenv("TOGETHER_AI_API_KEY", "")
}

func TestNew(t *testing.T) {
	tests := []struct {
		name  string
		model string
		opts  []together.Option
	}{
		{
			name:  "namespaced model name",
			model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
			opts:  []together.Option{together.WithAPIKey("test-key")},
		},
		{
			name:  "custom base URL",
			model: "deepseek-ai/DeepSeek-R1",
			opts: []together.Option{
				together.WithAPIKey("test-key"),
				together.WithBaseURL("https://custom.together.xyz/v1"),
			},
		},
		{
			name:  "unnamespaced model name",
			model: "test-model",
			opts:  []together.Option{together.WithAPIKey("test-key")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clearAPIKeyEnv(t)

			model, err := together.New(tt.model, tt.opts...)
			if err != nil {
				t.Fatalf("New(%q) unexpected error: %v", tt.model, err)
			}
			if model.Name() != tt.model {
				t.Errorf("Name() = %q, want %q", model.Name(), tt.model)
			}
		})
	}
}

// TestNewAPIKeySources pins which sources satisfy the API-key requirement.
// Which one wins when several are set is a wire-level fact, asserted by
// TestAuthorizationHeader in the transport test.
//
// The "no key anywhere" case is why every test here clears both variables: it
// previously cleared only TOGETHER_API_KEY, so New still found a key in a
// developer's TOGETHER_AI_API_KEY and the case passed only by accident of the
// environment.
func TestNewAPIKeySources(t *testing.T) {
	tests := []struct {
		name      string
		primary   string // TOGETHER_API_KEY
		secondary string // TOGETHER_AI_API_KEY
		opt       string // WithAPIKey, empty to omit
		wantErr   bool
	}{
		{name: "explicit option", opt: "from-option"},
		{name: "explicit option with both env vars set", primary: "a", secondary: "b", opt: "from-option"},
		{name: "primary env var", primary: "from-primary"},
		{name: "both env vars", primary: "from-primary", secondary: "from-secondary"},
		{name: "secondary env var only", secondary: "from-secondary"},
		{name: "no key anywhere is an error", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("TOGETHER_API_KEY", tt.primary)
			t.Setenv("TOGETHER_AI_API_KEY", tt.secondary)

			opts := []together.Option{}
			if tt.opt != "" {
				opts = append(opts, together.WithAPIKey(tt.opt))
			}

			model, err := together.New("test-model", opts...)
			if tt.wantErr {
				if err == nil {
					t.Fatal("New() = nil error, want an error when no API key is set")
				}
				if !strings.Contains(err.Error(), "TOGETHER_API_KEY") {
					t.Errorf("error %q does not name the TOGETHER_API_KEY env var", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("New() unexpected error: %v", err)
			}
			if model == nil {
				t.Fatal("New() returned a nil model")
			}
		})
	}
}

func TestMustNew(t *testing.T) {
	clearAPIKeyEnv(t)

	model := together.MustNew("test-model", together.WithAPIKey("test-key"))
	if model.Name() != "test-model" {
		t.Errorf("Name() = %q, want %q", model.Name(), "test-model")
	}
}

func TestMustNewPanics(t *testing.T) {
	clearAPIKeyEnv(t)

	defer func() {
		if r := recover(); r == nil {
			t.Error("MustNew() did not panic when no API key is set")
		}
	}()
	together.MustNew("test-model")
}

func TestImplementsInterfaces(t *testing.T) {
	clearAPIKeyEnv(t)

	model, err := together.New("test", together.WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}

	var _ core.Model = model
	var _ core.StreamModel = model
}
