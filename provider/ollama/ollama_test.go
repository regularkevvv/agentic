package ollama

import (
	"net/url"
	"strings"
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

func TestMustNewPanicsOnInvalidHost(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic, got none")
		}
		err, ok := r.(error)
		if !ok {
			t.Fatalf("expected panic value to be an error, got %T", r)
		}
		if !strings.Contains(err.Error(), "scheme must be http or https") {
			t.Errorf("panic error = %q, want it to mention the invalid scheme", err.Error())
		}
	}()

	MustNew("llama3.2", WithHost("ftp://gpu:11434"))
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

		// Ollama's own documented OLLAMA_HOST format omits the scheme. Before
		// normalization "127.0.0.1:11434/v1" failed to parse at all and
		// "localhost:11434/v1" parsed with scheme "localhost", losing the port.
		{"schemeless ipv4 host", "127.0.0.1:11434", "", "http://127.0.0.1:11434/v1"},
		{"schemeless named host", "localhost:11434", "", "http://localhost:11434/v1"},
		{"schemeless env host", "", "127.0.0.1:11434", "http://127.0.0.1:11434/v1"},
		{"schemeless host trailing slash", "localhost:11434/", "", "http://localhost:11434/v1"},
		{"schemeless without port", "gpu-server", "", "http://gpu-server/v1"},
		{"https preserved", "https://gpu:11434", "", "https://gpu:11434/v1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", tt.envHost)

			got, err := buildBaseURL(tt.host)
			if err != nil {
				t.Fatalf("buildBaseURL(%q) unexpected error: %v", tt.host, err)
			}
			if got != tt.expected {
				t.Errorf("buildBaseURL(%q) = %q, want %q", tt.host, got, tt.expected)
			}
		})
	}
}

// TestBuildBaseURLNormalizedHostIsParseable guards the two reported
// reproductions directly: the normalized base URL must survive url.Parse with
// its host and port intact, rather than erroring or being silently reinterpreted.
func TestBuildBaseURLNormalizedHostIsParseable(t *testing.T) {
	tests := []struct {
		name string
		host string
		want string
	}{
		{"ipv4 colon port", "127.0.0.1:11434", "127.0.0.1:11434"},
		{"named colon port", "localhost:11434", "localhost:11434"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", "")

			base, err := buildBaseURL(tt.host)
			if err != nil {
				t.Fatalf("buildBaseURL(%q) unexpected error: %v", tt.host, err)
			}

			u, err := url.Parse(base)
			if err != nil {
				t.Fatalf("url.Parse(%q) failed: %v", base, err)
			}
			if u.Scheme != "http" {
				t.Errorf("scheme = %q, want %q", u.Scheme, "http")
			}
			if u.Host != tt.want {
				t.Errorf("host = %q, want %q (port must not be lost)", u.Host, tt.want)
			}
			if u.Path != "/v1" {
				t.Errorf("path = %q, want %q", u.Path, "/v1")
			}
			if u.Opaque != "" {
				t.Errorf("opaque = %q, want empty (URL must not be opaque)", u.Opaque)
			}
		})
	}
}

func TestBuildBaseURLInvalidHost(t *testing.T) {
	tests := []struct {
		name    string
		host    string
		wantErr string
	}{
		{"unparseable escape", "http://exa%zzmple.com:11434", "invalid host"},
		{"unsupported scheme", "ftp://gpu:11434", "scheme must be http or https"},
		{"missing host", "http://", "missing host"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", "")

			got, err := buildBaseURL(tt.host)
			if err == nil {
				t.Fatalf("buildBaseURL(%q) = %q, want error", tt.host, got)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want it to contain %q", err.Error(), tt.wantErr)
			}
			if !strings.HasPrefix(err.Error(), "ollama: ") {
				t.Errorf("error = %q, want it to be prefixed with %q", err.Error(), "ollama: ")
			}
		})
	}
}

// TestNewRejectsInvalidHost checks that the error surfaces from New rather than
// being deferred to the first request.
func TestNewRejectsInvalidHost(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "")

	model, err := New("llama3.2", WithHost("ftp://gpu:11434"))
	if err == nil {
		t.Fatalf("expected error, got model %v", model)
	}
	if model != nil {
		t.Errorf("expected nil model on error, got %v", model)
	}
}

func TestNewSchemelessHost(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "")

	model, err := New("llama3.2", WithHost("127.0.0.1:11434"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "llama3.2" {
		t.Errorf("expected name %q, got %q", "llama3.2", model.Name())
	}
}

func TestImplementsInterfaces(t *testing.T) {
	model, _ := New("test")

	var _ core.Model = model
	var _ core.StreamModel = model
}
