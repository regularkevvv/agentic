package gemini

import (
	"encoding/json"
	"strings"
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

func TestJSONSchemaPassthrough(t *testing.T) {
	tests := []struct {
		name   string
		schema map[string]interface{}
		wantOK bool
	}{
		{name: "nil schema", schema: nil},
		{name: "empty schema", schema: map[string]interface{}{}},
		{
			name:   "unencodable value",
			schema: map[string]interface{}{"type": "object", "bad": make(chan int)},
		},
		{
			name: "full document survives",
			schema: map[string]interface{}{
				"type":  "object",
				"$defs": map[string]interface{}{"Leg": map[string]interface{}{"type": "string"}},
				"properties": map[string]interface{}{
					"leg":     map[string]interface{}{"$ref": "#/$defs/Leg"},
					"when":    map[string]interface{}{"type": "string", "format": "date-time"},
					"either":  map[string]interface{}{"anyOf": []interface{}{map[string]interface{}{"type": "string"}}},
					"howMany": map[string]interface{}{"type": "integer", "minimum": 1, "maximum": 10},
				},
				"required": []interface{}{"leg"},
			},
			wantOK: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := jsonSchema(tt.schema)
			if ok != tt.wantOK {
				t.Fatalf("jsonSchema() ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			// Passthrough must hand the SDK the original document untouched,
			// keeping $ref/$defs/anyOf/format/constraints that a hand-rolled
			// conversion drops.
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			for _, keyword := range []string{`"$ref"`, `"$defs"`, `"anyOf"`, `"format"`, `"minimum"`, `"maximum"`} {
				if !strings.Contains(string(encoded), keyword) {
					t.Errorf("expected %s to survive passthrough, got %s", keyword, encoded)
				}
			}
		})
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
