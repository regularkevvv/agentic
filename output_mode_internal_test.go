package agentic

import (
	"strings"
	"testing"
)

type outputModeCoverageValue struct {
	Value string `json:"value"`
}

func TestOutputModeAdditionalBranches(t *testing.T) {
	t.Run("native json schema adds object defaults", func(t *testing.T) {
		spec := NewNativeOutput[any]("generic", "generic output")
		schema := spec.jsonSchema()

		if got := schema["type"]; got != "object" {
			t.Fatalf("expected default object type, got %#v", got)
		}
		if _, ok := schema["properties"].(map[string]interface{}); !ok {
			t.Fatalf("expected default properties map, got %#v", schema["properties"])
		}
	})

	t.Run("prompted parse errors", func(t *testing.T) {
		type validated struct {
			Name string `json:"name" validate:"required"`
		}

		emptySpec := NewPromptedOutput[outputModeCoverageValue]()
		if _, err := emptySpec.Parse(Message{Role: RoleAssistant}); err == nil || err.Error() != "no text content in response for prompted output parsing" {
			t.Fatalf("expected empty text error, got %v", err)
		}

		invalidSpec := NewPromptedOutput[outputModeCoverageValue]()
		if _, err := invalidSpec.Parse(NewTextMessage(RoleAssistant, "{")); err == nil || !strings.Contains(err.Error(), "parse prompted JSON output") {
			t.Fatalf("expected JSON parse error, got %v", err)
		}

		validatedSpec := NewPromptedOutput[validated]()
		if _, err := validatedSpec.Parse(NewTextMessage(RoleAssistant, `{}`)); err == nil || !IsValidationError(err) {
			t.Fatalf("expected validation error, got %v", err)
		}
	})
}
