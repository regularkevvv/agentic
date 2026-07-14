package tool

import (
	"encoding/json"
	"fmt"

	"github.com/swaggest/jsonschema-go"

	"github.com/regularkevvv/agentic/internal/core"
)

// NewToolFromStruct creates a Tool definition from a struct type using JSON schema generation.
// The input parameter should be an instance of the struct type (can be zero value).
//
// Supported struct tags:
//   - json: field name in JSON schema
//   - description: field description
//   - required: marks field as required (use `required:"true"`)
//   - enum: comma-separated list of allowed values
//   - minimum: minimum value for numeric fields
//   - maximum: maximum value for numeric fields
//   - minLength: minimum length for string fields
//   - maxLength: maximum length for string fields
//   - pattern: regex pattern for string validation
func NewToolFromStruct(name, description string, input interface{}) (core.Tool, error) {
	if name == "" {
		return core.Tool{}, fmt.Errorf("tool name cannot be empty")
	}
	if description == "" {
		return core.Tool{}, fmt.Errorf("tool description cannot be empty")
	}

	reflector := jsonschema.Reflector{}
	schema, err := reflector.Reflect(input)
	if err != nil {
		return core.Tool{}, fmt.Errorf("failed to generate schema: %w", err)
	}

	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return core.Tool{}, fmt.Errorf("failed to marshal schema: %w", err)
	}

	var parameters map[string]interface{}
	if err := json.Unmarshal(schemaBytes, &parameters); err != nil {
		return core.Tool{}, fmt.Errorf("failed to unmarshal schema: %w", err)
	}

	if _, ok := parameters["type"]; !ok {
		parameters["type"] = "object"
	}
	if _, ok := parameters["properties"]; !ok {
		parameters["properties"] = map[string]interface{}{}
	}

	return core.Tool{
		Type: core.ToolTypeFunction,
		Function: core.Function{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}, nil
}

// MustNewToolFromStruct is like NewToolFromStruct but panics on error.
func MustNewToolFromStruct(name, description string, input interface{}) core.Tool {
	tool, err := NewToolFromStruct(name, description, input)
	if err != nil {
		panic(fmt.Sprintf("failed to create tool: %v", err))
	}
	return tool
}
