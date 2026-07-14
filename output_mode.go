package agentic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/swaggest/jsonschema-go"
)

// NativeOutputSpec uses the model's native JSON schema output (response_format)
// for structured output. This is supported by OpenAI (json_schema) and
// Anthropic (output_config) on newer models.
//
// The model is instructed via the API to produce JSON conforming to the schema
// of type T. No tool call is needed — the model returns JSON directly.
//
// Example:
//
//	type MovieReview struct {
//	    Title   string  `json:"title"   description:"Movie title"`
//	    Rating  float64 `json:"rating"  description:"Rating from 1-10"`
//	    Summary string  `json:"summary" description:"Brief review"`
//	}
//
//	output := agentic.NewNativeOutput[MovieReview]("movie_review", "A structured movie review")
type NativeOutputSpec[T any] struct {
	name        string
	description string
	strict      bool
}

// NewNativeOutput creates a NativeOutputSpec that uses native JSON schema output.
// The name identifies the schema (used in the API request).
// The description explains what the output represents.
func NewNativeOutput[T any](name, description string) *NativeOutputSpec[T] {
	return &NativeOutputSpec[T]{
		name:        name,
		description: description,
		strict:      true,
	}
}

// WithStrict sets whether to enforce strict schema adherence. Default is true.
func (s *NativeOutputSpec[T]) WithStrict(strict bool) *NativeOutputSpec[T] {
	s.strict = strict
	return s
}

// Tools returns nil — native output does not use tools.
func (s *NativeOutputSpec[T]) Tools() []Tool { return nil }

// Parse extracts the typed output from the assistant message text (JSON).
func (s *NativeOutputSpec[T]) Parse(msg Message) (T, error) {
	var zero T
	text := msg.GetTextContent()
	if text == "" {
		return zero, fmt.Errorf("no text content in response for native output parsing")
	}
	var result T
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return zero, fmt.Errorf("parse native JSON output: %w", err)
	}
	if err := ValidateStruct(result); err != nil {
		return zero, err
	}
	return result, nil
}

// ResponseFormat returns the ResponseFormat to include in the ChatRequest.
func (s *NativeOutputSpec[T]) ResponseFormat() *ResponseFormat {
	schema := s.jsonSchema()
	return &ResponseFormat{
		Type: "json_schema",
		JSONSchema: &JSONSchemaFormat{
			Name:        s.name,
			Description: s.description,
			Schema:      schema,
			Strict:      &s.strict,
		},
	}
}

// Mode returns the output mode for this spec.
func (s *NativeOutputSpec[T]) Mode() OutputMode { return OutputModeNative }

// jsonSchema generates the JSON schema for type T.
func (s *NativeOutputSpec[T]) jsonSchema() map[string]interface{} {
	var zero T
	reflector := jsonschema.Reflector{}
	schema, err := reflector.Reflect(zero)
	if err != nil {
		panic(fmt.Sprintf("failed to generate schema for %T: %v", zero, err))
	}
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		panic(fmt.Sprintf("failed to marshal schema: %v", err))
	}
	var result map[string]interface{}
	_ = json.Unmarshal(schemaBytes, &result)

	// Ensure it's an object type
	if _, ok := result["type"]; !ok {
		result["type"] = "object"
	}
	if _, ok := result["properties"]; !ok {
		result["properties"] = map[string]interface{}{}
	}

	return result
}

// PromptedOutputSpec instructs the model via the system prompt to output JSON
// conforming to a schema. The model receives the schema as text instructions
// and optionally uses the provider's JSON object mode (response_format: json_object)
// to ensure valid JSON output.
//
// This is useful for providers/models that don't support native JSON schema output
// but do support JSON object mode.
//
// Example:
//
//	output := agentic.NewPromptedOutput[MovieReview]()
type PromptedOutputSpec[T any] struct {
	template string // Custom template; empty uses default
}

// DefaultPromptedTemplate is the default template for prompted output.
// The {schema} placeholder is replaced with the JSON schema.
const DefaultPromptedTemplate = `Always respond with a JSON object that's compatible with this schema:

{schema}

Don't include any text or Markdown fencing before or after.`

// NewPromptedOutput creates a PromptedOutputSpec.
func NewPromptedOutput[T any]() *PromptedOutputSpec[T] {
	return &PromptedOutputSpec[T]{}
}

// WithTemplate sets a custom template. Use {schema} as a placeholder for the JSON schema.
func (s *PromptedOutputSpec[T]) WithTemplate(template string) *PromptedOutputSpec[T] {
	s.template = template
	return s
}

// Tools returns nil — prompted output does not use tools.
func (s *PromptedOutputSpec[T]) Tools() []Tool { return nil }

// Parse extracts the typed output from the assistant message text (JSON).
func (s *PromptedOutputSpec[T]) Parse(msg Message) (T, error) {
	var zero T
	text := msg.GetTextContent()
	if text == "" {
		return zero, fmt.Errorf("no text content in response for prompted output parsing")
	}
	// Strip markdown fencing if present
	text = stripJSONFencing(text)
	var result T
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return zero, fmt.Errorf("parse prompted JSON output: %w", err)
	}
	if err := ValidateStruct(result); err != nil {
		return zero, err
	}
	return result, nil
}

// SystemPromptSuffix returns the text to append to the system prompt
// with the JSON schema instructions.
func (s *PromptedOutputSpec[T]) SystemPromptSuffix() string {
	schema := s.jsonSchema()
	schemaBytes, _ := json.MarshalIndent(schema, "", "  ")

	tmpl := s.template
	if tmpl == "" {
		tmpl = DefaultPromptedTemplate
	}
	return strings.ReplaceAll(tmpl, "{schema}", string(schemaBytes))
}

// ResponseFormat returns a json_object response format hint for providers that support it.
func (s *PromptedOutputSpec[T]) ResponseFormat() *ResponseFormat {
	return &ResponseFormat{
		Type: "json_object",
	}
}

// Mode returns the output mode for this spec.
func (s *PromptedOutputSpec[T]) Mode() OutputMode { return OutputModePrompted }

// jsonSchema generates the JSON schema for type T.
func (s *PromptedOutputSpec[T]) jsonSchema() map[string]interface{} {
	var zero T
	reflector := jsonschema.Reflector{}
	schema, err := reflector.Reflect(zero)
	if err != nil {
		panic(fmt.Sprintf("failed to generate schema for %T: %v", zero, err))
	}
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		panic(fmt.Sprintf("failed to marshal schema: %v", err))
	}
	var result map[string]interface{}
	_ = json.Unmarshal(schemaBytes, &result)
	return result
}

// TextProcessorOutputSpec uses a custom function to convert raw text output
// into a typed result. The model returns plain text, and the processor function
// transforms it into the desired output type.
//
// Example:
//
//	output := agentic.NewTextProcessorOutput(func(text string) (int, error) {
//	    return strconv.Atoi(strings.TrimSpace(text))
//	})
type TextProcessorOutputSpec[T any] struct {
	processor func(string) (T, error)
}

// NewTextProcessorOutput creates a TextProcessorOutputSpec with the given processor function.
func NewTextProcessorOutput[T any](processor func(string) (T, error)) *TextProcessorOutputSpec[T] {
	return &TextProcessorOutputSpec[T]{processor: processor}
}

// Tools returns nil — text processor output does not use tools.
func (s *TextProcessorOutputSpec[T]) Tools() []Tool { return nil }

// Parse extracts text from the message and runs it through the processor function.
func (s *TextProcessorOutputSpec[T]) Parse(msg Message) (T, error) {
	var zero T
	text := msg.GetTextContent()
	if text == "" {
		return zero, fmt.Errorf("no text content in response for text processor output")
	}
	result, err := s.processor(text)
	if err != nil {
		return zero, fmt.Errorf("text processor: %w", err)
	}
	return result, nil
}

// Mode returns the output mode for this spec.
func (s *TextProcessorOutputSpec[T]) Mode() OutputMode { return OutputModeText }

// OutputModeSpec is implemented by output specs that carry an output mode.
// This is used by the agent to determine how to configure the request.
type OutputModeSpec[O any] interface {
	OutputSpec[O]
	Mode() OutputMode
}

// ResponseFormatSpec is implemented by output specs that need a response_format
// in the ChatRequest (NativeOutputSpec and PromptedOutputSpec).
type ResponseFormatSpec interface {
	ResponseFormat() *ResponseFormat
}

// SystemPromptAppender is implemented by output specs that need to append
// instructions to the system prompt (PromptedOutputSpec).
type SystemPromptAppender interface {
	SystemPromptSuffix() string
}

// stripJSONFencing removes markdown code fencing from JSON output.
func stripJSONFencing(text string) string {
	text = strings.TrimSpace(text)
	if strings.HasPrefix(text, "```json") {
		text = strings.TrimPrefix(text, "```json")
		text = strings.TrimSuffix(text, "```")
		text = strings.TrimSpace(text)
	} else if strings.HasPrefix(text, "```") {
		text = strings.TrimPrefix(text, "```")
		text = strings.TrimSuffix(text, "```")
		text = strings.TrimSpace(text)
	}
	return text
}
