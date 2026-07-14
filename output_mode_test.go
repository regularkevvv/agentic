package agentic_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

type Review struct {
	Title   string  `json:"title"   description:"Movie title"`
	Rating  float64 `json:"rating"  description:"Rating from 1-10"`
	Summary string  `json:"summary" description:"Brief review"`
}

// --- NativeOutputSpec tests ---

func TestNativeOutputSpec_Parse(t *testing.T) {
	spec := agentic.NewNativeOutput[Review]("review", "A movie review")

	msg := agentic.NewTextMessage(agentic.RoleAssistant, `{"title":"The Matrix","rating":9.5,"summary":"A great movie"}`)

	parsed, err := spec.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := parsed
	if result.Title != "The Matrix" {
		t.Errorf("expected title %q, got %q", "The Matrix", result.Title)
	}
	if result.Rating != 9.5 {
		t.Errorf("expected rating 9.5, got %f", result.Rating)
	}
}

func TestNativeOutputSpec_ParseInvalidJSON(t *testing.T) {
	spec := agentic.NewNativeOutput[Review]("review", "A movie review")

	msg := agentic.NewTextMessage(agentic.RoleAssistant, `not json`)
	_, err := spec.Parse(msg)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestNativeOutputSpec_ParseEmpty(t *testing.T) {
	spec := agentic.NewNativeOutput[Review]("review", "A movie review")

	msg := agentic.Message{Role: agentic.RoleAssistant}
	_, err := spec.Parse(msg)
	if err == nil {
		t.Fatal("expected error for empty message")
	}
}

func TestNativeOutputSpec_Tools(t *testing.T) {
	spec := agentic.NewNativeOutput[Review]("review", "A movie review")
	tools := spec.Tools()
	if len(tools) != 0 {
		t.Errorf("expected no tools, got %d", len(tools))
	}
}

func TestNativeOutputSpec_ResponseFormat(t *testing.T) {
	spec := agentic.NewNativeOutput[Review]("movie_review", "A structured movie review")

	rf := spec.ResponseFormat()
	if rf == nil {
		t.Fatal("expected non-nil ResponseFormat")
	}
	if rf.Type != "json_schema" {
		t.Errorf("expected type %q, got %q", "json_schema", rf.Type)
	}
	if rf.JSONSchema == nil {
		t.Fatal("expected non-nil JSONSchema")
	}
	if rf.JSONSchema.Name != "movie_review" {
		t.Errorf("expected name %q, got %q", "movie_review", rf.JSONSchema.Name)
	}
	if rf.JSONSchema.Schema == nil {
		t.Fatal("expected non-nil Schema")
	}
	if rf.JSONSchema.Strict == nil || *rf.JSONSchema.Strict != true {
		t.Error("expected strict to be true")
	}
}

func TestNativeOutputSpec_WithStrict(t *testing.T) {
	spec := agentic.NewNativeOutput[Review]("review", "test").WithStrict(false)

	rf := spec.ResponseFormat()
	if rf.JSONSchema.Strict == nil || *rf.JSONSchema.Strict != false {
		t.Error("expected strict to be false")
	}
}

func TestNativeOutputSpec_Mode(t *testing.T) {
	spec := agentic.NewNativeOutput[Review]("review", "test")
	if spec.Mode() != agentic.OutputModeNative {
		t.Errorf("expected mode %q, got %q", agentic.OutputModeNative, spec.Mode())
	}
}

func TestNativeOutputSpec_Validation(t *testing.T) {
	type Validated struct {
		Name  string `json:"name" validate:"required,min=2"`
		Score int    `json:"score" validate:"required,min=1,max=10"`
	}

	spec := agentic.NewNativeOutput[Validated]("test", "test")

	msg := agentic.NewTextMessage(agentic.RoleAssistant, `{"name":"","score":0}`)
	_, err := spec.Parse(msg)
	if err == nil {
		t.Fatal("expected validation error")
	}
	if !agentic.IsValidationError(err) {
		t.Errorf("expected ValidationError, got %T: %v", err, err)
	}
}

// --- PromptedOutputSpec tests ---

func TestPromptedOutputSpec_Parse(t *testing.T) {
	spec := agentic.NewPromptedOutput[Review]()

	msg := agentic.NewTextMessage(agentic.RoleAssistant, `{"title":"Inception","rating":8.5,"summary":"Dreams within dreams"}`)

	parsed, err := spec.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := parsed
	if result.Title != "Inception" {
		t.Errorf("expected title %q, got %q", "Inception", result.Title)
	}
}

func TestPromptedOutputSpec_ParseWithFencing(t *testing.T) {
	spec := agentic.NewPromptedOutput[Review]()

	// Model wraps JSON in markdown fencing
	msg := agentic.NewTextMessage(agentic.RoleAssistant, "```json\n{\"title\":\"Inception\",\"rating\":8.5,\"summary\":\"Amazing\"}\n```")

	parsed, err := spec.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := parsed
	if result.Title != "Inception" {
		t.Errorf("expected title %q, got %q", "Inception", result.Title)
	}
}

func TestPromptedOutputSpec_ParseWithGenericFencing(t *testing.T) {
	spec := agentic.NewPromptedOutput[Review]()

	msg := agentic.NewTextMessage(agentic.RoleAssistant, "```\n{\"title\":\"Inception\",\"rating\":8.5,\"summary\":\"Amazing\"}\n```")

	parsed, err := spec.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := parsed
	if result.Title != "Inception" {
		t.Errorf("expected title %q, got %q", "Inception", result.Title)
	}
}

func TestPromptedOutputSpec_Tools(t *testing.T) {
	spec := agentic.NewPromptedOutput[Review]()
	if len(spec.Tools()) != 0 {
		t.Error("expected no tools")
	}
}

func TestPromptedOutputSpec_SystemPromptSuffix(t *testing.T) {
	spec := agentic.NewPromptedOutput[Review]()

	suffix := spec.SystemPromptSuffix()
	if suffix == "" {
		t.Fatal("expected non-empty suffix")
	}
	if !strings.Contains(suffix, "JSON object") {
		t.Error("expected suffix to contain 'JSON object'")
	}
	if !strings.Contains(suffix, "properties") {
		t.Error("expected suffix to contain schema properties")
	}
}

func TestPromptedOutputSpec_CustomTemplate(t *testing.T) {
	spec := agentic.NewPromptedOutput[Review]().WithTemplate("Output JSON: {schema}")

	suffix := spec.SystemPromptSuffix()
	if !strings.HasPrefix(suffix, "Output JSON: ") {
		t.Errorf("expected custom template, got %q", suffix)
	}
}

func TestPromptedOutputSpec_ResponseFormat(t *testing.T) {
	spec := agentic.NewPromptedOutput[Review]()

	rf := spec.ResponseFormat()
	if rf == nil {
		t.Fatal("expected non-nil ResponseFormat")
	}
	if rf.Type != "json_object" {
		t.Errorf("expected type %q, got %q", "json_object", rf.Type)
	}
}

func TestPromptedOutputSpec_Mode(t *testing.T) {
	spec := agentic.NewPromptedOutput[Review]()
	if spec.Mode() != agentic.OutputModePrompted {
		t.Errorf("expected mode %q, got %q", agentic.OutputModePrompted, spec.Mode())
	}
}

// --- TextProcessorOutputSpec tests ---

func TestTextProcessorOutputSpec_Parse(t *testing.T) {
	spec := agentic.NewTextProcessorOutput(func(text string) (int, error) {
		return strconv.Atoi(strings.TrimSpace(text))
	})

	msg := agentic.NewTextMessage(agentic.RoleAssistant, "  42  ")

	parsed, err := spec.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := parsed
	if result != 42 {
		t.Errorf("expected 42, got %d", result)
	}
}

func TestTextProcessorOutputSpec_ParseError(t *testing.T) {
	spec := agentic.NewTextProcessorOutput(func(text string) (int, error) {
		return 0, fmt.Errorf("cannot parse: %s", text)
	})

	msg := agentic.NewTextMessage(agentic.RoleAssistant, "not a number")
	_, err := spec.Parse(msg)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestTextProcessorOutputSpec_EmptyText(t *testing.T) {
	spec := agentic.NewTextProcessorOutput(func(text string) (string, error) {
		return text, nil
	})

	msg := agentic.Message{Role: agentic.RoleAssistant}
	_, err := spec.Parse(msg)
	if err == nil {
		t.Fatal("expected error for empty message")
	}
}

func TestTextProcessorOutputSpec_Tools(t *testing.T) {
	spec := agentic.NewTextProcessorOutput(func(text string) (string, error) {
		return text, nil
	})
	if len(spec.Tools()) != 0 {
		t.Error("expected no tools")
	}
}

func TestTextProcessorOutputSpec_Mode(t *testing.T) {
	spec := agentic.NewTextProcessorOutput(func(text string) (string, error) {
		return text, nil
	})
	if spec.Mode() != agentic.OutputModeText {
		t.Errorf("expected mode %q, got %q", agentic.OutputModeText, spec.Mode())
	}
}

// --- TypedAgent with output modes integration tests ---

func TestTypedAgentWithNativeOutput(t *testing.T) {
	jsonOutput, _ := json.Marshal(Review{
		Title:   "The Matrix",
		Rating:  9.0,
		Summary: "A sci-fi masterpiece",
	})

	model := test.NewTestModel(
		test.ModelResponse{Text: string(jsonOutput)},
	)

	agent := agentic.NewTypedAgentWithMode[Review](
		"You review movies.",
		model,
		agentic.NewNativeOutput[Review]("movie_review", "A structured movie review"),
	)

	result, err := agent.Run(context.Background(), "Review The Matrix")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Title != "The Matrix" {
		t.Errorf("expected title %q, got %q", "The Matrix", result.Output.Title)
	}
	if result.Output.Rating != 9.0 {
		t.Errorf("expected rating 9.0, got %f", result.Output.Rating)
	}

	// Verify the request had response_format set
	calls := model.Calls()
	if len(calls) == 0 {
		t.Fatal("expected at least one call")
	}
	if calls[0].ResponseFormat == nil {
		t.Error("expected ResponseFormat to be set on request")
	} else if calls[0].ResponseFormat.Type != "json_schema" {
		t.Errorf("expected ResponseFormat.Type %q, got %q", "json_schema", calls[0].ResponseFormat.Type)
	}
}

func TestTypedAgentWithPromptedOutput(t *testing.T) {
	jsonOutput, _ := json.Marshal(Review{
		Title:   "Inception",
		Rating:  8.5,
		Summary: "Dreams within dreams",
	})

	model := test.NewTestModel(
		test.ModelResponse{Text: string(jsonOutput)},
	)

	agent := agentic.NewTypedAgentWithMode[Review](
		"You review movies.",
		model,
		agentic.NewPromptedOutput[Review](),
	)

	result, err := agent.Run(context.Background(), "Review Inception")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Title != "Inception" {
		t.Errorf("expected title %q, got %q", "Inception", result.Output.Title)
	}

	// Verify the request had response_format and schema in system prompt
	calls := model.Calls()
	if len(calls) == 0 {
		t.Fatal("expected at least one call")
	}
	if calls[0].ResponseFormat == nil {
		t.Error("expected ResponseFormat to be set on request")
	} else if calls[0].ResponseFormat.Type != "json_object" {
		t.Errorf("expected ResponseFormat.Type %q, got %q", "json_object", calls[0].ResponseFormat.Type)
	}

	// Check that the system prompt contains the schema
	systemMsg := calls[0].Messages[0]
	if systemMsg.Role != agentic.RoleSystem {
		t.Fatal("expected first message to be system")
	}
	systemText := systemMsg.GetTextContent()
	if !strings.Contains(systemText, "JSON object") {
		t.Error("expected system prompt to contain schema instructions")
	}
}

func TestTypedAgentWithTextProcessor(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{Text: "  42  "},
	)

	agent := agentic.NewTypedAgentWithMode[int](
		"You return numbers.",
		model,
		agentic.NewTextProcessorOutput(func(text string) (int, error) {
			return strconv.Atoi(strings.TrimSpace(text))
		}),
	)

	result, err := agent.Run(context.Background(), "What is 6 * 7?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output != 42 {
		t.Errorf("expected 42, got %d", result.Output)
	}
}

func TestTypedAgentWithToolOutput_BackwardCompat(t *testing.T) {
	// Verify the existing NewTypedAgent still works
	model := test.NewTestModel(
		test.ModelResponse{
			ToolCalls: []agentic.ToolUse{
				{
					ID:   "call_1",
					Name: "__output__",
					Input: map[string]interface{}{
						"title":   "The Matrix",
						"rating":  float64(9),
						"summary": "Classic sci-fi",
					},
				},
			},
		},
	)

	agent := agentic.NewTypedAgent[Review](
		"You review movies.",
		model,
		"Provide a structured movie review",
	)

	result, err := agent.Run(context.Background(), "Review The Matrix")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Title != "The Matrix" {
		t.Errorf("expected title %q, got %q", "The Matrix", result.Output.Title)
	}
}

func TestTypedAgentWithNativeOutput_ValidationRetry(t *testing.T) {
	type Validated struct {
		Name string `json:"name" validate:"required,min=2"`
	}

	// First response fails validation, second passes
	model := test.NewTestModel(
		test.ModelResponse{Text: `{"name":""}`},
		test.ModelResponse{Text: `{"name":"Alice"}`},
	)

	agent := agentic.NewTypedAgentWithMode[Validated](
		"Return names.",
		model,
		agentic.NewNativeOutput[Validated]("name", "A name"),
	)

	result, err := agent.Run(context.Background(), "Give me a name")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Name != "Alice" {
		t.Errorf("expected name %q, got %q", "Alice", result.Output.Name)
	}
}

// --- ResponseFormat in ChatRequest tests ---

func TestChatRequestResponseFormat(t *testing.T) {
	req := &agentic.ChatRequest{
		Model:    "test",
		Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "hi")},
		ResponseFormat: &agentic.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &agentic.JSONSchemaFormat{
				Name:   "test",
				Schema: map[string]interface{}{"type": "object"},
			},
		},
	}

	if err := req.Validate(); err != nil {
		t.Fatalf("validation should pass: %v", err)
	}
	if req.ResponseFormat.Type != "json_schema" {
		t.Errorf("expected type %q, got %q", "json_schema", req.ResponseFormat.Type)
	}
}
