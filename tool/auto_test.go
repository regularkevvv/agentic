package tool

import (
	"context"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

// ============================================================================
// camelToSnake tests
// ============================================================================

func TestCamelToSnake(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"", ""},
		{"Get", "get"},
		{"GetWeather", "get_weather"},
		{"getWeather", "get_weather"},
		{"HTTPServer", "http_server"},
		{"getURL", "get_url"},
		{"QueryDBByID", "query_db_by_id"},
		{"SimpleTest", "simple_test"},
		{"A", "a"},
		{"ABC", "abc"},
		{"ABCDef", "abc_def"},
		{"already_snake", "already_snake"},
		{"XMLParser", "xml_parser"},
		// Note: consecutive acronyms like HTTPSURL are ambiguous without a
		// dictionary. The algorithm treats it as a single uppercase run.
		{"getHTTPSURL", "get_httpsurl"},
		{"IOReader", "io_reader"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := camelToSnake(tt.input)
			if got != tt.expected {
				t.Errorf("camelToSnake(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

// ============================================================================
// inferToolName tests
// ============================================================================

type GetWeatherInput struct {
	Location string `json:"location"`
}

type SearchParams struct {
	Query string `json:"query"`
}

type QueryDBArgs struct {
	SQL string `json:"sql"`
}

type FetchDataRequest struct {
	URL string `json:"url"`
}

type SimpleStruct struct {
	X int `json:"x"`
}

func TestInferToolName(t *testing.T) {
	tests := []struct {
		name     string
		fn       func() string
		expected string
	}{
		{"strips Input suffix", inferToolName[GetWeatherInput], "get_weather"},
		{"strips Params suffix", inferToolName[SearchParams], "search"},
		{"strips Args suffix", inferToolName[QueryDBArgs], "query_db"},
		{"strips Request suffix", inferToolName[FetchDataRequest], "fetch_data"},
		{"no suffix to strip", inferToolName[SimpleStruct], "simple_struct"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.fn()
			if got != tt.expected {
				t.Errorf("got %q, want %q", got, tt.expected)
			}
		})
	}
}

// ============================================================================
// inferDescription tests
// ============================================================================

type DescribedToolInput struct {
	_    struct{} `tool:"Get the current weather for any city"`
	City string   `json:"city"`
}

type NoDescriptionInput struct {
	Name string `json:"name"`
}

type MultiFieldDescInput struct {
	_     struct{} `tool:"First description wins"`
	Value int      `json:"value"`
}

type PtrDescriptionInput struct {
	_    struct{} `tool:"Pointer description"`
	Name string   `json:"name"`
}

func TestInferDescription(t *testing.T) {
	tests := []struct {
		name     string
		fn       func() string
		expected string
	}{
		{"with tool tag", inferDescription[DescribedToolInput], "Get the current weather for any city"},
		{"without tool tag", inferDescription[NoDescriptionInput], ""},
		{"multi-field with tool tag", inferDescription[MultiFieldDescInput], "First description wins"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.fn()
			if got != tt.expected {
				t.Errorf("got %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestInferToolNameAndDescriptionAdditionalBranches(t *testing.T) {
	if got := inferToolName[struct{ Value int }](); got != "" {
		t.Fatalf("expected unnamed struct to infer empty name, got %q", got)
	}
	if got := inferDescription[*PtrDescriptionInput](); got != "Pointer description" {
		t.Fatalf("expected pointer description, got %q", got)
	}
	if got := inferDescription[string](); got != "" {
		t.Fatalf("expected non-struct description to be empty, got %q", got)
	}
}

// ============================================================================
// Auto tests
// ============================================================================

type GreetInput struct {
	_    struct{} `tool:"Greet a person by name"`
	Name string   `json:"name" description:"Person name"`
}

type GreetOutput struct {
	Greeting string `json:"greeting"`
}

func TestAutoInfersNameAndDescription(t *testing.T) {
	tool, handler, err := Auto(func(input GreetInput) (GreetOutput, error) {
		return GreetOutput{Greeting: "Hello " + input.Name}, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "greet" {
		t.Errorf("expected name %q, got %q", "greet", tool.Function.Name)
	}
	if tool.Function.Description != "Greet a person by name" {
		t.Errorf("expected description %q, got %q", "Greet a person by name", tool.Function.Description)
	}
	if handler.Name() != "greet" {
		t.Errorf("expected handler name %q, got %q", "greet", handler.Name())
	}

	// Execute the handler
	result, err := handler.Execute(context.Background(), map[string]interface{}{"name": "World"}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	output, ok := result.(GreetOutput)
	if !ok {
		t.Fatalf("expected GreetOutput, got %T", result)
	}
	if output.Greeting != "Hello World" {
		t.Errorf("expected %q, got %q", "Hello World", output.Greeting)
	}
}

func TestAutoWithNameOverride(t *testing.T) {
	tool, _, err := Auto(func(input GreetInput) (GreetOutput, error) {
		return GreetOutput{}, nil
	}, WithName("custom_greet"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "custom_greet" {
		t.Errorf("expected name %q, got %q", "custom_greet", tool.Function.Name)
	}
	// Description should still be inferred
	if tool.Function.Description != "Greet a person by name" {
		t.Errorf("expected description %q, got %q", "Greet a person by name", tool.Function.Description)
	}
}

func TestAutoWithDescriptionOverride(t *testing.T) {
	tool, _, err := Auto(func(input GreetInput) (GreetOutput, error) {
		return GreetOutput{}, nil
	}, WithDescription("Custom description"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "greet" {
		t.Errorf("expected name %q, got %q", "greet", tool.Function.Name)
	}
	if tool.Function.Description != "Custom description" {
		t.Errorf("expected description %q, got %q", "Custom description", tool.Function.Description)
	}
}

func TestAutoWithBothOverrides(t *testing.T) {
	tool, _, err := Auto(func(input GreetInput) (GreetOutput, error) {
		return GreetOutput{}, nil
	}, WithName("my_tool"), WithDescription("My tool"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "my_tool" {
		t.Errorf("expected name %q, got %q", "my_tool", tool.Function.Name)
	}
	if tool.Function.Description != "My tool" {
		t.Errorf("expected description %q, got %q", "My tool", tool.Function.Description)
	}
}

func TestAutoNoDescriptionFails(t *testing.T) {
	// NoDescriptionInput has no tool tag, so description will be empty, which should fail
	type Output struct{}
	_, _, err := Auto(func(input NoDescriptionInput) (Output, error) {
		return Output{}, nil
	})
	if err == nil {
		t.Fatal("expected error when description is empty")
	}
}

func TestMustAutoPanicsOnError(t *testing.T) {
	type Output struct{}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	// No description = error = panic
	MustAuto(func(input NoDescriptionInput) (Output, error) {
		return Output{}, nil
	})
}

func TestMustAutoSucceeds(t *testing.T) {
	tool, handler := MustAuto(func(input GreetInput) (GreetOutput, error) {
		return GreetOutput{Greeting: "Hi"}, nil
	})
	if tool.Function.Name != "greet" {
		t.Errorf("expected %q, got %q", "greet", tool.Function.Name)
	}
	if handler.Name() != "greet" {
		t.Errorf("expected handler name %q, got %q", "greet", handler.Name())
	}
}

// ============================================================================
// AutoWithDeps tests
// ============================================================================

type MyDeps struct {
	Prefix string
}

type PrefixInput struct {
	_    struct{} `tool:"Add a prefix to text"`
	Text string   `json:"text" description:"Text to prefix"`
}

type PrefixOutput struct {
	Result string `json:"result"`
}

func TestAutoWithDeps(t *testing.T) {
	tool, handler, err := AutoWithDeps(
		func(ctx core.RunContext[*MyDeps], input PrefixInput) (PrefixOutput, error) {
			return PrefixOutput{Result: ctx.Deps.Prefix + input.Text}, nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "prefix" {
		t.Errorf("expected name %q, got %q", "prefix", tool.Function.Name)
	}
	if tool.Function.Description != "Add a prefix to text" {
		t.Errorf("expected description %q, got %q", "Add a prefix to text", tool.Function.Description)
	}

	deps := &MyDeps{Prefix: ">> "}
	result, err := handler.Execute(context.Background(), map[string]interface{}{"text": "hello"}, core.NewDependencyEnvelope(deps))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	output, ok := result.(PrefixOutput)
	if !ok {
		t.Fatalf("expected PrefixOutput, got %T", result)
	}
	if output.Result != ">> hello" {
		t.Errorf("expected %q, got %q", ">> hello", output.Result)
	}
}

func TestAutoWithDepsOverrides(t *testing.T) {
	tool, _, err := AutoWithDeps(
		func(ctx core.RunContext[MyDeps], input PrefixInput) (PrefixOutput, error) {
			return PrefixOutput{}, nil
		},
		WithName("custom_prefix"),
		WithDescription("Custom desc"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "custom_prefix" {
		t.Errorf("expected name %q, got %q", "custom_prefix", tool.Function.Name)
	}
	if tool.Function.Description != "Custom desc" {
		t.Errorf("expected description %q, got %q", "Custom desc", tool.Function.Description)
	}
}

func TestMustAutoWithDepsSucceeds(t *testing.T) {
	tool, handler := MustAutoWithDeps(
		func(ctx core.RunContext[MyDeps], input PrefixInput) (PrefixOutput, error) {
			return PrefixOutput{Result: "ok"}, nil
		},
	)
	if tool.Function.Name != "prefix" {
		t.Errorf("expected %q, got %q", "prefix", tool.Function.Name)
	}
	if handler.Name() != "prefix" {
		t.Errorf("expected handler name %q, got %q", "prefix", handler.Name())
	}
}

func TestMustAutoWithDepsPanicsOnError(t *testing.T) {
	type Output struct{}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	MustAutoWithDeps(
		func(ctx core.RunContext[MyDeps], input NoDescriptionInput) (Output, error) {
			return Output{}, nil
		},
	)
}

// ============================================================================
// Schema generation tests (ensure Auto generates correct schema)
// ============================================================================

type DetailedInput struct {
	_        struct{} `tool:"A tool with detailed schema"`
	Query    string   `json:"query" description:"Search query" required:"true"`
	Limit    int      `json:"limit" description:"Max results" minimum:"1" maximum:"100"`
	Category string   `json:"category" description:"Category filter" enum:"news,blog,images"`
}

type DetailedOutput struct {
	Results []string `json:"results"`
}

func TestAutoPreservesSchemaDetails(t *testing.T) {
	tool, _, err := Auto(func(input DetailedInput) (DetailedOutput, error) {
		return DetailedOutput{}, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if tool.Function.Name != "detailed" {
		t.Errorf("expected name %q, got %q", "detailed", tool.Function.Name)
	}

	params := tool.Function.Parameters
	if params["type"] != "object" {
		t.Errorf("expected type 'object', got %v", params["type"])
	}

	props, ok := params["properties"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected properties map, got %T", params["properties"])
	}

	// Check query property exists
	if _, ok := props["query"]; !ok {
		t.Error("expected 'query' property")
	}
	if _, ok := props["limit"]; !ok {
		t.Error("expected 'limit' property")
	}
	if _, ok := props["category"]; !ok {
		t.Error("expected 'category' property")
	}
}

// ============================================================================
// Edge cases
// ============================================================================

// Test that suffix stripping doesn't strip the entire name
type Input struct {
	_ struct{} `tool:"A tool named just Input"`
	X int      `json:"x"`
}

func TestInferToolNameDoesNotStripEntireName(t *testing.T) {
	// "Input" alone should NOT strip to empty — the suffix is only stripped
	// if the name is longer than the suffix
	name := inferToolName[Input]()
	if name != "input" {
		t.Errorf("expected %q, got %q", "input", name)
	}
}

type HTTPServerInput struct {
	_   struct{} `tool:"Manage an HTTP server"`
	URL string   `json:"url"`
}

func TestAutoWithAcronyms(t *testing.T) {
	tool, _, err := Auto(func(input HTTPServerInput) (GreetOutput, error) {
		return GreetOutput{}, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// HTTPServer → http_server (acronym followed by lowercase)
	if tool.Function.Name != "http_server" {
		t.Errorf("expected name %q, got %q", "http_server", tool.Function.Name)
	}
}
