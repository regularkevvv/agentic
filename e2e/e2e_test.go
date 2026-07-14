//go:build e2e
// +build e2e

// Package e2e contains end-to-end tests that make real LLM API calls.
//
// These tests require API keys to be set as environment variables:
//   - ANTHROPIC_API_KEY for Anthropic tests
//   - OPENAI_API_KEY for OpenAI tests
//
// Run with: go test -v ./e2e/ -tags=e2e -timeout=120s
//
// Tests are skipped automatically if the corresponding API key is not set.
package e2e

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/anthropic"
	"github.com/regularkevvv/agentic/provider/grok"
	"github.com/regularkevvv/agentic/provider/ollama"
	"github.com/regularkevvv/agentic/provider/openai"
	"github.com/regularkevvv/agentic/provider/openrouter"
	"github.com/regularkevvv/agentic/provider/together"
)

// TestMain loads .env from the repo root before running tests.
func TestMain(m *testing.M) {
	// Walk up from the test directory to find the .env file at repo root
	dir, _ := os.Getwd()
	for {
		envPath := filepath.Join(dir, ".env")
		if err := loadDotEnv(envPath); err == nil {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break // reached filesystem root
		}
		dir = parent
	}
	os.Exit(m.Run())
}

func loadDotEnv(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if _, exists := os.LookupEnv(key); exists {
			continue // don't override existing env vars
		}
		os.Setenv(key, value)
	}
	return scanner.Err()
}

// helpers

func skipIfNoAnthropicKey(t *testing.T) {
	t.Helper()
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping Anthropic e2e test")
	}
}

func skipIfNoOpenAIKey(t *testing.T) {
	t.Helper()
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set, skipping OpenAI e2e test")
	}
}

func newAnthropicModel(t *testing.T) agentic.Model {
	t.Helper()
	m, err := anthropic.New("claude-sonnet-4-6")
	if err != nil {
		t.Fatalf("failed to create anthropic model: %v", err)
	}
	return m
}

func newOpenAIModel(t *testing.T) agentic.Model {
	t.Helper()
	m, err := openai.New("gpt-4o-mini")
	if err != nil {
		t.Fatalf("failed to create openai model: %v", err)
	}
	return m
}

func skipIfNoTogetherKey(t *testing.T) {
	t.Helper()
	if os.Getenv("TOGETHER_API_KEY") == "" && os.Getenv("TOGETHER_AI_API_KEY") == "" {
		t.Skip("TOGETHER_API_KEY / TOGETHER_AI_API_KEY not set, skipping Together AI e2e test")
	}
}

func skipIfNoOllama(t *testing.T) {
	t.Helper()
	// Quick check: try connecting to the default Ollama endpoint.
	resp, err := http.Get("http://localhost:11434/api/tags")
	if err != nil {
		t.Skip("Ollama not running locally, skipping Ollama e2e test")
	}
	resp.Body.Close()
}

func skipIfNoOpenRouterKey(t *testing.T) {
	t.Helper()
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping OpenRouter e2e test")
	}
}

func skipIfNoGrokKey(t *testing.T) {
	t.Helper()
	if os.Getenv("GROK_API_KEY") == "" && os.Getenv("XAI_API_KEY") == "" {
		t.Skip("GROK_API_KEY / XAI_API_KEY not set, skipping Grok e2e test")
	}
}

func newTogetherModel(t *testing.T) agentic.Model {
	t.Helper()
	m, err := together.New("meta-llama/Llama-3.3-70B-Instruct-Turbo")
	if err != nil {
		t.Fatalf("failed to create together model: %v", err)
	}
	return m
}

func newOpenAIResponsesModel(t *testing.T) agentic.Model {
	t.Helper()
	m, err := openai.NewResponses("gpt-4o-mini")
	if err != nil {
		t.Fatalf("failed to create openai responses model: %v", err)
	}
	return m
}

func ctxWithTimeout(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// ============================================================================
// Dynamic System Prompts — E2E Tests
// ============================================================================

func TestE2E_Anthropic_StaticPrompt(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx := ctxWithTimeout(t)
	model := newAnthropicModel(t)

	agent := agentic.NewAgent(
		"You are a pirate. Always respond with pirate language including 'Arrr'. Keep responses under 50 words.",
		model,
		agentic.WithMaxTokens(100),
	)

	result, err := agent.Run(ctx, "Say hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lower := strings.ToLower(result.Output)
	if !strings.Contains(lower, "arr") {
		t.Errorf("expected pirate language with 'arr', got: %s", result.Output)
	}
}

func TestE2E_Anthropic_MultipleSystemPrompts(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx := ctxWithTimeout(t)
	model := newAnthropicModel(t)

	agent := agentic.NewAgent("", model,
		agentic.WithSystemPrompts(
			"You are a helpful assistant that always responds in uppercase only.",
			"Keep your responses to exactly one sentence.",
		),
		agentic.WithMaxTokens(100),
	)

	result, err := agent.Run(ctx, "What is 2+2?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check that the response is uppercase (at least mostly)
	upper := strings.ToUpper(result.Output)
	// Allow some flexibility — model might include punctuation
	cleaned := strings.Map(func(r rune) rune {
		if r >= 'A' && r <= 'Z' {
			return r
		}
		return -1
	}, result.Output)
	cleanedUpper := strings.Map(func(r rune) rune {
		if r >= 'A' && r <= 'Z' {
			return r
		}
		return -1
	}, upper)

	if cleaned != cleanedUpper {
		t.Errorf("expected uppercase response, got: %s", result.Output)
	}

	t.Logf("Response: %s", result.Output)
}

func TestE2E_Anthropic_DynamicPromptWithDeps(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx := ctxWithTimeout(t)
	model := newAnthropicModel(t)

	type Deps struct {
		Language string
	}

	agent := agentic.NewAgentWithDepsDynamic[*Deps](func(ctx agentic.RunContext[*Deps]) (string, error) {
		return fmt.Sprintf("You translate text. Keep translations short.\n\nTranslate to %s. Respond with ONLY the translation, no explanation.", ctx.Deps.Language), nil
	}, model,
		agentic.WithMaxTokens(50),
	)

	deps := &Deps{Language: "Spanish"}
	result, err := agent.Run(ctx, "Hello, how are you?", deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lower := strings.ToLower(result.Output)
	// Should contain Spanish words
	if !strings.Contains(lower, "hola") && !strings.Contains(lower, "cmo") && !strings.Contains(lower, "est") {
		t.Logf("Warning: expected Spanish translation, got: %s", result.Output)
	}
	t.Logf("Translation: %s", result.Output)
}

func TestE2E_Anthropic_MixedStaticDynamic(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx := ctxWithTimeout(t)
	model := newAnthropicModel(t)

	type Deps struct {
		Topic string
	}

	agent := agentic.NewAgentWithDepsDynamic[*Deps](func(ctx agentic.RunContext[*Deps]) (string, error) {
		return fmt.Sprintf("You are an expert educator.\n\nThe current topic is: %s. Stay focused on this topic.\n\nKeep responses under 3 sentences.", ctx.Deps.Topic), nil
	}, model,
		agentic.WithMaxTokens(200),
	)

	deps := &Deps{Topic: "photosynthesis"}
	result, err := agent.Run(ctx, "Explain the basic process", deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lower := strings.ToLower(result.Output)
	// Should mention something related to photosynthesis
	if !strings.Contains(lower, "light") && !strings.Contains(lower, "plant") && !strings.Contains(lower, "photo") && !strings.Contains(lower, "sun") && !strings.Contains(lower, "energy") {
		t.Errorf("expected response about photosynthesis, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
}

func TestE2E_OpenAI_MultipleSystemPrompts(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx := ctxWithTimeout(t)
	model := newOpenAIModel(t)

	agent := agentic.NewAgent("", model,
		agentic.WithSystemPrompts(
			"You always respond as a medieval knight.",
			"End every response with 'For honor and glory!'",
		),
		agentic.WithMaxTokens(100),
	)

	result, err := agent.Run(ctx, "Greet me")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lower := strings.ToLower(result.Output)
	if !strings.Contains(lower, "honor") && !strings.Contains(lower, "glory") && !strings.Contains(lower, "knight") {
		t.Logf("Warning: expected medieval knight response, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
}

// ============================================================================
// Output Validators — E2E Tests
// ============================================================================

func TestE2E_Anthropic_OutputValidator_PassesFirstTry(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx := ctxWithTimeout(t)
	model := newAnthropicModel(t)

	validatorCalled := false
	agent := agentic.NewAgent(
		"You are a helpful assistant. Respond concisely.",
		model,
		agentic.WithMaxTokens(100),
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			validatorCalled = true
			if len(output) == 0 {
				return agentic.NewValidationError("Response cannot be empty")
			}
			return nil
		}),
	)

	result, err := agent.Run(ctx, "What is 2+2?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !validatorCalled {
		t.Error("expected validator to be called")
	}
	if result.Output == "" {
		t.Error("expected non-empty output")
	}
	t.Logf("Response: %s", result.Output)
}

func TestE2E_Anthropic_OutputValidator_RetryOnce(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newAnthropicModel(t)

	attempt := 0
	agent := agentic.NewAgent(
		"You answer math questions. Always include the numeric result in your response.",
		model,
		agentic.WithMaxTokens(200),
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			attempt++
			if attempt == 1 {
				// Reject the first response to force a retry
				return agentic.NewValidationError("Please include the word 'ANSWER:' followed by the number in your response.")
			}
			return nil
		}),
		agentic.WithMaxValidationRetries(3),
	)

	result, err := agent.Run(ctx, "What is 15 * 7?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if attempt < 2 {
		t.Errorf("expected at least 2 validation attempts, got %d", attempt)
	}
	t.Logf("Response after %d attempts: %s", attempt, result.Output)
}

func TestE2E_Anthropic_OutputValidator_RequiresJSON(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newAnthropicModel(t)

	agent := agentic.NewAgent(
		`You answer questions. You MUST respond with valid JSON in the format: {"answer": "your answer here"}. Nothing else.`,
		model,
		agentic.WithMaxTokens(200),
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			trimmed := strings.TrimSpace(output)
			if !strings.HasPrefix(trimmed, "{") || !strings.HasSuffix(trimmed, "}") {
				return agentic.NewValidationError("Response must be valid JSON starting with { and ending with }. Do not include any other text.")
			}
			return nil
		}),
		agentic.WithMaxValidationRetries(3),
	)

	result, err := agent.Run(ctx, "What is the capital of France?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	trimmed := strings.TrimSpace(result.Output)
	if !strings.HasPrefix(trimmed, "{") {
		t.Errorf("expected JSON response, got: %s", result.Output)
	}
	t.Logf("JSON response: %s", result.Output)
}

func TestE2E_Anthropic_OutputValidator_MultipleValidators(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx := ctxWithTimeout(t)
	model := newAnthropicModel(t)

	v1Called := false
	v2Called := false

	agent := agentic.NewAgent(
		"You are a helpful assistant. Keep responses brief.",
		model,
		agentic.WithMaxTokens(100),
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			v1Called = true
			if len(output) == 0 {
				return agentic.NewValidationError("cannot be empty")
			}
			return nil
		}),
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			v2Called = true
			return nil
		}),
	)

	result, err := agent.Run(ctx, "Hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !v1Called || !v2Called {
		t.Error("expected both validators to be called")
	}
	if result.Output == "" {
		t.Error("expected non-empty output")
	}
}

func TestE2E_OpenAI_OutputValidator_RetryOnce(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newOpenAIModel(t)

	attempt := 0
	agent := agentic.NewAgent(
		"You answer questions concisely.",
		model,
		agentic.WithMaxTokens(200),
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			attempt++
			if attempt == 1 {
				return agentic.NewValidationError("Please start your response with 'Answer:'")
			}
			return nil
		}),
		agentic.WithMaxValidationRetries(3),
	)

	result, err := agent.Run(ctx, "What is 3+3?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if attempt < 2 {
		t.Errorf("expected at least 2 validation attempts, got %d", attempt)
	}
	t.Logf("Response after %d attempts: %s", attempt, result.Output)
}

type publicAPISmokeContextKey struct{}

type publicAPISmokeDeps struct {
	Marker string
}

type publicAPISmokeInput struct {
	Value string `json:"value" description:"The probe value"`
}

type publicAPISmokeAnswer struct {
	Value string `json:"value" validate:"required" description:"A short answer"`
}

// TestE2E_OpenAI_PublicAPISmoke is the release proof for the public API
// graph. It intentionally crosses the public binding, tool, typed-validation,
// and handoff boundaries using a real provider.
func TestE2E_OpenAI_PublicAPISmoke(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	ctx = context.WithValue(ctx, publicAPISmokeContextKey{}, "context-marker")
	model := newOpenAIModel(t)
	deps := &publicAPISmokeDeps{Marker: "dependency-marker"}

	var contextCalls atomic.Int32
	var dependencyCalls atomic.Int32
	toolAgent := agentic.NewAgentWithDeps[*publicAPISmokeDeps](
		"You are a tool verification agent. You MUST call context_probe and dependency_probe exactly once each before giving a short final answer.",
		model,
		agentic.WithMaxTokens(200),
	)
	agentic.AddTool(toolAgent, func(runCtx context.Context, input publicAPISmokeInput) (string, error) {
		if got, _ := runCtx.Value(publicAPISmokeContextKey{}).(string); got != "context-marker" {
			return "", fmt.Errorf("context marker missing: %q", got)
		}
		contextCalls.Add(1)
		return "context-ok:" + input.Value, nil
	}, agentic.AutoToolName("context_probe"), agentic.AutoToolDescription("Verify run context propagation"))
	agentic.AddToolWithDeps(toolAgent, func(runCtx agentic.RunContext[*publicAPISmokeDeps], input publicAPISmokeInput) (string, error) {
		if runCtx.Deps.Marker != deps.Marker {
			return "", fmt.Errorf("dependency marker mismatch: %q", runCtx.Deps.Marker)
		}
		dependencyCalls.Add(1)
		return runCtx.Deps.Marker + ":" + input.Value, nil
	}, agentic.AutoToolName("dependency_probe"), agentic.AutoToolDescription("Verify exact dependency propagation"))

	bound := toolAgent.Bind(deps)
	toolResult, err := bound.Run(ctx, "Run both probes with the value 'api', then report success.")
	if err != nil {
		t.Fatalf("bound tool run failed: %v", err)
	}
	if contextCalls.Load() == 0 || dependencyCalls.Load() == 0 {
		t.Fatalf("expected both tool forms to run, context=%d deps=%d; calls=%v", contextCalls.Load(), dependencyCalls.Load(), toolResult.ToolCalls)
	}

	var validationAttempts atomic.Int32
	typed := agentic.NewTypedAgentWithDeps[publicAPISmokeAnswer, *publicAPISmokeDeps](
		"Return a short structured answer using the provided output tool.",
		model,
		"The public API smoke answer",
		agentic.WithMaxTokens(200),
		agentic.WithMaxValidationRetries(2),
	).AddOutputValidatorWithDeps(
		agentic.TypedOutputValidatorWithDepsFunc[*publicAPISmokeDeps, publicAPISmokeAnswer](func(runCtx agentic.RunContext[*publicAPISmokeDeps], output publicAPISmokeAnswer) error {
			if runCtx.Deps.Marker != deps.Marker {
				return fmt.Errorf("typed validator dependency mismatch: %q", runCtx.Deps.Marker)
			}
			if validationAttempts.Add(1) == 1 {
				return agentic.NewValidationError("retry once to prove structured validation recovery")
			}
			if output.Value == "" {
				return agentic.NewValidationError("value must not be empty")
			}
			return nil
		}),
	)
	structuredResult, err := typed.Run(ctx, "Return value 'structured-ok'.", deps)
	if err != nil {
		t.Fatalf("structured validation run failed: %v", err)
	}
	if validationAttempts.Load() < 2 || structuredResult.Output.Value == "" {
		t.Fatalf("expected a successful validation retry, attempts=%d output=%+v", validationAttempts.Load(), structuredResult.Output)
	}

	var childPromptCalls atomic.Int32
	child := agentic.NewAgentWithDepsDynamic[*publicAPISmokeDeps](func(runCtx agentic.RunContext[*publicAPISmokeDeps]) (string, error) {
		childPromptCalls.Add(1)
		return "You are the delegated child. Include this marker in your short answer: " + runCtx.Deps.Marker, nil
	}, model, agentic.WithMaxTokens(100))
	parent := agentic.NewAgent(
		"You are the parent. You MUST call the delegate tool exactly once for the user's task, then return a short final answer based on its result.",
		model,
		agentic.WithMaxTokens(200),
	)
	parent.AddHandoff(agentic.NewHandoff(
		"delegate",
		"Delegate the task to the bound dependency-aware child",
		child.Bind(deps),
	))
	handoffResult, err := parent.Run(ctx, "Delegate a request to repeat the dependency marker.")
	if err != nil {
		t.Fatalf("handoff run failed: %v", err)
	}
	if childPromptCalls.Load() == 0 {
		t.Fatalf("expected bound child handoff to run; calls=%v", handoffResult.ToolCalls)
	}
}

// ============================================================================
// Combined — Dynamic Prompts + Validators
// ============================================================================

func TestE2E_Anthropic_DynamicPrompts_And_Validator(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newAnthropicModel(t)

	type Deps struct {
		Format string
	}

	agent := agentic.NewAgentWithDepsDynamic[*Deps](func(ctx agentic.RunContext[*Deps]) (string, error) {
		return fmt.Sprintf("You are a math assistant.\n\nAlways format your final answer as: %s <number>", ctx.Deps.Format), nil
	}, model,
		agentic.WithMaxTokens(200),
		agentic.WithMaxValidationRetries(3),
	).AddOutputValidator(agentic.OutputValidatorWithDepsFunc[*Deps](func(ctx agentic.RunContext[*Deps], output string) error {
		if !strings.Contains(strings.ToLower(output), strings.ToLower(ctx.Deps.Format)) {
			return agentic.NewValidationErrorf("Response must contain '%s'", ctx.Deps.Format)
		}
		return nil
	}))

	deps := &Deps{Format: "RESULT:"}
	result, err := agent.Run(ctx, "What is 25 * 4?", deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Output, "RESULT:") {
		t.Errorf("expected 'RESULT:' in output, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
}

// ============================================================================
// Streaming — Dynamic Prompts + Validators
// ============================================================================

func TestE2E_Anthropic_Stream_DynamicPrompts(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx := ctxWithTimeout(t)

	m, err := anthropic.New("claude-sonnet-4-6")
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}

	agent := agentic.NewAgent("", m,
		agentic.WithSystemPrompts(
			"You are a pirate. Always say 'Arrr'. Keep responses short (under 30 words).",
		),
		agentic.WithMaxTokens(100),
	)

	sr, err := agent.RunStream(ctx, "Greet me")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text, err := sr.Text()
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}

	lower := strings.ToLower(text)
	if !strings.Contains(lower, "arr") {
		t.Errorf("expected pirate language, got: %s", text)
	}
	t.Logf("Streamed response: %s", text)
}

func TestE2E_Anthropic_WithToolsAndValidator(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newAnthropicModel(t)

	type WeatherInput struct {
		City string `json:"city" description:"City name"`
	}
	type WeatherOutput struct {
		Temperature int    `json:"temperature"`
		Condition   string `json:"condition"`
	}

	tool, handler := agentic.MustToolPlain("get_weather", "Get current weather for a city", func(input WeatherInput) (WeatherOutput, error) {
		// Fake weather data
		return WeatherOutput{Temperature: 72, Condition: "sunny"}, nil
	})

	agent := agentic.NewAgent(
		"You are a weather assistant. When asked about weather, use the get_weather tool. Always include the temperature in your response.",
		model,
		agentic.WithMaxTokens(200),
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			if !strings.Contains(output, "72") {
				return agentic.NewValidationError("Response must include the temperature (72)")
			}
			return nil
		}),
		agentic.WithMaxValidationRetries(3),
	).AddTool(tool, handler)

	result, err := agent.Run(ctx, "What's the weather in San Francisco?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.ToolCalls) == 0 {
		t.Error("expected at least one tool call")
	}
	if !strings.Contains(result.Output, "72") {
		t.Errorf("expected temperature in output, got: %s", result.Output)
	}
	t.Logf("Response with tools + validation: %s", result.Output)
}

// ============================================================================
// Struct Tag Validation — E2E Tests
// ============================================================================

func TestE2E_Anthropic_StructTagValidation(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newAnthropicModel(t)

	type MovieReview struct {
		Title  string  `json:"title"  validate:"required,min=1"                     description:"Movie title"`
		Rating float64 `json:"rating" validate:"required,min=1,max=10"              description:"Rating from 1 to 10"`
		Genre  string  `json:"genre"  validate:"required,oneof=action comedy drama"  description:"Movie genre"`
	}

	agent := agentic.NewTypedAgent[MovieReview](
		"You are a movie critic. Review the movie as requested.",
		model,
		"Provide a movie review",
		agentic.WithMaxTokens(300),
	)

	result, err := agent.Run(ctx, "Review The Matrix")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Title == "" {
		t.Error("expected non-empty title")
	}
	if result.Output.Rating < 1 || result.Output.Rating > 10 {
		t.Errorf("expected rating between 1-10, got %f", result.Output.Rating)
	}
	validGenres := map[string]bool{"action": true, "comedy": true, "drama": true}
	if !validGenres[result.Output.Genre] {
		t.Errorf("expected genre to be one of action/comedy/drama, got %q", result.Output.Genre)
	}
	t.Logf("Movie review: %+v", result.Output)
}

func TestE2E_OpenAI_StructTagValidation(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newOpenAIModel(t)

	type CityInfo struct {
		Name       string `json:"name"       validate:"required"                       description:"City name"`
		Country    string `json:"country"    validate:"required"                       description:"Country name"`
		Population int    `json:"population" validate:"required,min=1"                 description:"Approximate population"`
		Continent  string `json:"continent"  validate:"required,oneof=Africa Asia Europe North_America South_America Oceania Antarctica" description:"Continent"`
	}

	agent := agentic.NewTypedAgent[CityInfo](
		"You provide factual information about cities.",
		model,
		"Provide city information",
		agentic.WithMaxTokens(300),
	)

	result, err := agent.Run(ctx, "Tell me about Tokyo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Name == "" {
		t.Error("expected non-empty name")
	}
	if result.Output.Population < 1 {
		t.Errorf("expected positive population, got %d", result.Output.Population)
	}
	t.Logf("City info: %+v", result.Output)
}

// ============================================================================
// Phase 1: Auto-Registration E2E Tests
// ============================================================================

// Auto-registered tool input types with tool tags

type CalculateInput struct {
	_          struct{} `tool:"Calculate the result of a mathematical expression"`
	Expression string   `json:"expression" description:"A mathematical expression like '2 + 2' or '10 * 5'"`
}

type CalculateOutput struct {
	Result float64 `json:"result"`
}

type TranslateWordInput struct {
	_    struct{} `tool:"Translate a word from English to Spanish"`
	Word string   `json:"word" description:"The English word to translate"`
}

type TranslateWordOutput struct {
	Translation string `json:"translation"`
}

func TestAutoTool_Anthropic(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	model := newAnthropicModel(t)

	// Create tool using auto-registration — no name or description needed
	tool, handler := agentic.MustAutoTool(func(input CalculateInput) (CalculateOutput, error) {
		// Simple evaluator for basic expressions
		return CalculateOutput{Result: 42}, nil
	})

	t.Logf("Auto-registered tool: name=%q, desc=%q", tool.Function.Name, tool.Function.Description)

	// Verify inferred metadata
	if tool.Function.Name != "calculate" {
		t.Errorf("expected name %q, got %q", "calculate", tool.Function.Name)
	}
	if tool.Function.Description != "Calculate the result of a mathematical expression" {
		t.Errorf("expected description from tool tag, got %q", tool.Function.Description)
	}

	agent := agentic.NewAgent(
		"You are a calculator assistant. Use the calculate tool for math questions.",
		model,
		agentic.WithMaxTokens(200),
	).AddAutoTool(tool, handler)

	result, err := agent.Run(ctx, "What is 6 times 7?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The model should have called the calculate tool
	if len(result.ToolCalls) == 0 {
		t.Fatal("expected at least one tool call")
	}
	if result.ToolCalls[0].Name != "calculate" {
		t.Errorf("expected tool call to 'calculate', got %q", result.ToolCalls[0].Name)
	}
	t.Logf("Output: %s", result.Output)
	t.Logf("Tool calls: %d", len(result.ToolCalls))
}

func TestAutoTool_OpenAI(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	model := newOpenAIModel(t)

	tool, handler := agentic.MustAutoTool(func(input TranslateWordInput) (TranslateWordOutput, error) {
		return TranslateWordOutput{Translation: "hola"}, nil
	})

	t.Logf("Auto-registered tool: name=%q, desc=%q", tool.Function.Name, tool.Function.Description)

	if tool.Function.Name != "translate_word" {
		t.Errorf("expected name %q, got %q", "translate_word", tool.Function.Name)
	}

	agent := agentic.NewAgent(
		"You are a translation assistant. Use the translate_word tool to translate words.",
		model,
		agentic.WithMaxTokens(200),
	).AddAutoTool(tool, handler)

	result, err := agent.Run(ctx, "Translate 'hello' to Spanish")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.ToolCalls) == 0 {
		t.Fatal("expected at least one tool call")
	}
	if result.ToolCalls[0].Name != "translate_word" {
		t.Errorf("expected tool call to 'translate_word', got %q", result.ToolCalls[0].Name)
	}
	t.Logf("Output: %s", result.Output)
}

func TestAutoTool_MultipleTools_Anthropic(t *testing.T) {
	skipIfNoAnthropicKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	model := newAnthropicModel(t)

	calcTool, calcHandler := agentic.MustAutoTool(func(input CalculateInput) (CalculateOutput, error) {
		return CalculateOutput{Result: 15}, nil
	})

	translateTool, translateHandler := agentic.MustAutoTool(func(input TranslateWordInput) (TranslateWordOutput, error) {
		return TranslateWordOutput{Translation: "quince"}, nil
	})

	agent := agentic.NewAgent(
		"You are a multi-purpose assistant. Use the appropriate tool for each task.",
		model,
		agentic.WithMaxTokens(300),
	).AddAutoTool(calcTool, calcHandler).AddAutoTool(translateTool, translateHandler)

	result, err := agent.Run(ctx, "What is 5 + 10?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have used the calculate tool
	found := false
	for _, tc := range result.ToolCalls {
		if tc.Name == "calculate" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected the 'calculate' tool to be called")
	}
	t.Logf("Output: %s", result.Output)
	t.Logf("Tool calls: %v", result.ToolCalls)
}

// ============================================================================
// OpenAI Responses API — E2E Tests
// ============================================================================

func TestE2E_OpenAI_Responses_BasicRun(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx := ctxWithTimeout(t)
	model := newOpenAIResponsesModel(t)

	agent := agentic.NewAgent(
		"You are a concise assistant. Keep answers to one sentence.",
		model,
		agentic.WithMaxTokens(100),
	)

	result, err := agent.Run(ctx, "What is 2+2?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output == "" {
		t.Error("expected non-empty output")
	}
	if !strings.Contains(result.Output, "4") {
		t.Errorf("expected '4' in response, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
}

func TestE2E_OpenAI_Responses_Streaming(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx := ctxWithTimeout(t)
	model := newOpenAIResponsesModel(t)

	agent := agentic.NewAgent(
		"You are a pirate. Always say 'Arrr'. Keep responses under 30 words.",
		model,
		agentic.WithMaxTokens(100),
	)

	sr, err := agent.RunStream(ctx, "Greet me")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text, err := sr.Text()
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}

	lower := strings.ToLower(text)
	if !strings.Contains(lower, "arr") {
		t.Errorf("expected pirate language, got: %s", text)
	}
	t.Logf("Streamed response: %s", text)
}

func TestE2E_OpenAI_Responses_WithTools(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newOpenAIResponsesModel(t)

	type WeatherInput struct {
		City string `json:"city" description:"City name"`
	}
	type WeatherOutput struct {
		Temperature int    `json:"temperature"`
		Condition   string `json:"condition"`
	}

	tool, handler := agentic.MustToolPlain("get_weather", "Get current weather for a city", func(input WeatherInput) (WeatherOutput, error) {
		return WeatherOutput{Temperature: 72, Condition: "sunny"}, nil
	})

	agent := agentic.NewAgent(
		"You are a weather assistant. When asked about weather, use the get_weather tool. Always include the temperature in your response.",
		model,
		agentic.WithMaxTokens(200),
	).AddTool(tool, handler)

	result, err := agent.Run(ctx, "What's the weather in San Francisco?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.ToolCalls) == 0 {
		t.Error("expected at least one tool call")
	}
	if !strings.Contains(result.Output, "72") {
		t.Errorf("expected temperature in output, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
	t.Logf("Tool calls: %d", len(result.ToolCalls))
}

func TestE2E_OpenAI_Responses_StructuredOutput(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newOpenAIResponsesModel(t)

	type CityInfo struct {
		Name       string `json:"name"       validate:"required"   description:"City name"`
		Country    string `json:"country"    validate:"required"   description:"Country name"`
		Population int    `json:"population" validate:"required,min=1" description:"Approximate population"`
	}

	agent := agentic.NewTypedAgent[CityInfo](
		"You provide factual information about cities.",
		model,
		"Provide city information",
		agentic.WithMaxTokens(300),
	)

	result, err := agent.Run(ctx, "Tell me about Tokyo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output.Name == "" {
		t.Error("expected non-empty name")
	}
	if result.Output.Population < 1 {
		t.Errorf("expected positive population, got %d", result.Output.Population)
	}
	t.Logf("City info: %+v", result.Output)
}

func TestE2E_OpenAI_Responses_StreamWithTools(t *testing.T) {
	skipIfNoOpenAIKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newOpenAIResponsesModel(t)

	type LookupInput struct {
		Query string `json:"query" description:"Search query"`
	}

	tool, handler := agentic.MustToolPlain("lookup", "Look up information", func(input LookupInput) (string, error) {
		return "The population of Paris is approximately 2.1 million.", nil
	})

	agent := agentic.NewAgent(
		"You are a research assistant. Use the lookup tool to answer questions. Include the facts from the tool in your response.",
		model,
		agentic.WithMaxTokens(200),
	).AddTool(tool, handler)

	sr, err := agent.RunStream(ctx, "What is the population of Paris?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text, err := sr.Text()
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if text == "" {
		t.Error("expected non-empty streamed text")
	}
	t.Logf("Streamed response: %s", text)
}

// ============================================================================
// Together AI — E2E Tests
// ============================================================================

func TestE2E_Together_BasicRun(t *testing.T) {
	skipIfNoTogetherKey(t)
	ctx := ctxWithTimeout(t)
	model := newTogetherModel(t)

	agent := agentic.NewAgent(
		"You are a concise assistant. Keep answers to one sentence.",
		model,
		agentic.WithMaxTokens(100),
	)

	result, err := agent.Run(ctx, "What is 2+2?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output == "" {
		t.Error("expected non-empty output")
	}
	if !strings.Contains(result.Output, "4") {
		t.Errorf("expected '4' in response, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
}

func TestE2E_Together_Streaming(t *testing.T) {
	skipIfNoTogetherKey(t)
	ctx := ctxWithTimeout(t)
	model := newTogetherModel(t)

	agent := agentic.NewAgent(
		"You are a pirate. Always say 'Arrr'. Keep responses under 30 words.",
		model,
		agentic.WithMaxTokens(100),
	)

	sr, err := agent.RunStream(ctx, "Greet me")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text, err := sr.Text()
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if text == "" {
		t.Error("expected non-empty streamed text")
	}
	t.Logf("Streamed response: %s", text)
}

func TestE2E_Together_WithTools(t *testing.T) {
	skipIfNoTogetherKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	model := newTogetherModel(t)

	type WeatherInput struct {
		City string `json:"city" description:"City name"`
	}
	type WeatherOutput struct {
		Temperature int    `json:"temperature"`
		Condition   string `json:"condition"`
	}

	tool, handler := agentic.MustToolPlain("get_weather", "Get current weather for a city", func(input WeatherInput) (WeatherOutput, error) {
		return WeatherOutput{Temperature: 72, Condition: "sunny"}, nil
	})

	agent := agentic.NewAgent(
		"You are a weather assistant. When asked about weather, use the get_weather tool. Always include the temperature in your response.",
		model,
		agentic.WithMaxTokens(200),
	).AddTool(tool, handler)

	result, err := agent.Run(ctx, "What's the weather in San Francisco?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.ToolCalls) == 0 {
		t.Error("expected at least one tool call")
	}
	if !strings.Contains(result.Output, "72") {
		t.Errorf("expected temperature in output, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
	t.Logf("Tool calls: %d", len(result.ToolCalls))
}

// ============================================================================
// Grok (xAI) — E2E Tests
// ============================================================================

func TestE2E_Grok_BasicRun(t *testing.T) {
	skipIfNoGrokKey(t)
	ctx := ctxWithTimeout(t)

	m, err := grok.New("grok-3-mini")
	if err != nil {
		t.Fatalf("failed to create grok model: %v", err)
	}

	agent := agentic.NewAgent(
		"You are a concise assistant. Keep answers to one sentence.",
		m,
		agentic.WithMaxTokens(100),
	)

	result, err := agent.Run(ctx, "What is 2+2?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output == "" {
		t.Error("expected non-empty output")
	}
	if !strings.Contains(result.Output, "4") {
		t.Errorf("expected '4' in response, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
}

func TestE2E_Grok_Streaming(t *testing.T) {
	skipIfNoGrokKey(t)
	ctx := ctxWithTimeout(t)

	m, err := grok.New("grok-3-mini")
	if err != nil {
		t.Fatalf("failed to create grok model: %v", err)
	}

	agent := agentic.NewAgent(
		"You are a pirate. Always say 'Arrr'. Keep responses under 30 words.",
		m,
		agentic.WithMaxTokens(100),
	)

	sr, err := agent.RunStream(ctx, "Greet me")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text, err := sr.Text()
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if text == "" {
		t.Error("expected non-empty streamed text")
	}
	t.Logf("Streamed response: %s", text)
}

func TestE2E_Grok_WithTools(t *testing.T) {
	skipIfNoGrokKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	m, err := grok.New("grok-3-mini")
	if err != nil {
		t.Fatalf("failed to create grok model: %v", err)
	}

	type WeatherInput struct {
		City string `json:"city" description:"City name"`
	}
	type WeatherOutput struct {
		Temperature int    `json:"temperature"`
		Condition   string `json:"condition"`
	}

	tool, handler := agentic.MustToolPlain("get_weather", "Get current weather for a city", func(input WeatherInput) (WeatherOutput, error) {
		return WeatherOutput{Temperature: 72, Condition: "sunny"}, nil
	})

	agent := agentic.NewAgent(
		"You are a weather assistant. When asked about weather, use the get_weather tool. Always include the temperature in your response.",
		m,
		agentic.WithMaxTokens(200),
	).AddTool(tool, handler)

	result, err := agent.Run(ctx, "What's the weather in San Francisco?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.ToolCalls) == 0 {
		t.Error("expected at least one tool call")
	}
	if !strings.Contains(result.Output, "72") {
		t.Errorf("expected temperature in output, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
	t.Logf("Tool calls: %d", len(result.ToolCalls))
}

// ============================================================================
// OpenRouter — E2E Tests
// ============================================================================

func TestE2E_OpenRouter_BasicRun(t *testing.T) {
	skipIfNoOpenRouterKey(t)
	ctx := ctxWithTimeout(t)

	m, err := openrouter.New("openai/gpt-4o-mini")
	if err != nil {
		t.Fatalf("failed to create openrouter model: %v", err)
	}

	agent := agentic.NewAgent(
		"You are a concise assistant. Keep answers to one sentence.",
		m,
		agentic.WithMaxTokens(100),
	)

	result, err := agent.Run(ctx, "What is 2+2?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output == "" {
		t.Error("expected non-empty output")
	}
	if !strings.Contains(result.Output, "4") {
		t.Errorf("expected '4' in response, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
}

func TestE2E_OpenRouter_Streaming(t *testing.T) {
	skipIfNoOpenRouterKey(t)
	ctx := ctxWithTimeout(t)

	m, err := openrouter.New("openai/gpt-4o-mini")
	if err != nil {
		t.Fatalf("failed to create openrouter model: %v", err)
	}

	agent := agentic.NewAgent(
		"You are a pirate. Always say 'Arrr'. Keep responses under 30 words.",
		m,
		agentic.WithMaxTokens(100),
	)

	sr, err := agent.RunStream(ctx, "Greet me")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text, err := sr.Text()
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if text == "" {
		t.Error("expected non-empty streamed text")
	}
	t.Logf("Streamed response: %s", text)
}

func TestE2E_OpenRouter_WithTools(t *testing.T) {
	skipIfNoOpenRouterKey(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	m, err := openrouter.New("openai/gpt-4o-mini")
	if err != nil {
		t.Fatalf("failed to create openrouter model: %v", err)
	}

	type ORWeatherInput struct {
		City string `json:"city" description:"City name"`
	}
	type ORWeatherOutput struct {
		Temperature int    `json:"temperature"`
		Condition   string `json:"condition"`
	}

	tool, handler := agentic.MustToolPlain("get_weather", "Get current weather for a city", func(input ORWeatherInput) (ORWeatherOutput, error) {
		return ORWeatherOutput{Temperature: 72, Condition: "sunny"}, nil
	})

	agent := agentic.NewAgent(
		"You are a weather assistant. When asked about weather, use the get_weather tool. Always include the temperature in your response.",
		m,
		agentic.WithMaxTokens(200),
	).AddTool(tool, handler)

	result, err := agent.Run(ctx, "What's the weather in San Francisco?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.ToolCalls) == 0 {
		t.Error("expected at least one tool call")
	}
	if !strings.Contains(result.Output, "72") {
		t.Errorf("expected temperature in output, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
	t.Logf("Tool calls: %d", len(result.ToolCalls))
}

// ============================================================================
// Ollama (local) — E2E Tests
// ============================================================================

func TestE2E_Ollama_BasicRun(t *testing.T) {
	skipIfNoOllama(t)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	m, err := ollama.New("qwen3:8b")
	if err != nil {
		t.Fatalf("failed to create ollama model: %v", err)
	}

	agent := agentic.NewAgent(
		"You are a concise assistant. Keep answers to one sentence. Do not think out loud.",
		m,
		agentic.WithMaxTokens(200),
	)

	result, err := agent.Run(ctx, "What is 2+2? Reply with just the answer.")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Output == "" {
		t.Error("expected non-empty output")
	}
	if !strings.Contains(result.Output, "4") {
		t.Errorf("expected '4' in response, got: %s", result.Output)
	}
	t.Logf("Response: %s", result.Output)
}

func TestE2E_Ollama_Streaming(t *testing.T) {
	skipIfNoOllama(t)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	m, err := ollama.New("qwen3:8b")
	if err != nil {
		t.Fatalf("failed to create ollama model: %v", err)
	}

	agent := agentic.NewAgent(
		"You are a concise assistant. Keep responses under 30 words. Do not think out loud.",
		m,
		agentic.WithMaxTokens(200),
	)

	sr, err := agent.RunStream(ctx, "Say hello in one sentence.")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text, err := sr.Text()
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if text == "" {
		t.Error("expected non-empty streamed text")
	}
	t.Logf("Streamed response: %s", text)
}

func TestE2E_Ollama_WithTools(t *testing.T) {
	skipIfNoOllama(t)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	m, err := ollama.New("qwen3:8b")
	if err != nil {
		t.Fatalf("failed to create ollama model: %v", err)
	}

	type OllamaWeatherInput struct {
		City string `json:"city" description:"City name"`
	}
	type OllamaWeatherOutput struct {
		Temperature int    `json:"temperature"`
		Condition   string `json:"condition"`
	}

	tool, handler := agentic.MustToolPlain("get_weather", "Get current weather for a city", func(input OllamaWeatherInput) (OllamaWeatherOutput, error) {
		return OllamaWeatherOutput{Temperature: 72, Condition: "sunny"}, nil
	})

	agent := agentic.NewAgent(
		"You are a weather assistant. When asked about weather, use the get_weather tool. Always include the temperature number in your response. Do not think out loud.",
		m,
		agentic.WithMaxTokens(300),
	).AddTool(tool, handler)

	result, err := agent.Run(ctx, "What's the weather in San Francisco?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.ToolCalls) == 0 {
		t.Error("expected at least one tool call")
	}
	t.Logf("Response: %s", result.Output)
	t.Logf("Tool calls: %d", len(result.ToolCalls))
}
