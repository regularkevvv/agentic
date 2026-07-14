package agentic_test

import (
	"context"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/test"
)

func TestValidatorAlwaysPasses(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "Hello!"})

	agent := agentic.NewAgent("test", model,
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			return nil // always pass
		}),
	)

	result, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "Hello!" {
		t.Errorf("expected %q, got %q", "Hello!", result.Output)
	}
	if model.CallCount() != 1 {
		t.Errorf("expected 1 call, got %d", model.CallCount())
	}
}

func TestValidatorRejectsFirstAttemptPassesSecond(t *testing.T) {
	callCount := 0
	model := test.NewTestModel(
		test.ModelResponse{Text: "bad response"},         // first attempt
		test.ModelResponse{Text: "good answer response"}, // second attempt after validation failure
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			callCount++
			if !strings.Contains(output, "good") {
				return agentic.NewValidationError("Response must contain 'good'")
			}
			return nil
		}),
	)

	result, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "good answer response" {
		t.Errorf("expected %q, got %q", "good answer response", result.Output)
	}
	if model.CallCount() != 2 {
		t.Errorf("expected 2 model calls, got %d", model.CallCount())
	}
	if callCount != 2 {
		t.Errorf("expected validator called 2 times, got %d", callCount)
	}

	// Verify the validation error was sent back to the model
	calls := model.Calls()
	lastCallMsgs := calls[1].Messages
	// Find the user message with validation error
	found := false
	for _, msg := range lastCallMsgs {
		if msg.Role == agentic.RoleUser && strings.Contains(msg.GetTextContent(), "Output validation error") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected validation error message sent back to model")
	}
}

func TestMultipleValidatorsAllMustPass(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "HELLO WORLD"})

	v1Called := false
	v2Called := false

	agent := agentic.NewAgent("test", model,
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			v1Called = true
			return nil
		}),
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			v2Called = true
			return nil
		}),
	)

	_, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !v1Called || !v2Called {
		t.Error("expected both validators to be called")
	}
}

func TestMultipleValidatorsFirstFails(t *testing.T) {
	model := test.NewTestModel(
		test.ModelResponse{Text: "bad"},
		test.ModelResponse{Text: "good"},
	)

	v2CallCount := 0

	agent := agentic.NewAgent("test", model,
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			if output == "bad" {
				return agentic.NewValidationError("rejected")
			}
			return nil
		}),
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			v2CallCount++
			return nil
		}),
	)

	result, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "good" {
		t.Errorf("expected %q, got %q", "good", result.Output)
	}
	// Second validator should only be called for the "good" response
	if v2CallCount != 1 {
		t.Errorf("expected second validator called 1 time, got %d", v2CallCount)
	}
}

func TestMaxValidationRetriesExceeded(t *testing.T) {
	// Model always returns "bad" — validator should fail up to maxValidationRetries
	model := test.NewTestModel(
		test.ModelResponse{Text: "bad"},
	)

	agent := agentic.NewAgent("test", model,
		agentic.WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			return agentic.NewValidationError("always fails")
		}),
		agentic.WithMaxValidationRetries(2),
	)

	_, err := agent.Run(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected error after max validation retries")
	}
	if !strings.Contains(err.Error(), "output validation failed after 2 retries") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidatorWithInterface(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "response"})

	validator := agentic.OutputValidatorFunc(func(ctx context.Context, output string) error {
		return nil
	})

	agent := agentic.NewAgent("test", model,
		agentic.WithOutputValidator(validator),
	)

	result, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "response" {
		t.Errorf("expected %q, got %q", "response", result.Output)
	}
}

func TestValidationErrorHelpers(t *testing.T) {
	err := agentic.NewValidationError("test error")
	if err.Message != "test error" {
		t.Errorf("expected %q, got %q", "test error", err.Message)
	}
	if err.Error() != "test error" {
		t.Errorf("expected Error() %q, got %q", "test error", err.Error())
	}

	err2 := agentic.NewValidationErrorf("error %d", 42)
	if err2.Message != "error 42" {
		t.Errorf("expected %q, got %q", "error 42", err2.Message)
	}

	if !agentic.IsValidationError(err) {
		t.Error("expected IsValidationError to return true")
	}
	if agentic.IsValidationError(nil) {
		t.Error("expected IsValidationError(nil) to return false")
	}
}

func TestValidatorWithDeps(t *testing.T) {
	type MyDeps struct {
		MinLength int
	}

	model := test.NewTestModel(
		test.ModelResponse{Text: "hi"},
		test.ModelResponse{Text: "hello world, this is a longer response"},
	)

	agent := agentic.NewAgentWithDeps[*MyDeps]("test", model).AddOutputValidator(
		agentic.OutputValidatorWithDepsFunc[*MyDeps](func(ctx agentic.RunContext[*MyDeps], output string) error {
			if len(output) < ctx.Deps.MinLength {
				return agentic.NewValidationErrorf("response too short (min %d chars)", ctx.Deps.MinLength)
			}
			return nil
		}),
	)

	deps := &MyDeps{MinLength: 10}
	result, err := agent.Run(context.Background(), "say something", deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "hello world, this is a longer response" {
		t.Errorf("unexpected output: %q", result.Output)
	}
}

func TestNoValidatorsDoesNothing(t *testing.T) {
	model := test.NewTestModel(test.ModelResponse{Text: "ok"})
	agent := agentic.NewAgent("test", model)

	result, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Output != "ok" {
		t.Errorf("expected %q, got %q", "ok", result.Output)
	}
}

// ============================================================================
// Struct tag validation (ValidateStruct) tests
// ============================================================================

func TestValidateStructRequired(t *testing.T) {
	type Input struct {
		Name string `json:"name" validate:"required"`
	}
	err := agentic.ValidateStruct(Input{Name: ""})
	if err == nil {
		t.Fatal("expected validation error for empty required field")
	}
	if !strings.Contains(err.Error(), "name is required") {
		t.Errorf("expected 'name is required', got: %s", err.Error())
	}

	// Valid case
	err = agentic.ValidateStruct(Input{Name: "Alice"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateStructMinMax(t *testing.T) {
	type Input struct {
		Rating float64 `json:"rating" validate:"required,min=1,max=10"`
	}

	// Too low
	err := agentic.ValidateStruct(Input{Rating: 0})
	if err == nil {
		t.Fatal("expected error for rating=0")
	}
	if !strings.Contains(err.Error(), "rating") {
		t.Errorf("expected error about rating, got: %s", err.Error())
	}

	// Too high
	err = agentic.ValidateStruct(Input{Rating: 15})
	if err == nil {
		t.Fatal("expected error for rating=15")
	}

	// Valid
	err = agentic.ValidateStruct(Input{Rating: 7.5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateStructOneof(t *testing.T) {
	type Input struct {
		Genre string `json:"genre" validate:"required,oneof=action comedy drama"`
	}

	err := agentic.ValidateStruct(Input{Genre: "thriller"})
	if err == nil {
		t.Fatal("expected error for invalid genre")
	}
	if !strings.Contains(err.Error(), "must be one of") {
		t.Errorf("expected 'must be one of' error, got: %s", err.Error())
	}

	err = agentic.ValidateStruct(Input{Genre: "comedy"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateStructMultipleErrors(t *testing.T) {
	type Input struct {
		Title  string  `json:"title" validate:"required"`
		Rating float64 `json:"rating" validate:"required,min=1,max=10"`
	}

	err := agentic.ValidateStruct(Input{Title: "", Rating: 0})
	if err == nil {
		t.Fatal("expected validation errors")
	}
	errMsg := err.Error()
	if !strings.Contains(errMsg, "title") {
		t.Errorf("expected error about title, got: %s", errMsg)
	}
	if !strings.Contains(errMsg, "rating") {
		t.Errorf("expected error about rating, got: %s", errMsg)
	}
}

func TestValidateStructNoTags(t *testing.T) {
	type Input struct {
		Name string `json:"name"`
	}
	// No validate tags — should pass even with zero values
	err := agentic.ValidateStruct(Input{Name: ""})
	if err != nil {
		t.Fatalf("expected no error for struct without validate tags, got: %v", err)
	}
}

func TestValidateStructNilAndNonStruct(t *testing.T) {
	// nil should pass
	if err := agentic.ValidateStruct(nil); err != nil {
		t.Fatalf("unexpected error for nil: %v", err)
	}

	// Non-struct should pass
	if err := agentic.ValidateStruct("hello"); err != nil {
		t.Fatalf("unexpected error for string: %v", err)
	}
}

func TestValidateStructReturnsValidationError(t *testing.T) {
	type Input struct {
		Name string `json:"name" validate:"required"`
	}
	err := agentic.ValidateStruct(Input{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !agentic.IsValidationError(err) {
		t.Errorf("expected ValidationError, got %T", err)
	}
}

func TestValidateStructUsesJSONFieldNames(t *testing.T) {
	type Input struct {
		FirstName string `json:"first_name" validate:"required"`
	}
	err := agentic.ValidateStruct(Input{})
	if err == nil {
		t.Fatal("expected error")
	}
	// Should use json field name "first_name", not Go name "FirstName"
	if !strings.Contains(err.Error(), "first_name") {
		t.Errorf("expected JSON field name 'first_name' in error, got: %s", err.Error())
	}
}
