package agentic

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/go-playground/validator/v10"
)

type textValidatorAdapter func(context.Context, dependencyEnvelope, string) error

// structValidator is a package-level validator instance (goroutine-safe).
// It uses JSON field names in error messages so the LLM sees the same names
// it used when generating the output.
var structValidator *validator.Validate

func init() {
	structValidator = validator.New()
	// Use json tag names in error messages so they match what the LLM sees.
	structValidator.RegisterTagNameFunc(func(fld reflect.StructField) string {
		name := strings.SplitN(fld.Tag.Get("json"), ",", 2)[0]
		if name == "-" || name == "" {
			return fld.Name
		}
		return name
	})
}

// ValidateStruct validates a struct using `validate` struct tags.
// Returns a *ValidationError with human-readable messages if validation fails,
// or nil if validation passes (or the value has no validate tags).
//
// This is used automatically by ToolOutputSpec when parsing structured output,
// but can also be called directly.
func ValidateStruct(v any) error {
	if v == nil {
		return nil
	}
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil
	}

	err := structValidator.Struct(v)
	if err == nil {
		return nil
	}

	var validationErrors validator.ValidationErrors
	if !errors.As(err, &validationErrors) {
		return err
	}

	return &ValidationError{
		Message: formatValidationErrors(validationErrors),
	}
}

// formatValidationErrors converts validator.ValidationErrors into a clear,
// actionable message for the LLM.
func formatValidationErrors(errs validator.ValidationErrors) string {
	messages := make([]string, 0, len(errs))
	for _, fe := range errs {
		messages = append(messages, formatFieldError(fe))
	}
	return "Validation failed:\n" + strings.Join(messages, "\n")
}

// formatFieldError produces a human-readable message for a single field error.
func formatFieldError(fe validator.FieldError) string {
	field := fe.Field()
	switch fe.Tag() {
	case "required":
		return fmt.Sprintf("- %s is required", field)
	case "min":
		if fe.Kind() == reflect.String {
			return fmt.Sprintf("- %s must be at least %s characters long", field, fe.Param())
		}
		return fmt.Sprintf("- %s must be at least %s", field, fe.Param())
	case "max":
		if fe.Kind() == reflect.String {
			return fmt.Sprintf("- %s must be at most %s characters long", field, fe.Param())
		}
		return fmt.Sprintf("- %s must be at most %s", field, fe.Param())
	case "gt":
		return fmt.Sprintf("- %s must be greater than %s", field, fe.Param())
	case "gte":
		return fmt.Sprintf("- %s must be greater than or equal to %s", field, fe.Param())
	case "lt":
		return fmt.Sprintf("- %s must be less than %s", field, fe.Param())
	case "lte":
		return fmt.Sprintf("- %s must be less than or equal to %s", field, fe.Param())
	case "oneof":
		return fmt.Sprintf("- %s must be one of [%s]", field, fe.Param())
	case "len":
		return fmt.Sprintf("- %s must have exactly %s elements", field, fe.Param())
	case "email":
		return fmt.Sprintf("- %s must be a valid email address", field)
	case "url":
		return fmt.Sprintf("- %s must be a valid URL", field)
	case "contains":
		return fmt.Sprintf("- %s must contain '%s'", field, fe.Param())
	default:
		return fmt.Sprintf("- %s failed '%s' validation", field, fe.Tag())
	}
}

// OutputValidator validates agent text output and can request retries.
// When validation fails, the error message is sent back to the model
// and the agent re-enters the loop (up to maxValidationRetries).
//
// For structured output, prefer using `validate` struct tags instead —
// they are applied automatically. Use OutputValidator for custom logic
// on the raw text response.
type OutputValidator interface {
	Validate(ctx context.Context, output string) error
}

// OutputValidatorFunc is a function adapter for OutputValidator.
type OutputValidatorFunc func(ctx context.Context, output string) error

// Validate implements OutputValidator.
func (f OutputValidatorFunc) Validate(ctx context.Context, output string) error {
	return f(ctx, output)
}

// OutputValidatorWithDeps validates text output with the exact run dependency.
type OutputValidatorWithDeps[D any] interface {
	Validate(RunContext[D], string) error
}

// OutputValidatorWithDepsFunc adapts a dependency-aware validation function.
type OutputValidatorWithDepsFunc[D any] func(RunContext[D], string) error

func (f OutputValidatorWithDepsFunc[D]) Validate(ctx RunContext[D], output string) error {
	return f(ctx, output)
}

// TypedOutputValidator validates typed structured output.
// Use this for programmatic validation that goes beyond what struct tags can express.
type TypedOutputValidator[O any] interface {
	ValidateTyped(context.Context, O) error
}

// TypedOutputValidatorFunc is a function adapter for TypedOutputValidator.
type TypedOutputValidatorFunc[O any] func(context.Context, O) error

// ValidateTyped implements TypedOutputValidator.
func (f TypedOutputValidatorFunc[O]) ValidateTyped(ctx context.Context, output O) error {
	return f(ctx, output)
}

// TypedOutputValidatorWithDeps validates structured output with exact D and O.
type TypedOutputValidatorWithDeps[D, O any] interface {
	ValidateTyped(RunContext[D], O) error
}

// TypedOutputValidatorWithDepsFunc adapts a dependency-aware typed validator.
type TypedOutputValidatorWithDepsFunc[D, O any] func(RunContext[D], O) error

func (f TypedOutputValidatorWithDepsFunc[D, O]) ValidateTyped(ctx RunContext[D], output O) error {
	return f(ctx, output)
}

// ValidationError signals output validation failed and the model should retry.
// Return this from a validator to have the agent send the message back to the
// model and request a new response.
type ValidationError struct {
	Message string
}

func (e *ValidationError) Error() string {
	return e.Message
}

// NewValidationError creates a ValidationError with the given message.
func NewValidationError(msg string) *ValidationError {
	return &ValidationError{Message: msg}
}

// NewValidationErrorf creates a ValidationError with a formatted message.
func NewValidationErrorf(format string, args ...interface{}) *ValidationError {
	return &ValidationError{Message: fmt.Sprintf(format, args...)}
}

// IsValidationError checks if an error is a ValidationError.
func IsValidationError(err error) bool {
	var validationErr *ValidationError
	return errors.As(err, &validationErr)
}
