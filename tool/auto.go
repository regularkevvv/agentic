package tool

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"unicode"

	"github.com/regularkevvv/agentic/internal/core"
)

// AutoToolOption configures auto-registration behavior.
type AutoToolOption func(*autoToolConfig)

type autoToolConfig struct {
	name        string // override inferred name
	description string // override inferred description
}

// WithName overrides the auto-inferred tool name.
func WithName(name string) AutoToolOption {
	return func(c *autoToolConfig) {
		c.name = name
	}
}

// WithDescription overrides the auto-inferred description.
func WithDescription(desc string) AutoToolOption {
	return func(c *autoToolConfig) {
		c.description = desc
	}
}

// Auto creates a plain tool by reflecting on the handler function's input type.
// Name is inferred from TInput type name (CamelCase -> snake_case, "Input" suffix stripped).
// Description is inferred from a `tool` struct tag on a `struct{}` field of TInput.
func Auto[TInput any, TOutput any](
	handler func(input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (core.Tool, core.ToolHandler, error) {
	cfg := autoToolConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	name := cfg.name
	if name == "" {
		name = inferToolName[TInput]()
	}

	description := cfg.description
	if description == "" {
		description = inferDescription[TInput]()
	}

	return ToolPlain(name, description, handler)
}

// AutoWithContext creates a context-aware tool with inferred metadata.
func AutoWithContext[TInput any, TOutput any](
	handler func(ctx context.Context, input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (core.Tool, core.ToolHandler, error) {
	cfg := autoToolConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	name := cfg.name
	if name == "" {
		name = inferToolName[TInput]()
	}
	description := cfg.description
	if description == "" {
		description = inferDescription[TInput]()
	}
	return ToolWithContext(name, description, handler)
}

// MustAutoWithContext is like AutoWithContext but panics on error.
func MustAutoWithContext[TInput any, TOutput any](
	handler func(ctx context.Context, input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (core.Tool, core.ToolHandler) {
	tool, h, err := AutoWithContext(handler, opts...)
	if err != nil {
		panic(fmt.Sprintf("MustAutoWithContext: %v", err))
	}
	return tool, h
}

// MustAuto is like Auto but panics on error.
func MustAuto[TInput any, TOutput any](
	handler func(input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (core.Tool, core.ToolHandler) {
	tool, h, err := Auto(handler, opts...)
	if err != nil {
		panic(fmt.Sprintf("MustAuto: %v", err))
	}
	return tool, h
}

// AutoWithDeps creates a deps-aware tool with auto-inferred metadata.
func AutoWithDeps[TInput any, TOutput any, DepsT any](
	handler func(ctx core.RunContext[DepsT], input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (core.Tool, core.ToolHandler, error) {
	cfg := autoToolConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	name := cfg.name
	if name == "" {
		name = inferToolName[TInput]()
	}

	description := cfg.description
	if description == "" {
		description = inferDescription[TInput]()
	}

	return ToolWithDeps[TInput, TOutput, DepsT](name, description, handler)
}

// MustAutoWithDeps is like AutoWithDeps but panics on error.
func MustAutoWithDeps[TInput any, TOutput any, DepsT any](
	handler func(ctx core.RunContext[DepsT], input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (core.Tool, core.ToolHandler) {
	tool, h, err := AutoWithDeps[TInput, TOutput, DepsT](handler, opts...)
	if err != nil {
		panic(fmt.Sprintf("MustAutoWithDeps: %v", err))
	}
	return tool, h
}

// inferToolName infers a snake_case tool name from the TInput type name.
// It strips common suffixes ("Input", "Params", "Args", "Request") and
// converts CamelCase to snake_case.
func inferToolName[T any]() string {
	t := reflect.TypeOf((*T)(nil)).Elem()
	name := t.Name()
	if name == "" {
		return ""
	}

	// Strip common suffixes
	for _, suffix := range []string{"Input", "Params", "Args", "Request"} {
		if strings.HasSuffix(name, suffix) && len(name) > len(suffix) {
			name = strings.TrimSuffix(name, suffix)
			break
		}
	}

	return camelToSnake(name)
}

// inferDescription extracts a tool description from a `tool` struct tag on TInput.
func inferDescription[T any]() string {
	t := reflect.TypeOf((*T)(nil)).Elem()
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return ""
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if tag := field.Tag.Get("tool"); tag != "" {
			return tag
		}
	}
	return ""
}

// camelToSnake converts a CamelCase string to snake_case.
func camelToSnake(s string) string {
	if s == "" {
		return ""
	}

	var buf strings.Builder
	runes := []rune(s)

	for i, r := range runes {
		if unicode.IsUpper(r) {
			if i > 0 {
				prev := runes[i-1]
				if unicode.IsLower(prev) || unicode.IsDigit(prev) {
					buf.WriteRune('_')
				} else if unicode.IsUpper(prev) {
					if i+1 < len(runes) && unicode.IsLower(runes[i+1]) {
						buf.WriteRune('_')
					}
				}
			}
			buf.WriteRune(unicode.ToLower(r))
		} else {
			buf.WriteRune(r)
		}
	}

	return buf.String()
}
