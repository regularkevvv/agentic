package tool

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestRegistryValidatesArgumentsBeforeHandler(t *testing.T) {
	type input struct {
		Query    string `json:"query" required:"true" minLength:"3" pattern:"^[a-z]+$"`
		Limit    int    `json:"limit" minimum:"1" maximum:"10"`
		Category string `json:"category" enum:"news,blog"`
	}
	var calls atomic.Int32
	definition, handler := MustToolPlain("search", "search tool", func(input input) (string, error) {
		calls.Add(1)
		return input.Query, nil
	})
	registry := NewRegistry()
	if err := registry.Register(definition, handler); err != nil {
		t.Fatalf("Register: %v", err)
	}

	invalid := []map[string]interface{}{
		{"limit": float64(1), "category": "news"},
		{"query": "UP", "limit": float64(1), "category": "news"},
		{"query": "valid", "limit": float64(0), "category": "news"},
		{"query": "valid", "limit": float64(1), "category": "other"},
	}
	for i, args := range invalid {
		result, err := registry.Execute(context.Background(), core.ToolUse{ID: "bad", Name: "search", Input: args}, nil)
		if err != nil {
			t.Fatalf("case %d Execute: %v", i, err)
		}
		var retry *core.ModelRetry
		if !result.IsError || !errors.As(result.Error, &retry) {
			t.Fatalf("case %d expected model-retry validation result, got %#v", i, result)
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("handler ran for invalid arguments %d times", calls.Load())
	}

	result, err := registry.Execute(context.Background(), core.ToolUse{
		ID: "good", Name: "search", Input: map[string]interface{}{
			"query": "valid", "limit": float64(2), "category": "blog",
		},
	}, nil)
	if err != nil || result.IsError || result.Content != "valid" || calls.Load() != 1 {
		t.Fatalf("valid arguments failed: result=%#v err=%v calls=%d", result, err, calls.Load())
	}
}

func TestRegistryExecuteBatchRunsConcurrentlyAndPreservesOrder(t *testing.T) {
	type input struct {
		Value string `json:"value"`
	}
	entered := make(chan string, 2)
	release := make(chan struct{})
	makeHandler := func(name string) (core.Tool, core.ToolHandler) {
		return MustToolWithContext(name, name+" tool", func(ctx context.Context, input input) (string, error) {
			entered <- name
			select {
			case <-release:
				return name, nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		})
	}
	first, firstHandler := makeHandler("first")
	second, secondHandler := makeHandler("second")
	registry := NewRegistry()
	if err := registry.Register(first, firstHandler); err != nil {
		t.Fatalf("register first: %v", err)
	}
	if err := registry.Register(second, secondHandler); err != nil {
		t.Fatalf("register second: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	done := make(chan []core.ToolExecutionResult, 1)
	errs := make(chan error, 1)
	go func() {
		results, err := registry.ExecuteBatch(ctx, []core.ToolUse{
			{ID: "1", Name: "first", Input: map[string]interface{}{"value": "a"}},
			{ID: "2", Name: "second", Input: map[string]interface{}{"value": "b"}},
		}, nil)
		if err != nil {
			errs <- err
			return
		}
		done <- results
	}()

	seen := map[string]bool{}
	for len(seen) < 2 {
		select {
		case name := <-entered:
			seen[name] = true
		case <-ctx.Done():
			t.Fatal("both handlers did not enter concurrently")
		}
	}
	close(release)
	select {
	case err := <-errs:
		t.Fatalf("ExecuteBatch: %v", err)
	case results := <-done:
		if len(results) != 2 || results[0].ToolName != "first" || results[1].ToolName != "second" {
			t.Fatalf("result order changed: %#v", results)
		}
	case <-ctx.Done():
		t.Fatal("ExecuteBatch did not finish")
	}
}

func TestRegistryToolsPreserveRegistrationOrder(t *testing.T) {
	type input struct {
		Value string `json:"value"`
	}
	registry := NewRegistry()
	for _, name := range []string{"third", "first", "second"} {
		definition, handler := MustToolPlain(name, name+" tool", func(input input) (string, error) { return name, nil })
		if err := registry.Register(definition, handler); err != nil {
			t.Fatalf("Register %s: %v", name, err)
		}
	}
	tools := registry.Tools()
	for i, want := range []string{"third", "first", "second"} {
		if tools[i].Function.Name != want {
			t.Fatalf("tool %d: want %q, got %q", i, want, tools[i].Function.Name)
		}
	}
}

func TestSchemaValidationCoversComposedAndNestedSchemas(t *testing.T) {
	root := map[string]interface{}{
		"type":                 "object",
		"required":             []interface{}{"name", "count", "ratio", "enabled", "nothing", "tags", "address", "optional", "choice"},
		"additionalProperties": false,
		"properties": map[string]interface{}{
			"name": map[string]interface{}{
				"type": "string", "minLength": float64(2), "maxLength": float64(5),
				"pattern": "^[a-z]+$", "enum": []interface{}{"valid", "other"},
			},
			"count":    map[string]interface{}{"type": "integer", "minimum": float64(1), "maximum": float64(3)},
			"ratio":    map[string]interface{}{"type": "number", "minimum": float64(1), "maximum": float64(2)},
			"enabled":  map[string]interface{}{"type": "boolean"},
			"nothing":  map[string]interface{}{"type": "null"},
			"tags":     map[string]interface{}{"type": "array", "minItems": float64(1), "maxItems": float64(2), "items": map[string]interface{}{"type": "string"}},
			"address":  map[string]interface{}{"$ref": "#/$defs/address"},
			"optional": map[string]interface{}{"anyOf": []interface{}{map[string]interface{}{"type": "string"}, map[string]interface{}{"type": "null"}}},
			"choice":   map[string]interface{}{"oneOf": []interface{}{map[string]interface{}{"type": "string"}, map[string]interface{}{"type": "integer"}}},
			"metadata": "schema extensions that are not objects are ignored",
		},
		"$defs": map[string]interface{}{
			"address": map[string]interface{}{
				"type":       "object",
				"required":   []string{"city"},
				"properties": map[string]interface{}{"city": map[string]interface{}{"type": "string"}},
			},
		},
	}
	valid := map[string]interface{}{
		"name":     "valid",
		"count":    float64(2),
		"ratio":    float64(1.5),
		"enabled":  true,
		"nothing":  nil,
		"tags":     []interface{}{"go"},
		"address":  map[string]interface{}{"city": "Lima"},
		"optional": nil,
		"choice":   "text",
		"metadata": 123,
	}
	if err := validateSchemaValue(valid, root, root, "input"); err != nil {
		t.Fatalf("valid composed input: %v", err)
	}

	tests := []struct {
		name   string
		value  interface{}
		schema map[string]interface{}
	}{
		{"object type", "wrong", map[string]interface{}{"type": "object"}},
		{"required", map[string]interface{}{}, map[string]interface{}{"type": "object", "required": []string{"value"}}},
		{"additional property", map[string]interface{}{"extra": true}, map[string]interface{}{"type": "object", "properties": map[string]interface{}{}, "additionalProperties": false}},
		{"any of", true, map[string]interface{}{"anyOf": []interface{}{map[string]interface{}{"type": "string"}, map[string]interface{}{"type": "null"}}}},
		{"one of none", true, map[string]interface{}{"oneOf": []interface{}{map[string]interface{}{"type": "string"}, map[string]interface{}{"type": "integer"}}}},
		{"one of multiple", float64(1), map[string]interface{}{"oneOf": []interface{}{map[string]interface{}{"type": "number"}, map[string]interface{}{"type": "number"}}}},
		{"enum", "missing", map[string]interface{}{"enum": []interface{}{"allowed"}}},
		{"array type", "wrong", map[string]interface{}{"type": "array"}},
		{"array minimum", []interface{}{}, map[string]interface{}{"type": "array", "minItems": float64(1)}},
		{"array maximum", []interface{}{1, 2}, map[string]interface{}{"type": "array", "maxItems": float64(1)}},
		{"array item", []interface{}{true}, map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "string"}}},
		{"string type", true, map[string]interface{}{"type": "string"}},
		{"string minimum", "a", map[string]interface{}{"type": "string", "minLength": float64(2)}},
		{"string maximum", "long", map[string]interface{}{"type": "string", "maxLength": float64(2)}},
		{"string pattern", "123", map[string]interface{}{"type": "string", "pattern": "^[a-z]+$"}},
		{"invalid schema pattern", "abc", map[string]interface{}{"type": "string", "pattern": "["}},
		{"integer type", float64(1.5), map[string]interface{}{"type": "integer"}},
		{"number type", "one", map[string]interface{}{"type": "number"}},
		{"number minimum", float64(0), map[string]interface{}{"type": "number", "minimum": float64(1)}},
		{"number maximum", float64(2), map[string]interface{}{"type": "number", "maximum": float64(1)}},
		{"boolean type", "true", map[string]interface{}{"type": "boolean"}},
		{"null type", false, map[string]interface{}{"type": "null"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateSchemaValue(tt.value, tt.schema, tt.schema, "input"); err == nil {
				t.Fatal("expected schema validation error")
			}
		})
	}
}

func TestSchemaValidationHelpers(t *testing.T) {
	root := map[string]interface{}{"a/b~c": map[string]interface{}{"type": "string"}}
	resolved, err := resolveLocalRef(root, "#/a~1b~0c")
	if err != nil || resolved["type"] != "string" {
		t.Fatalf("escaped local reference: resolved=%#v err=%v", resolved, err)
	}
	for _, ref := range []struct {
		root map[string]interface{}
		ref  string
	}{
		{root, "https://example.com/schema"},
		{map[string]interface{}{}, "#/missing"},
		{map[string]interface{}{"value": 1}, "#/value/child"},
		{map[string]interface{}{"value": 1}, "#/value"},
	} {
		if _, err := resolveLocalRef(ref.root, ref.ref); err == nil {
			t.Fatalf("expected reference error for %q", ref.ref)
		}
	}

	if _, ok := schemaList("wrong"); ok {
		t.Fatal("string is not a schema list")
	}
	if _, ok := schemaList([]interface{}{true}); ok {
		t.Fatal("non-object entry is not a schema list")
	}
	if got, ok := schemaList([]interface{}{map[string]interface{}{"type": "string"}}); !ok || len(got) != 1 {
		t.Fatalf("unexpected schema list %#v, %v", got, ok)
	}
	if got := stringList([]interface{}{"a", 1, "b"}); len(got) != 2 || got[1] != "b" {
		t.Fatalf("unexpected interface string list %#v", got)
	}
	if got := stringList("wrong"); got != nil {
		t.Fatalf("expected nil string list, got %#v", got)
	}

	numbers := []interface{}{float64(1), float32(1), int(1), int8(1), int16(1), int32(1), int64(1), uint(1), uint8(1), uint16(1), uint32(1), uint64(1)}
	for _, value := range numbers {
		if got, ok := number(value); !ok || got != 1 {
			t.Fatalf("number(%T): got=%v ok=%v", value, got, ok)
		}
	}
	if _, ok := number("one"); ok {
		t.Fatal("string must not be accepted as a number")
	}
}
