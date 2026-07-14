package tool

import (
	"fmt"
	"math"
	"reflect"
	"regexp"
	"strings"
)

// validateToolInput validates model-produced arguments against the same JSON
// Schema sent to the provider. Provider-side strict modes are not universal,
// so handlers must not assume the wire schema was enforced for them.
func validateToolInput(input map[string]interface{}, schema map[string]interface{}) error {
	return validateSchemaValue(input, schema, schema, "input")
}

func validateSchemaValue(value interface{}, schema, root map[string]interface{}, path string) error {
	if ref, _ := schema["$ref"].(string); ref != "" {
		resolved, err := resolveLocalRef(root, ref)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		return validateSchemaValue(value, resolved, root, path)
	}

	if variants, ok := schemaList(schema["anyOf"]); ok {
		if validateAnyVariant(value, variants, root, path) {
			return nil
		}
		return fmt.Errorf("%s does not match any allowed schema", path)
	}
	if variants, ok := schemaList(schema["oneOf"]); ok {
		matches := 0
		for _, variant := range variants {
			if validateSchemaValue(value, variant, root, path) == nil {
				matches++
			}
		}
		if matches != 1 {
			return fmt.Errorf("%s must match exactly one allowed schema", path)
		}
	}

	if enum, ok := schema["enum"].([]interface{}); ok {
		matched := false
		for _, allowed := range enum {
			if reflect.DeepEqual(value, allowed) || fmt.Sprint(value) == fmt.Sprint(allowed) {
				matched = true
				break
			}
		}
		if !matched {
			return fmt.Errorf("%s must be one of %v", path, enum)
		}
	}

	typeName, _ := schema["type"].(string)
	switch typeName {
	case "object":
		object, ok := value.(map[string]interface{})
		if !ok {
			return fmt.Errorf("%s must be an object", path)
		}
		for _, required := range stringList(schema["required"]) {
			if _, exists := object[required]; !exists {
				return fmt.Errorf("%s.%s is required", path, required)
			}
		}
		properties, _ := schema["properties"].(map[string]interface{})
		for name, child := range object {
			childSchema, exists := properties[name]
			if !exists {
				if allowed, ok := schema["additionalProperties"].(bool); ok && !allowed {
					return fmt.Errorf("%s.%s is not allowed", path, name)
				}
				continue
			}
			childMap, ok := childSchema.(map[string]interface{})
			if ok {
				if err := validateSchemaValue(child, childMap, root, path+"."+name); err != nil {
					return err
				}
			}
		}
	case "array":
		items, ok := value.([]interface{})
		if !ok {
			return fmt.Errorf("%s must be an array", path)
		}
		if min, ok := number(schema["minItems"]); ok && float64(len(items)) < min {
			return fmt.Errorf("%s must contain at least %s items", path, formatNumber(min))
		}
		if max, ok := number(schema["maxItems"]); ok && float64(len(items)) > max {
			return fmt.Errorf("%s must contain at most %s items", path, formatNumber(max))
		}
		if itemSchema, ok := schema["items"].(map[string]interface{}); ok {
			for i, item := range items {
				if err := validateSchemaValue(item, itemSchema, root, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
	case "string":
		text, ok := value.(string)
		if !ok {
			return fmt.Errorf("%s must be a string", path)
		}
		if min, ok := number(schema["minLength"]); ok && float64(len([]rune(text))) < min {
			return fmt.Errorf("%s must contain at least %s characters", path, formatNumber(min))
		}
		if max, ok := number(schema["maxLength"]); ok && float64(len([]rune(text))) > max {
			return fmt.Errorf("%s must contain at most %s characters", path, formatNumber(max))
		}
		if pattern, _ := schema["pattern"].(string); pattern != "" {
			re, err := regexp.Compile(pattern)
			if err != nil {
				return fmt.Errorf("%s has invalid schema pattern: %w", path, err)
			}
			if !re.MatchString(text) {
				return fmt.Errorf("%s must match %q", path, pattern)
			}
		}
	case "integer":
		n, ok := number(value)
		if !ok || math.Trunc(n) != n {
			return fmt.Errorf("%s must be an integer", path)
		}
		if err := validateNumber(n, schema, path); err != nil {
			return err
		}
	case "number":
		n, ok := number(value)
		if !ok {
			return fmt.Errorf("%s must be a number", path)
		}
		if err := validateNumber(n, schema, path); err != nil {
			return err
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("%s must be a boolean", path)
		}
	case "null":
		if value != nil {
			return fmt.Errorf("%s must be null", path)
		}
	}

	return nil
}

func validateAnyVariant(value interface{}, variants []map[string]interface{}, root map[string]interface{}, path string) bool {
	for _, variant := range variants {
		if validateSchemaValue(value, variant, root, path) == nil {
			return true
		}
	}
	return false
}

func validateNumber(value float64, schema map[string]interface{}, path string) error {
	if min, ok := number(schema["minimum"]); ok && value < min {
		return fmt.Errorf("%s must be at least %s", path, formatNumber(min))
	}
	if max, ok := number(schema["maximum"]); ok && value > max {
		return fmt.Errorf("%s must be at most %s", path, formatNumber(max))
	}
	return nil
}

func resolveLocalRef(root map[string]interface{}, ref string) (map[string]interface{}, error) {
	if !strings.HasPrefix(ref, "#/") {
		return nil, fmt.Errorf("unsupported schema reference %q", ref)
	}
	var current interface{} = root
	for _, segment := range strings.Split(strings.TrimPrefix(ref, "#/"), "/") {
		segment = strings.ReplaceAll(strings.ReplaceAll(segment, "~1", "/"), "~0", "~")
		object, ok := current.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid schema reference %q", ref)
		}
		current, ok = object[segment]
		if !ok {
			return nil, fmt.Errorf("unresolved schema reference %q", ref)
		}
	}
	resolved, ok := current.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("schema reference %q is not an object", ref)
	}
	return resolved, nil
}

func schemaList(value interface{}) ([]map[string]interface{}, bool) {
	items, ok := value.([]interface{})
	if !ok {
		return nil, false
	}
	result := make([]map[string]interface{}, 0, len(items))
	for _, item := range items {
		object, ok := item.(map[string]interface{})
		if !ok {
			return nil, false
		}
		result = append(result, object)
	}
	return result, true
}

func stringList(value interface{}) []string {
	switch values := value.(type) {
	case []string:
		return values
	case []interface{}:
		result := make([]string, 0, len(values))
		for _, item := range values {
			if text, ok := item.(string); ok {
				result = append(result, text)
			}
		}
		return result
	default:
		return nil
	}
}

func number(value interface{}) (float64, bool) {
	switch n := value.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int8:
		return float64(n), true
	case int16:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint:
		return float64(n), true
	case uint8:
		return float64(n), true
	case uint16:
		return float64(n), true
	case uint32:
		return float64(n), true
	case uint64:
		return float64(n), true
	default:
		return 0, false
	}
}

func formatNumber(value float64) string {
	return fmt.Sprintf("%g", value)
}
