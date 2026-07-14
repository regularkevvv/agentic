package core

import (
	"fmt"
	"reflect"
)

// DependencyEnvelope carries a dependency value through the non-generic
// execution core after a typed public boundary has validated it.
type DependencyEnvelope struct {
	value any
	typ   reflect.Type
}

// NewDependencyEnvelope preserves D's static type even when value is a typed
// nil. Callers must validate nil before allowing external effects.
func NewDependencyEnvelope[D any](value D) DependencyEnvelope {
	return DependencyEnvelope{value: value, typ: reflect.TypeFor[D]()}
}

// DependencyType returns the static type recorded in the envelope.
func (e DependencyEnvelope) DependencyType() reflect.Type {
	return e.typ
}

// DependencyIsNil reports whether the dependency is nil, including typed nil
// pointers, maps, slices, functions, channels, interfaces, and unsafe pointers.
func (e DependencyEnvelope) DependencyIsNil() bool {
	if e.value == nil {
		return true
	}
	v := reflect.ValueOf(e.value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice, reflect.UnsafePointer:
		return v.IsNil()
	default:
		return false
	}
}

// ExtractDependency performs the single checked type-erasure crossing used by
// dependency-aware adapters. A mismatch is an internal wiring error.
func ExtractDependency[D any](envelope any) (D, error) {
	var zero D
	e, ok := envelope.(DependencyEnvelope)
	if !ok {
		return zero, fmt.Errorf("invalid dependency envelope: got %T", envelope)
	}
	expected := reflect.TypeFor[D]()
	if e.typ != expected {
		return zero, fmt.Errorf("invalid dependency type: expected %v, got %v", expected, e.typ)
	}
	value, ok := e.value.(D)
	if !ok {
		return zero, fmt.Errorf("invalid dependency value: expected %v, got %T", expected, e.value)
	}
	return value, nil
}
