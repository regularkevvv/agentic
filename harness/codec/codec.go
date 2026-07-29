// Package codec defines the representation boundary for durable harness data.
//
// Core runtime packages treat encoded payloads as opaque bytes. A deployment
// chooses the concrete representation independently from its journal backend.
package codec

import (
	"errors"
	"fmt"
)

// Codec encodes and decodes versioned harness-domain payloads.
//
// Implementations must be deterministic for the same value: durable event
// replay, transcript validation, and adapter conformance rely on byte-stable
// encodings.
type Codec interface {
	Encode(any) ([]byte, error)
	Decode([]byte, any) error
}

// Encode provides contextual errors and defensively copies an implementation's
// output before it crosses into the durable-store boundary.
func Encode(c Codec, value any) ([]byte, error) {
	if c == nil {
		return nil, errors.New("payload codec is required")
	}
	encoded, err := c.Encode(value)
	if err != nil {
		return nil, err
	}
	return append([]byte(nil), encoded...), nil
}

// Decode decodes an opaque payload into T.
func Decode[T any](c Codec, payload []byte) (T, error) {
	var value T
	if c == nil {
		return value, errors.New("payload codec is required")
	}
	if err := c.Decode(append([]byte(nil), payload...), &value); err != nil {
		return value, fmt.Errorf("decode payload: %w", err)
	}
	return value, nil
}
