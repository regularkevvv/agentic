// Package json supplies the standard JSON payload-codec adapter.
package json

import stdjson "encoding/json"

// Codec implements the harness codec contract with encoding/json.
type Codec struct{}

func New() Codec { return Codec{} }

func (Codec) Encode(value any) ([]byte, error) {
	return stdjson.Marshal(value)
}

func (Codec) Decode(payload []byte, target any) error {
	return stdjson.Unmarshal(payload, target)
}
