package codec

import (
	"errors"
	"testing"
)

type aliasingCodec struct {
	buffer []byte
}

func (c *aliasingCodec) Encode(any) ([]byte, error) {
	return c.buffer, nil
}

func (c *aliasingCodec) Decode(payload []byte, target any) error {
	value, ok := target.(*[]byte)
	if !ok {
		return errors.New("unexpected target")
	}
	*value = payload
	return nil
}

func TestHelpersKeepCodecBuffersOpaqueAndUnaliased(t *testing.T) {
	t.Parallel()
	implementation := &aliasingCodec{buffer: []byte("encoded")}
	encoded, err := Encode(implementation, struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	implementation.buffer[0] = 'X'
	if string(encoded) != "encoded" {
		t.Fatalf("Encode returned aliased bytes: %q", encoded)
	}

	payload := []byte("decoded")
	decoded, err := Decode[[]byte](implementation, payload)
	if err != nil {
		t.Fatal(err)
	}
	payload[0] = 'X'
	if string(decoded) != "decoded" {
		t.Fatalf("Decode exposed its input buffer: %q", decoded)
	}
}

func TestHelpersRequireCodec(t *testing.T) {
	t.Parallel()
	if _, err := Encode(nil, struct{}{}); err == nil {
		t.Fatal("Encode accepted a nil codec")
	}
	if _, err := Decode[struct{}](nil, nil); err == nil {
		t.Fatal("Decode accepted a nil codec")
	}
}
