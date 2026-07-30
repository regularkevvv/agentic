package json

import (
	"bytes"
	"testing"
)

func TestCodecRoundTripIsByteStable(t *testing.T) {
	t.Parallel()
	codec := New()
	input := struct {
		Name  string
		Count int
	}{Name: "value", Count: 3}
	first, err := codec.Encode(input)
	if err != nil {
		t.Fatal(err)
	}
	second, err := codec.Encode(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("encoding is not stable: %q != %q", first, second)
	}
	var output struct {
		Name  string
		Count int
	}
	if err := codec.Decode(first, &output); err != nil {
		t.Fatal(err)
	}
	if output != input {
		t.Fatalf("round trip = %#v, want %#v", output, input)
	}
}
