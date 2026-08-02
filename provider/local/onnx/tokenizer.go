package onnx

import (
	"fmt"
	"os"

	"github.com/daulet/tokenizers"
)

// tokenizer produces the model's input ids from raw text.
//
// It reads tokenizer.json, the portable fast-tokenizer format that carries the
// normalizer, the pre-tokenizer, the vocabulary, and the post-processor that
// adds the special tokens. That file is what makes a Go tokenizer viable at
// all: reimplementing a model's byte-level BPE and its post-processing is how
// ids drift by one special token and every coordinate moves.
//
// The binding is CGO over the same Rust crate `transformers` uses, so agreement
// with the reference implementation is a property to verify rather than to
// assume — TestLiveTokenizerMatchesPyTorchIDs does exactly that, separately
// from inference, so a mismatch says which half broke.
type tokenizer struct {
	inner *tokenizers.Tokenizer
}

// newTokenizer loads tokenizer.json from path.
func newTokenizer(path string) (*tokenizer, error) {
	// The binding reports a Rust-side parse failure for a file it cannot read,
	// which reads as a corrupt tokenizer rather than a missing one. Stat first
	// so the common mistake — a path to the model directory instead of to the
	// file inside it — says so.
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("onnx: tokenizer: %w", err)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("onnx: tokenizer path %s is a directory; "+
			"point it at the tokenizer.json inside it", path)
	}

	inner, err := tokenizers.FromFile(path)
	if err != nil {
		return nil, fmt.Errorf("onnx: load tokenizer %s: %w", path, err)
	}
	return &tokenizer{inner: inner}, nil
}

// encode returns the model's input ids for text, including the special tokens
// the post-processor adds.
//
// Special tokens are added because the reference does: `tokenizer([text])` in
// transformers wraps the sequence, and a SPLADE head pools over every position
// including those, so omitting them would change the result rather than only
// the length.
func (t *tokenizer) encode(text string) ([]int64, error) {
	encoded, _, err := t.inner.EncodeErr(text, true)
	if err != nil {
		return nil, err
	}
	ids := make([]int64, len(encoded))
	for i, id := range encoded {
		ids[i] = int64(id)
	}
	return ids, nil
}

// vocabularySize reports how many distinct ids this tokenizer can emit. It is
// compared against the graph's output width to catch a tokenizer paired with a
// model it does not belong to, which otherwise fails as ids indexing past the
// embedding table.
func (t *tokenizer) vocabularySize() int {
	return int(t.inner.VocabSize())
}

// close releases the Rust-side allocation, which is outside Go's heap and not
// collected.
func (t *tokenizer) close() error {
	return t.inner.Close()
}
