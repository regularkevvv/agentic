package deepinfra

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

// realisticSparseRow builds a row shaped like the live one: BGE-M3's full
// 250002-token vocabulary with only a handful of nonzero weights, which is
// what a short input actually produces.
func realisticSparseRow(vocabulary, nonzero int) json.RawMessage {
	var b strings.Builder
	b.Grow(vocabulary * 4)
	b.WriteByte('[')
	stride := vocabulary / nonzero
	for i := range vocabulary {
		if i > 0 {
			b.WriteByte(',')
		}
		if stride > 0 && i%stride == 0 {
			b.WriteString(strconv.FormatFloat(0.25+float64(i%7)/10, 'f', 4, 64))
			continue
		}
		b.WriteString("0.0")
	}
	b.WriteByte(']')
	return json.RawMessage(b.String())
}

// streamSparseRow is the tempting implementation: walk the row token by token
// so the full vocabulary width is never held. It exists only to keep the
// measurement that rejected it — encoding/json boxes every number into an
// interface, so it allocates by the million.
func streamSparseRow(raw json.RawMessage) (*core.SparseVector, int, error) {
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	if _, err := decoder.Token(); err != nil {
		return nil, 0, err
	}
	vec := &core.SparseVector{}
	width := 0
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, 0, err
		}
		if value := token.(float64); value != 0 {
			vec.Indices = append(vec.Indices, uint32(width))
			vec.Values = append(vec.Values, float32(value))
		}
		width++
	}
	return vec, width, nil
}

// BenchmarkDecodeSparseRow measures one row three ways, at BGE-M3's real
// vocabulary width with a realistic handful of nonzeros.
//
// "stream" is the intuitive optimization and the slowest by far. "fresh
// buffer" is the straightforward decode. "reused buffer" is what the provider
// does across a batch, and is why the scratch slice is threaded through
// buildResponse rather than allocated per input.
func BenchmarkDecodeSparseRow(b *testing.B) {
	row := realisticSparseRow(BGEM3SparseVocabulary, 8)

	b.Run("stream", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, width, err := streamSparseRow(row); err != nil || width != BGEM3SparseVocabulary {
				b.Fatalf("decode: %v, width %d", err, width)
			}
		}
	})

	b.Run("fresh buffer", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			var scratch []float32
			if _, width, err := decodeSparseRow(row, 0, &scratch); err != nil || width != BGEM3SparseVocabulary {
				b.Fatalf("decode: %v, width %d", err, width)
			}
		}
	})

	b.Run("reused buffer", func(b *testing.B) {
		b.ReportAllocs()
		scratch := make([]float32, 0, BGEM3SparseVocabulary)
		for b.Loop() {
			if _, width, err := decodeSparseRow(row, 0, &scratch); err != nil || width != BGEM3SparseVocabulary {
				b.Fatalf("decode: %v, width %d", err, width)
			}
		}
	})
}

// The two implementations must agree, or the benchmark is comparing a
// correctness difference rather than a cost one.
func TestBufferedAndStreamingDecodeAgree(t *testing.T) {
	row := realisticSparseRow(4096, 6)

	var scratch []float32
	materialized, materializedWidth, err := decodeSparseRow(row, 0, &scratch)
	if err != nil {
		t.Fatalf("buffered decode: %v", err)
	}
	streamed, streamedWidth, err := streamSparseRow(row)
	if err != nil {
		t.Fatalf("streaming decode: %v", err)
	}

	if streamedWidth != materializedWidth {
		t.Fatalf("widths differ: %d vs %d", streamedWidth, materializedWidth)
	}
	if streamed.Len() != materialized.Len() {
		t.Fatalf("nonzero counts differ: %d vs %d", streamed.Len(), materialized.Len())
	}
	for i := range streamed.Indices {
		if streamed.Indices[i] != materialized.Indices[i] {
			t.Fatalf("index %d differs: %d vs %d", i, streamed.Indices[i], materialized.Indices[i])
		}
		if streamed.Values[i] != materialized.Values[i] {
			t.Fatalf("value %d differs: %v vs %v", i, streamed.Values[i], materialized.Values[i])
		}
	}
}
