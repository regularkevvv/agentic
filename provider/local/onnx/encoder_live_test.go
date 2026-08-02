package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"sort"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

// These tests run the real graph and are the gate for this package. They skip
// cleanly without it, because the graph is a 117 MiB artifact that cannot live
// in the repository — see README.md for the one-time export.
//
//	AGENTIC_ONNX_MODEL       the exported .onnx file
//	AGENTIC_ONNX_TOKENIZER   the matching tokenizer.json
//	AGENTIC_ONNX_LIBRARY     libonnxruntime.dylib or libonnxruntime.so
//
// testdata/granite_reference.json is what they assert against: PyTorch's own
// input ids, nonzero count, top terms, and every coordinate for each input,
// produced by provider/local/onnx/export_onnx.py alongside the graph.
//
// The gate is exact agreement on every coordinate index and agreement on every
// weight to within 1e-4. Measured 4.56e-06 on darwin/arm64, 2026-08-01, so a
// looser gate would be accepting a regression rather than allowing for float
// arithmetic.
const (
	modelEnvVar     = "AGENTIC_ONNX_MODEL"
	tokenizerEnvVar = "AGENTIC_ONNX_TOKENIZER"

	// weightTolerance is the largest disagreement with PyTorch this package
	// accepts on any single coordinate weight.
	weightTolerance = 1e-4
)

// coordinate is one [index, weight] pair from the reference file.
type coordinate struct {
	Index  int
	Weight float64
}

// UnmarshalJSON reads the two-element array the exporter writes. The pair form
// keeps a file with tens of thousands of coordinates readable, which matters
// because a human diffing two references is the fallback when this test fails.
func (c *coordinate) UnmarshalJSON(raw []byte) error {
	var pair [2]float64
	if err := json.Unmarshal(raw, &pair); err != nil {
		return err
	}
	c.Index, c.Weight = int(pair[0]), pair[1]
	return nil
}

type referenceInput struct {
	Text         string  `json:"text"`
	InputIDs     []int64 `json:"input_ids"`
	NonzeroCount int     `json:"nonzero_count"`
	Top          []struct {
		Index  int     `json:"index"`
		Weight float64 `json:"weight"`
		Term   string  `json:"term"`
	} `json:"top"`
	Coordinates []coordinate `json:"all_coordinates"`
}

type reference struct {
	Model      string           `json:"model"`
	Vocabulary int              `json:"vocabulary"`
	Inputs     []referenceInput `json:"inputs"`
}

func loadReference(t *testing.T) reference {
	t.Helper()
	raw, err := os.ReadFile("testdata/granite_reference.json")
	if err != nil {
		t.Fatalf("read reference: %v", err)
	}
	var ref reference
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatalf("decode reference: %v", err)
	}
	if len(ref.Inputs) == 0 {
		t.Fatal("reference has no inputs")
	}
	// The golden comparison walks the reference's own coordinate list, so a file
	// carrying fewer coordinates than the count it declares would quietly reduce
	// that gate to a prefix and still report a pass — with a smaller reported
	// disagreement, which reads as an improvement.
	for _, input := range ref.Inputs {
		if len(input.Coordinates) != input.NonzeroCount {
			t.Fatalf("%q: reference carries %d coordinates for a declared nonzero count of %d",
				input.Text, len(input.Coordinates), input.NonzeroCount)
		}
	}
	return ref
}

// liveTokenizerPath skips the test when the tokenizer gate is absent.
func liveTokenizerPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv(tokenizerEnvVar)
	if path == "" {
		t.Skipf("%s is not set", tokenizerEnvVar)
	}
	return path
}

// liveEncoder builds an encoder against the reference model, or skips.
//
// The space is declared the way an operator would declare one, with the
// revision the reference was produced from, so what the test stores is what a
// deployment would store.
func liveEncoder(t *testing.T, ref reference, opts ...Option) *Encoder {
	t.Helper()
	modelPath := os.Getenv(modelEnvVar)
	if modelPath == "" {
		t.Skipf("%s is not set", modelEnvVar)
	}
	tokenizerPath := liveTokenizerPath(t)

	encoder, err := New(modelPath, tokenizerPath, agentic.VectorSpace{
		Model:  ref.Model,
		Metric: agentic.SimilarityDotProduct,
	}, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() {
		if err := encoder.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return encoder
}

// TestLiveTokenizerMatchesPyTorchIDs checks the Go tokenizer reproduces the ids
// `transformers` produced from the same raw text.
//
// This is a separate claim from inference and fails separately on purpose. If
// the ids drift by one special token every coordinate moves, and a combined
// test would report that as an inference disagreement.
func TestLiveTokenizerMatchesPyTorchIDs(t *testing.T) {
	ref := loadReference(t)
	tk, err := newTokenizer(liveTokenizerPath(t))
	if err != nil {
		t.Fatalf("newTokenizer: %v", err)
	}
	defer func() {
		if err := tk.close(); err != nil {
			t.Errorf("close tokenizer: %v", err)
		}
	}()

	for _, input := range ref.Inputs {
		ids, err := tk.encode(input.Text)
		if err != nil {
			t.Fatalf("encode %q: %v", input.Text, err)
		}
		if len(ids) != len(input.InputIDs) {
			t.Errorf("%q: %d ids, PyTorch produced %d\n go      %v\n pytorch %v",
				input.Text, len(ids), len(input.InputIDs), ids, input.InputIDs)
			continue
		}
		for i := range ids {
			if ids[i] != input.InputIDs[i] {
				t.Errorf("%q: id %d is %d, PyTorch produced %d\n go      %v\n pytorch %v",
					input.Text, i, ids[i], input.InputIDs[i], ids, input.InputIDs)
				break
			}
		}
	}
}

// TestLiveEncodeMatchesPyTorchGolden is the exit gate: every coordinate index
// identical, every weight within 1e-4.
func TestLiveEncodeMatchesPyTorchGolden(t *testing.T) {
	ref := loadReference(t)
	encoder := liveEncoder(t, ref)

	texts := make([]string, len(ref.Inputs))
	tokens := 0
	for i, input := range ref.Inputs {
		texts[i] = input.Text
		tokens += len(input.InputIDs)
	}

	resp, err := agentic.EncodeDocuments(context.Background(), encoder, texts, agentic.RepresentationSparse)
	if err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}
	if len(resp.Data) != len(texts) {
		t.Fatalf("got %d representations for %d inputs", len(resp.Data), len(texts))
	}

	space, ok := resp.Space(agentic.RepresentationSparse)
	if !ok {
		t.Fatal("response describes no sparse space")
	}
	if space.Dimensions != ref.Vocabulary {
		t.Errorf("space declares a vocabulary of %d, PyTorch reported %d",
			space.Dimensions, ref.Vocabulary)
	}
	if resp.Usage.InputTokens != tokens {
		t.Errorf("usage reports %d input tokens, the reference ids total %d",
			resp.Usage.InputTokens, tokens)
	}

	worst := 0.0
	for i, input := range ref.Inputs {
		delta := compareToGolden(t, input, resp.Data[i].Sparse)
		if delta > worst {
			worst = delta
		}
	}
	t.Logf("largest weight disagreement with PyTorch across %d inputs: %.2e (gate %.0e)",
		len(ref.Inputs), worst, weightTolerance)
}

// compareToGolden asserts one input's vector against the reference and returns
// the largest weight difference it saw.
func compareToGolden(t *testing.T, input referenceInput, got *agentic.SparseVector) float64 {
	t.Helper()
	if got == nil {
		t.Errorf("%q: no sparse vector", input.Text)
		return 0
	}
	if got.Len() != input.NonzeroCount {
		t.Errorf("%q: %d coordinates, PyTorch produced %d",
			input.Text, got.Len(), input.NonzeroCount)
		return 0
	}

	worst := 0.0
	for i, want := range input.Coordinates {
		if int(got.Indices[i]) != want.Index {
			t.Fatalf("%q: coordinate %d is index %d, PyTorch produced %d",
				input.Text, i, got.Indices[i], want.Index)
		}
		if delta := math.Abs(float64(got.Values[i]) - want.Weight); delta > worst {
			worst = delta
		}
	}
	if worst > weightTolerance {
		t.Errorf("%q: largest weight difference %.2e exceeds %.0e",
			input.Text, worst, weightTolerance)
	}

	compareTopTerms(t, input, got)
	return worst
}

// compareTopTerms checks the heaviest coordinates rank the same way.
//
// The index comparison above already covers correctness; this covers the claim
// the model is here for. The top terms for "automobile" are auto, Motor, car —
// expansion, from a model that never saw the word "car" in the input.
func compareTopTerms(t *testing.T, input referenceInput, got *agentic.SparseVector) {
	t.Helper()
	const topTerms = 8

	ranked := make([]int, got.Len())
	for i := range ranked {
		ranked[i] = i
	}
	// Ties break by index, which is how the reference was sorted: Python's sort
	// is stable and the coordinates were built in index order.
	sort.SliceStable(ranked, func(a, b int) bool {
		return got.Values[ranked[a]] > got.Values[ranked[b]]
	})

	for rank := 0; rank < topTerms && rank < len(input.Top) && rank < len(ranked); rank++ {
		want := input.Top[rank]
		index := int(got.Indices[ranked[rank]])
		if index != want.Index {
			t.Errorf("%q: rank %d is index %d, PyTorch ranked %q (index %d) there",
				input.Text, rank, index, want.Term, want.Index)
		}
	}
	if len(input.Top) > 0 {
		t.Logf("%q -> %s", input.Text, topTermNames(input, topTerms))
	}
}

func topTermNames(input referenceInput, count int) string {
	names := ""
	for i := 0; i < count && i < len(input.Top); i++ {
		if i > 0 {
			names += ", "
		}
		names += input.Top[i].Term
	}
	return names
}

// paddedWidths reports, for each text, the width its row will be padded to and
// the length it tokenizes to. It asks the encoder's own grouping rule rather
// than restating the rule here, so a test built on it describes the batches the
// encoder really forms.
func paddedWidths(t *testing.T, encoder *Encoder, texts []string) (widths, lengths []int) {
	t.Helper()
	rows, _, err := encoder.tokenize(texts)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	lengths = make([]int, len(rows))
	for i, row := range rows {
		lengths[i] = len(row)
	}
	widths = make([]int, len(lengths))
	for _, group := range groupByWidth(lengths, encoder.vocabulary, encoder.batchBytes) {
		width := 0
		for _, i := range group {
			if lengths[i] > width {
				width = lengths[i]
			}
		}
		for _, i := range group {
			widths[i] = width
		}
	}
	return widths, lengths
}

// TestLiveBatchedRowsMatchRowsAlone is the assertion that makes batching safe.
//
// Rows in a batch are padded to the widest one, and if the attention mask were
// not honored a padded row would encode differently from the same row alone —
// silently, and only for callers who batch.
//
// The texts are chosen for their token lengths rather than their content. At 10
// and 5 the first two are one group at exactly the [maxPadRatio] bound, so the
// short row is padded to twice its length, which is the most padding this
// encoder ever applies; the reference document is 18 tokens and runs in a pass
// of its own, which makes this also the assertion that results come back in
// request order rather than group order. The reference inputs alone would not
// do: at 5 and 18 tokens the grouping rule separates them and nothing is padded
// at all.
func TestLiveBatchedRowsMatchRowsAlone(t *testing.T) {
	ref := loadReference(t)
	encoder := liveEncoder(t, ref)

	texts := []string{
		"the automobile was recalibrated before launch",
		"automobile",
		ref.Inputs[len(ref.Inputs)-1].Text,
	}
	widths, lengths := paddedWidths(t, encoder, texts)
	// Asserted rather than assumed: a version of this test whose rows are not
	// padded passes for the wrong reason, and its output reads like this one's.
	if widths[1] != lengths[1]*maxPadRatio {
		t.Fatalf("row 1 is padded from %d to %d; these texts no longer produce the %dx padding this test exists to exercise (lengths %v, widths %v)",
			lengths[1], widths[1], maxPadRatio, lengths, widths)
	}

	batched, err := agentic.EncodeDocuments(context.Background(), encoder, texts, agentic.RepresentationSparse)
	if err != nil {
		t.Fatalf("batched encode: %v", err)
	}

	worst := 0.0
	for i, text := range texts {
		alone, err := agentic.EncodeDocuments(context.Background(), encoder, []string{text}, agentic.RepresentationSparse)
		if err != nil {
			t.Fatalf("solo encode %q: %v", text, err)
		}
		one, many := alone.Data[0].Sparse, batched.Data[i].Sparse
		if one.Len() != many.Len() {
			t.Fatalf("%q: %d coordinates alone, %d in a batch", text, one.Len(), many.Len())
		}
		for j := range one.Indices {
			if one.Indices[j] != many.Indices[j] {
				t.Fatalf("%q: coordinate %d is index %d alone and %d in a batch",
					text, j, one.Indices[j], many.Indices[j])
			}
			if delta := math.Abs(float64(one.Values[j] - many.Values[j])); delta > worst {
				worst = delta
			}
		}
	}
	if worst > weightTolerance {
		t.Errorf("padding changed a weight by %.2e, which exceeds %.0e", worst, weightTolerance)
	}
	// The coordinate indices are identical and the weights are not: measured
	// 1.46e-06 on darwin/arm64, 2026-08-01. That residual is the wider forward
	// pass's float arithmetic rather than padding reaching the result, which the
	// test below holds to the bit.
	t.Logf("a %d-token row padded to %d differs from the same row alone by at most %.2e (gate %.0e)",
		lengths[1], widths[1], worst, weightTolerance)
}

// TestLivePadTokenIDDoesNotReachTheResult checks that what excludes a padded
// position is the attention mask rather than the id written into it, which is
// the contract [WithPadTokenID] states.
//
// It is also what makes the residual above explainable. If the mask were
// ignored, changing the padding id would move the padded row's weights; here it
// moves nothing, so the difference between a padded row and the same row alone
// is float arithmetic over a wider pass and not padding leaking into the answer.
// The comparison is therefore exact: anything but bit equality is a leak.
func TestLivePadTokenIDDoesNotReachTheResult(t *testing.T) {
	ref := loadReference(t)
	texts := []string{"the automobile was recalibrated before launch", "automobile"}

	withDefault := liveEncoder(t, ref)
	widths, lengths := paddedWidths(t, withDefault, texts)
	if widths[1] <= lengths[1] {
		t.Fatalf("nothing is padded, so this asserts nothing (lengths %v, widths %v)", lengths, widths)
	}
	// The far end of the vocabulary is as far from the default id of 0 as a
	// valid id goes; one the embedding table does not have would fault inside
	// the graph rather than show up as a difference.
	other := int64(ref.Vocabulary - 1)
	withOther := liveEncoder(t, ref, WithPadTokenID(other))

	padded := func(encoder *Encoder) *agentic.SparseVector {
		t.Helper()
		resp, err := agentic.EncodeDocuments(context.Background(), encoder, texts, agentic.RepresentationSparse)
		if err != nil {
			t.Fatalf("batched encode: %v", err)
		}
		return resp.Data[1].Sparse
	}
	want, got := padded(withDefault), padded(withOther)

	if got.Len() != want.Len() {
		t.Fatalf("%d coordinates with padding id %d, %d with the default", got.Len(), other, want.Len())
	}
	for i := range want.Indices {
		if got.Indices[i] != want.Indices[i] || got.Values[i] != want.Values[i] {
			t.Fatalf("coordinate %d is index %d weight %v with padding id %d, and index %d weight %v with the default",
				i, got.Indices[i], got.Values[i], other, want.Indices[i], want.Values[i])
		}
	}
	t.Logf("a %d-token row padded to %d is identical under padding ids 0 and %d",
		lengths[1], widths[1], other)
}

// TestLiveGroupingReportsOneRequestPerForwardPass checks the usage a caller
// budgets against reflects how the encoder actually split the work.
func TestLiveGroupingReportsOneRequestPerForwardPass(t *testing.T) {
	ref := loadReference(t)
	encoder := liveEncoder(t, ref)

	texts := []string{"a", "b"}
	for _, input := range ref.Inputs {
		texts = append(texts, input.Text)
	}
	resp, err := agentic.EncodeDocuments(context.Background(), encoder, texts, agentic.RepresentationSparse)
	if err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}
	if resp.Usage.RequestCount < 1 || resp.Usage.RequestCount > len(texts) {
		t.Errorf("usage reports %d forward passes for %d inputs",
			resp.Usage.RequestCount, len(texts))
	}
	t.Logf("%d inputs encoded in %d forward passes", len(texts), resp.Usage.RequestCount)
}

// TestLiveRejectsUnsupportedKinds checks the live encoder refuses what it
// cannot produce, rather than the unit test's hand-built one.
func TestLiveRejectsUnsupportedKinds(t *testing.T) {
	ref := loadReference(t)
	encoder := liveEncoder(t, ref)

	for _, kind := range []agentic.RepresentationKind{
		agentic.RepresentationDense,
		agentic.RepresentationMultiVector,
	} {
		_, err := agentic.EncodeQueries(context.Background(), encoder, []string{"automobile"}, kind)
		if !errors.Is(err, agentic.ErrUnsupportedRepresentation) {
			t.Errorf("%s: got %v, want ErrUnsupportedRepresentation", kind, err)
		}
	}
}

// TestLiveRejectsAnOverLongInput checks the sequence limit is enforced before
// inference, with the token count in the message and without echoing the text.
func TestLiveRejectsAnOverLongInput(t *testing.T) {
	ref := loadReference(t)
	encoder := liveEncoder(t, ref, WithMaxTokens(4))

	_, err := agentic.EncodeDocuments(context.Background(), encoder,
		[]string{ref.Inputs[len(ref.Inputs)-1].Text}, agentic.RepresentationSparse)
	if !errors.Is(err, agentic.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want ErrInvalidRepresentationRequest", err)
	}
	t.Logf("rejected: %v", err)
}
