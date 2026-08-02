package onnx

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

// These tests need no model, no ONNX Runtime, and no tokenizer file. They cover
// what the encoder decides before it touches any of the three: argument
// validation, vector-space reconciliation, how a request is split into forward
// passes, and the pooling reduction. Agreement with the reference model is a
// different claim and lives in encoder_live_test.go.

// testVocabulary is small enough to write expected pooling results by hand and
// large enough that an off-by-one in the stride would show.
const testVocabulary = 5

func testEncoder(t *testing.T) *Encoder {
	t.Helper()
	space, err := resolveSpace(agentic.VectorSpace{Model: "test/model"}, testVocabulary)
	if err != nil {
		t.Fatalf("resolveSpace: %v", err)
	}
	return &Encoder{
		space:      space,
		vocabulary: testVocabulary,
		maxTokens:  defaultMaxTokens,
		batchBytes: defaultBatchBytes,
		limits:     agentic.DefaultRepresentationLimits(),
	}
}

func sparseRequest(texts ...string) *agentic.RepresentationRequest {
	return &agentic.RepresentationRequest{
		Input:   texts,
		Outputs: []agentic.RepresentationKind{agentic.RepresentationSparse},
	}
}

func TestNewRejectsMissingPaths(t *testing.T) {
	space := agentic.VectorSpace{Model: "test/model"}
	if _, err := New("", "tokenizer.json", space); err == nil {
		t.Error("expected an error for an empty model path")
	}
	if _, err := New("model.onnx", "", space); err == nil {
		t.Error("expected an error for an empty tokenizer path")
	}
}

func TestNewRejectsOutOfRangeOptions(t *testing.T) {
	space := agentic.VectorSpace{Model: "test/model"}
	cases := map[string]Option{
		"zero max tokens":     WithMaxTokens(0),
		"negative pad id":     WithPadTokenID(-1),
		"zero batch bytes":    WithMaxBatchBytes(0),
		"negative batch byte": WithMaxBatchBytes(-1),
	}
	for name, option := range cases {
		if _, err := New("model.onnx", "tokenizer.json", space, option); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

// A dense space would be accepted by VectorSpace.Validate and produce an
// encoder that declares an identity it cannot fill, so it is refused with the
// same typed error a dense request gets.
func TestNewRejectsANonSparseSpace(t *testing.T) {
	space := agentic.VectorSpace{Model: "test/model", Kind: agentic.RepresentationDense}
	_, err := New("model.onnx", "tokenizer.json", space)
	if !errors.Is(err, agentic.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

func TestNewRequiresAModelName(t *testing.T) {
	if _, err := New("model.onnx", "tokenizer.json", agentic.VectorSpace{}); err == nil {
		t.Fatal("expected an error for a vector space with no model")
	}
}

func TestResolveSpaceCompletesTheDescriptorFromTheGraph(t *testing.T) {
	space, err := resolveSpace(agentic.VectorSpace{Model: "ibm-granite/x"}, 50265)
	if err != nil {
		t.Fatalf("resolveSpace: %v", err)
	}
	if space.Provider != providerName {
		t.Errorf("provider = %q, want %q", space.Provider, providerName)
	}
	if space.Dimensions != 50265 {
		t.Errorf("dimensions = %d, want the graph's 50265", space.Dimensions)
	}
	if space.Metric != agentic.SimilarityDotProduct {
		t.Errorf("metric = %q, want dot product", space.Metric)
	}
	if space.ID == "" {
		t.Error("ID was not derived")
	}
	if space.ID != space.CanonicalID() {
		t.Errorf("ID %q is not the canonical one", space.ID)
	}
}

func TestResolveSpaceKeepsACallerSuppliedIdentity(t *testing.T) {
	declared := agentic.VectorSpace{
		ID:       "granite-sparse-2026-08",
		Provider: "internal",
		Model:    "ibm-granite/x",
		Metric:   agentic.SimilarityCosine,
		Kind:     agentic.RepresentationSparse,
	}
	space, err := resolveSpace(declared, 16)
	if err != nil {
		t.Fatalf("resolveSpace: %v", err)
	}
	if space.ID != declared.ID || space.Provider != declared.Provider || space.Metric != declared.Metric {
		t.Errorf("resolveSpace overwrote a declared identity: %+v", space)
	}
}

// A declared width that disagrees with the graph almost always means the space
// ID belongs to a different model, so it is reported rather than corrected.
func TestResolveSpaceRejectsAWidthTheGraphContradicts(t *testing.T) {
	_, err := resolveSpace(agentic.VectorSpace{Model: "x", Dimensions: 30522}, 50265)
	if err == nil {
		t.Fatal("expected an error for a contradicted vocabulary")
	}
}

func TestEncodeRejectsUnsupportedKinds(t *testing.T) {
	encoder := testEncoder(t)
	for _, kind := range []agentic.RepresentationKind{
		agentic.RepresentationDense,
		agentic.RepresentationMultiVector,
	} {
		req := &agentic.RepresentationRequest{
			Input:   []string{"automobile"},
			Outputs: []agentic.RepresentationKind{kind},
		}
		_, err := encoder.Encode(context.Background(), req)
		if !errors.Is(err, agentic.ErrUnsupportedRepresentation) {
			t.Errorf("%s: got %v, want ErrUnsupportedRepresentation", kind, err)
		}
		var unsupported *agentic.UnsupportedRepresentationError
		if errors.As(err, &unsupported) && unsupported.Provider != providerName {
			t.Errorf("%s: error names provider %q", kind, unsupported.Provider)
		}
	}
}

func TestEncodeRejectsMalformedRequests(t *testing.T) {
	encoder := testEncoder(t)
	cases := map[string]*agentic.RepresentationRequest{
		"nil request": nil,
		"no inputs":   sparseRequest(),
		"empty input": sparseRequest("automobile", ""),
		"no outputs":  {Input: []string{"automobile"}},
	}
	for name, req := range cases {
		_, err := encoder.Encode(context.Background(), req)
		if !errors.Is(err, agentic.ErrInvalidRepresentationRequest) {
			t.Errorf("%s: got %v, want ErrInvalidRepresentationRequest", name, err)
		}
	}
}

// Encoding through a destroyed session is a crash rather than an error, so the
// closed encoder has to refuse before it reaches one.
func TestEncodeAfterCloseFails(t *testing.T) {
	encoder := testEncoder(t)
	if err := encoder.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := encoder.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := encoder.Encode(context.Background(), sparseRequest("automobile")); err == nil {
		t.Fatal("expected an error from a closed encoder")
	}
}

func TestCapabilitiesDeclareSparseOnly(t *testing.T) {
	capabilities := testEncoder(t).Capabilities()
	if !capabilities.Supports(agentic.RepresentationSparse) {
		t.Error("sparse is not declared")
	}
	if capabilities.Supports(agentic.RepresentationDense) ||
		capabilities.Supports(agentic.RepresentationMultiVector) {
		t.Errorf("declared more than sparse: %v", capabilities.Outputs)
	}
	if capabilities.SupportsTruncation {
		t.Error("truncation is declared but over-long inputs are rejected, not clipped")
	}
	if capabilities.SupportsMultiOutput {
		t.Error("multi-output is declared but only one kind exists")
	}
}

func TestNameReportsTheSpaceModel(t *testing.T) {
	if got := testEncoder(t).Name(); got != "test/model" {
		t.Errorf("Name() = %q", got)
	}
}

func TestGroupByWidthOrdersByLength(t *testing.T) {
	// Lengths 4, 1, 3, 2 at a budget too small for any pair, so every group is
	// one row and the grouping shows the ordering alone.
	groups := groupByWidth([]int{4, 1, 3, 2}, testVocabulary, testVocabulary*bytesPerLogit)
	got := make([]int, 0, len(groups))
	for _, group := range groups {
		if len(group) != 1 {
			t.Fatalf("expected single-row groups, got %v", groups)
		}
		got = append(got, group[0])
	}
	if want := []int{1, 3, 2, 0}; !reflect.DeepEqual(got, want) {
		t.Errorf("order = %v, want %v (shortest first)", got, want)
	}
}

// The bound the comment on maxPadRatio claims: no row is padded past twice its
// own length.
func TestGroupByWidthBoundsPadding(t *testing.T) {
	lengths := []int{3, 3, 18}
	groups := groupByWidth(lengths, testVocabulary, 1<<30)
	if len(groups) != 2 {
		t.Fatalf("groups = %v, want the two short rows apart from the long one", groups)
	}
	for _, group := range groups {
		width := 0
		for _, i := range group {
			if lengths[i] > width {
				width = lengths[i]
			}
		}
		for _, i := range group {
			if width > lengths[i]*maxPadRatio {
				t.Errorf("row %d of length %d padded to %d", i, lengths[i], width)
			}
		}
	}
}

func TestGroupByWidthHonorsTheByteBudget(t *testing.T) {
	lengths := []int{10, 10, 10, 10}
	// Room for two rows of width 10 and not for three.
	budget := 2 * 10 * testVocabulary * bytesPerLogit
	groups := groupByWidth(lengths, testVocabulary, budget)
	if len(groups) != 2 {
		t.Fatalf("groups = %v, want two passes of two rows", groups)
	}
	for _, group := range groups {
		if len(group) != 2 {
			t.Errorf("group %v exceeds the budget", group)
		}
	}
}

// A row too large for the budget on its own still has to be encoded: refusing
// an input because a memory ceiling would rather it did not is not a useful
// failure.
func TestGroupByWidthEmitsAnOversizedRowAlone(t *testing.T) {
	groups := groupByWidth([]int{512}, testVocabulary, 1)
	if len(groups) != 1 || len(groups[0]) != 1 || groups[0][0] != 0 {
		t.Fatalf("groups = %v, want the single row emitted anyway", groups)
	}
}

func TestGroupByWidthOnNoInputs(t *testing.T) {
	if groups := groupByWidth(nil, testVocabulary, 1<<30); len(groups) != 0 {
		t.Errorf("groups = %v, want none", groups)
	}
}

// logitsFor lays out one row's [sequence, vocabulary] slab from per-position
// vocabulary rows.
func logitsFor(positions ...[testVocabulary]float32) []float32 {
	flat := make([]float32, 0, len(positions)*testVocabulary)
	for _, position := range positions {
		flat = append(flat, position[:]...)
	}
	return flat
}

func TestReduceTakesTheMaximumOverPositions(t *testing.T) {
	var pool pooler
	vector := pool.reduce(logitsFor(
		[testVocabulary]float32{0, 1, 3, 0, 0},
		[testVocabulary]float32{0, 7, 2, 0, 0},
	), []int64{1, 1}, testVocabulary)

	if want := []uint32{1, 2}; !reflect.DeepEqual(vector.Indices, want) {
		t.Fatalf("indices = %v, want %v", vector.Indices, want)
	}
	expect := []float64{math.Log1p(7), math.Log1p(3)}
	for i, weight := range vector.Values {
		if math.Abs(float64(weight)-expect[i]) > 1e-6 {
			t.Errorf("value %d = %v, want log1p of the larger logit %v", i, weight, expect[i])
		}
	}
}

func TestReduceIgnoresMaskedPositions(t *testing.T) {
	var pool pooler
	vector := pool.reduce(logitsFor(
		[testVocabulary]float32{0, 1, 0, 0, 0},
		[testVocabulary]float32{0, 0, 0, 0, 9},
	), []int64{1, 0}, testVocabulary)

	if want := []uint32{1}; !reflect.DeepEqual(vector.Indices, want) {
		t.Errorf("indices = %v, want %v — the padded position contributed", vector.Indices, want)
	}
}

// relu, and the contract's rule that an explicit zero is a coordinate that
// should have been omitted.
func TestReduceDropsNonPositiveLogits(t *testing.T) {
	var pool pooler
	vector := pool.reduce(logitsFor(
		[testVocabulary]float32{-3, 0, 2, -0.5, 0},
	), []int64{1}, testVocabulary)

	if want := []uint32{2}; !reflect.DeepEqual(vector.Indices, want) {
		t.Fatalf("indices = %v, want %v", vector.Indices, want)
	}
	for i, weight := range vector.Values {
		if weight <= 0 {
			t.Errorf("value %d is %v; zero and negative weights must not be emitted", i, weight)
		}
	}
}

// The accumulator is reused across every row of every pass, so a row that
// leaked into the next one would show as coordinates nobody's logits produced.
func TestReduceClearsItsAccumulatorBetweenRows(t *testing.T) {
	var pool pooler
	pool.reduce(logitsFor([testVocabulary]float32{5, 5, 5, 5, 5}), []int64{1}, testVocabulary)
	vector := pool.reduce(logitsFor([testVocabulary]float32{0, 0, 4, 0, 0}), []int64{1}, testVocabulary)

	if want := []uint32{2}; !reflect.DeepEqual(vector.Indices, want) {
		t.Errorf("indices = %v, want %v — the previous row survived", vector.Indices, want)
	}
}

func TestReduceEmitsStrictlyIncreasingIndices(t *testing.T) {
	var pool pooler
	vector := pool.reduce(logitsFor(
		[testVocabulary]float32{1, 0, 3, 0, 2},
		[testVocabulary]float32{0, 4, 0, 6, 0},
	), []int64{1, 1}, testVocabulary)

	if len(vector.Indices) != testVocabulary {
		t.Fatalf("indices = %v, want every coordinate", vector.Indices)
	}
	for i := 1; i < len(vector.Indices); i++ {
		if vector.Indices[i] <= vector.Indices[i-1] {
			t.Fatalf("indices are not strictly increasing: %v", vector.Indices)
		}
	}
}
