package test_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	testprovider "github.com/regularkevvv/agentic/provider/test"
	"github.com/regularkevvv/agentic/provider/test/conformance"
)

func allOutputs() []core.RepresentationKind {
	return []core.RepresentationKind{
		core.RepresentationDense,
		core.RepresentationSparse,
		core.RepresentationMultiVector,
	}
}

func TestRepresentationConformance(t *testing.T) {
	cases := []struct {
		name string
		opts []testprovider.RepresentationOption
	}{
		{"all outputs", nil},
		{"dense only", []testprovider.RepresentationOption{
			testprovider.WithRepresentationOutputs(core.RepresentationDense),
		}},
		{"sparse only", []testprovider.RepresentationOption{
			testprovider.WithRepresentationOutputs(core.RepresentationSparse),
		}},
		{"multi vector only", []testprovider.RepresentationOption{
			testprovider.WithRepresentationOutputs(core.RepresentationMultiVector),
		}},
		{"single output per request", []testprovider.RepresentationOption{
			testprovider.WithRepresentationMultiOutput(false),
		}},
		{"batch limited", []testprovider.RepresentationOption{
			testprovider.WithRepresentationMaxBatchSize(2),
		}},
		{"large batch limit", []testprovider.RepresentationOption{
			testprovider.WithRepresentationMaxBatchSize(2048),
		}},
		{"untyped inputs only", []testprovider.RepresentationOption{
			testprovider.WithRepresentationInputTypes(core.EmbeddingInputNone),
		}},
		{"empty inputs allowed", []testprovider.RepresentationOption{
			testprovider.WithRepresentationEmptyInput(true),
		}},
		{"no truncation", []testprovider.RepresentationOption{
			testprovider.WithRepresentationTruncation(false),
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conformance.RunRepresentation(t, conformance.RepresentationOptions{
				NewEncoder: func(*testing.T) core.RepresentationEncoder {
					return testprovider.NewTestRepresentationEncoder(tc.opts...)
				},
				Deterministic: true,
			})
		})
	}
}

// The suite must also work for an encoder that declares itself nondeterministic
// and unable to observe cancellation, which is how a live provider runs it.
func TestRepresentationConformanceRelaxedOptions(t *testing.T) {
	conformance.RunRepresentation(t, conformance.RepresentationOptions{
		NewEncoder: func(*testing.T) core.RepresentationEncoder {
			return testprovider.NewTestRepresentationEncoder()
		},
		Corpus:           []string{"one input only"},
		SkipCancellation: true,
	})
}

func TestTestRepresentationEncoderDefaults(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()
	if encoder.Name() != "test:encoder" {
		t.Errorf("Name() = %q", encoder.Name())
	}
	caps := encoder.Capabilities()
	if len(caps.Outputs) != 3 || !caps.SupportsMultiOutput || !caps.SupportsTruncation {
		t.Fatalf("unexpected default capabilities: %+v", caps)
	}

	// Capabilities must hand back copies, or a caller could quietly widen the
	// encoder's declared support.
	caps.Outputs[0] = "colbert"
	caps.InputTypes[0] = "passage"
	if encoder.Capabilities().Outputs[0] != core.RepresentationDense {
		t.Error("Capabilities() shares its Outputs slice")
	}
	if encoder.Capabilities().InputTypes[0] != core.EmbeddingInputNone {
		t.Error("Capabilities() shares its InputTypes slice")
	}
}

func TestTestRepresentationEncoderName(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder(
		testprovider.WithRepresentationName("BAAI/bge-m3"),
	)
	if encoder.Name() != "BAAI/bge-m3" {
		t.Fatalf("Name() = %q", encoder.Name())
	}
	if encoder.Space(core.RepresentationDense).Model != "BAAI/bge-m3" {
		t.Error("the configured name should reach the vector space")
	}
}

func TestTestRepresentationEncoderIsDeterministic(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder(testprovider.WithRepresentationDimensions(4))
	req := &core.RepresentationRequest{
		Input:     []string{"sparse vectors in postgres", "sparse vectors in postgres"},
		InputType: core.EmbeddingInputDocument,
		Outputs:   allOutputs(),
	}
	resp, err := encoder.Encode(context.Background(), req)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	first, second := resp.Data[0], resp.Data[1]
	if first.Dense[0] != second.Dense[0] || len(first.Dense) != 4 {
		t.Errorf("equal inputs produced different dense vectors")
	}
	if first.Sparse.Len() != second.Sparse.Len() {
		t.Error("equal inputs produced different sparse vectors")
	}
	if len(first.MultiVector) != 4 {
		t.Errorf("got %d token vectors for a four-word input", len(first.MultiVector))
	}
}

// The input role must reach the values, not just the request record: a fake
// whose query and document encodings are identical cannot stand in for a
// provider whose are not.
func TestTestRepresentationEncoderSeparatesInputRoles(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()
	encode := func(inputType core.EmbeddingInputType) core.Representation {
		resp, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
			Input:     []string{"engineering budget"},
			InputType: inputType,
			Outputs:   allOutputs(),
		})
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		return resp.Data[0]
	}
	query := encode(core.EmbeddingInputQuery)
	document := encode(core.EmbeddingInputDocument)

	if query.Dense[0] == document.Dense[0] {
		t.Error("query and document dense encodings are identical")
	}
	if query.Sparse.Indices[0] == document.Sparse.Indices[0] {
		t.Error("query and document sparse coordinates are identical")
	}
}

// Sharing words must share sparse coordinates; that is the property a lexical
// retrieval test depends on.
func TestTestRepresentationEncoderSparseSharesCoordinates(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()
	resp, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"quarterly budget report", "budget"},
		Outputs: []core.RepresentationKind{core.RepresentationSparse},
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	shared := resp.Data[1].Sparse.Indices[0]
	found := false
	for _, index := range resp.Data[0].Sparse.Indices {
		if index == shared {
			found = true
		}
	}
	if !found {
		t.Error("inputs sharing a word do not share a sparse coordinate")
	}

	// A repeated word weighs more than one that occurs once.
	repeated, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"budget budget budget"},
		Outputs: []core.RepresentationKind{core.RepresentationSparse},
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if repeated.Data[0].Sparse.Values[0] <= resp.Data[1].Sparse.Values[0] {
		t.Error("repeating a word did not increase its weight")
	}
}

func TestTestRepresentationEncoderSpaces(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder(
		testprovider.WithRepresentationDimensions(16),
		testprovider.WithRepresentationVocabulary(500),
	)
	dense := encoder.Space(core.RepresentationDense)
	sparse := encoder.Space(core.RepresentationSparse)

	if dense.Dimensions != 16 || dense.Metric != core.SimilarityCosine {
		t.Errorf("dense space = %+v", dense)
	}
	if sparse.Dimensions != 500 || sparse.Metric != core.SimilarityDotProduct {
		t.Errorf("sparse space = %+v", sparse)
	}
	if dense.ID == sparse.ID {
		t.Error("two kinds share a vector space ID")
	}
	if dense.ID != dense.CanonicalID() {
		t.Error("space ID is not canonical")
	}
}

// Changing a revision changes the space, so a consumer keyed on the ID stops
// mixing old and new vectors without being told to.
func TestTestRepresentationEncoderRevisionChangesSpace(t *testing.T) {
	first := testprovider.NewTestRepresentationEncoder().Space(core.RepresentationSparse)
	second := testprovider.NewTestRepresentationEncoder(
		testprovider.WithRepresentationRevision("rev-2", "tok-2"),
	).Space(core.RepresentationSparse)
	if first.ID == second.ID {
		t.Fatal("a revision change produced the same space ID")
	}
}

func TestTestRepresentationEncoderRejectsInvalidOptions(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder(
		testprovider.WithRepresentationDimensions(0),
		testprovider.WithRepresentationVocabulary(-5),
	)
	if encoder.Space(core.RepresentationDense).Dimensions != 8 {
		t.Error("a non-positive dimension override should be ignored")
	}
	if encoder.Space(core.RepresentationSparse).Dimensions != 4096 {
		t.Error("a non-positive vocabulary override should be ignored")
	}
}

func TestTestRepresentationEncoderRecordsCalls(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()
	req := &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
	if _, err := encoder.Encode(context.Background(), req); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if encoder.CallCount() != 1 {
		t.Fatalf("call count = %d", encoder.CallCount())
	}

	calls := encoder.Calls()
	if len(calls) != 1 || calls[0].Input[0] != "a" {
		t.Fatalf("calls = %+v", calls)
	}
	// The recorded request is a copy, so mutating it cannot rewrite history.
	calls[0].Input[0] = "changed"
	if encoder.Calls()[0].Input[0] != "a" {
		t.Error("recorded calls share the caller's slices")
	}

	encoder.Reset()
	if encoder.CallCount() != 0 {
		t.Error("Reset did not clear the recorded calls")
	}
}

func TestTestRepresentationEncoderInjectedFailures(t *testing.T) {
	t.Run("every call", func(t *testing.T) {
		sentinel := errors.New("injected")
		encoder := testprovider.NewTestRepresentationEncoder(testprovider.WithRepresentationError(sentinel))
		_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
			Input:   []string{"a"},
			Outputs: []core.RepresentationKind{core.RepresentationDense},
		})
		if !errors.Is(err, sentinel) {
			t.Fatalf("got %v, want the injected error", err)
		}
	})

	t.Run("chosen call", func(t *testing.T) {
		sentinel := errors.New("second call fails")
		encoder := testprovider.NewTestRepresentationEncoder(
			testprovider.WithRepresentationFailure(func(call int, req *core.RepresentationRequest) error {
				if call == 1 {
					return sentinel
				}
				return nil
			}),
		)
		req := &core.RepresentationRequest{
			Input:   []string{"a"},
			Outputs: []core.RepresentationKind{core.RepresentationDense},
		}
		if _, err := encoder.Encode(context.Background(), req); err != nil {
			t.Fatalf("first call: %v", err)
		}
		if _, err := encoder.Encode(context.Background(), req); !errors.Is(err, sentinel) {
			t.Fatalf("second call: got %v, want the injected error", err)
		}
	})
}

func TestTestRepresentationEncoderRejectsUnsupportedOutputs(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder(
		testprovider.WithRepresentationOutputs(core.RepresentationDense),
	)
	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationSparse},
	})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
	if encoder.CallCount() != 0 {
		t.Error("a rejected request should not be recorded as a call")
	}
}

func TestTestRepresentationEncoderHonorsCancellation(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := encoder.Encode(ctx, &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}

func TestTestRepresentationEncoderEncodesEmptyInputWhenAllowed(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder(testprovider.WithRepresentationEmptyInput(true))
	resp, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{""},
		Outputs: allOutputs(),
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if resp.Data[0].Sparse.Len() != 0 {
		t.Error("an input with no words should have no sparse coordinates")
	}
	if len(resp.Data[0].MultiVector) != 1 {
		t.Error("an input with no words should still have one token vector")
	}
}

func TestTestRepresentationEncoderCapsTokenVectors(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()
	long := strings.TrimSpace(strings.Repeat("word ", 600))
	resp, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{long},
		Outputs: []core.RepresentationKind{core.RepresentationMultiVector},
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(resp.Data[0].MultiVector) != 512 {
		t.Fatalf("got %d token vectors, want the 512 cap", len(resp.Data[0].MultiVector))
	}
}

func TestTestRepresentationEncoderIsRaceSafe(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()
	req := &core.RepresentationRequest{
		Input:   []string{"concurrent"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
	done := make(chan struct{})
	for i := 0; i < 8; i++ {
		go func() {
			defer func() { done <- struct{}{} }()
			if _, err := encoder.Encode(context.Background(), req); err != nil {
				t.Errorf("Encode: %v", err)
			}
		}()
	}
	for i := 0; i < 8; i++ {
		<-done
	}
	if encoder.CallCount() != 8 {
		t.Errorf("call count = %d, want 8", encoder.CallCount())
	}
}
