package agentic_test

import (
	"context"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	testprovider "github.com/regularkevvv/agentic/provider/test"
)

func TestEncodeQueriesAndDocumentsSetInputType(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()

	if _, err := agentic.EncodeQueries(context.Background(), encoder,
		[]string{"what is the engineering budget?"},
		agentic.RepresentationDense, agentic.RepresentationSparse,
	); err != nil {
		t.Fatalf("EncodeQueries: %v", err)
	}
	if _, err := agentic.EncodeDocuments(context.Background(), encoder,
		[]string{"The monthly engineering budget is $2,000."},
		agentic.RepresentationDense,
	); err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}

	calls := encoder.Calls()
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(calls))
	}
	if calls[0].InputType != agentic.EmbeddingInputQuery {
		t.Errorf("query call input type = %q", calls[0].InputType)
	}
	if len(calls[0].Outputs) != 2 {
		t.Errorf("query call outputs = %v, want both requested kinds", calls[0].Outputs)
	}
	if calls[1].InputType != agentic.EmbeddingInputDocument {
		t.Errorf("document call input type = %q", calls[1].InputType)
	}
}

// The helpers set the input role and nothing else: they do not split batches
// or add outputs the caller did not ask for.
func TestEncodeHelpersDoNotChooseOutputs(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()
	_, err := agentic.EncodeDocuments(context.Background(), encoder, []string{"a"})
	if !errors.Is(err, agentic.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want a request error for an empty output list", err)
	}
}

func TestEncodeHelpersRejectNilEncoder(t *testing.T) {
	_, err := agentic.EncodeQueries(context.Background(), nil, []string{"a"}, agentic.RepresentationDense)
	if !errors.Is(err, agentic.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want ErrInvalidRepresentationRequest", err)
	}
	var typed *agentic.InvalidRepresentationRequestError
	if !errors.As(err, &typed) || typed.Invariant != "encoder.nil" {
		t.Fatalf("got %v, want the encoder.nil invariant", err)
	}
	if _, err := agentic.EncodeDocuments(context.Background(), nil, []string{"a"}, agentic.RepresentationDense); err == nil {
		t.Fatal("EncodeDocuments should reject a nil encoder too")
	}
}

// The documented consumer flow: encode, then persist each value beside the ID
// of the space it came from.
func TestEncodeDocumentsCarriesSpaceIdentity(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder()
	resp, err := agentic.EncodeDocuments(context.Background(), encoder,
		[]string{"PostgreSQL supports sparse vectors"},
		agentic.RepresentationDense,
		agentic.RepresentationSparse,
	)
	if err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}

	item := resp.Data[0]
	denseSpace := resp.Spaces[agentic.RepresentationDense]
	sparseSpace := resp.Spaces[agentic.RepresentationSparse]

	if len(item.Dense) != denseSpace.Dimensions {
		t.Errorf("dense width %d does not match the declared space width %d",
			len(item.Dense), denseSpace.Dimensions)
	}
	if item.Sparse.Len() == 0 {
		t.Error("sparse output is empty")
	}
	if denseSpace.ID == "" || sparseSpace.ID == "" {
		t.Error("a space is missing its ID; the values cannot be safely stored")
	}
	if denseSpace.Metric != agentic.SimilarityCosine {
		t.Errorf("dense metric = %q", denseSpace.Metric)
	}
	if sparseSpace.Metric != agentic.SimilarityDotProduct {
		t.Errorf("sparse metric = %q", sparseSpace.Metric)
	}
	if resp.Usage.RequestCount != 1 {
		t.Errorf("request count = %d, want one call for both kinds", resp.Usage.RequestCount)
	}
}

func TestEmbedderAsRepresentationEncoder(t *testing.T) {
	embedder := testprovider.NewTestEmbedder(4)
	encoder, err := agentic.EmbedderAsRepresentationEncoder(embedder, agentic.VectorSpace{
		Provider: "test",
		Revision: "2026-01",
	})
	if err != nil {
		t.Fatalf("EmbedderAsRepresentationEncoder: %v", err)
	}

	resp, err := agentic.EncodeDocuments(context.Background(), encoder,
		[]string{"alpha", "beta"}, agentic.RepresentationDense)
	if err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}
	if len(resp.Data) != 2 || len(resp.Data[0].Dense) != 4 {
		t.Fatalf("unexpected response shape: %d items", len(resp.Data))
	}
	if resp.Spaces[agentic.RepresentationDense].Dimensions != 4 {
		t.Error("adapted space did not record the observed width")
	}
	if embedder.Calls()[0].InputType != agentic.EmbeddingInputDocument {
		t.Error("the input role did not reach the underlying embedder")
	}

	// The adapter must not claim a capability the embedder does not have.
	_, err = agentic.EncodeDocuments(context.Background(), encoder,
		[]string{"alpha"}, agentic.RepresentationSparse)
	if !errors.Is(err, agentic.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

func TestEmbedderAsRepresentationEncoderRequiresProvider(t *testing.T) {
	_, err := agentic.EmbedderAsRepresentationEncoder(testprovider.NewTestEmbedder(4), agentic.VectorSpace{})
	if err == nil {
		t.Fatal("a vector space without a provider should be rejected")
	}
}

func TestRepresentationEncoderAsEmbedder(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder(testprovider.WithRepresentationDimensions(6))
	embedder, err := agentic.RepresentationEncoderAsEmbedder(encoder)
	if err != nil {
		t.Fatalf("RepresentationEncoderAsEmbedder: %v", err)
	}

	resp, err := agentic.EmbedQuery(context.Background(), embedder, "alpha", "beta")
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if len(resp.Vectors) != 2 || len(resp.Vectors[0]) != 6 {
		t.Fatalf("unexpected vectors: %d items", len(resp.Vectors))
	}

	call := encoder.Calls()[0]
	if len(call.Outputs) != 1 || call.Outputs[0] != agentic.RepresentationDense {
		t.Errorf("projection requested %v, want dense only", call.Outputs)
	}
	if call.InputType != agentic.EmbeddingInputQuery {
		t.Errorf("input role = %q, want query", call.InputType)
	}
}

func TestRepresentationEncoderAsEmbedderRejectsSparseOnlyEncoders(t *testing.T) {
	encoder := testprovider.NewTestRepresentationEncoder(
		testprovider.WithRepresentationOutputs(agentic.RepresentationSparse),
	)
	_, err := agentic.RepresentationEncoderAsEmbedder(encoder)
	if !errors.Is(err, agentic.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation at construction", err)
	}
}

// Round-tripping an embedder through both adapters must preserve its vectors
// exactly, so adopting the new interface changes nothing for existing callers.
func TestAdapterRoundTripPreservesVectors(t *testing.T) {
	embedder := testprovider.NewTestEmbedder(4)
	direct, err := agentic.EmbedDocuments(context.Background(), embedder, "alpha", "beta")
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}

	encoder, err := agentic.EmbedderAsRepresentationEncoder(embedder, agentic.VectorSpace{Provider: "test"})
	if err != nil {
		t.Fatalf("EmbedderAsRepresentationEncoder: %v", err)
	}
	projected, err := agentic.RepresentationEncoderAsEmbedder(encoder)
	if err != nil {
		t.Fatalf("RepresentationEncoderAsEmbedder: %v", err)
	}
	round, err := agentic.EmbedDocuments(context.Background(), projected, "alpha", "beta")
	if err != nil {
		t.Fatalf("round-trip EmbedDocuments: %v", err)
	}

	if len(round.Vectors) != len(direct.Vectors) {
		t.Fatalf("round trip returned %d vectors, want %d", len(round.Vectors), len(direct.Vectors))
	}
	for i := range direct.Vectors {
		for j := range direct.Vectors[i] {
			if round.Vectors[i][j] != direct.Vectors[i][j] {
				t.Fatalf("vector %d differs at %d after a round trip", i, j)
			}
		}
	}
}

// The facade must expose the representation surface without leaking internal
// package paths into a consumer's code.
func TestRepresentationFacadeTypesAreUsable(t *testing.T) {
	var (
		_ agentic.RepresentationEncoder
		_ agentic.RepresentationRequest
		_ agentic.RepresentationResponse
		_ agentic.Representation
		_ agentic.RepresentationCapabilities
		_ agentic.RepresentationUsage
		_ agentic.RepresentationLimits
		_ agentic.RepresentationValidator
		_ agentic.SparseVector
	)

	space := agentic.VectorSpace{
		Provider:   "example",
		Model:      "BAAI/bge-m3",
		Revision:   "immutable-revision",
		Tokenizer:  "immutable-tokenizer",
		Kind:       agentic.RepresentationSparse,
		Dimensions: 250002,
		Metric:     agentic.SimilarityDotProduct,
	}.WithCanonicalID()
	if err := space.Validate(); err != nil {
		t.Fatalf("facade-built space is invalid: %v", err)
	}
	if space.ID != space.CanonicalID() {
		t.Error("canonical ID is not reproducible through the facade")
	}

	limits := agentic.RepresentationLimits{MaxInputs: 1}
	validator := agentic.RepresentationValidator{
		Provider:     "example",
		Capabilities: agentic.RepresentationCapabilities{Outputs: []agentic.RepresentationKind{agentic.RepresentationDense}},
		Limits:       limits,
	}
	err := validator.ValidateRequest(&agentic.RepresentationRequest{
		Input:   []string{"a", "b"},
		Outputs: []agentic.RepresentationKind{agentic.RepresentationDense},
	})
	if !errors.Is(err, agentic.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want the facade validator to enforce its limits", err)
	}
}

// An encoder compiled outside this module — provider/local/onnx is one — reaches the
// validator through the facade and must reach the same defaults with it.
// Without this the bounds a consumer sees would depend on where its provider
// was built.
func TestDefaultRepresentationLimitsAreReachableThroughTheFacade(t *testing.T) {
	limits := agentic.DefaultRepresentationLimits()
	if limits.MaxInputs <= 0 || limits.MaxInputBytes <= 0 || limits.MaxTotalInputBytes <= 0 ||
		limits.MaxSparseNonZero <= 0 || limits.MaxTokenVectors <= 0 {
		t.Fatalf("a bound is disabled by default: %+v", limits)
	}

	validator := agentic.RepresentationValidator{
		Provider:     "example",
		Capabilities: agentic.RepresentationCapabilities{Outputs: []agentic.RepresentationKind{agentic.RepresentationSparse}},
		Limits:       limits,
	}
	oversized := make([]string, limits.MaxInputs+1)
	for i := range oversized {
		oversized[i] = "a"
	}
	err := validator.ValidateRequest(&agentic.RepresentationRequest{
		Input:   oversized,
		Outputs: []agentic.RepresentationKind{agentic.RepresentationSparse},
	})
	if !errors.Is(err, agentic.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want the default input ceiling to be enforced", err)
	}
}
