// Package conformance holds the shared contract suite for representation
// encoders.
//
// It lives beside the test doubles rather than inside them so that importing a
// fake does not drag the testing package into a consumer's binary, and so that
// its assertion branches — which by construction only run when a provider is
// broken — are not measured as uncovered library code.
package conformance

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

// RepresentationOptions configures [RunRepresentation].
type RepresentationOptions struct {
	// NewEncoder returns the encoder under test. It is called once per
	// subtest, so an encoder that records calls starts each check clean.
	// Required.
	NewEncoder func(t *testing.T) core.RepresentationEncoder

	// Corpus is the text encoded by the shape checks. It defaults to a short
	// multilingual set; keep any override small, because a live provider runs
	// this suite against a metered API.
	Corpus []string

	// Deterministic says that encoding the same text twice returns identical
	// values. Test doubles and locally hosted models set this; hosted services
	// generally should not, since batching and hardware can perturb the last
	// bits of a float without anything being wrong.
	Deterministic bool

	// SkipCancellation omits the context-cancellation check, for encoders that
	// answer from memory and never reach a cancellation point.
	SkipCancellation bool
}

// defaultConformanceCorpus is deliberately small and multilingual: it exercises
// tokenization without turning a live conformance run into a real bill.
var defaultConformanceCorpus = []string{
	"PostgreSQL stores sparse vectors in a sparsevec column.",
	"El zorro marrón salta sobre el perro perezoso.",
	"late interaction scoring compares every query token against every document token",
}

// RunRepresentation checks an encoder against the parts of the
// representation contract that hold for every provider.
//
// It is exported so that provider packages, this repository's test doubles,
// and downstream retrieval systems all assert the same behavior. A consumer
// that writes its indexing code against these guarantees can swap encoders
// without re-testing the guarantees themselves.
func RunRepresentation(t *testing.T, opts RepresentationOptions) {
	t.Helper()
	if opts.NewEncoder == nil {
		t.Fatal("conformance: NewEncoder is required")
	}
	corpus := opts.Corpus
	if len(corpus) == 0 {
		corpus = defaultConformanceCorpus
	}
	// An encoder with a small batch limit still has to pass every other
	// check, so the shared corpus is trimmed to fit rather than tripping the
	// limit in each one.
	if limit := opts.NewEncoder(t).Capabilities().MaximumBatchSize; limit > 0 && len(corpus) > limit {
		corpus = corpus[:limit]
	}

	t.Run("capabilities are well formed", func(t *testing.T) {
		checkCapabilities(t, opts.NewEncoder(t))
	})
	t.Run("each supported output encodes alone", func(t *testing.T) {
		checkSingleOutputs(t, opts, corpus)
	})
	t.Run("supported outputs encode together", func(t *testing.T) {
		checkCombinedOutputs(t, opts, corpus)
	})
	t.Run("input roles are accepted", func(t *testing.T) {
		checkInputRoles(t, opts, corpus)
	})
	t.Run("output order follows input order", func(t *testing.T) {
		checkOrder(t, opts, corpus)
	})
	t.Run("unsupported outputs are typed errors", func(t *testing.T) {
		checkUnsupportedOutput(t, opts, corpus)
	})
	t.Run("invalid requests are typed errors", func(t *testing.T) {
		checkInvalidRequests(t, opts, corpus)
	})
	t.Run("request slices are not mutated", func(t *testing.T) {
		checkNoMutation(t, opts, corpus)
	})
	t.Run("batch limit is enforced", func(t *testing.T) {
		checkBatchLimit(t, opts)
	})
	t.Run("cancellation is honored", func(t *testing.T) {
		checkCancellation(t, opts, corpus)
	})
	t.Run("dense output projects to an embedder", func(t *testing.T) {
		checkEmbedderProjection(t, opts, corpus)
	})
}

func checkCapabilities(t *testing.T, encoder core.RepresentationEncoder) {
	t.Helper()
	if encoder.Name() == "" {
		t.Error("Name() is empty")
	}
	caps := encoder.Capabilities()
	if len(caps.Outputs) == 0 {
		t.Fatal("Capabilities().Outputs is empty; an encoder must produce something")
	}
	seen := make(map[core.RepresentationKind]bool, len(caps.Outputs))
	for _, kind := range caps.Outputs {
		if !kind.Valid() {
			t.Errorf("Capabilities().Outputs contains unknown kind %q", string(kind))
		}
		if seen[kind] {
			t.Errorf("Capabilities().Outputs lists %s more than once", kind)
		}
		seen[kind] = true
	}
	for _, inputType := range caps.InputTypes {
		switch inputType {
		case core.EmbeddingInputNone, core.EmbeddingInputQuery, core.EmbeddingInputDocument:
		default:
			t.Errorf("Capabilities().InputTypes contains unknown type %q", string(inputType))
		}
	}
	if caps.MaximumBatchSize < 0 {
		t.Errorf("Capabilities().MaximumBatchSize is negative: %d", caps.MaximumBatchSize)
	}
}

func checkSingleOutputs(t *testing.T, opts RepresentationOptions, corpus []string) {
	t.Helper()
	for _, kind := range opts.NewEncoder(t).Capabilities().Outputs {
		t.Run(string(kind), func(t *testing.T) {
			encoder := opts.NewEncoder(t)
			req := &core.RepresentationRequest{
				Input:     corpus,
				InputType: preferredInputType(encoder.Capabilities()),
				Outputs:   []core.RepresentationKind{kind},
			}
			resp, err := encoder.Encode(context.Background(), req)
			if err != nil {
				t.Fatalf("Encode(%s): %v", kind, err)
			}
			checkResponse(t, req, resp)
		})
	}
}

func checkCombinedOutputs(t *testing.T, opts RepresentationOptions, corpus []string) {
	t.Helper()
	encoder := opts.NewEncoder(t)
	caps := encoder.Capabilities()
	if !caps.SupportsMultiOutput || len(caps.Outputs) < 2 {
		t.Skip("encoder returns one representation kind per request")
	}
	req := &core.RepresentationRequest{
		Input:     corpus,
		InputType: preferredInputType(caps),
		Outputs:   append([]core.RepresentationKind(nil), caps.Outputs...),
	}
	resp, err := encoder.Encode(context.Background(), req)
	if err != nil {
		t.Fatalf("Encode(all outputs): %v", err)
	}
	checkResponse(t, req, resp)
	if resp.Usage.RequestCount > 1 {
		t.Errorf("combined request reported %d provider calls; the point of multi-output "+
			"is one forward pass", resp.Usage.RequestCount)
	}
}

func checkInputRoles(t *testing.T, opts RepresentationOptions, corpus []string) {
	t.Helper()
	caps := opts.NewEncoder(t).Capabilities()
	for _, inputType := range []core.EmbeddingInputType{
		core.EmbeddingInputNone,
		core.EmbeddingInputQuery,
		core.EmbeddingInputDocument,
	} {
		if !caps.SupportsInputType(inputType) {
			continue
		}
		t.Run(inputTypeName(inputType), func(t *testing.T) {
			encoder := opts.NewEncoder(t)
			req := &core.RepresentationRequest{
				Input:     corpus[:1],
				InputType: inputType,
				Outputs:   []core.RepresentationKind{caps.Outputs[0]},
			}
			resp, err := encoder.Encode(context.Background(), req)
			if err != nil {
				t.Fatalf("Encode(input type %q): %v", string(inputType), err)
			}
			checkResponse(t, req, resp)
		})
	}
}

// checkOrder proves that Data[i] belongs to Input[i] rather than to whatever
// position the provider happened to return it in. Reversing the batch and
// comparing the same text's output is the only check that catches a provider
// whose response carries an index field the adapter ignored.
func checkOrder(t *testing.T, opts RepresentationOptions, corpus []string) {
	t.Helper()
	if !opts.Deterministic {
		t.Skip("encoder is not declared deterministic")
	}
	if len(corpus) < 2 {
		t.Skip("order check needs at least two inputs")
	}
	encoder := opts.NewEncoder(t)
	caps := encoder.Capabilities()
	kind := caps.Outputs[0]

	forward, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:     corpus,
		InputType: preferredInputType(caps),
		Outputs:   []core.RepresentationKind{kind},
	})
	if err != nil {
		t.Fatalf("Encode(forward): %v", err)
	}
	reversed := make([]string, len(corpus))
	for i, text := range corpus {
		reversed[len(corpus)-1-i] = text
	}
	backward, err := opts.NewEncoder(t).Encode(context.Background(), &core.RepresentationRequest{
		Input:     reversed,
		InputType: preferredInputType(caps),
		Outputs:   []core.RepresentationKind{kind},
	})
	if err != nil {
		t.Fatalf("Encode(reversed): %v", err)
	}
	for i := range corpus {
		got := backward.Data[len(corpus)-1-i]
		if !sameRepresentation(kind, forward.Data[i], got) {
			t.Errorf("input %d encodes differently when the batch is reversed; "+
				"the provider's output order does not follow its input order", i)
		}
	}
}

func checkUnsupportedOutput(t *testing.T, opts RepresentationOptions, corpus []string) {
	t.Helper()
	encoder := opts.NewEncoder(t)
	caps := encoder.Capabilities()
	var missing core.RepresentationKind
	for _, kind := range []core.RepresentationKind{
		core.RepresentationMultiVector,
		core.RepresentationSparse,
		core.RepresentationDense,
	} {
		if !caps.Supports(kind) {
			missing = kind
			break
		}
	}
	if missing == "" {
		t.Skip("encoder supports every representation kind")
	}

	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:     corpus[:1],
		InputType: preferredInputType(caps),
		Outputs:   []core.RepresentationKind{missing},
	})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("requesting unsupported %s: got %v, want ErrUnsupportedRepresentation", missing, err)
	}
	var typed *core.UnsupportedRepresentationError
	if !errors.As(err, &typed) {
		t.Fatalf("requesting unsupported %s: error is not *UnsupportedRepresentationError", missing)
	}
	if typed.Kind != missing {
		t.Errorf("error reports kind %s, want %s", typed.Kind, missing)
	}
	if typed.Provider == "" {
		t.Error("error does not name the provider")
	}
}

func checkInvalidRequests(t *testing.T, opts RepresentationOptions, corpus []string) {
	t.Helper()
	encoder := opts.NewEncoder(t)
	caps := encoder.Capabilities()
	inputType := preferredInputType(caps)
	kind := caps.Outputs[0]

	cases := []struct {
		name string
		req  *core.RepresentationRequest
	}{
		{"nil request", nil},
		{"no inputs", &core.RepresentationRequest{
			InputType: inputType,
			Outputs:   []core.RepresentationKind{kind},
		}},
		{"no outputs", &core.RepresentationRequest{
			Input:     corpus[:1],
			InputType: inputType,
		}},
		{"duplicate outputs", &core.RepresentationRequest{
			Input:     corpus[:1],
			InputType: inputType,
			Outputs:   []core.RepresentationKind{kind, kind},
		}},
		{"unknown output kind", &core.RepresentationRequest{
			Input:     corpus[:1],
			InputType: inputType,
			Outputs:   []core.RepresentationKind{"colbert"},
		}},
		{"unknown input type", &core.RepresentationRequest{
			Input:     corpus[:1],
			InputType: core.EmbeddingInputType("passage"),
			Outputs:   []core.RepresentationKind{kind},
		}},
	}
	if !caps.AllowsEmptyInput {
		cases = append(cases, struct {
			name string
			req  *core.RepresentationRequest
		}{"empty input string", &core.RepresentationRequest{
			Input:     []string{""},
			InputType: inputType,
			Outputs:   []core.RepresentationKind{kind},
		}})
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := encoder.Encode(context.Background(), tc.req)
			if !errors.Is(err, core.ErrInvalidRepresentationRequest) &&
				!errors.Is(err, core.ErrUnsupportedRepresentation) {
				t.Fatalf("got %v, want a representation request error", err)
			}
			// The same invalid request must fail the same way every time:
			// a caller logging or matching on the message needs it stable.
			_, again := encoder.Encode(context.Background(), tc.req)
			if again == nil || again.Error() != err.Error() {
				t.Errorf("error is not deterministic: first %v, second %v", err, again)
			}
		})
	}
}

func checkNoMutation(t *testing.T, opts RepresentationOptions, corpus []string) {
	t.Helper()
	encoder := opts.NewEncoder(t)
	caps := encoder.Capabilities()

	outputs := append([]core.RepresentationKind(nil), caps.Outputs[:1]...)
	if caps.SupportsMultiOutput {
		outputs = append([]core.RepresentationKind(nil), caps.Outputs...)
	}
	input := append([]string(nil), corpus...)

	inputBefore := append([]string(nil), input...)
	outputsBefore := append([]core.RepresentationKind(nil), outputs...)

	if _, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:     input,
		InputType: preferredInputType(caps),
		Outputs:   outputs,
	}); err != nil {
		t.Fatalf("Encode: %v", err)
	}

	for i := range inputBefore {
		if input[i] != inputBefore[i] {
			t.Errorf("Encode mutated Input[%d]", i)
		}
	}
	for i := range outputsBefore {
		if outputs[i] != outputsBefore[i] {
			t.Errorf("Encode mutated Outputs[%d]", i)
		}
	}
}

func checkBatchLimit(t *testing.T, opts RepresentationOptions) {
	t.Helper()
	encoder := opts.NewEncoder(t)
	caps := encoder.Capabilities()
	limit := caps.MaximumBatchSize
	if limit <= 0 {
		t.Skip("encoder declares no batch limit")
	}
	if limit > 256 {
		t.Skipf("batch limit of %d is too large to probe cheaply", limit)
	}

	input := make([]string, limit+1)
	for i := range input {
		input[i] = "batch limit probe"
	}
	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:     input,
		InputType: preferredInputType(caps),
		Outputs:   []core.RepresentationKind{caps.Outputs[0]},
	})
	if !errors.Is(err, core.ErrInvalidRepresentationRequest) {
		t.Fatalf("%d inputs against a limit of %d: got %v, want ErrInvalidRepresentationRequest",
			limit+1, limit, err)
	}
}

// checkCancellation proves the encoder returns the context's own error rather
// than wrapping it into a generic provider failure. A retry wrapper that
// flattens context.Canceled makes a caller's shutdown path indistinguishable
// from a transient outage, and it will be retried as one.
func checkCancellation(t *testing.T, opts RepresentationOptions, corpus []string) {
	t.Helper()
	if opts.SkipCancellation {
		t.Skip("encoder does not observe cancellation")
	}
	encoder := opts.NewEncoder(t)
	caps := encoder.Capabilities()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := encoder.Encode(ctx, &core.RepresentationRequest{
		Input:     corpus[:1],
		InputType: preferredInputType(caps),
		Outputs:   []core.RepresentationKind{caps.Outputs[0]},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled context: got %v, want context.Canceled", err)
	}
}

func checkEmbedderProjection(t *testing.T, opts RepresentationOptions, corpus []string) {
	t.Helper()
	encoder := opts.NewEncoder(t)
	if !encoder.Capabilities().Supports(core.RepresentationDense) {
		t.Skip("encoder produces no dense output")
	}
	embedder, err := core.NewEncoderEmbedder(encoder)
	if err != nil {
		t.Fatalf("NewEncoderEmbedder: %v", err)
	}
	resp, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input:     corpus,
		InputType: preferredInputType(encoder.Capabilities()),
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != len(corpus) {
		t.Fatalf("got %d vectors for %d inputs", len(resp.Vectors), len(corpus))
	}
	width := len(resp.Vectors[0])
	if width == 0 {
		t.Fatal("projected vectors have zero width")
	}
	for i, vec := range resp.Vectors {
		if len(vec) != width {
			t.Errorf("vector %d has width %d, want %d", i, len(vec), width)
		}
	}
}

// checkResponse asserts the invariants a caller relies on for every response,
// independently of whether the provider validated its own output.
func checkResponse(t *testing.T, req *core.RepresentationRequest, resp *core.RepresentationResponse) {
	t.Helper()
	if resp == nil {
		t.Fatal("response is nil")
	}
	if len(resp.Data) != len(req.Input) {
		t.Fatalf("got %d representations for %d inputs", len(resp.Data), len(req.Input))
	}
	if resp.Model == "" {
		t.Error("response does not name a model")
	}
	checkUsage(t, resp.Usage)

	for _, kind := range req.Outputs {
		space, ok := resp.Spaces[kind]
		if !ok {
			t.Fatalf("response has no vector space for requested kind %s", kind)
		}
		if err := space.Validate(); err != nil {
			t.Errorf("vector space for %s: %v", kind, err)
		}
		if space.Kind != kind {
			t.Errorf("vector space for %s declares kind %s", kind, space.Kind)
		}
		if space.ID != "" && space.ID != space.CanonicalID() && space.Revision == "" {
			t.Logf("space %s has a caller-supplied ID and no revision; "+
				"a silent model change cannot be detected", kind)
		}
	}
	for kind := range resp.Spaces {
		if !req.Wants(kind) {
			t.Errorf("response describes unrequested vector space %s", kind)
		}
	}

	for i, item := range resp.Data {
		for _, kind := range req.Outputs {
			if !item.Has(kind) {
				t.Fatalf("item %d is missing requested representation %s", i, kind)
			}
			checkShape(t, i, kind, item, resp.Spaces[kind])
		}
		for _, kind := range item.Kinds() {
			if !req.Wants(kind) {
				t.Errorf("item %d carries unrequested representation %s", i, kind)
			}
		}
	}
}

func checkShape(t *testing.T, item int, kind core.RepresentationKind, rep core.Representation, space core.VectorSpace) {
	t.Helper()
	switch kind {
	case core.RepresentationDense:
		if len(rep.Dense) != space.Dimensions {
			t.Errorf("item %d dense width %d, space declares %d", item, len(rep.Dense), space.Dimensions)
		}
		checkFinite(t, item, "dense", rep.Dense)
	case core.RepresentationSparse:
		if len(rep.Sparse.Indices) != len(rep.Sparse.Values) {
			t.Errorf("item %d sparse has %d indices and %d values",
				item, len(rep.Sparse.Indices), len(rep.Sparse.Values))
			return
		}
		for j, index := range rep.Sparse.Indices {
			if j > 0 && index <= rep.Sparse.Indices[j-1] {
				t.Errorf("item %d sparse indices are not strictly increasing at %d", item, j)
			}
			if int(index) >= space.Dimensions {
				t.Errorf("item %d sparse index %d exceeds the declared vocabulary %d",
					item, index, space.Dimensions)
			}
		}
		checkFinite(t, item, "sparse", rep.Sparse.Values)
	case core.RepresentationMultiVector:
		for token, vec := range rep.MultiVector {
			if len(vec) != space.Dimensions {
				t.Errorf("item %d token vector %d width %d, space declares %d",
					item, token, len(vec), space.Dimensions)
			}
			checkFinite(t, item, "multi_vector", vec)
		}
	}
}

func checkFinite(t *testing.T, item int, kind string, values []float32) {
	t.Helper()
	for i, value := range values {
		v := float64(value)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("item %d %s value %d is not finite", item, kind, i)
			return
		}
	}
}

func checkUsage(t *testing.T, usage core.RepresentationUsage) {
	t.Helper()
	if usage.InputTokens < 0 || usage.InputBytes < 0 || usage.OutputBytes < 0 {
		t.Errorf("usage has a negative measurement: %+v", usage)
	}
	if usage.RequestCount < 1 {
		t.Errorf("usage reports %d provider requests for a completed call", usage.RequestCount)
	}
}

// preferredInputType picks a role the encoder accepts, favoring document
// because that is the side an index is built from.
func preferredInputType(caps core.RepresentationCapabilities) core.EmbeddingInputType {
	for _, inputType := range []core.EmbeddingInputType{
		core.EmbeddingInputDocument,
		core.EmbeddingInputNone,
		core.EmbeddingInputQuery,
	} {
		if caps.SupportsInputType(inputType) {
			return inputType
		}
	}
	return core.EmbeddingInputNone
}

func inputTypeName(inputType core.EmbeddingInputType) string {
	if inputType == core.EmbeddingInputNone {
		return "none"
	}
	return string(inputType)
}

// sameRepresentation compares two representations of one kind exactly. It is
// only used when the caller declares the encoder deterministic.
func sameRepresentation(kind core.RepresentationKind, a, b core.Representation) bool {
	switch kind {
	case core.RepresentationDense:
		return equalFloats(a.Dense, b.Dense)
	case core.RepresentationSparse:
		if a.Sparse.Len() != b.Sparse.Len() {
			return false
		}
		for i := range a.Sparse.Indices {
			if a.Sparse.Indices[i] != b.Sparse.Indices[i] {
				return false
			}
		}
		return equalFloats(a.Sparse.Values, b.Sparse.Values)
	case core.RepresentationMultiVector:
		if len(a.MultiVector) != len(b.MultiVector) {
			return false
		}
		for i := range a.MultiVector {
			if !equalFloats(a.MultiVector[i], b.MultiVector[i]) {
				return false
			}
		}
		return true
	}
	return false
}

func equalFloats(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
