package test

import (
	"context"
	"hash/fnv"
	"sort"
	"strings"
	"sync"

	"github.com/regularkevvv/agentic/internal/core"
)

// Defaults for a TestRepresentationEncoder built without options.
const (
	defaultRepresentationDims  = 8
	defaultRepresentationVocab = 4096

	// defaultMaxTokenVectors caps how many token vectors one input produces,
	// so a pathological test input cannot allocate without bound.
	defaultMaxTokenVectors = 512
)

// TestRepresentationEncoder is a deterministic core.RepresentationEncoder for
// testing without API calls.
//
// Every output is a pure function of the input text, the input type, and the
// configured space, so equal inputs encode identically and different inputs
// encode differently. That is enough structure to test a retrieval pipeline
// end to end — a sparse vector here really does share coordinates with another
// input that shares words — without any of it depending on a live model.
//
// The zero value is not usable; call NewTestRepresentationEncoder.
type TestRepresentationEncoder struct {
	mu    sync.Mutex
	calls []core.RepresentationRequest

	name       string
	dims       int
	vocabulary int
	caps       core.RepresentationCapabilities
	revision   string
	tokenizer  string
	failure    func(call int, req *core.RepresentationRequest) error
}

// RepresentationOption configures a TestRepresentationEncoder.
type RepresentationOption func(*TestRepresentationEncoder)

// WithRepresentationName sets the model name the encoder reports.
func WithRepresentationName(name string) RepresentationOption {
	return func(e *TestRepresentationEncoder) { e.name = name }
}

// WithRepresentationDimensions sets the dense and per-token vector width.
func WithRepresentationDimensions(dims int) RepresentationOption {
	return func(e *TestRepresentationEncoder) {
		if dims > 0 {
			e.dims = dims
		}
	}
}

// WithRepresentationVocabulary sets the sparse vocabulary size, which is the
// declared Dimensions of the sparse space.
func WithRepresentationVocabulary(size int) RepresentationOption {
	return func(e *TestRepresentationEncoder) {
		if size > 0 {
			e.vocabulary = size
		}
	}
}

// WithRepresentationOutputs restricts which kinds the encoder produces.
// Requesting an omitted kind returns core.UnsupportedRepresentationError.
func WithRepresentationOutputs(kinds ...core.RepresentationKind) RepresentationOption {
	return func(e *TestRepresentationEncoder) {
		e.caps.Outputs = append([]core.RepresentationKind(nil), kinds...)
	}
}

// WithRepresentationInputTypes restricts which input types the encoder accepts.
func WithRepresentationInputTypes(types ...core.EmbeddingInputType) RepresentationOption {
	return func(e *TestRepresentationEncoder) {
		e.caps.InputTypes = append([]core.EmbeddingInputType(nil), types...)
	}
}

// WithRepresentationMaxBatchSize caps inputs per request, so a caller can
// exercise the batch-limit rejection path.
func WithRepresentationMaxBatchSize(size int) RepresentationOption {
	return func(e *TestRepresentationEncoder) { e.caps.MaximumBatchSize = size }
}

// WithRepresentationMultiOutput sets whether several kinds may be requested in
// one call. It defaults to true.
func WithRepresentationMultiOutput(supported bool) RepresentationOption {
	return func(e *TestRepresentationEncoder) { e.caps.SupportsMultiOutput = supported }
}

// WithRepresentationTruncation sets whether the encoder accepts the truncate
// option. It defaults to true.
func WithRepresentationTruncation(supported bool) RepresentationOption {
	return func(e *TestRepresentationEncoder) { e.caps.SupportsTruncation = supported }
}

// WithRepresentationEmptyInput sets whether empty input strings are accepted.
// An encoder that accepts them also documents empty sparse output, since an
// input with no words has no coordinates.
func WithRepresentationEmptyInput(allowed bool) RepresentationOption {
	return func(e *TestRepresentationEncoder) {
		e.caps.AllowsEmptyInput = allowed
		e.caps.AllowsEmptySparse = allowed
	}
}

// WithRepresentationRevision sets the model and tokenizer revisions recorded
// in the vector spaces, which changes the canonical space IDs.
func WithRepresentationRevision(model, tokenizer string) RepresentationOption {
	return func(e *TestRepresentationEncoder) {
		e.revision = model
		e.tokenizer = tokenizer
	}
}

// WithRepresentationError makes every Encode call fail with err.
func WithRepresentationError(err error) RepresentationOption {
	return WithRepresentationFailure(func(int, *core.RepresentationRequest) error { return err })
}

// WithRepresentationFailure installs a hook consulted before each Encode call,
// receiving the zero-based call index and the request. A non-nil return
// becomes the call's error, which is how a test drives a failure at a chosen
// point in a chunked batch.
func WithRepresentationFailure(fn func(call int, req *core.RepresentationRequest) error) RepresentationOption {
	return func(e *TestRepresentationEncoder) { e.failure = fn }
}

// NewTestRepresentationEncoder creates an encoder that produces all three
// representation kinds at 8 dimensions over a 4096-entry vocabulary, accepting
// every input type, unless options say otherwise.
func NewTestRepresentationEncoder(opts ...RepresentationOption) *TestRepresentationEncoder {
	e := &TestRepresentationEncoder{
		name:       "test:encoder",
		dims:       defaultRepresentationDims,
		vocabulary: defaultRepresentationVocab,
		revision:   "test-revision-1",
		tokenizer:  "test-tokenizer-1",
		caps: core.RepresentationCapabilities{
			Outputs: []core.RepresentationKind{
				core.RepresentationDense,
				core.RepresentationSparse,
				core.RepresentationMultiVector,
			},
			InputTypes: []core.EmbeddingInputType{
				core.EmbeddingInputNone,
				core.EmbeddingInputQuery,
				core.EmbeddingInputDocument,
			},
			SupportsTruncation:  true,
			SupportsMultiOutput: true,
		},
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// Name implements core.RepresentationEncoder.
func (e *TestRepresentationEncoder) Name() string { return e.name }

// Capabilities implements core.RepresentationEncoder.
func (e *TestRepresentationEncoder) Capabilities() core.RepresentationCapabilities {
	caps := e.caps
	caps.Outputs = append([]core.RepresentationKind(nil), e.caps.Outputs...)
	caps.InputTypes = append([]core.EmbeddingInputType(nil), e.caps.InputTypes...)
	return caps
}

// Space returns the vector space this encoder reports for kind.
func (e *TestRepresentationEncoder) Space(kind core.RepresentationKind) core.VectorSpace {
	space := core.VectorSpace{
		Provider:   "test",
		Model:      e.name,
		Revision:   e.revision,
		Tokenizer:  e.tokenizer,
		Kind:       kind,
		Dimensions: e.dims,
		Metric:     core.SimilarityCosine,
	}
	if kind == core.RepresentationSparse {
		space.Dimensions = e.vocabulary
		space.Metric = core.SimilarityDotProduct
	}
	return space.WithCanonicalID()
}

// Encode implements core.RepresentationEncoder.
func (e *TestRepresentationEncoder) Encode(ctx context.Context, req *core.RepresentationRequest) (*core.RepresentationResponse, error) {
	validator := core.RepresentationValidator{
		Provider:     e.name,
		Capabilities: e.Capabilities(),
		Limits:       core.DefaultRepresentationLimits(),
	}
	if err := validator.ValidateRequest(req); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	e.mu.Lock()
	call := len(e.calls)
	e.calls = append(e.calls, *req.Clone())
	failure := e.failure
	e.mu.Unlock()

	if failure != nil {
		if err := failure(call, req); err != nil {
			return nil, err
		}
	}

	spaces := make(map[core.RepresentationKind]core.VectorSpace, len(req.Outputs))
	for _, kind := range req.Outputs {
		spaces[kind] = e.Space(kind)
	}

	data := make([]core.Representation, len(req.Input))
	inputBytes := 0
	for i, text := range req.Input {
		inputBytes += len(text)
		for _, kind := range req.Outputs {
			switch kind {
			case core.RepresentationDense:
				data[i].Dense = deterministicVector(text, string(req.InputType), e.dims)
			case core.RepresentationSparse:
				data[i].Sparse = e.sparseVector(text, req.InputType)
			case core.RepresentationMultiVector:
				data[i].MultiVector = e.tokenVectors(text, req.InputType)
			}
		}
	}

	resp := &core.RepresentationResponse{
		Data:   data,
		Spaces: spaces,
		Model:  e.name,
		Usage: core.RepresentationUsage{
			InputTokens:  inputBytes,
			RequestCount: 1,
			InputBytes:   inputBytes,
		},
	}
	if err := validator.ValidateResponse(req, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// Calls returns deep copies of all requests received, so that inspecting the
// history cannot rewrite it.
func (e *TestRepresentationEncoder) Calls() []core.RepresentationRequest {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]core.RepresentationRequest, len(e.calls))
	for i := range e.calls {
		out[i] = *e.calls[i].Clone()
	}
	return out
}

// CallCount returns the number of Encode calls made.
func (e *TestRepresentationEncoder) CallCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.calls)
}

// Reset clears the recorded calls.
func (e *TestRepresentationEncoder) Reset() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.calls = nil
}

// sparseVector derives a canonical sparse vector from the words in text.
//
// Each distinct word hashes to one vocabulary coordinate and carries a weight
// that grows with how often the word occurs, which reproduces the property a
// real learned sparse model has and a retrieval test needs: two inputs sharing
// a rare word share a coordinate, and an input repeating it weighs it higher.
// Two words that collide onto one coordinate combine by taking the larger
// weight, so the result stays strictly increasing and duplicate-free.
func (e *TestRepresentationEncoder) sparseVector(text string, inputType core.EmbeddingInputType) *core.SparseVector {
	counts := make(map[uint32]float32)
	words := strings.Fields(strings.ToLower(text))
	for _, word := range words {
		index := uint32(hashString(word, string(inputType)) % uint64(e.vocabulary))
		counts[index]++
	}

	indices := make([]uint32, 0, len(counts))
	for index := range counts {
		indices = append(indices, index)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })

	vec := &core.SparseVector{
		Indices: indices,
		Values:  make([]float32, len(indices)),
	}
	for i, index := range indices {
		vec.Values[i] = counts[index]
	}
	return vec
}

// tokenVectors derives one vector per word, for late-interaction scoring. An
// input with no words yields a single vector for the input as a whole, because
// an empty multi-vector is not a valid representation.
func (e *TestRepresentationEncoder) tokenVectors(text string, inputType core.EmbeddingInputType) [][]float32 {
	words := strings.Fields(strings.ToLower(text))
	if len(words) == 0 {
		words = []string{text}
	}
	if len(words) > defaultMaxTokenVectors {
		words = words[:defaultMaxTokenVectors]
	}
	vectors := make([][]float32, len(words))
	for i, word := range words {
		vectors[i] = deterministicVector(word, string(inputType), e.dims)
	}
	return vectors
}

func hashString(parts ...string) uint64 {
	h := fnv.New64a()
	for _, part := range parts {
		_, _ = h.Write([]byte(part))
		_, _ = h.Write([]byte{0})
	}
	return h.Sum64()
}

// Compile-time check that TestRepresentationEncoder implements the contract.
var _ core.RepresentationEncoder = (*TestRepresentationEncoder)(nil)
