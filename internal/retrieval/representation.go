package retrieval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// RepresentationKind names one of the vector shapes a retrieval model can
// produce for a single input.
//
// The three kinds are not interchangeable and are never converted into one
// another by this package: a sparse vector is not a dense vector with mostly
// zeroes elided, and a multi-vector is not a dense vector waiting to be pooled.
type RepresentationKind string

const (
	// RepresentationDense is a single fixed-width vector of floats, compared
	// with cosine or dot product.
	RepresentationDense RepresentationKind = "dense"

	// RepresentationSparse is a learned sparse vector: a small set of
	// vocabulary coordinates with nonzero weights, in the shape SPLADE,
	// BGE-M3's lexical head, and Pinecone's sparse models produce.
	RepresentationSparse RepresentationKind = "sparse"

	// RepresentationMultiVector is one vector per token, for late-interaction
	// scoring such as ColBERT's MaxSim.
	//
	// It is not a flattened dense vector. Averaging it into one vector throws
	// away exactly the information late interaction exists to use, so nothing
	// in this package does so implicitly.
	RepresentationMultiVector RepresentationKind = "multi_vector"
)

// Valid reports whether k is one of the three known kinds.
func (k RepresentationKind) Valid() bool {
	switch k {
	case RepresentationDense, RepresentationSparse, RepresentationMultiVector:
		return true
	}
	return false
}

// SimilarityMetric is how two values in a vector space are compared.
//
// Agentic reports the metric a space expects; it does not compute similarity.
// Scoring belongs to whichever index holds the vectors.
type SimilarityMetric string

const (
	// SimilarityCosine compares direction, ignoring magnitude.
	SimilarityCosine SimilarityMetric = "cosine"

	// SimilarityDotProduct compares direction and magnitude together, which is
	// what learned sparse weights need: a term's weight is part of its score.
	SimilarityDotProduct SimilarityMetric = "dot_product"
)

// Valid reports whether m is one of the known metrics.
func (m SimilarityMetric) Valid() bool {
	switch m {
	case SimilarityCosine, SimilarityDotProduct:
		return true
	}
	return false
}

// SparseVector is a learned sparse representation in coordinate form.
//
// Indices and Values are parallel: the pair at position i is one vocabulary
// coordinate and its weight. A learned sparse model has a large logical
// vocabulary (BGE-M3: 250002 tokens) but assigns nonzero weight to only a
// handful of coordinates per input, so storing the dense form would be
// thousands of times larger for no gain.
//
// Indices are strictly increasing. That makes the value canonical — one input
// has exactly one encoding — which in turn makes fixtures comparable and
// duplicate coordinates detectable rather than silently last-write-wins.
type SparseVector struct {
	// Indices are vocabulary coordinates, strictly increasing and within the
	// space's declared vocabulary size.
	Indices []uint32

	// Values are the weights at the matching Indices position. Every weight is
	// finite and nonzero; an explicit zero is a coordinate that should have
	// been omitted.
	Values []float32
}

// Len returns the number of nonzero coordinates, or 0 for a nil vector.
func (v *SparseVector) Len() int {
	if v == nil {
		return 0
	}
	return len(v.Indices)
}

// Clone returns a deep copy, or nil for a nil vector.
func (v *SparseVector) Clone() *SparseVector {
	if v == nil {
		return nil
	}
	out := &SparseVector{
		Indices: make([]uint32, len(v.Indices)),
		Values:  make([]float32, len(v.Values)),
	}
	copy(out.Indices, v.Indices)
	copy(out.Values, v.Values)
	return out
}

// RepresentationRequest is a batch encoding request.
//
// Batch is the only shape. A provider that natively accepts one input per call
// adapts to this contract rather than the other way round, so callers never
// have two code paths to keep in step.
type RepresentationRequest struct {
	// Input is the list of texts to encode. Must be non-empty. Provider batch
	// limits apply and are checked before transport.
	Input []string

	// InputType marks the inputs as queries or documents. Models with
	// asymmetric query and document encoders need this; models without the
	// concept ignore it.
	InputType EmbeddingInputType

	// Outputs selects which representations to compute. Must be non-empty,
	// free of duplicates, and supported by the encoder.
	//
	// Selection is explicit because the kinds differ in cost by orders of
	// magnitude: a multi-vector response for a long document is thousands of
	// floats per input, and no caller should pay for one by accident.
	Outputs []RepresentationKind

	// Truncate controls what a provider does with an input longer than the
	// model's context window. True truncates it to fit; false rejects the
	// request. Nil uses the provider's own default.
	Truncate *bool
}

// Validate checks that the request is well-formed independently of any
// provider. It applies the strict default for empty inputs; encoders that
// document support for empty strings use [RepresentationValidator] instead.
func (r *RepresentationRequest) Validate() error {
	return validateRequestShape(r, false)
}

// Clone returns a deep copy of the request, so a provider can normalize inputs
// without mutating the caller's slices.
func (r *RepresentationRequest) Clone() *RepresentationRequest {
	if r == nil {
		return nil
	}
	out := *r
	out.Input = make([]string, len(r.Input))
	copy(out.Input, r.Input)
	out.Outputs = make([]RepresentationKind, len(r.Outputs))
	copy(out.Outputs, r.Outputs)
	if r.Truncate != nil {
		truncate := *r.Truncate
		out.Truncate = &truncate
	}
	return &out
}

// Wants reports whether kind was requested.
func (r *RepresentationRequest) Wants(kind RepresentationKind) bool {
	if r == nil {
		return false
	}
	for _, k := range r.Outputs {
		if k == kind {
			return true
		}
	}
	return false
}

// Representation holds every representation computed for one input. Only the
// requested and supported fields are populated; the rest are zero.
type Representation struct {
	// Dense is a single fixed-width vector, set when RepresentationDense was
	// requested.
	Dense []float32

	// Sparse is the learned sparse vector, set when RepresentationSparse was
	// requested.
	Sparse *SparseVector

	// MultiVector is one vector per token, set when RepresentationMultiVector
	// was requested. Outer index is token position; every inner vector has the
	// same width.
	MultiVector [][]float32
}

// Has reports whether the representation for kind is populated.
func (r Representation) Has(kind RepresentationKind) bool {
	switch kind {
	case RepresentationDense:
		return len(r.Dense) > 0
	case RepresentationSparse:
		return r.Sparse != nil
	case RepresentationMultiVector:
		return len(r.MultiVector) > 0
	}
	return false
}

// Kinds returns the populated kinds in a stable order.
func (r Representation) Kinds() []RepresentationKind {
	var kinds []RepresentationKind
	for _, kind := range allRepresentationKinds {
		if r.Has(kind) {
			kinds = append(kinds, kind)
		}
	}
	return kinds
}

// allRepresentationKinds is the canonical iteration order for the three kinds,
// so error messages and space maps are reported the same way every time.
var allRepresentationKinds = []RepresentationKind{
	RepresentationDense,
	RepresentationSparse,
	RepresentationMultiVector,
}

// VectorSpace identifies the space a set of values lives in, closely enough
// that a consumer can refuse to query an index built with something else.
//
// The marketing model name is not enough. Two vectors from the same named
// model but different weights revisions, or two sparse vectors from different
// tokenizer vocabularies, are not comparable — and the failure is silent, in
// the form of quietly worse recall rather than an error. Persist the descriptor
// beside the vectors and compare before querying.
type VectorSpace struct {
	// ID is a stable opaque identifier for the space. Callers may supply their
	// own; [VectorSpace.WithCanonicalID] derives one from the fields below.
	ID string

	// Provider is the Agentic provider package that produced the values,
	// e.g. "deepinfra".
	Provider string

	// Model is the model identifier, e.g. "BAAI/bge-m3".
	Model string

	// Revision pins the model weights: a commit hash, an immutable endpoint
	// deployment revision, or another identifier the operator can prove does
	// not change under them. Empty means the provider could not prove one,
	// which is itself information — an index keyed on an unrevisioned space
	// cannot detect a silent model swap.
	Revision string

	// Tokenizer pins the tokenizer and its vocabulary. It matters most for
	// sparse output, where the index values *are* vocabulary positions and a
	// vocabulary change silently reassigns every coordinate.
	Tokenizer string

	// Kind is the representation kind this space holds.
	Kind RepresentationKind

	// Dimensions is the logical width: the vector width for dense, the
	// vocabulary size for sparse, and the per-token vector width for
	// multi-vector.
	Dimensions int

	// Metric is the similarity metric the space expects.
	Metric SimilarityMetric
}

// vectorSpaceKeyVersion prefixes the canonical key so that a future change to
// which fields are material produces different IDs rather than colliding with
// the old scheme.
const vectorSpaceKeyVersion = "agentic.vectorspace.v1"

// CanonicalKey returns the inspectable string form of the space's material
// compatibility fields, excluding ID itself.
//
// It is the input to [VectorSpace.CanonicalID]. Having it be readable is the
// point: when two IDs differ and an operator wants to know why, diffing two
// canonical keys answers the question without a lookup table.
func (s VectorSpace) CanonicalKey() string {
	var b strings.Builder
	b.WriteString(vectorSpaceKeyVersion)
	writeSpaceField(&b, "provider", s.Provider)
	writeSpaceField(&b, "model", s.Model)
	writeSpaceField(&b, "revision", s.Revision)
	writeSpaceField(&b, "tokenizer", s.Tokenizer)
	writeSpaceField(&b, "kind", string(s.Kind))
	writeSpaceField(&b, "dimensions", strconv.Itoa(s.Dimensions))
	writeSpaceField(&b, "metric", string(s.Metric))
	return b.String()
}

// writeSpaceField appends one quoted key/value line to the canonical key.
// Quoting keeps the encoding unambiguous when a model or revision identifier
// contains a newline or a quote.
func writeSpaceField(b *strings.Builder, name, value string) {
	b.WriteByte('\n')
	b.WriteString(name)
	b.WriteByte('=')
	b.WriteString(strconv.Quote(value))
}

// CanonicalID derives a deterministic ID from the material fields.
//
// The same fields always produce the same ID, on any machine and in any
// process, so two services can agree on space identity without coordinating.
// The hash is truncated to 128 bits, which is far more than enough to keep
// distinct spaces distinct and short enough to sit in a database column.
func (s VectorSpace) CanonicalID() string {
	sum := sha256.Sum256([]byte(s.CanonicalKey()))
	return "vs1_" + hex.EncodeToString(sum[:16])
}

// WithCanonicalID returns a copy of s with ID set to its canonical value,
// leaving an already-set ID alone.
func (s VectorSpace) WithCanonicalID() VectorSpace {
	if s.ID == "" {
		s.ID = s.CanonicalID()
	}
	return s
}

// Compatible reports whether values from s and other may share an index. It
// compares the material fields, not the ID, so a caller-supplied ID cannot
// assert a compatibility the underlying fields contradict.
func (s VectorSpace) Compatible(other VectorSpace) bool {
	return s.CanonicalKey() == other.CanonicalKey()
}

// String returns a short human-readable form for logs and errors. It carries
// no vector values and no input text.
func (s VectorSpace) String() string {
	return fmt.Sprintf("%s(%s/%s kind=%s dims=%d metric=%s)",
		s.ID, s.Provider, s.Model, s.Kind, s.Dimensions, s.Metric)
}

// Validate checks that the descriptor is internally consistent.
func (s VectorSpace) Validate() error {
	switch {
	case s.ID == "":
		return errors.New("vector space id cannot be empty")
	case s.Provider == "":
		return errors.New("vector space provider cannot be empty")
	case s.Model == "":
		return errors.New("vector space model cannot be empty")
	case !s.Kind.Valid():
		return fmt.Errorf("vector space kind %q is not a known representation kind", s.Kind)
	case s.Dimensions <= 0:
		return errors.New("vector space dimensions must be positive")
	case !s.Metric.Valid():
		return fmt.Errorf("vector space metric %q is not a known similarity metric", s.Metric)
	}
	return nil
}

// RepresentationUsage reports what a request consumed. These are measurements,
// not money: Agentic has no price table and would be wrong about it within a
// quarter if it did. Combine them with a dated external price catalog.
type RepresentationUsage struct {
	// InputTokens is the token count reported by the service, or zero when it
	// reports none. It is never estimated locally.
	InputTokens int

	// RequestCount is how many provider calls the response covers, which is
	// more than one when a batch was chunked. Endpoint-hour and per-request
	// billing shapes are invisible to token counts, so this is the only
	// measurement that makes them comparable.
	RequestCount int

	// InputBytes is the total UTF-8 size of the request inputs.
	InputBytes int

	// OutputBytes is the size of the provider's response payloads, where the
	// provider can observe it cheaply.
	OutputBytes int
}

// Add accumulates other into u.
func (u *RepresentationUsage) Add(other RepresentationUsage) {
	u.InputTokens += other.InputTokens
	u.RequestCount += other.RequestCount
	u.InputBytes += other.InputBytes
	u.OutputBytes += other.OutputBytes
}

// RepresentationResponse holds one Representation per request input, in input
// order.
type RepresentationResponse struct {
	// Data has exactly one entry per request input, Data[i] corresponding to
	// Input[i].
	Data []Representation

	// Spaces describes each populated output kind. Every requested kind has an
	// entry; no other kind does.
	Spaces map[RepresentationKind]VectorSpace

	// Model is the model name reported by the provider, or the configured name
	// for providers that do not echo one.
	Model string

	// Usage reports what the request consumed.
	Usage RepresentationUsage
}

// Space returns the descriptor for kind and whether it is present.
func (r *RepresentationResponse) Space(kind RepresentationKind) (VectorSpace, bool) {
	if r == nil {
		return VectorSpace{}, false
	}
	space, ok := r.Spaces[kind]
	return space, ok
}

// RepresentationCapabilities declares what an encoder can actually do.
//
// It is a runtime contract, not documentation: [RepresentationValidator]
// rejects requests that exceed it before transport, and the conformance suite
// checks that behavior matches the declaration. A provider that claims sparse
// output must return sparse output.
type RepresentationCapabilities struct {
	// Outputs lists the representation kinds the encoder can produce.
	Outputs []RepresentationKind

	// InputTypes lists the accepted input types. An encoder without the
	// query/document distinction lists only EmbeddingInputNone.
	InputTypes []EmbeddingInputType

	// MaximumBatchSize caps inputs per request. Zero means the encoder itself
	// imposes no limit that the caller must respect.
	MaximumBatchSize int

	// SupportsTruncation reports whether the encoder honors
	// RepresentationRequest.Truncate.
	SupportsTruncation bool

	// SupportsMultiOutput reports whether more than one kind can be requested
	// in a single call. When false, each kind costs a separate request.
	SupportsMultiOutput bool

	// AllowsEmptyInput reports whether an empty input string is accepted.
	// Most services reject one; a few encode it as a zero vector.
	AllowsEmptyInput bool

	// AllowsEmptySparse reports whether a sparse vector with no nonzero
	// coordinates is a documented outcome rather than a decoding bug. It is
	// true for encoders that filter stopwords and can legitimately empty a
	// short input.
	AllowsEmptySparse bool
}

// Supports reports whether kind is in Outputs.
func (c RepresentationCapabilities) Supports(kind RepresentationKind) bool {
	for _, k := range c.Outputs {
		if k == kind {
			return true
		}
	}
	return false
}

// SupportsInputType reports whether inputType is in InputTypes. An encoder
// that declares no input types accepts only EmbeddingInputNone.
func (c RepresentationCapabilities) SupportsInputType(inputType EmbeddingInputType) bool {
	if len(c.InputTypes) == 0 {
		return inputType == EmbeddingInputNone
	}
	for _, t := range c.InputTypes {
		if t == inputType {
			return true
		}
	}
	return false
}

// RepresentationEncoder is the core abstraction for models that turn text into
// one or more retrieval representations.
//
// It sits beside [Embedder] rather than replacing it. An Embedder returns one
// dense vector and is the right contract for the many models that produce only
// that; an encoder is for models like BGE-M3 that produce dense, sparse, and
// token-level output from a single forward pass, where issuing three requests
// to collect three views of the same text would be both slower and wrong.
//
// Implementations own inference and transport only. Indexing, lexical search,
// candidate fusion, and final ranking belong to the consuming retrieval
// system, which is why nothing here returns a score.
type RepresentationEncoder interface {
	// Encode returns one Representation per input, in input order, populating
	// exactly the requested kinds.
	Encode(ctx context.Context, req *RepresentationRequest) (*RepresentationResponse, error)

	// Name returns the model identifier, e.g. "BAAI/bge-m3".
	Name() string

	// Capabilities declares what this encoder can produce and accept.
	Capabilities() RepresentationCapabilities
}
