package retrieval

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// Sentinel errors for the representation contract. Every typed error below
// matches one of these under errors.Is, so a caller can branch on the class of
// failure without depending on the concrete type.
var (
	// ErrUnsupportedRepresentation reports a kind the encoder cannot produce.
	ErrUnsupportedRepresentation = errors.New("unsupported representation")

	// ErrInvalidRepresentationRequest reports a request that violates the
	// contract before any transport happens.
	ErrInvalidRepresentationRequest = errors.New("invalid representation request")

	// ErrInvalidRepresentationResponse reports a provider response that
	// violates the contract after decoding.
	ErrInvalidRepresentationResponse = errors.New("invalid representation response")
)

// UnsupportedRepresentationError reports that a caller asked an encoder for a
// representation kind it does not produce.
type UnsupportedRepresentationError struct {
	// Provider is the encoder that rejected the request.
	Provider string

	// Kind is the unsupported kind.
	Kind RepresentationKind

	// Supported lists what the encoder does produce, so the error is
	// actionable without a second lookup.
	Supported []RepresentationKind
}

func (e *UnsupportedRepresentationError) Error() string {
	if len(e.Supported) == 0 {
		return fmt.Sprintf("%s: does not support the %s representation", e.Provider, e.Kind)
	}
	names := make([]string, len(e.Supported))
	for i, kind := range e.Supported {
		names[i] = string(kind)
	}
	return fmt.Sprintf("%s: does not support the %s representation (supports %s)",
		e.Provider, e.Kind, strings.Join(names, ", "))
}

// Unwrap makes the error match ErrUnsupportedRepresentation under errors.Is.
func (e *UnsupportedRepresentationError) Unwrap() error { return ErrUnsupportedRepresentation }

// InvalidRepresentationRequestError reports a request that violates the
// contract. It names the invariant rather than echoing the input, so the error
// is safe to log next to user documents.
type InvalidRepresentationRequestError struct {
	// Invariant is a short stable identifier for the rule that was broken,
	// e.g. "input.empty" or "outputs.duplicate".
	Invariant string

	// Detail explains the violation without reproducing input text.
	Detail string
}

func (e *InvalidRepresentationRequestError) Error() string {
	return fmt.Sprintf("invalid representation request [%s]: %s", e.Invariant, e.Detail)
}

// Unwrap makes the error match ErrInvalidRepresentationRequest under errors.Is.
func (e *InvalidRepresentationRequestError) Unwrap() error { return ErrInvalidRepresentationRequest }

// InvalidRepresentationResponseError reports a provider response that cannot
// be trusted: wrong cardinality, wrong shape, or values that would corrupt an
// index if stored.
//
// It carries positions and shapes only. A malformed response body is not
// echoed, because it may contain the caller's documents.
type InvalidRepresentationResponseError struct {
	// Provider is the encoder that produced the response.
	Provider string

	// Item is the index of the offending input, or -1 when the problem is with
	// the response as a whole.
	Item int

	// Kind is the offending representation kind, or empty when the problem is
	// not specific to one.
	Kind RepresentationKind

	// Problem describes the shape violation.
	Problem string
}

func (e *InvalidRepresentationResponseError) Error() string {
	var b strings.Builder
	b.WriteString(e.Provider)
	b.WriteString(": invalid representation response")
	if e.Item >= 0 {
		fmt.Fprintf(&b, " for item %d", e.Item)
	}
	if e.Kind != "" {
		fmt.Fprintf(&b, " (%s)", e.Kind)
	}
	b.WriteString(": ")
	b.WriteString(e.Problem)
	return b.String()
}

// Unwrap makes the error match ErrInvalidRepresentationResponse under
// errors.Is.
func (e *InvalidRepresentationResponseError) Unwrap() error {
	return ErrInvalidRepresentationResponse
}

// RepresentationLimits bounds the size of what a provider will send and
// accept.
//
// The limits exist because the multi-vector kind makes a small request into a
// very large response: a 512-token document at 1024 dimensions is half a
// million floats, and a batch of a hundred of them decoded without a ceiling
// is a straightforward way to exhaust a process's memory from a remote
// response body.
type RepresentationLimits struct {
	// MaxInputs caps items per request. Zero disables the check.
	MaxInputs int

	// MaxInputBytes caps the UTF-8 size of a single input. Zero disables the
	// check.
	MaxInputBytes int

	// MaxTotalInputBytes caps the summed size of all inputs in a request.
	// Zero disables the check.
	MaxTotalInputBytes int

	// MaxSparseNonZero caps nonzero coordinates per sparse vector. Zero
	// disables the check.
	MaxSparseNonZero int

	// MaxTokenVectors caps token vectors per multi-vector representation.
	// Zero disables the check.
	MaxTokenVectors int
}

// DefaultRepresentationLimits returns limits generous enough for normal
// retrieval workloads and tight enough that a malformed or hostile response
// cannot exhaust memory before validation runs.
func DefaultRepresentationLimits() RepresentationLimits {
	return RepresentationLimits{
		MaxInputs:          2048,
		MaxInputBytes:      1 << 20,  // 1 MiB per input
		MaxTotalInputBytes: 32 << 20, // 32 MiB per request
		MaxSparseNonZero:   65536,
		MaxTokenVectors:    8192,
	}
}

// RepresentationValidator applies the provider-independent parts of the
// representation contract. Every encoder runs the same checks, including the
// deterministic test double, so a consumer sees identical error behavior from
// a fake and from a live provider.
type RepresentationValidator struct {
	// Provider names the encoder in error messages.
	Provider string

	// Capabilities is what the encoder declares it can do.
	Capabilities RepresentationCapabilities

	// Limits bounds request and response size. The zero value disables every
	// bound; use DefaultRepresentationLimits for a sensible ceiling.
	Limits RepresentationLimits
}

// ValidateRequest checks a request against the contract and the encoder's
// declared capabilities, before any transport happens.
func (v RepresentationValidator) ValidateRequest(req *RepresentationRequest) error {
	if err := validateRequestShape(req, v.Capabilities.AllowsEmptyInput); err != nil {
		return err
	}

	if !v.Capabilities.SupportsInputType(req.InputType) {
		return &InvalidRepresentationRequestError{
			Invariant: "input_type.unsupported",
			Detail: fmt.Sprintf("%s does not accept input type %q",
				v.Provider, string(req.InputType)),
		}
	}

	for _, kind := range req.Outputs {
		if !v.Capabilities.Supports(kind) {
			return &UnsupportedRepresentationError{
				Provider:  v.Provider,
				Kind:      kind,
				Supported: v.Capabilities.Outputs,
			}
		}
	}

	if len(req.Outputs) > 1 && !v.Capabilities.SupportsMultiOutput {
		return &InvalidRepresentationRequestError{
			Invariant: "outputs.multi_unsupported",
			Detail: fmt.Sprintf("%s returns one representation kind per request, got %d",
				v.Provider, len(req.Outputs)),
		}
	}

	if req.Truncate != nil && !v.Capabilities.SupportsTruncation {
		return &InvalidRepresentationRequestError{
			Invariant: "truncate.unsupported",
			Detail:    fmt.Sprintf("%s does not support the truncate option", v.Provider),
		}
	}

	if max := v.Capabilities.MaximumBatchSize; max > 0 && len(req.Input) > max {
		return &InvalidRepresentationRequestError{
			Invariant: "input.batch_limit",
			Detail: fmt.Sprintf("%s accepts at most %d inputs per request, got %d",
				v.Provider, max, len(req.Input)),
		}
	}

	return v.validateRequestSize(req)
}

// validateRequestSize applies the byte ceilings. It reports sizes and
// positions only; input text never reaches the error.
func (v RepresentationValidator) validateRequestSize(req *RepresentationRequest) error {
	limits := v.Limits
	if limits.MaxInputs > 0 && len(req.Input) > limits.MaxInputs {
		return &InvalidRepresentationRequestError{
			Invariant: "input.limit",
			Detail: fmt.Sprintf("request has %d inputs, limit is %d",
				len(req.Input), limits.MaxInputs),
		}
	}

	total := 0
	for i, text := range req.Input {
		if limits.MaxInputBytes > 0 && len(text) > limits.MaxInputBytes {
			return &InvalidRepresentationRequestError{
				Invariant: "input.item_bytes",
				Detail: fmt.Sprintf("input %d is %d bytes, limit is %d",
					i, len(text), limits.MaxInputBytes),
			}
		}
		total += len(text)
	}
	if limits.MaxTotalInputBytes > 0 && total > limits.MaxTotalInputBytes {
		return &InvalidRepresentationRequestError{
			Invariant: "input.total_bytes",
			Detail: fmt.Sprintf("request is %d bytes, limit is %d",
				total, limits.MaxTotalInputBytes),
		}
	}
	return nil
}

// validateRequestShape checks the parts of the contract that hold for every
// encoder, with the empty-input rule as the one provider-dependent knob.
func validateRequestShape(req *RepresentationRequest, allowEmptyInput bool) error {
	if req == nil {
		return &InvalidRepresentationRequestError{
			Invariant: "request.nil",
			Detail:    "request cannot be nil",
		}
	}
	if len(req.Input) == 0 {
		return &InvalidRepresentationRequestError{
			Invariant: "input.empty",
			Detail:    "input cannot be empty",
		}
	}
	if !allowEmptyInput {
		for i, text := range req.Input {
			if text == "" {
				return &InvalidRepresentationRequestError{
					Invariant: "input.empty_item",
					Detail:    fmt.Sprintf("input %d is an empty string", i),
				}
			}
		}
	}
	switch req.InputType {
	case EmbeddingInputNone, EmbeddingInputQuery, EmbeddingInputDocument:
	default:
		return &InvalidRepresentationRequestError{
			Invariant: "input_type.unknown",
			Detail:    "input type must be query, document, or empty",
		}
	}
	if len(req.Outputs) == 0 {
		return &InvalidRepresentationRequestError{
			Invariant: "outputs.empty",
			Detail:    "outputs cannot be empty",
		}
	}
	seen := make(map[RepresentationKind]bool, len(req.Outputs))
	for _, kind := range req.Outputs {
		if !kind.Valid() {
			return &InvalidRepresentationRequestError{
				Invariant: "outputs.unknown",
				Detail:    fmt.Sprintf("output kind %q is not dense, sparse, or multi_vector", string(kind)),
			}
		}
		if seen[kind] {
			return &InvalidRepresentationRequestError{
				Invariant: "outputs.duplicate",
				Detail:    fmt.Sprintf("output kind %s is listed more than once", kind),
			}
		}
		seen[kind] = true
	}
	return nil
}

// ValidateResponse checks a decoded response against the request that produced
// it. A response that fails here is discarded whole: a partially valid batch
// written into an index is worse than an error, because the damage is silent.
func (v RepresentationValidator) ValidateResponse(req *RepresentationRequest, resp *RepresentationResponse) error {
	if resp == nil {
		return &InvalidRepresentationResponseError{
			Provider: v.Provider,
			Item:     -1,
			Problem:  "response is nil",
		}
	}
	if len(resp.Data) != len(req.Input) {
		return &InvalidRepresentationResponseError{
			Provider: v.Provider,
			Item:     -1,
			Problem: fmt.Sprintf("got %d representations for %d inputs",
				len(resp.Data), len(req.Input)),
		}
	}
	if err := v.validateSpaces(req, resp); err != nil {
		return err
	}
	for i := range resp.Data {
		if err := v.validateItem(req, resp, i); err != nil {
			return err
		}
	}
	return nil
}

// validateSpaces checks that the response describes exactly the requested
// kinds, and that each descriptor is usable as a persisted index key.
func (v RepresentationValidator) validateSpaces(req *RepresentationRequest, resp *RepresentationResponse) error {
	for _, kind := range req.Outputs {
		space, ok := resp.Spaces[kind]
		if !ok {
			return &InvalidRepresentationResponseError{
				Provider: v.Provider,
				Item:     -1,
				Kind:     kind,
				Problem:  "response describes no vector space for a requested output",
			}
		}
		if space.Kind != kind {
			return &InvalidRepresentationResponseError{
				Provider: v.Provider,
				Item:     -1,
				Kind:     kind,
				Problem:  fmt.Sprintf("vector space declares kind %q", string(space.Kind)),
			}
		}
		if err := space.Validate(); err != nil {
			return &InvalidRepresentationResponseError{
				Provider: v.Provider,
				Item:     -1,
				Kind:     kind,
				Problem:  err.Error(),
			}
		}
	}
	for kind := range resp.Spaces {
		if !req.Wants(kind) {
			return &InvalidRepresentationResponseError{
				Provider: v.Provider,
				Item:     -1,
				Kind:     kind,
				Problem:  "response describes a vector space that was not requested",
			}
		}
	}
	return nil
}

// validateItem checks one input's representations: every requested kind
// present, no unrequested kind present, and every value storable.
func (v RepresentationValidator) validateItem(req *RepresentationRequest, resp *RepresentationResponse, i int) error {
	item := resp.Data[i]
	for _, kind := range allRepresentationKinds {
		wanted := req.Wants(kind)
		if item.Has(kind) && !wanted {
			return &InvalidRepresentationResponseError{
				Provider: v.Provider,
				Item:     i,
				Kind:     kind,
				Problem:  "representation was returned but not requested",
			}
		}
		if !wanted {
			continue
		}
		space := resp.Spaces[kind]

		var err error
		switch kind {
		case RepresentationDense:
			err = v.validateDense(item.Dense, space, i)
		case RepresentationSparse:
			err = v.validateSparse(item.Sparse, space, i)
		case RepresentationMultiVector:
			err = v.validateMultiVector(item.MultiVector, space, i)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (v RepresentationValidator) validateDense(vec []float32, space VectorSpace, i int) error {
	fail := func(problem string) error {
		return &InvalidRepresentationResponseError{
			Provider: v.Provider,
			Item:     i,
			Kind:     RepresentationDense,
			Problem:  problem,
		}
	}
	if len(vec) == 0 {
		return fail("dense vector is missing")
	}
	if len(vec) != space.Dimensions {
		return fail(fmt.Sprintf("dense vector has width %d, space declares %d",
			len(vec), space.Dimensions))
	}
	if pos, ok := firstNonFinite(vec); ok {
		return fail(fmt.Sprintf("dense value at position %d is not finite", pos))
	}
	return nil
}

func (v RepresentationValidator) validateSparse(vec *SparseVector, space VectorSpace, i int) error {
	fail := func(problem string) error {
		return &InvalidRepresentationResponseError{
			Provider: v.Provider,
			Item:     i,
			Kind:     RepresentationSparse,
			Problem:  problem,
		}
	}
	if vec == nil {
		return fail("sparse vector is missing")
	}
	if len(vec.Indices) != len(vec.Values) {
		return fail(fmt.Sprintf("sparse vector has %d indices and %d values",
			len(vec.Indices), len(vec.Values)))
	}
	if len(vec.Indices) == 0 && !v.Capabilities.AllowsEmptySparse {
		return fail("sparse vector has no nonzero coordinates")
	}
	if limit := v.Limits.MaxSparseNonZero; limit > 0 && len(vec.Indices) > limit {
		return fail(fmt.Sprintf("sparse vector has %d nonzero coordinates, limit is %d",
			len(vec.Indices), limit))
	}
	for j, index := range vec.Indices {
		if j > 0 && index <= vec.Indices[j-1] {
			return fail(fmt.Sprintf("sparse indices are not strictly increasing at position %d", j))
		}
		if space.Dimensions > 0 && int64(index) >= int64(space.Dimensions) {
			return fail(fmt.Sprintf("sparse index %d at position %d is outside the declared vocabulary of %d",
				index, j, space.Dimensions))
		}
		value := vec.Values[j]
		if !isFinite(value) {
			return fail(fmt.Sprintf("sparse value at position %d is not finite", j))
		}
		if value == 0 {
			return fail(fmt.Sprintf("sparse value at position %d is zero", j))
		}
	}
	return nil
}

func (v RepresentationValidator) validateMultiVector(vectors [][]float32, space VectorSpace, i int) error {
	fail := func(problem string) error {
		return &InvalidRepresentationResponseError{
			Provider: v.Provider,
			Item:     i,
			Kind:     RepresentationMultiVector,
			Problem:  problem,
		}
	}
	if len(vectors) == 0 {
		return fail("multi-vector representation is missing")
	}
	if limit := v.Limits.MaxTokenVectors; limit > 0 && len(vectors) > limit {
		return fail(fmt.Sprintf("multi-vector representation has %d token vectors, limit is %d",
			len(vectors), limit))
	}
	for t, vec := range vectors {
		if len(vec) != space.Dimensions {
			return fail(fmt.Sprintf("token vector %d has width %d, space declares %d",
				t, len(vec), space.Dimensions))
		}
		if pos, ok := firstNonFinite(vec); ok {
			return fail(fmt.Sprintf("token vector %d has a non-finite value at position %d", t, pos))
		}
	}
	return nil
}

// firstNonFinite returns the position of the first NaN or infinity in vec.
//
// NaN and infinity are checked rather than tolerated because they survive
// storage: a single NaN coordinate poisons every distance computed against
// that vector, and the index reports no error while quietly returning nothing.
func firstNonFinite(vec []float32) (int, bool) {
	for i, value := range vec {
		if !isFinite(value) {
			return i, true
		}
	}
	return 0, false
}

func isFinite(value float32) bool {
	v := float64(value)
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
