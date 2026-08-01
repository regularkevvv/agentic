package core

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
)

func denseSpace(dims int) VectorSpace {
	return VectorSpace{
		Provider:   "test",
		Model:      "model",
		Revision:   "rev-1",
		Tokenizer:  "tok-1",
		Kind:       RepresentationDense,
		Dimensions: dims,
		Metric:     SimilarityCosine,
	}.WithCanonicalID()
}

func sparseSpace(vocab int) VectorSpace {
	return VectorSpace{
		Provider:   "test",
		Model:      "model",
		Revision:   "rev-1",
		Tokenizer:  "tok-1",
		Kind:       RepresentationSparse,
		Dimensions: vocab,
		Metric:     SimilarityDotProduct,
	}.WithCanonicalID()
}

func multiVectorSpace(dims int) VectorSpace {
	return VectorSpace{
		Provider:   "test",
		Model:      "model",
		Revision:   "rev-1",
		Tokenizer:  "tok-1",
		Kind:       RepresentationMultiVector,
		Dimensions: dims,
		Metric:     SimilarityCosine,
	}.WithCanonicalID()
}

func TestRepresentationKindValid(t *testing.T) {
	for _, kind := range []RepresentationKind{RepresentationDense, RepresentationSparse, RepresentationMultiVector} {
		if !kind.Valid() {
			t.Errorf("%s should be valid", kind)
		}
	}
	for _, kind := range []RepresentationKind{"", "colbert", "Dense"} {
		if RepresentationKind(kind).Valid() {
			t.Errorf("%q should not be valid", kind)
		}
	}
}

func TestSimilarityMetricValid(t *testing.T) {
	if !SimilarityCosine.Valid() || !SimilarityDotProduct.Valid() {
		t.Error("known metrics should be valid")
	}
	if SimilarityMetric("euclidean").Valid() {
		t.Error("euclidean should not be valid")
	}
}

func TestSparseVectorLenAndClone(t *testing.T) {
	var nilVec *SparseVector
	if nilVec.Len() != 0 {
		t.Error("nil sparse vector should have length 0")
	}
	if nilVec.Clone() != nil {
		t.Error("cloning a nil sparse vector should return nil")
	}

	vec := &SparseVector{Indices: []uint32{1, 5}, Values: []float32{0.5, 0.25}}
	clone := vec.Clone()
	if clone.Len() != 2 {
		t.Fatalf("clone length = %d, want 2", clone.Len())
	}
	clone.Indices[0] = 99
	clone.Values[0] = 9
	if vec.Indices[0] != 1 || vec.Values[0] != 0.5 {
		t.Error("clone shares backing arrays with the original")
	}
}

func TestRepresentationHasAndKinds(t *testing.T) {
	empty := Representation{}
	for _, kind := range allRepresentationKinds {
		if empty.Has(kind) {
			t.Errorf("empty representation should not have %s", kind)
		}
	}
	if len(empty.Kinds()) != 0 {
		t.Error("empty representation should report no kinds")
	}
	if empty.Has(RepresentationKind("unknown")) {
		t.Error("unknown kind should never be reported as present")
	}

	full := Representation{
		Dense:       []float32{1},
		Sparse:      &SparseVector{Indices: []uint32{0}, Values: []float32{1}},
		MultiVector: [][]float32{{1}},
	}
	kinds := full.Kinds()
	if len(kinds) != 3 || kinds[0] != RepresentationDense || kinds[2] != RepresentationMultiVector {
		t.Fatalf("Kinds() = %v, want dense, sparse, multi_vector in order", kinds)
	}
}

func TestRepresentationRequestCloneIsDeep(t *testing.T) {
	var nilReq *RepresentationRequest
	if nilReq.Clone() != nil {
		t.Error("cloning a nil request should return nil")
	}

	truncate := true
	req := &RepresentationRequest{
		Input:     []string{"a", "b"},
		InputType: EmbeddingInputQuery,
		Outputs:   []RepresentationKind{RepresentationDense},
		Truncate:  &truncate,
	}
	clone := req.Clone()
	clone.Input[0] = "changed"
	clone.Outputs[0] = RepresentationSparse
	*clone.Truncate = false

	if req.Input[0] != "a" {
		t.Error("clone shares the input slice")
	}
	if req.Outputs[0] != RepresentationDense {
		t.Error("clone shares the outputs slice")
	}
	if !*req.Truncate {
		t.Error("clone shares the truncate pointer")
	}
}

func TestRepresentationRequestWants(t *testing.T) {
	var nilReq *RepresentationRequest
	if nilReq.Wants(RepresentationDense) {
		t.Error("nil request wants nothing")
	}
	req := &RepresentationRequest{Outputs: []RepresentationKind{RepresentationSparse}}
	if !req.Wants(RepresentationSparse) || req.Wants(RepresentationDense) {
		t.Error("Wants does not match Outputs")
	}
}

func TestRepresentationRequestValidate(t *testing.T) {
	valid := &RepresentationRequest{
		Input:   []string{"hello"},
		Outputs: []RepresentationKind{RepresentationDense},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid request: %v", err)
	}

	tests := []struct {
		name      string
		req       *RepresentationRequest
		invariant string
	}{
		{"nil", nil, "request.nil"},
		{"no input", &RepresentationRequest{Outputs: []RepresentationKind{RepresentationDense}}, "input.empty"},
		{"empty input string", &RepresentationRequest{
			Input:   []string{"ok", ""},
			Outputs: []RepresentationKind{RepresentationDense},
		}, "input.empty_item"},
		{"unknown input type", &RepresentationRequest{
			Input:     []string{"hello"},
			InputType: EmbeddingInputType("passage"),
			Outputs:   []RepresentationKind{RepresentationDense},
		}, "input_type.unknown"},
		{"no outputs", &RepresentationRequest{Input: []string{"hello"}}, "outputs.empty"},
		{"unknown output", &RepresentationRequest{
			Input:   []string{"hello"},
			Outputs: []RepresentationKind{"colbert"},
		}, "outputs.unknown"},
		{"duplicate output", &RepresentationRequest{
			Input:   []string{"hello"},
			Outputs: []RepresentationKind{RepresentationDense, RepresentationDense},
		}, "outputs.duplicate"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.req.Validate()
			var typed *InvalidRepresentationRequestError
			if !errors.As(err, &typed) {
				t.Fatalf("got %v, want *InvalidRepresentationRequestError", err)
			}
			if typed.Invariant != tc.invariant {
				t.Errorf("invariant = %q, want %q", typed.Invariant, tc.invariant)
			}
			if !errors.Is(err, ErrInvalidRepresentationRequest) {
				t.Error("error does not match ErrInvalidRepresentationRequest")
			}
		})
	}
}

func TestVectorSpaceCanonicalIdentity(t *testing.T) {
	space := denseSpace(4)
	if !strings.HasPrefix(space.ID, "vs1_") {
		t.Errorf("canonical ID %q does not carry the scheme prefix", space.ID)
	}
	if space.ID != space.CanonicalID() {
		t.Error("WithCanonicalID did not assign the canonical ID")
	}

	// An already-set ID is left alone: callers with an existing index key
	// must be able to keep using it.
	custom := VectorSpace{ID: "mine", Provider: "p", Model: "m", Kind: RepresentationDense, Dimensions: 4, Metric: SimilarityCosine}
	if custom.WithCanonicalID().ID != "mine" {
		t.Error("WithCanonicalID overwrote a caller-supplied ID")
	}

	// Every material field participates in identity.
	base := VectorSpace{Provider: "p", Model: "m", Revision: "r", Tokenizer: "t", Kind: RepresentationDense, Dimensions: 4, Metric: SimilarityCosine}
	mutations := []struct {
		name   string
		mutate func(VectorSpace) VectorSpace
	}{
		{"provider", func(s VectorSpace) VectorSpace { s.Provider = "other"; return s }},
		{"model", func(s VectorSpace) VectorSpace { s.Model = "other"; return s }},
		{"revision", func(s VectorSpace) VectorSpace { s.Revision = "other"; return s }},
		{"tokenizer", func(s VectorSpace) VectorSpace { s.Tokenizer = "other"; return s }},
		{"kind", func(s VectorSpace) VectorSpace { s.Kind = RepresentationSparse; return s }},
		{"dimensions", func(s VectorSpace) VectorSpace { s.Dimensions = 8; return s }},
		{"metric", func(s VectorSpace) VectorSpace { s.Metric = SimilarityDotProduct; return s }},
	}
	for _, m := range mutations {
		t.Run(m.name, func(t *testing.T) {
			changed := m.mutate(base)
			if changed.CanonicalID() == base.CanonicalID() {
				t.Errorf("changing %s did not change the canonical ID", m.name)
			}
			if base.Compatible(changed) {
				t.Errorf("spaces differing in %s should not be compatible", m.name)
			}
		})
	}

	if !base.Compatible(base) {
		t.Error("a space should be compatible with itself")
	}
	// A caller-supplied ID cannot assert a compatibility the fields deny.
	relabeled := base
	relabeled.ID = base.WithCanonicalID().ID
	relabeled.Tokenizer = "different"
	if base.Compatible(relabeled) {
		t.Error("identical IDs must not override differing material fields")
	}
}

// A quoted canonical key keeps two different field sets from colliding when a
// value contains the separator characters.
func TestVectorSpaceCanonicalKeyIsUnambiguous(t *testing.T) {
	a := VectorSpace{Provider: "p", Model: "m\nrevision=x", Kind: RepresentationDense, Dimensions: 1, Metric: SimilarityCosine}
	b := VectorSpace{Provider: "p", Model: "m", Revision: "x", Kind: RepresentationDense, Dimensions: 1, Metric: SimilarityCosine}
	if a.CanonicalKey() == b.CanonicalKey() {
		t.Fatal("a newline in a field value collided two distinct spaces")
	}
	if !strings.Contains(a.CanonicalKey(), vectorSpaceKeyVersion) {
		t.Error("canonical key does not carry its scheme version")
	}
}

func TestVectorSpaceValidate(t *testing.T) {
	valid := denseSpace(4)
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid space: %v", err)
	}
	if !strings.Contains(valid.String(), "dense") {
		t.Errorf("String() = %q, want the kind included", valid.String())
	}

	tests := []struct {
		name  string
		space VectorSpace
		want  string
	}{
		{"no id", VectorSpace{Provider: "p", Model: "m", Kind: RepresentationDense, Dimensions: 1, Metric: SimilarityCosine}, "id cannot be empty"},
		{"no provider", VectorSpace{ID: "x", Model: "m", Kind: RepresentationDense, Dimensions: 1, Metric: SimilarityCosine}, "provider cannot be empty"},
		{"no model", VectorSpace{ID: "x", Provider: "p", Kind: RepresentationDense, Dimensions: 1, Metric: SimilarityCosine}, "model cannot be empty"},
		{"bad kind", VectorSpace{ID: "x", Provider: "p", Model: "m", Kind: "colbert", Dimensions: 1, Metric: SimilarityCosine}, "not a known representation kind"},
		{"zero dimensions", VectorSpace{ID: "x", Provider: "p", Model: "m", Kind: RepresentationDense, Metric: SimilarityCosine}, "dimensions must be positive"},
		{"bad metric", VectorSpace{ID: "x", Provider: "p", Model: "m", Kind: RepresentationDense, Dimensions: 1, Metric: "l2"}, "not a known similarity metric"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.space.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestRepresentationUsageAdd(t *testing.T) {
	usage := RepresentationUsage{InputTokens: 1, RequestCount: 1, InputBytes: 10, OutputBytes: 100}
	usage.Add(RepresentationUsage{InputTokens: 2, RequestCount: 3, InputBytes: 20, OutputBytes: 200})
	want := RepresentationUsage{InputTokens: 3, RequestCount: 4, InputBytes: 30, OutputBytes: 300}
	if usage != want {
		t.Fatalf("usage = %+v, want %+v", usage, want)
	}
}

func TestRepresentationResponseSpace(t *testing.T) {
	var nilResp *RepresentationResponse
	if _, ok := nilResp.Space(RepresentationDense); ok {
		t.Error("nil response should report no space")
	}
	resp := &RepresentationResponse{Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(4)}}
	if _, ok := resp.Space(RepresentationDense); !ok {
		t.Error("present space should be reported")
	}
	if _, ok := resp.Space(RepresentationSparse); ok {
		t.Error("absent space should not be reported")
	}
}

func TestRepresentationCapabilities(t *testing.T) {
	caps := RepresentationCapabilities{
		Outputs:    []RepresentationKind{RepresentationDense},
		InputTypes: []EmbeddingInputType{EmbeddingInputQuery},
	}
	if !caps.Supports(RepresentationDense) || caps.Supports(RepresentationSparse) {
		t.Error("Supports does not match Outputs")
	}
	if !caps.SupportsInputType(EmbeddingInputQuery) || caps.SupportsInputType(EmbeddingInputDocument) {
		t.Error("SupportsInputType does not match InputTypes")
	}

	// An encoder that declares no input types accepts only the untyped form,
	// rather than silently accepting a role it will not honor.
	bare := RepresentationCapabilities{Outputs: []RepresentationKind{RepresentationDense}}
	if !bare.SupportsInputType(EmbeddingInputNone) {
		t.Error("an encoder with no declared input types should accept the untyped form")
	}
	if bare.SupportsInputType(EmbeddingInputQuery) {
		t.Error("an encoder with no declared input types should not accept query")
	}
}

func TestUnsupportedRepresentationError(t *testing.T) {
	withList := &UnsupportedRepresentationError{
		Provider:  "deepinfra",
		Kind:      RepresentationMultiVector,
		Supported: []RepresentationKind{RepresentationDense, RepresentationSparse},
	}
	msg := withList.Error()
	if !strings.Contains(msg, "deepinfra") || !strings.Contains(msg, "multi_vector") ||
		!strings.Contains(msg, "dense, sparse") {
		t.Errorf("error message %q lacks provider, kind, or supported list", msg)
	}
	if !errors.Is(withList, ErrUnsupportedRepresentation) {
		t.Error("error does not match its sentinel")
	}

	bare := &UnsupportedRepresentationError{Provider: "p", Kind: RepresentationSparse}
	if strings.Contains(bare.Error(), "supports") {
		t.Errorf("error message %q should omit an empty supported list", bare.Error())
	}
}

func TestInvalidRepresentationResponseErrorMessage(t *testing.T) {
	whole := &InvalidRepresentationResponseError{Provider: "p", Item: -1, Problem: "response is nil"}
	if strings.Contains(whole.Error(), "for item") {
		t.Errorf("whole-response error %q should not name an item", whole.Error())
	}

	item := &InvalidRepresentationResponseError{Provider: "p", Item: 2, Kind: RepresentationSparse, Problem: "bad"}
	msg := item.Error()
	for _, want := range []string{"p", "for item 2", "sparse", "bad"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error message %q lacks %q", msg, want)
		}
	}
	if !errors.Is(item, ErrInvalidRepresentationResponse) {
		t.Error("error does not match its sentinel")
	}
}

func denseValidator() RepresentationValidator {
	return RepresentationValidator{
		Provider: "test",
		Capabilities: RepresentationCapabilities{
			Outputs: []RepresentationKind{RepresentationDense, RepresentationSparse, RepresentationMultiVector},
			InputTypes: []EmbeddingInputType{
				EmbeddingInputNone, EmbeddingInputQuery, EmbeddingInputDocument,
			},
			SupportsTruncation:  true,
			SupportsMultiOutput: true,
		},
		Limits: DefaultRepresentationLimits(),
	}
}

func TestValidateRequestAgainstCapabilities(t *testing.T) {
	truncate := true

	tests := []struct {
		name      string
		caps      func(RepresentationCapabilities) RepresentationCapabilities
		limits    *RepresentationLimits
		req       *RepresentationRequest
		invariant string
		unsupport bool
	}{
		{
			name: "unsupported kind",
			caps: func(c RepresentationCapabilities) RepresentationCapabilities {
				c.Outputs = []RepresentationKind{RepresentationDense}
				return c
			},
			req: &RepresentationRequest{
				Input:   []string{"a"},
				Outputs: []RepresentationKind{RepresentationSparse},
			},
			unsupport: true,
		},
		{
			name: "unsupported input type",
			caps: func(c RepresentationCapabilities) RepresentationCapabilities {
				c.InputTypes = []EmbeddingInputType{EmbeddingInputNone}
				return c
			},
			req: &RepresentationRequest{
				Input:     []string{"a"},
				InputType: EmbeddingInputQuery,
				Outputs:   []RepresentationKind{RepresentationDense},
			},
			invariant: "input_type.unsupported",
		},
		{
			name: "multi output unsupported",
			caps: func(c RepresentationCapabilities) RepresentationCapabilities {
				c.SupportsMultiOutput = false
				return c
			},
			req: &RepresentationRequest{
				Input:   []string{"a"},
				Outputs: []RepresentationKind{RepresentationDense, RepresentationSparse},
			},
			invariant: "outputs.multi_unsupported",
		},
		{
			name: "truncation unsupported",
			caps: func(c RepresentationCapabilities) RepresentationCapabilities {
				c.SupportsTruncation = false
				return c
			},
			req: &RepresentationRequest{
				Input:    []string{"a"},
				Outputs:  []RepresentationKind{RepresentationDense},
				Truncate: &truncate,
			},
			invariant: "truncate.unsupported",
		},
		{
			name: "provider batch limit",
			caps: func(c RepresentationCapabilities) RepresentationCapabilities {
				c.MaximumBatchSize = 1
				return c
			},
			req: &RepresentationRequest{
				Input:   []string{"a", "b"},
				Outputs: []RepresentationKind{RepresentationDense},
			},
			invariant: "input.batch_limit",
		},
		{
			name:   "configured input count limit",
			caps:   func(c RepresentationCapabilities) RepresentationCapabilities { return c },
			limits: &RepresentationLimits{MaxInputs: 1},
			req: &RepresentationRequest{
				Input:   []string{"a", "b"},
				Outputs: []RepresentationKind{RepresentationDense},
			},
			invariant: "input.limit",
		},
		{
			name:   "per item byte limit",
			caps:   func(c RepresentationCapabilities) RepresentationCapabilities { return c },
			limits: &RepresentationLimits{MaxInputBytes: 2},
			req: &RepresentationRequest{
				Input:   []string{"abc"},
				Outputs: []RepresentationKind{RepresentationDense},
			},
			invariant: "input.item_bytes",
		},
		{
			name:   "total byte limit",
			caps:   func(c RepresentationCapabilities) RepresentationCapabilities { return c },
			limits: &RepresentationLimits{MaxTotalInputBytes: 3},
			req: &RepresentationRequest{
				Input:   []string{"ab", "cd"},
				Outputs: []RepresentationKind{RepresentationDense},
			},
			invariant: "input.total_bytes",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := denseValidator()
			v.Capabilities = tc.caps(v.Capabilities)
			if tc.limits != nil {
				v.Limits = *tc.limits
			}
			err := v.ValidateRequest(tc.req)
			if tc.unsupport {
				var typed *UnsupportedRepresentationError
				if !errors.As(err, &typed) {
					t.Fatalf("got %v, want *UnsupportedRepresentationError", err)
				}
				return
			}
			var typed *InvalidRepresentationRequestError
			if !errors.As(err, &typed) {
				t.Fatalf("got %v, want *InvalidRepresentationRequestError", err)
			}
			if typed.Invariant != tc.invariant {
				t.Errorf("invariant = %q, want %q", typed.Invariant, tc.invariant)
			}
		})
	}
}

func TestValidateRequestAllowsEmptyInputWhenDeclared(t *testing.T) {
	v := denseValidator()
	req := &RepresentationRequest{
		Input:   []string{""},
		Outputs: []RepresentationKind{RepresentationDense},
	}
	if err := v.ValidateRequest(req); err == nil {
		t.Fatal("an encoder that does not allow empty inputs should reject one")
	}

	v.Capabilities.AllowsEmptyInput = true
	if err := v.ValidateRequest(req); err != nil {
		t.Fatalf("an encoder that allows empty inputs should accept one: %v", err)
	}
}

func TestValidateRequestZeroLimitsDisableSizeChecks(t *testing.T) {
	v := denseValidator()
	v.Limits = RepresentationLimits{}
	req := &RepresentationRequest{
		Input:   []string{strings.Repeat("x", 4096)},
		Outputs: []RepresentationKind{RepresentationDense},
	}
	if err := v.ValidateRequest(req); err != nil {
		t.Fatalf("zero limits should impose no ceiling: %v", err)
	}
}

func TestValidateRequestAcceptsWellFormed(t *testing.T) {
	v := denseValidator()
	truncate := false
	req := &RepresentationRequest{
		Input:     []string{"a", "b"},
		InputType: EmbeddingInputDocument,
		Outputs:   []RepresentationKind{RepresentationDense, RepresentationSparse},
		Truncate:  &truncate,
	}
	if err := v.ValidateRequest(req); err != nil {
		t.Fatalf("well-formed request rejected: %v", err)
	}
}

func TestValidateResponse(t *testing.T) {
	req := &RepresentationRequest{
		Input:   []string{"a", "b"},
		Outputs: []RepresentationKind{RepresentationDense},
	}
	good := &RepresentationResponse{
		Data: []Representation{
			{Dense: []float32{1, 2}},
			{Dense: []float32{3, 4}},
		},
		Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(2)},
		Model:  "model",
	}
	if err := denseValidator().ValidateResponse(req, good); err != nil {
		t.Fatalf("well-formed response rejected: %v", err)
	}

	tests := []struct {
		name string
		resp *RepresentationResponse
		want string
	}{
		{"nil response", nil, "response is nil"},
		{
			name: "short batch",
			resp: &RepresentationResponse{
				Data:   []Representation{{Dense: []float32{1, 2}}},
				Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(2)},
			},
			want: "got 1 representations for 2 inputs",
		},
		{
			name: "long batch",
			resp: &RepresentationResponse{
				Data: []Representation{
					{Dense: []float32{1, 2}}, {Dense: []float32{1, 2}}, {Dense: []float32{1, 2}},
				},
				Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(2)},
			},
			want: "got 3 representations for 2 inputs",
		},
		{
			name: "missing space",
			resp: &RepresentationResponse{
				Data:   []Representation{{Dense: []float32{1, 2}}, {Dense: []float32{1, 2}}},
				Spaces: map[RepresentationKind]VectorSpace{},
			},
			want: "describes no vector space",
		},
		{
			name: "space declares another kind",
			resp: &RepresentationResponse{
				Data:   []Representation{{Dense: []float32{1, 2}}, {Dense: []float32{1, 2}}},
				Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: sparseSpace(100)},
			},
			want: `declares kind "sparse"`,
		},
		{
			name: "invalid space",
			resp: &RepresentationResponse{
				Data: []Representation{{Dense: []float32{1, 2}}, {Dense: []float32{1, 2}}},
				Spaces: map[RepresentationKind]VectorSpace{
					RepresentationDense: {Provider: "p", Model: "m", Kind: RepresentationDense, Dimensions: 2, Metric: SimilarityCosine},
				},
			},
			want: "id cannot be empty",
		},
		{
			name: "unrequested space",
			resp: &RepresentationResponse{
				Data: []Representation{{Dense: []float32{1, 2}}, {Dense: []float32{1, 2}}},
				Spaces: map[RepresentationKind]VectorSpace{
					RepresentationDense:  denseSpace(2),
					RepresentationSparse: sparseSpace(100),
				},
			},
			want: "not requested",
		},
		{
			name: "unrequested representation",
			resp: &RepresentationResponse{
				Data: []Representation{
					{Dense: []float32{1, 2}, Sparse: &SparseVector{Indices: []uint32{1}, Values: []float32{1}}},
					{Dense: []float32{1, 2}},
				},
				Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(2)},
			},
			want: "returned but not requested",
		},
		{
			name: "missing dense",
			resp: &RepresentationResponse{
				Data:   []Representation{{Dense: []float32{1, 2}}, {}},
				Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(2)},
			},
			want: "dense vector is missing",
		},
		{
			name: "inconsistent dense width",
			resp: &RepresentationResponse{
				Data:   []Representation{{Dense: []float32{1, 2}}, {Dense: []float32{1, 2, 3}}},
				Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(2)},
			},
			want: "width 3, space declares 2",
		},
		{
			name: "non-finite dense value",
			resp: &RepresentationResponse{
				Data: []Representation{
					{Dense: []float32{1, 2}},
					{Dense: []float32{1, float32(math.NaN())}},
				},
				Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(2)},
			},
			want: "position 1 is not finite",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := denseValidator().ValidateResponse(req, tc.resp)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
			if !errors.Is(err, ErrInvalidRepresentationResponse) {
				t.Error("error does not match ErrInvalidRepresentationResponse")
			}
		})
	}
}

func TestValidateResponseInfiniteDenseValue(t *testing.T) {
	req := &RepresentationRequest{Input: []string{"a"}, Outputs: []RepresentationKind{RepresentationDense}}
	resp := &RepresentationResponse{
		Data:   []Representation{{Dense: []float32{float32(math.Inf(1)), 1}}},
		Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(2)},
	}
	err := denseValidator().ValidateResponse(req, resp)
	if err == nil || !strings.Contains(err.Error(), "position 0 is not finite") {
		t.Fatalf("got %v, want a non-finite value error", err)
	}
}

func TestValidateResponseSparse(t *testing.T) {
	req := &RepresentationRequest{Input: []string{"a"}, Outputs: []RepresentationKind{RepresentationSparse}}
	spaces := map[RepresentationKind]VectorSpace{RepresentationSparse: sparseSpace(100)}

	good := &RepresentationResponse{
		Data:   []Representation{{Sparse: &SparseVector{Indices: []uint32{3, 17}, Values: []float32{0.5, -0.25}}}},
		Spaces: spaces,
	}
	if err := denseValidator().ValidateResponse(req, good); err != nil {
		t.Fatalf("well-formed sparse response rejected: %v", err)
	}

	tests := []struct {
		name   string
		sparse *SparseVector
		limits *RepresentationLimits
		want   string
	}{
		{"missing", nil, nil, "sparse vector is missing"},
		{
			name:   "length mismatch",
			sparse: &SparseVector{Indices: []uint32{1, 2}, Values: []float32{1}},
			want:   "2 indices and 1 values",
		},
		{
			name:   "empty",
			sparse: &SparseVector{},
			want:   "no nonzero coordinates",
		},
		{
			name:   "not strictly increasing",
			sparse: &SparseVector{Indices: []uint32{5, 5}, Values: []float32{1, 1}},
			want:   "not strictly increasing at position 1",
		},
		{
			name:   "descending",
			sparse: &SparseVector{Indices: []uint32{9, 2}, Values: []float32{1, 1}},
			want:   "not strictly increasing at position 1",
		},
		{
			name:   "out of vocabulary",
			sparse: &SparseVector{Indices: []uint32{100}, Values: []float32{1}},
			want:   "outside the declared vocabulary of 100",
		},
		{
			name:   "non-finite weight",
			sparse: &SparseVector{Indices: []uint32{1}, Values: []float32{float32(math.Inf(-1))}},
			want:   "position 0 is not finite",
		},
		{
			name:   "zero weight",
			sparse: &SparseVector{Indices: []uint32{1}, Values: []float32{0}},
			want:   "position 0 is zero",
		},
		{
			name:   "over the nonzero limit",
			sparse: &SparseVector{Indices: []uint32{1, 2, 3}, Values: []float32{1, 1, 1}},
			limits: &RepresentationLimits{MaxSparseNonZero: 2},
			want:   "3 nonzero coordinates, limit is 2",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := denseValidator()
			if tc.limits != nil {
				v.Limits = *tc.limits
			}
			resp := &RepresentationResponse{
				Data:   []Representation{{Sparse: tc.sparse}},
				Spaces: spaces,
			}
			err := v.ValidateResponse(req, resp)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestValidateResponseAllowsDocumentedEmptySparse(t *testing.T) {
	req := &RepresentationRequest{Input: []string{"the"}, Outputs: []RepresentationKind{RepresentationSparse}}
	resp := &RepresentationResponse{
		Data:   []Representation{{Sparse: &SparseVector{}}},
		Spaces: map[RepresentationKind]VectorSpace{RepresentationSparse: sparseSpace(100)},
	}
	v := denseValidator()
	v.Capabilities.AllowsEmptySparse = true
	if err := v.ValidateResponse(req, resp); err != nil {
		t.Fatalf("an encoder that documents empty sparse output should accept one: %v", err)
	}
}

func TestValidateResponseMultiVector(t *testing.T) {
	req := &RepresentationRequest{Input: []string{"a"}, Outputs: []RepresentationKind{RepresentationMultiVector}}
	spaces := map[RepresentationKind]VectorSpace{RepresentationMultiVector: multiVectorSpace(2)}

	good := &RepresentationResponse{
		Data:   []Representation{{MultiVector: [][]float32{{1, 2}, {3, 4}}}},
		Spaces: spaces,
	}
	if err := denseValidator().ValidateResponse(req, good); err != nil {
		t.Fatalf("well-formed multi-vector response rejected: %v", err)
	}

	tests := []struct {
		name    string
		vectors [][]float32
		limits  *RepresentationLimits
		want    string
	}{
		{"missing", nil, nil, "multi-vector representation is missing"},
		{
			name:    "inconsistent token width",
			vectors: [][]float32{{1, 2}, {1}},
			want:    "token vector 1 has width 1, space declares 2",
		},
		{
			name:    "non-finite token value",
			vectors: [][]float32{{1, float32(math.NaN())}},
			want:    "token vector 0 has a non-finite value at position 1",
		},
		{
			name:    "over the token limit",
			vectors: [][]float32{{1, 2}, {3, 4}, {5, 6}},
			limits:  &RepresentationLimits{MaxTokenVectors: 2},
			want:    "3 token vectors, limit is 2",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := denseValidator()
			if tc.limits != nil {
				v.Limits = *tc.limits
			}
			resp := &RepresentationResponse{
				Data:   []Representation{{MultiVector: tc.vectors}},
				Spaces: spaces,
			}
			err := v.ValidateResponse(req, resp)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

// Errors must be safe to log beside user documents, so nothing that reaches
// them may carry input text.
func TestValidationErrorsDoNotEchoInput(t *testing.T) {
	const secret = "the launch code is hunter2"
	v := denseValidator()
	v.Limits = RepresentationLimits{MaxInputBytes: 4}

	err := v.ValidateRequest(&RepresentationRequest{
		Input:   []string{secret},
		Outputs: []RepresentationKind{RepresentationDense},
	})
	if err == nil {
		t.Fatal("oversized input should be rejected")
	}
	if strings.Contains(err.Error(), "hunter2") {
		t.Fatalf("error echoes input text: %q", err.Error())
	}
}

func TestDefaultRepresentationLimits(t *testing.T) {
	limits := DefaultRepresentationLimits()
	if limits.MaxInputs <= 0 || limits.MaxInputBytes <= 0 || limits.MaxTotalInputBytes <= 0 ||
		limits.MaxSparseNonZero <= 0 || limits.MaxTokenVectors <= 0 {
		t.Fatalf("default limits leave a dimension unbounded: %+v", limits)
	}
}

// stubEmbedder is a minimal Embedder for the adapter tests.
type stubEmbedder struct {
	name    string
	vectors [][]float32
	usage   EmbeddingUsage
	model   string
	err     error
	calls   []EmbeddingRequest
	nilResp bool
}

func (s *stubEmbedder) Embed(_ context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	s.calls = append(s.calls, *req)
	if s.err != nil {
		return nil, s.err
	}
	if s.nilResp {
		return nil, nil
	}
	return &EmbeddingResponse{Vectors: s.vectors, Model: s.model, Usage: s.usage}, nil
}

func (s *stubEmbedder) Name() string { return s.name }

func TestNewEmbedderEncoderDefaults(t *testing.T) {
	embedder := &stubEmbedder{name: "text-embedding-3-small"}
	encoder, err := NewEmbedderEncoder(embedder, VectorSpace{Provider: "openai"})
	if err != nil {
		t.Fatalf("NewEmbedderEncoder: %v", err)
	}
	if encoder.Name() != "text-embedding-3-small" {
		t.Errorf("Name() = %q", encoder.Name())
	}
	if encoder.space.Model != "text-embedding-3-small" {
		t.Errorf("model = %q, want the embedder name", encoder.space.Model)
	}
	if encoder.space.Kind != RepresentationDense || encoder.space.Metric != SimilarityCosine {
		t.Errorf("defaults not applied: %+v", encoder.space)
	}

	caps := encoder.Capabilities()
	if len(caps.Outputs) != 1 || caps.Outputs[0] != RepresentationDense {
		t.Errorf("adapter should declare dense only, got %v", caps.Outputs)
	}
	if !caps.SupportsTruncation {
		t.Error("adapter should pass truncation through")
	}
	if caps.SupportsMultiOutput {
		t.Error("a dense-only adapter cannot support multi-output")
	}
}

func TestNewEmbedderEncoderRejectsBadConfiguration(t *testing.T) {
	tests := []struct {
		name     string
		embedder Embedder
		space    VectorSpace
		want     string
	}{
		{"nil embedder", nil, VectorSpace{Provider: "p"}, "embedder cannot be nil"},
		{"no provider", &stubEmbedder{name: "m"}, VectorSpace{}, "provider is required"},
		{"no model", &stubEmbedder{}, VectorSpace{Provider: "p"}, "model is required"},
		{"non-dense kind", &stubEmbedder{name: "m"}, VectorSpace{Provider: "p", Kind: RepresentationSparse}, "produces dense output"},
		{"bad metric", &stubEmbedder{name: "m"}, VectorSpace{Provider: "p", Metric: "l2"}, "not a known similarity metric"},
		{"negative dimensions", &stubEmbedder{name: "m"}, VectorSpace{Provider: "p", Dimensions: -1}, "cannot be negative"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewEmbedderEncoder(tc.embedder, tc.space)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestEmbedderEncoderEncode(t *testing.T) {
	embedder := &stubEmbedder{
		name:    "voyage-3.5",
		vectors: [][]float32{{1, 2, 3}, {4, 5, 6}},
		model:   "voyage-3.5",
		usage:   EmbeddingUsage{PromptTokens: 11, TotalTokens: 11},
	}
	encoder, err := NewEmbedderEncoder(embedder, VectorSpace{Provider: "voyageai", Revision: "2025-01"})
	if err != nil {
		t.Fatalf("NewEmbedderEncoder: %v", err)
	}

	resp, err := encoder.Encode(context.Background(), &RepresentationRequest{
		Input:     []string{"alpha", "beta"},
		InputType: EmbeddingInputDocument,
		Outputs:   []RepresentationKind{RepresentationDense},
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("got %d representations", len(resp.Data))
	}
	space := resp.Spaces[RepresentationDense]
	if space.Dimensions != 3 {
		t.Errorf("space width = %d, want the observed 3", space.Dimensions)
	}
	if space.ID == "" {
		t.Error("space has no ID")
	}
	if resp.Usage.InputTokens != 11 || resp.Usage.RequestCount != 1 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.Usage.InputBytes != len("alpha")+len("beta") {
		t.Errorf("input bytes = %d", resp.Usage.InputBytes)
	}
	if len(embedder.calls) != 1 || embedder.calls[0].InputType != EmbeddingInputDocument {
		t.Errorf("input role was not propagated: %+v", embedder.calls)
	}
}

func TestEmbedderEncoderUsesTotalTokensWhenPromptTokensAbsent(t *testing.T) {
	embedder := &stubEmbedder{
		name:    "voyage-3.5",
		vectors: [][]float32{{1, 2}},
		usage:   EmbeddingUsage{TotalTokens: 7},
	}
	encoder, _ := NewEmbedderEncoder(embedder, VectorSpace{Provider: "voyageai"})
	resp, err := encoder.Encode(context.Background(), &RepresentationRequest{
		Input:   []string{"alpha"},
		Outputs: []RepresentationKind{RepresentationDense},
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if resp.Usage.InputTokens != 7 {
		t.Errorf("input tokens = %d, want the reported total of 7", resp.Usage.InputTokens)
	}
	if resp.Model != "voyage-3.5" {
		t.Errorf("model = %q, want the embedder name when the response echoes none", resp.Model)
	}
}

func TestEmbedderEncoderHonorsConfiguredDimensions(t *testing.T) {
	embedder := &stubEmbedder{name: "m", vectors: [][]float32{{1, 2, 3}}}
	encoder, _ := NewEmbedderEncoder(embedder, VectorSpace{Provider: "p", Dimensions: 4})
	_, err := encoder.Encode(context.Background(), &RepresentationRequest{
		Input:   []string{"alpha"},
		Outputs: []RepresentationKind{RepresentationDense},
	})
	if err == nil || !strings.Contains(err.Error(), "width 3, space declares 4") {
		t.Fatalf("got %v, want a width mismatch against the configured space", err)
	}
}

func TestEmbedderEncoderRejectsUnsupportedOutputs(t *testing.T) {
	encoder, _ := NewEmbedderEncoder(&stubEmbedder{name: "m", vectors: [][]float32{{1}}}, VectorSpace{Provider: "p"})
	_, err := encoder.Encode(context.Background(), &RepresentationRequest{
		Input:   []string{"alpha"},
		Outputs: []RepresentationKind{RepresentationSparse},
	})
	if !errors.Is(err, ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

func TestEmbedderEncoderPropagatesEmbedderErrors(t *testing.T) {
	sentinel := errors.New("upstream failure")
	encoder, _ := NewEmbedderEncoder(&stubEmbedder{name: "m", err: sentinel}, VectorSpace{Provider: "p"})
	_, err := encoder.Encode(context.Background(), &RepresentationRequest{
		Input:   []string{"alpha"},
		Outputs: []RepresentationKind{RepresentationDense},
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("got %v, want the embedder's own error", err)
	}
}

func TestEmbedderEncoderRejectsEmptyResponses(t *testing.T) {
	t.Run("nil response", func(t *testing.T) {
		encoder, _ := NewEmbedderEncoder(&stubEmbedder{name: "m", nilResp: true}, VectorSpace{Provider: "p"})
		_, err := encoder.Encode(context.Background(), &RepresentationRequest{
			Input:   []string{"alpha"},
			Outputs: []RepresentationKind{RepresentationDense},
		})
		if err == nil || !strings.Contains(err.Error(), "returned no response") {
			t.Fatalf("got %v, want a missing-response error", err)
		}
	})

	t.Run("no vectors to infer width from", func(t *testing.T) {
		encoder, _ := NewEmbedderEncoder(&stubEmbedder{name: "m"}, VectorSpace{Provider: "p"})
		_, err := encoder.Encode(context.Background(), &RepresentationRequest{
			Input:   []string{"alpha"},
			Outputs: []RepresentationKind{RepresentationDense},
		})
		if err == nil || !strings.Contains(err.Error(), "no vector to infer the space width") {
			t.Fatalf("got %v, want a width-inference error", err)
		}
	})

	t.Run("zero-width vector", func(t *testing.T) {
		encoder, _ := NewEmbedderEncoder(&stubEmbedder{name: "m", vectors: [][]float32{{}}}, VectorSpace{Provider: "p"})
		_, err := encoder.Encode(context.Background(), &RepresentationRequest{
			Input:   []string{"alpha"},
			Outputs: []RepresentationKind{RepresentationDense},
		})
		if err == nil || !strings.Contains(err.Error(), "no vector to infer the space width") {
			t.Fatalf("got %v, want a width-inference error", err)
		}
	})
}

// A dense encoder built without a declared width lands in a different space
// when the model's output dimension changes, rather than silently corrupting
// the existing index.
func TestEmbedderEncoderWidthChangeChangesSpace(t *testing.T) {
	narrow := &stubEmbedder{name: "m", vectors: [][]float32{{1, 2}}}
	wide := &stubEmbedder{name: "m", vectors: [][]float32{{1, 2, 3, 4}}}

	encodeWith := func(e Embedder) VectorSpace {
		encoder, _ := NewEmbedderEncoder(e, VectorSpace{Provider: "p"})
		resp, err := encoder.Encode(context.Background(), &RepresentationRequest{
			Input:   []string{"alpha"},
			Outputs: []RepresentationKind{RepresentationDense},
		})
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		return resp.Spaces[RepresentationDense]
	}

	if encodeWith(narrow).ID == encodeWith(wide).ID {
		t.Fatal("a changed output width produced the same space ID")
	}
}

// stubEncoder is a minimal RepresentationEncoder for the reverse adapter.
type stubEncoder struct {
	caps  RepresentationCapabilities
	resp  *RepresentationResponse
	err   error
	calls []RepresentationRequest
}

func (s *stubEncoder) Encode(_ context.Context, req *RepresentationRequest) (*RepresentationResponse, error) {
	s.calls = append(s.calls, *req)
	if s.err != nil {
		return nil, s.err
	}
	return s.resp, nil
}

func (s *stubEncoder) Name() string                             { return "stub-encoder" }
func (s *stubEncoder) Capabilities() RepresentationCapabilities { return s.caps }

func TestNewEncoderEmbedder(t *testing.T) {
	if _, err := NewEncoderEmbedder(nil); err == nil {
		t.Error("nil encoder should be rejected")
	}

	sparseOnly := &stubEncoder{caps: RepresentationCapabilities{Outputs: []RepresentationKind{RepresentationSparse}}}
	_, err := NewEncoderEmbedder(sparseOnly)
	if !errors.Is(err, ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation at construction", err)
	}
}

func TestEncoderEmbedderEmbed(t *testing.T) {
	encoder := &stubEncoder{
		caps: RepresentationCapabilities{
			Outputs:    []RepresentationKind{RepresentationDense, RepresentationSparse},
			InputTypes: []EmbeddingInputType{EmbeddingInputNone, EmbeddingInputQuery, EmbeddingInputDocument},
		},
		resp: &RepresentationResponse{
			Data:   []Representation{{Dense: []float32{1, 2}}, {Dense: []float32{3, 4}}},
			Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: denseSpace(2)},
			Model:  "bge-m3",
			Usage:  RepresentationUsage{InputTokens: 9, RequestCount: 1},
		},
	}
	embedder, err := NewEncoderEmbedder(encoder)
	if err != nil {
		t.Fatalf("NewEncoderEmbedder: %v", err)
	}
	if embedder.Name() != "stub-encoder" {
		t.Errorf("Name() = %q", embedder.Name())
	}

	resp, err := embedder.Embed(context.Background(), &EmbeddingRequest{
		Input:     []string{"a", "b"},
		InputType: EmbeddingInputQuery,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 2 || resp.Vectors[1][1] != 4 {
		t.Fatalf("vectors = %v", resp.Vectors)
	}
	if resp.Model != "bge-m3" {
		t.Errorf("model = %q", resp.Model)
	}
	if resp.Usage.PromptTokens != 9 || resp.Usage.TotalTokens != 9 {
		t.Errorf("usage = %+v", resp.Usage)
	}

	call := encoder.calls[0]
	if len(call.Outputs) != 1 || call.Outputs[0] != RepresentationDense {
		t.Errorf("projection requested %v, want dense only", call.Outputs)
	}
	if call.InputType != EmbeddingInputQuery {
		t.Errorf("input role was not propagated: %q", call.InputType)
	}
}

func TestEncoderEmbedderRejectsDimensionOverride(t *testing.T) {
	encoder := &stubEncoder{caps: RepresentationCapabilities{Outputs: []RepresentationKind{RepresentationDense}}}
	embedder, _ := NewEncoderEmbedder(encoder)

	_, err := embedder.Embed(context.Background(), &EmbeddingRequest{
		Input:      []string{"a"},
		Dimensions: 256,
	})
	if !errors.Is(err, ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want ErrInvalidRepresentationRequest", err)
	}
	if len(encoder.calls) != 0 {
		t.Error("the request should be rejected before it reaches the encoder")
	}
}

func TestEncoderEmbedderPropagatesErrors(t *testing.T) {
	t.Run("invalid request", func(t *testing.T) {
		encoder := &stubEncoder{caps: RepresentationCapabilities{Outputs: []RepresentationKind{RepresentationDense}}}
		embedder, _ := NewEncoderEmbedder(encoder)
		if _, err := embedder.Embed(context.Background(), &EmbeddingRequest{}); err == nil {
			t.Fatal("an empty request should be rejected")
		}
	})

	t.Run("encoder failure", func(t *testing.T) {
		sentinel := errors.New("encode failed")
		encoder := &stubEncoder{
			caps: RepresentationCapabilities{Outputs: []RepresentationKind{RepresentationDense}},
			err:  sentinel,
		}
		embedder, _ := NewEncoderEmbedder(encoder)
		_, err := embedder.Embed(context.Background(), &EmbeddingRequest{Input: []string{"a"}})
		if !errors.Is(err, sentinel) {
			t.Fatalf("got %v, want the encoder's own error", err)
		}
	})
}
