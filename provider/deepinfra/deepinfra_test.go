package deepinfra_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/deepinfra"
	"github.com/regularkevvv/agentic/provider/test/conformance"
)

// denseResponse is a well-formed two-input reply carrying dense output only.
const denseResponse = `{
	"embeddings": [[0.1, 0.2], [0.3, 0.4]],
	"input_tokens": 6,
	"inference_status": {"status": "unknown", "tokens_input": 6}
}`

func newTestEncoder(t *testing.T, handler http.HandlerFunc, opts ...deepinfra.Option) (*deepinfra.Encoder, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]deepinfra.Option{
		deepinfra.WithAPIToken("test-token"),
		deepinfra.WithBaseURL(server.URL),
	}, opts...)

	encoder, err := deepinfra.New(deepinfra.BGEM3Model, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return encoder, server
}

// respondWith serves body for every request.
func respondWith(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}
}

func documentRequest(outputs ...core.RepresentationKind) *core.RepresentationRequest {
	return &core.RepresentationRequest{
		Input:     []string{"sparse vectors", "dense vectors"},
		InputType: core.EmbeddingInputDocument,
		Outputs:   outputs,
	}
}

func TestNewRequiresToken(t *testing.T) {
	t.Setenv("DEEPINFRA_TOKEN", "")
	if _, err := deepinfra.New(deepinfra.BGEM3Model); err == nil {
		t.Fatal("expected an error when no token is configured")
	}
}

func TestNewReadsTokenFromEnvironment(t *testing.T) {
	t.Setenv("DEEPINFRA_TOKEN", "env-token")

	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		fmt.Fprint(w, denseResponse)
	}))
	t.Cleanup(server.Close)

	encoder, err := deepinfra.New(deepinfra.BGEM3Model, deepinfra.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense)); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if authorization != "Bearer env-token" {
		t.Errorf("Authorization = %q", authorization)
	}
}

// An explicit token wins over the environment, so one process can talk to two
// accounts.
func TestNewPrefersExplicitToken(t *testing.T) {
	t.Setenv("DEEPINFRA_TOKEN", "env-token")

	var authorization string
	encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		fmt.Fprint(w, denseResponse)
	})
	if _, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense)); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if authorization != "Bearer test-token" {
		t.Errorf("Authorization = %q, want the explicit token", authorization)
	}
}

func TestNewValidatesConfiguration(t *testing.T) {
	tests := []struct {
		name string
		opts []deepinfra.Option
		want string
	}{
		{"negative retries", []deepinfra.Option{deepinfra.WithMaxRetries(-1)}, "max retries"},
		{"zero response limit", []deepinfra.Option{deepinfra.WithMaxResponseBytes(0)}, "max response bytes"},
		{"negative batch size", []deepinfra.Option{deepinfra.WithBatchSize(-1)}, "batch size"},
		{"negative vocabulary", []deepinfra.Option{deepinfra.WithSparseVocabulary(-1)}, "sparse vocabulary"},
		{"unknown output", []deepinfra.Option{deepinfra.WithOutputs("colbert")}, "not dense, sparse, or multi_vector"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := append([]deepinfra.Option{deepinfra.WithAPIToken("t")}, tc.opts...)
			_, err := deepinfra.New(deepinfra.BGEM3Model, opts...)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}

	if _, err := deepinfra.New("", deepinfra.WithAPIToken("t")); err == nil {
		t.Error("an empty model should be rejected")
	}
}

func TestMustNewPanicsOnError(t *testing.T) {
	t.Setenv("DEEPINFRA_TOKEN", "")
	defer func() {
		if recover() == nil {
			t.Error("MustNew should panic without a token")
		}
	}()
	deepinfra.MustNew(deepinfra.BGEM3Model)
}

func TestMustNewReturnsEncoder(t *testing.T) {
	encoder := deepinfra.MustNew(deepinfra.BGEM3Model, deepinfra.WithAPIToken("t"))
	if encoder.Name() != deepinfra.BGEM3Model {
		t.Errorf("Name() = %q", encoder.Name())
	}
}

// The native route is what exposes sparse and multi-vector output; the
// OpenAI-compatible route would silently return dense only.
func TestEncodeUsesNativeInferenceRoute(t *testing.T) {
	var path, contentType string
	var body map[string]any

	encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		contentType = r.Header.Get("Content-Type")
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, denseResponse)
	})
	if _, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense)); err != nil {
		t.Fatalf("Encode: %v", err)
	}

	if path != "/inference/"+deepinfra.BGEM3Model {
		t.Errorf("path = %q", path)
	}
	if contentType != "application/json" {
		t.Errorf("Content-Type = %q", contentType)
	}
	inputs, _ := body["inputs"].([]any)
	if len(inputs) != 2 || inputs[0] != "sparse vectors" {
		t.Errorf("inputs = %v", body["inputs"])
	}
}

// The output flags must be sent explicitly. DeepInfra defaults dense to true,
// so an omitted flag would return and bill for a representation nobody asked
// for.
func TestEncodeSendsEveryOutputFlag(t *testing.T) {
	tests := []struct {
		name    string
		outputs []core.RepresentationKind
		want    map[string]bool
	}{
		{"dense only", []core.RepresentationKind{core.RepresentationDense},
			map[string]bool{"dense": true, "sparse": false, "colbert": false}},
		{"sparse only", []core.RepresentationKind{core.RepresentationSparse},
			map[string]bool{"dense": false, "sparse": true, "colbert": false}},
		{"multi vector only", []core.RepresentationKind{core.RepresentationMultiVector},
			map[string]bool{"dense": false, "sparse": false, "colbert": true}},
		{"dense and sparse", []core.RepresentationKind{core.RepresentationDense, core.RepresentationSparse},
			map[string]bool{"dense": true, "sparse": true, "colbert": false}},
	}

	fixture := readFixture(t)
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var body map[string]any
			encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewDecoder(r.Body).Decode(&body)
				fmt.Fprint(w, fixture)
			})
			if _, err := encoder.Encode(context.Background(), documentRequest(tc.outputs...)); err != nil {
				t.Fatalf("Encode: %v", err)
			}
			for flag, want := range tc.want {
				got, present := body[flag].(bool)
				if !present {
					t.Fatalf("request omitted the %q flag", flag)
				}
				if got != want {
					t.Errorf("%q = %v, want %v", flag, got, want)
				}
			}
		})
	}
}

func TestEncodeNormalizeFlag(t *testing.T) {
	t.Run("omitted by default", func(t *testing.T) {
		var body map[string]any
		encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&body)
			fmt.Fprint(w, denseResponse)
		})
		if _, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense)); err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if _, present := body["normalize"]; present {
			t.Error("normalize should be omitted so the API default applies")
		}
	})

	t.Run("sent when configured", func(t *testing.T) {
		var body map[string]any
		encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&body)
			fmt.Fprint(w, denseResponse)
		}, deepinfra.WithNormalize(true))
		if _, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense)); err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if body["normalize"] != true {
			t.Errorf("normalize = %v, want true", body["normalize"])
		}
	})
}

// BGE-M3 is symmetric, so the role changes nothing on the wire — but it must
// also not be silently rejected, because a caller writing role-aware retrieval
// code has to be able to pass it.
func TestEncodeAcceptsEveryInputRole(t *testing.T) {
	for _, inputType := range []core.EmbeddingInputType{
		core.EmbeddingInputNone, core.EmbeddingInputQuery, core.EmbeddingInputDocument,
	} {
		encoder, _ := newTestEncoder(t, respondWith(denseResponse))
		_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
			Input:     []string{"a", "b"},
			InputType: inputType,
			Outputs:   []core.RepresentationKind{core.RepresentationDense},
		})
		if err != nil {
			t.Errorf("input type %q: %v", inputType, err)
		}
	}
}

func TestEncodeDecodesEveryRepresentation(t *testing.T) {
	encoder, _ := newTestEncoder(t, respondWith(readFixture(t)))
	resp, err := encoder.Encode(context.Background(), documentRequest(
		core.RepresentationDense, core.RepresentationSparse, core.RepresentationMultiVector,
	))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	if len(resp.Data) != 2 {
		t.Fatalf("got %d representations", len(resp.Data))
	}
	if resp.Model != deepinfra.BGEM3Model {
		t.Errorf("model = %q", resp.Model)
	}

	// Dense, in input order.
	if resp.Data[0].Dense[0] != 0.12 || resp.Data[1].Dense[0] != 0.08 {
		t.Errorf("dense output is not in input order: %v", resp.Data)
	}
	if got := resp.Spaces[core.RepresentationDense].Dimensions; got != 4 {
		t.Errorf("dense space width = %d, want the observed 4", got)
	}

	// Sparse: the vocabulary-width row becomes coordinates, zeros dropped.
	sparse := resp.Data[0].Sparse
	if len(sparse.Indices) != 2 || sparse.Indices[0] != 1 || sparse.Indices[1] != 4 {
		t.Errorf("sparse indices = %v, want the nonzero positions", sparse.Indices)
	}
	if sparse.Values[0] != 0.91 {
		t.Errorf("sparse values = %v", sparse.Values)
	}
	sparseSpace := resp.Spaces[core.RepresentationSparse]
	if sparseSpace.Dimensions != 8 {
		t.Errorf("sparse vocabulary = %d, want the observed row width", sparseSpace.Dimensions)
	}
	if sparseSpace.Metric != core.SimilarityDotProduct {
		t.Errorf("sparse metric = %q, want dot_product", sparseSpace.Metric)
	}

	// Multi-vector: one vector per token, not pooled into one.
	if len(resp.Data[0].MultiVector) != 2 || len(resp.Data[1].MultiVector) != 1 {
		t.Errorf("multi-vector shape = %d, %d token vectors",
			len(resp.Data[0].MultiVector), len(resp.Data[1].MultiVector))
	}
	if resp.Spaces[core.RepresentationMultiVector].Dimensions != 4 {
		t.Error("multi-vector space did not record the observed token width")
	}

	if resp.Usage.InputTokens != 6 || resp.Usage.RequestCount != 1 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.Usage.OutputBytes == 0 {
		t.Error("usage did not record the response size")
	}
}

// Each kind lands in its own space; mixing them in one index would be a silent
// corruption.
func TestEncodeSpacesAreDistinct(t *testing.T) {
	encoder, _ := newTestEncoder(t, respondWith(readFixture(t)),
		deepinfra.WithModelRevision("rev-abc", "xlm-roberta-250002"))
	resp, err := encoder.Encode(context.Background(), documentRequest(
		core.RepresentationDense, core.RepresentationSparse, core.RepresentationMultiVector,
	))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	ids := make(map[string]bool)
	for kind, space := range resp.Spaces {
		if space.Revision != "rev-abc" || space.Tokenizer != "xlm-roberta-250002" {
			t.Errorf("%s space did not record the configured revisions: %+v", kind, space)
		}
		if err := space.Validate(); err != nil {
			t.Errorf("%s space: %v", kind, err)
		}
		if ids[space.ID] {
			t.Errorf("%s reuses another kind's space ID", kind)
		}
		ids[space.ID] = true
	}
}

// A recorded revision is what lets a consumer notice that the deployment
// behind a model name changed.
func TestEncodeRevisionChangesSpaceIdentity(t *testing.T) {
	spaceID := func(opts ...deepinfra.Option) string {
		encoder, _ := newTestEncoder(t, respondWith(denseResponse), opts...)
		resp, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense))
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		return resp.Spaces[core.RepresentationDense].ID
	}

	if spaceID() == spaceID(deepinfra.WithModelRevision("rev-1", "tok-1")) {
		t.Error("recording a revision did not change the space ID")
	}
	if spaceID(deepinfra.WithModelRevision("rev-1", "tok-1")) ==
		spaceID(deepinfra.WithModelRevision("rev-2", "tok-1")) {
		t.Error("a revision change did not change the space ID")
	}
}

func TestEncodeSparseCoordinateMap(t *testing.T) {
	const body = `{
		"sparse": [{"8271": 0.37, "1012": 0.91}, {"914": 0.63}],
		"input_tokens": 4
	}`

	t.Run("needs a declared vocabulary", func(t *testing.T) {
		encoder, _ := newTestEncoder(t, respondWith(body))
		_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationSparse))
		if err == nil || !strings.Contains(err.Error(), "WithSparseVocabulary") {
			t.Fatalf("got %v, want an error naming the missing vocabulary", err)
		}
	})

	t.Run("sorted into canonical order", func(t *testing.T) {
		encoder, _ := newTestEncoder(t, respondWith(body),
			deepinfra.WithSparseVocabulary(deepinfra.BGEM3SparseVocabulary))
		resp, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationSparse))
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		sparse := resp.Data[0].Sparse
		if len(sparse.Indices) != 2 || sparse.Indices[0] != 1012 || sparse.Indices[1] != 8271 {
			t.Fatalf("indices = %v, want ascending order", sparse.Indices)
		}
		if sparse.Values[0] != 0.91 || sparse.Values[1] != 0.37 {
			t.Errorf("values = %v, want the weights to follow their indices", sparse.Values)
		}
		if resp.Spaces[core.RepresentationSparse].Dimensions != deepinfra.BGEM3SparseVocabulary {
			t.Error("sparse space did not use the declared vocabulary")
		}
	})
}

// A declared vocabulary that contradicts the observed row width means one of
// the two is wrong, and storing either would be a guess.
func TestEncodeRejectsVocabularyMismatch(t *testing.T) {
	encoder, _ := newTestEncoder(t, respondWith(readFixture(t)), deepinfra.WithSparseVocabulary(250002))
	_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationSparse))
	if err == nil || !strings.Contains(err.Error(), "declared as 250002") {
		t.Fatalf("got %v, want a vocabulary mismatch error", err)
	}
}

func TestEncodeRejectsMalformedSparse(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"null row", `{"sparse": [null, null]}`, "sparse row is null"},
		{"string row", `{"sparse": ["oops", "oops"]}`, "neither a weight array nor a coordinate map"},
		{"non-numeric key", `{"sparse": [{"cat": 0.5}, {"1": 0.5}]}`, "is not a token index"},
		{"non-numeric weight", `{"sparse": [{"1": "heavy"}, {"1": 0.5}]}`, "is not a number"},
		{"duplicate coordinate", `{"sparse": [{"1": 0.5, "1": 0.9}, {"1": 0.5}]}`, "appears more than once"},
		{"nested array row", `{"sparse": [[[1, 0.5]], [[2, 0.5]]]}`, "not an array of weights"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			encoder, _ := newTestEncoder(t, respondWith(tc.body), deepinfra.WithSparseVocabulary(100))
			_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationSparse))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

// An out-of-range coordinate would index past the vocabulary of any store that
// accepted it.
func TestEncodeRejectsOutOfRangeSparseIndex(t *testing.T) {
	encoder, _ := newTestEncoder(t,
		respondWith(`{"sparse": [{"500": 0.5}, {"1": 0.5}]}`),
		deepinfra.WithSparseVocabulary(100))
	_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationSparse))
	if err == nil || !strings.Contains(err.Error(), "outside the declared vocabulary") {
		t.Fatalf("got %v, want an out-of-vocabulary error", err)
	}
}

func TestEncodeRejectsMalformedResponses(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		outputs []core.RepresentationKind
		want    string
	}{
		{
			name:    "truncated json",
			body:    `{"embeddings": [[0.1, 0.2]`,
			outputs: []core.RepresentationKind{core.RepresentationDense},
			want:    "decode response",
		},
		{
			name:    "short dense batch",
			body:    `{"embeddings": [[0.1, 0.2]]}`,
			outputs: []core.RepresentationKind{core.RepresentationDense},
			want:    "got 1 dense representations for 2 inputs",
		},
		{
			name:    "long dense batch",
			body:    `{"embeddings": [[0.1], [0.2], [0.3]]}`,
			outputs: []core.RepresentationKind{core.RepresentationDense},
			want:    "got 3 dense representations for 2 inputs",
		},
		{
			name:    "missing requested output",
			body:    `{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}`,
			outputs: []core.RepresentationKind{core.RepresentationDense, core.RepresentationSparse},
			want:    "got 0 sparse representations for 2 inputs",
		},
		{
			name:    "short multi-vector batch",
			body:    `{"colbert": [[[0.1, 0.2]]]}`,
			outputs: []core.RepresentationKind{core.RepresentationMultiVector},
			want:    "got 1 multi_vector representations for 2 inputs",
		},
		{
			name:    "inconsistent dense width",
			body:    `{"embeddings": [[0.1, 0.2], [0.3]]}`,
			outputs: []core.RepresentationKind{core.RepresentationDense},
			want:    "width 1, space declares 2",
		},
		{
			name:    "not a number",
			body:    `{"embeddings": [[0.1, "x"], [0.3, 0.4]]}`,
			outputs: []core.RepresentationKind{core.RepresentationDense},
			want:    "decode response",
		},
		{
			name:    "inconsistent token vector width",
			body:    `{"colbert": [[[0.1, 0.2], [0.3]], [[0.4, 0.5]]]}`,
			outputs: []core.RepresentationKind{core.RepresentationMultiVector},
			want:    "token vector 1 has width 1",
		},
		{
			name:    "empty multi-vector item",
			body:    `{"colbert": [[[0.1, 0.2]], []]}`,
			outputs: []core.RepresentationKind{core.RepresentationMultiVector},
			want:    "multi-vector representation is missing",
		},
		{
			name:    "inference reported failure",
			body:    `{"embeddings": [[0.1], [0.2]], "inference_status": {"status": "failed"}}`,
			outputs: []core.RepresentationKind{core.RepresentationDense},
			want:    "status failed",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			encoder, _ := newTestEncoder(t, respondWith(tc.body))
			_, err := encoder.Encode(context.Background(), documentRequest(tc.outputs...))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestEncodeRejectsNonFiniteValues(t *testing.T) {
	// JSON has no NaN literal, so a provider that emits one sends it as a
	// bare token; encoding/json rejects that before the validator sees it.
	// A very large exponent is the reachable path to an infinity.
	encoder, _ := newTestEncoder(t, respondWith(`{"embeddings": [[1e400, 0.2], [0.3, 0.4]]}`))
	_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense))
	if err == nil {
		t.Fatal("an out-of-range float should not be accepted")
	}
}

func TestEncodeRejectsUnsupportedOutputs(t *testing.T) {
	encoder, _ := newTestEncoder(t, respondWith(denseResponse),
		deepinfra.WithOutputs(core.RepresentationDense))

	_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationSparse))
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
	var typed *core.UnsupportedRepresentationError
	if !errors.As(err, &typed) || typed.Provider != "deepinfra" {
		t.Fatalf("error does not name the provider: %v", err)
	}
}

// The native API has no truncate parameter, so accepting the option would mean
// silently ignoring a caller's instruction not to clip a document.
func TestEncodeRejectsTruncateOption(t *testing.T) {
	encoder, _ := newTestEncoder(t, respondWith(denseResponse))
	truncate := false
	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:    []string{"a"},
		Outputs:  []core.RepresentationKind{core.RepresentationDense},
		Truncate: &truncate,
	})
	if !errors.Is(err, core.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want the truncate option to be rejected", err)
	}
}

func TestEncodeBatchesLargeRequests(t *testing.T) {
	var batches [][]string
	encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Inputs []string `json:"inputs"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		batches = append(batches, body.Inputs)

		vectors := make([]string, len(body.Inputs))
		for i, text := range body.Inputs {
			vectors[i] = fmt.Sprintf("[%d, 0.5]", len(text))
		}
		fmt.Fprintf(w, `{"embeddings": [%s], "input_tokens": %d}`,
			strings.Join(vectors, ","), len(body.Inputs))
	}, deepinfra.WithBatchSize(2))

	resp, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a", "bb", "ccc", "dddd", "eeeee"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(batches) != 3 {
		t.Fatalf("made %d provider calls, want 3", len(batches))
	}
	if len(resp.Data) != 5 {
		t.Fatalf("got %d representations for 5 inputs", len(resp.Data))
	}
	for i, want := range []float32{1, 2, 3, 4, 5} {
		if resp.Data[i].Dense[0] != want {
			t.Errorf("item %d = %v, want the encoding of input %d", i, resp.Data[i].Dense, i)
		}
	}
	if resp.Usage.RequestCount != 3 {
		t.Errorf("request count = %d, want one per provider call", resp.Usage.RequestCount)
	}
	if resp.Usage.InputTokens != 5 {
		t.Errorf("input tokens = %d, want the summed 5", resp.Usage.InputTokens)
	}
}

func TestEncodeRejectsOversizedResponses(t *testing.T) {
	big := strings.Repeat("0.1,", 1000) + "0.1"
	encoder, _ := newTestEncoder(t,
		respondWith(fmt.Sprintf(`{"embeddings": [[%s], [%s]]}`, big, big)),
		deepinfra.WithMaxResponseBytes(256))

	_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense))
	if err == nil || !strings.Contains(err.Error(), "exceeds the 256 byte limit") {
		t.Fatalf("got %v, want an oversized-response error", err)
	}
}

func TestEncodeDoesNotRetryClientErrors(t *testing.T) {
	calls := 0
	encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"detail": "inputs must be a list of strings"}`)
	})

	_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense))
	var apiErr *deepinfra.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("got %v, want *deepinfra.APIError", err)
	}
	if apiErr.Status != http.StatusBadRequest {
		t.Errorf("status = %d", apiErr.Status)
	}
	if apiErr.Detail != "inputs must be a list of strings" {
		t.Errorf("detail = %q", apiErr.Detail)
	}
	if calls != 1 {
		t.Errorf("made %d calls; a 400 will fail identically on a retry", calls)
	}
}

// An error must be safe to log next to the documents that produced it, so the
// token never appears and neither does the response body verbatim.
func TestAPIErrorsRedactSecretsAndInput(t *testing.T) {
	const document = "the launch code is hunter2"
	encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		fmt.Fprintf(w, `{"error": "cannot embed %q", "internal": "token test-token"}`, document)
	})

	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{document},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	message := err.Error()
	if strings.Contains(message, "test-token") {
		t.Errorf("error leaks the API token: %q", message)
	}
	if strings.Contains(message, "internal") {
		t.Errorf("error echoes the raw response body: %q", message)
	}
}

func TestEncodeHonorsCancellation(t *testing.T) {
	encoder, _ := newTestEncoder(t, respondWith(denseResponse))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := encoder.Encode(ctx, documentRequest(core.RepresentationDense))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}

func TestEmbedProjectsDenseOutput(t *testing.T) {
	var body map[string]any
	encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, denseResponse)
	})

	resp, err := encoder.Embed(context.Background(), &core.EmbeddingRequest{
		Input:     []string{"a", "b"},
		InputType: core.EmbeddingInputQuery,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 2 || len(resp.Vectors[0]) != 2 {
		t.Fatalf("vectors = %v", resp.Vectors)
	}
	if resp.Usage.TotalTokens != 6 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if body["sparse"] != false || body["colbert"] != false {
		t.Error("the dense projection should not request sparse or multi-vector output")
	}
}

func TestEmbedFailsWhenDenseIsNotAdvertised(t *testing.T) {
	encoder, _ := newTestEncoder(t, respondWith(denseResponse),
		deepinfra.WithOutputs(core.RepresentationSparse))

	_, err := encoder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"a"}})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

func TestCapabilitiesDescribeTheDeployment(t *testing.T) {
	full := deepinfra.MustNew(deepinfra.BGEM3Model, deepinfra.WithAPIToken("t")).Capabilities()
	if len(full.Outputs) != 3 || !full.SupportsMultiOutput || full.SupportsTruncation {
		t.Fatalf("default capabilities = %+v", full)
	}

	narrowed := deepinfra.MustNew(deepinfra.BGEM3Model,
		deepinfra.WithAPIToken("t"),
		deepinfra.WithOutputs(core.RepresentationDense),
	).Capabilities()
	if len(narrowed.Outputs) != 1 {
		t.Fatalf("narrowed capabilities = %+v", narrowed)
	}

	// Capabilities must hand back a copy.
	narrowed.Outputs[0] = core.RepresentationSparse
	if deepinfra.MustNew(deepinfra.BGEM3Model,
		deepinfra.WithAPIToken("t"),
		deepinfra.WithOutputs(core.RepresentationDense),
	).Capabilities().Outputs[0] != core.RepresentationDense {
		t.Error("Capabilities() shares its Outputs slice")
	}
}

// The provider must satisfy the same contract every other encoder does.
func TestConformance(t *testing.T) {
	fixture := readFixture(t)
	conformance.RunRepresentation(t, conformance.RepresentationOptions{
		NewEncoder: func(t *testing.T) core.RepresentationEncoder {
			encoder, _ := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Inputs  []string `json:"inputs"`
					Dense   bool     `json:"dense"`
					Sparse  bool     `json:"sparse"`
					Colbert bool     `json:"colbert"`
				}
				_ = json.NewDecoder(r.Body).Decode(&body)
				writeSyntheticResponse(w, body.Inputs, body.Dense, body.Sparse, body.Colbert)
			}, deepinfra.WithSparseVocabulary(256))
			_ = fixture
			return encoder
		},
		Corpus:        []string{"alpha beta", "gamma", "delta epsilon zeta"},
		Deterministic: true,
	})
}

// writeSyntheticResponse answers with output derived from the inputs, so the
// conformance suite's order and determinism checks have something to compare.
func writeSyntheticResponse(w http.ResponseWriter, inputs []string, dense, sparse, colbert bool) {
	var parts []string
	if dense {
		rows := make([]string, len(inputs))
		for i, text := range inputs {
			rows[i] = fmt.Sprintf("[%d, 0.5]", len(text))
		}
		parts = append(parts, `"embeddings": [`+strings.Join(rows, ",")+`]`)
	}
	if sparse {
		rows := make([]string, len(inputs))
		for i, text := range inputs {
			rows[i] = fmt.Sprintf(`{"%d": 0.75}`, len(text)%256)
		}
		parts = append(parts, `"sparse": [`+strings.Join(rows, ",")+`]`)
	}
	if colbert {
		rows := make([]string, len(inputs))
		for i, text := range inputs {
			rows[i] = fmt.Sprintf("[[%d, 0.25]]", len(text))
		}
		parts = append(parts, `"colbert": [`+strings.Join(rows, ",")+`]`)
	}
	parts = append(parts, `"input_tokens": 3`)
	fmt.Fprint(w, "{"+strings.Join(parts, ",")+"}")
}

func readFixture(t *testing.T) string {
	t.Helper()
	body, err := os.ReadFile("testdata/bge_m3_multi_response.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return string(body)
}
