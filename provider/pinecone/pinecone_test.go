package pinecone_test

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
	"github.com/regularkevvv/agentic/provider/pinecone"
	"github.com/regularkevvv/agentic/provider/test/conformance"
)

// The sparse vocabulary a deployment measures for the hosted English model.
// It is a test constant, not a package default: this package will not invent a
// bound it cannot observe in a response.
const testVocabulary = 100000

const denseResponse = `{
	"model": "llama-text-embed-v2",
	"vector_type": "dense",
	"data": [
		{"vector_type": "dense", "values": [0.1, 0.2]},
		{"vector_type": "dense", "values": [0.3, 0.4]}
	],
	"usage": {"total_tokens": 6}
}`

func newTestEncoder(t *testing.T, handler http.HandlerFunc, opts ...pinecone.Option) *pinecone.Encoder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]pinecone.Option{
		pinecone.WithAPIKey("pc-test"),
		pinecone.WithBaseURL(server.URL),
	}, opts...)

	encoder, err := pinecone.New("llama-text-embed-v2", opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return encoder
}

func newSparseEncoder(t *testing.T, handler http.HandlerFunc, opts ...pinecone.Option) *pinecone.Encoder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]pinecone.Option{
		pinecone.WithAPIKey("pc-test"),
		pinecone.WithBaseURL(server.URL),
		pinecone.WithOutputs(core.RepresentationSparse),
		pinecone.WithSparseVocabulary(testVocabulary),
	}, opts...)

	encoder, err := pinecone.New(pinecone.SparseEnglishModel, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return encoder
}

func respondWith(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}
}

func documentRequest(kind core.RepresentationKind, inputs ...string) *core.RepresentationRequest {
	return &core.RepresentationRequest{
		Input:     inputs,
		InputType: core.EmbeddingInputDocument,
		Outputs:   []core.RepresentationKind{kind},
	}
}

// --------------------------------------------------------------------------
// Construction
// --------------------------------------------------------------------------

func TestNewRequiresAPIKey(t *testing.T) {
	t.Setenv("PINECONE_API_KEY", "")
	if _, err := pinecone.New("llama-text-embed-v2"); err == nil {
		t.Fatal("expected an error when no API key is configured")
	}
}

func TestNewReadsKeyFromEnvironment(t *testing.T) {
	t.Setenv("PINECONE_API_KEY", "env-key")

	var apiKey, apiVersion string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		apiKey = r.Header.Get("Api-Key")
		apiVersion = r.Header.Get("X-Pinecone-Api-Version")
		fmt.Fprint(w, denseResponse)
	}))
	t.Cleanup(server.Close)

	encoder, err := pinecone.New("llama-text-embed-v2", pinecone.WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := encoder.Encode(context.Background(),
		documentRequest(core.RepresentationDense, "a", "b")); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if apiKey != "env-key" {
		t.Errorf("Api-Key = %q", apiKey)
	}
	// The dated version header is what keeps a response shape from changing
	// under a deployment without warning.
	if apiVersion != "2025-04" {
		t.Errorf("X-Pinecone-Api-Version = %q", apiVersion)
	}
}

// A sparse model has no usable space without a declared vocabulary, and this
// package will not guess one.
func TestNewRequiresSparseVocabulary(t *testing.T) {
	_, err := pinecone.New(pinecone.SparseEnglishModel,
		pinecone.WithAPIKey("k"),
		pinecone.WithOutputs(core.RepresentationSparse),
	)
	if err == nil || !strings.Contains(err.Error(), "WithSparseVocabulary") {
		t.Fatalf("got %v, want an error naming the missing vocabulary", err)
	}
}

func TestNewValidatesConfiguration(t *testing.T) {
	tests := []struct {
		name  string
		model string
		opts  []pinecone.Option
		want  string
	}{
		{"empty model", "", nil, "model cannot be empty"},
		{"two output kinds", "m", []pinecone.Option{
			pinecone.WithOutputs(core.RepresentationDense, core.RepresentationSparse),
		}, "one vector type"},
		{"no output kinds", "m", []pinecone.Option{pinecone.WithOutputs()}, "one vector type"},
		{"negative dimensions", "m", []pinecone.Option{pinecone.WithDimensions(-1)}, "dimensions cannot be negative"},
		{"negative batch size", "m", []pinecone.Option{pinecone.WithBatchSize(-1)}, "batch size"},
		{"negative retries", "m", []pinecone.Option{pinecone.WithMaxRetries(-1)}, "max retries"},
		{"zero response limit", "m", []pinecone.Option{pinecone.WithMaxResponseBytes(0)}, "max response bytes"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := append([]pinecone.Option{pinecone.WithAPIKey("k")}, tc.opts...)
			_, err := pinecone.New(tc.model, opts...)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

// Pinecone Inference has no late-interaction model, so asking for one is a
// typed capability error rather than a runtime surprise.
func TestNewRejectsMultiVector(t *testing.T) {
	_, err := pinecone.New("m", pinecone.WithAPIKey("k"),
		pinecone.WithOutputs(core.RepresentationMultiVector))
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

func TestMustNew(t *testing.T) {
	if pinecone.MustNew("m", pinecone.WithAPIKey("k")).Name() != "m" {
		t.Error("MustNew did not return the configured model")
	}
	defer func() {
		if recover() == nil {
			t.Error("MustNew should panic on an invalid configuration")
		}
	}()
	pinecone.MustNew("")
}

// --------------------------------------------------------------------------
// Dense
// --------------------------------------------------------------------------

func TestEncodeDense(t *testing.T) {
	var path string
	var body map[string]any

	encoder := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, denseResponse)
	})

	resp, err := encoder.Encode(context.Background(),
		documentRequest(core.RepresentationDense, "a", "b"))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	if path != "/embed" {
		t.Errorf("path = %q", path)
	}
	if body["model"] != "llama-text-embed-v2" {
		t.Errorf("model = %v", body["model"])
	}
	inputs, _ := body["inputs"].([]any)
	if len(inputs) != 2 {
		t.Fatalf("inputs = %v", body["inputs"])
	}
	first, _ := inputs[0].(map[string]any)
	if first["text"] != "a" {
		t.Errorf("inputs are not wrapped in {text}: %v", inputs[0])
	}

	if len(resp.Data) != 2 || resp.Data[1].Dense[1] != 0.4 {
		t.Errorf("data = %v", resp.Data)
	}
	space := resp.Spaces[core.RepresentationDense]
	if space.Dimensions != 2 || space.Metric != core.SimilarityCosine {
		t.Errorf("space = %+v", space)
	}
	if resp.Usage.InputTokens != 6 || resp.Usage.RequestCount != 1 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

// The roles are distinct on the wire, which is what an asymmetric retrieval
// model needs, and they encode into the same space.
func TestEncodeMapsInputRoles(t *testing.T) {
	tests := []struct {
		inputType core.EmbeddingInputType
		want      any
	}{
		{core.EmbeddingInputQuery, "query"},
		{core.EmbeddingInputDocument, "passage"},
		{core.EmbeddingInputNone, nil},
	}
	for _, tc := range tests {
		t.Run(string(tc.inputType), func(t *testing.T) {
			var body map[string]any
			encoder := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewDecoder(r.Body).Decode(&body)
				fmt.Fprint(w, denseResponse)
			})
			if _, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
				Input:     []string{"a", "b"},
				InputType: tc.inputType,
				Outputs:   []core.RepresentationKind{core.RepresentationDense},
			}); err != nil {
				t.Fatalf("Encode: %v", err)
			}
			params, _ := body["parameters"].(map[string]any)
			if params["input_type"] != tc.want {
				t.Errorf("input_type = %v, want %v", params["input_type"], tc.want)
			}
		})
	}
}

func TestEncodeMapsTruncation(t *testing.T) {
	tests := []struct {
		name     string
		truncate *bool
		want     any
	}{
		{"unset", nil, nil},
		{"true", boolPtr(true), "END"},
		{"false", boolPtr(false), "NONE"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var body map[string]any
			encoder := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewDecoder(r.Body).Decode(&body)
				fmt.Fprint(w, denseResponse)
			})
			if _, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
				Input:    []string{"a", "b"},
				Outputs:  []core.RepresentationKind{core.RepresentationDense},
				Truncate: tc.truncate,
			}); err != nil {
				t.Fatalf("Encode: %v", err)
			}
			params, _ := body["parameters"].(map[string]any)
			if params["truncate"] != tc.want {
				t.Errorf("truncate = %v, want %v", params["truncate"], tc.want)
			}
		})
	}
}

func TestEncodeSendsConfiguredDimension(t *testing.T) {
	var body map[string]any
	encoder := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, denseResponse)
	}, pinecone.WithDimensions(2))

	if _, err := encoder.Encode(context.Background(),
		documentRequest(core.RepresentationDense, "a", "b")); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	params, _ := body["parameters"].(map[string]any)
	if params["dimension"] != float64(2) {
		t.Errorf("dimension = %v", params["dimension"])
	}
}

func TestEmbedProjectsDenseOutput(t *testing.T) {
	encoder := newTestEncoder(t, respondWith(denseResponse))
	resp, err := encoder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"a", "b"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 2 || resp.Usage.TotalTokens != 6 {
		t.Fatalf("response = %+v", resp)
	}
}

func TestEmbedFailsForASparseModel(t *testing.T) {
	encoder := newSparseEncoder(t, respondWith("{}"))
	_, err := encoder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"a"}})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

// --------------------------------------------------------------------------
// Sparse
// --------------------------------------------------------------------------

// Pinecone returns coordinates in the model's own order; canonical form is
// strictly increasing, so they have to be sorted with their weights and tokens
// carried along.
func TestEncodeSparseCanonicalizesCoordinates(t *testing.T) {
	const body = `{
		"model": "pinecone-sparse-english-v0",
		"vector_type": "sparse",
		"data": [
			{"vector_type": "sparse", "sparse_indices": [822, 10, 156], "sparse_values": [0.2, 0.9, 0.4],
			 "sparse_tokens": ["fox", "quick", "brown"]}
		],
		"usage": {"total_tokens": 3}
	}`
	encoder := newSparseEncoder(t, respondWith(body), pinecone.WithReturnTokens(true))

	resp, tokens, err := encoder.EncodeWithTokens(context.Background(),
		documentRequest(core.RepresentationSparse, "the quick brown fox"))
	if err != nil {
		t.Fatalf("EncodeWithTokens: %v", err)
	}

	sparse := resp.Data[0].Sparse
	wantIndices := []uint32{10, 156, 822}
	for i, want := range wantIndices {
		if sparse.Indices[i] != want {
			t.Fatalf("indices = %v, want %v", sparse.Indices, wantIndices)
		}
	}
	// The weights and tokens must follow their coordinates through the sort.
	if sparse.Values[0] != 0.9 || sparse.Values[2] != 0.2 {
		t.Errorf("values = %v", sparse.Values)
	}
	if tokens[0][0] != "quick" || tokens[0][2] != "fox" {
		t.Errorf("tokens = %v", tokens[0])
	}

	space := resp.Spaces[core.RepresentationSparse]
	if space.Dimensions != testVocabulary || space.Metric != core.SimilarityDotProduct {
		t.Errorf("space = %+v", space)
	}
}

// Expansion is what a learned sparse model does beyond lexical matching. This
// pins that it is observable, not that any particular synonym appears — that
// is a live-model property, asserted in the e2e test.
func TestEncodeSparseExposesExpansionTokens(t *testing.T) {
	const body = `{
		"model": "pinecone-sparse-english-v0",
		"vector_type": "sparse",
		"data": [
			{"vector_type": "sparse", "sparse_indices": [11, 47, 900], "sparse_values": [0.8, 0.5, 0.3],
			 "sparse_tokens": ["car", "automobile", "vehicle"]}
		],
		"usage": {"total_tokens": 2}
	}`
	encoder := newSparseEncoder(t, respondWith(body), pinecone.WithReturnTokens(true))

	_, tokens, err := encoder.EncodeWithTokens(context.Background(),
		documentRequest(core.RepresentationSparse, "car"))
	if err != nil {
		t.Fatalf("EncodeWithTokens: %v", err)
	}
	if len(tokens[0]) != 3 {
		t.Fatalf("tokens = %v", tokens[0])
	}
	expanded := false
	for _, token := range tokens[0] {
		if token == "automobile" {
			expanded = true
		}
	}
	if !expanded {
		t.Error("expansion tokens are not reachable through EncodeWithTokens")
	}
}

func TestEncodeWithTokensRequiresTheOption(t *testing.T) {
	encoder := newSparseEncoder(t, respondWith("{}"))
	_, _, err := encoder.EncodeWithTokens(context.Background(),
		documentRequest(core.RepresentationSparse, "a"))
	if err == nil || !strings.Contains(err.Error(), "WithReturnTokens") {
		t.Fatalf("got %v, want an error naming the missing option", err)
	}
}

func TestEncodeSparseRejectsMalformedVectors(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "mismatched arrays",
			body: `{"vector_type":"sparse","data":[{"vector_type":"sparse","sparse_indices":[1,2],"sparse_values":[0.5]}]}`,
			want: "2 sparse indices and 1 values",
		},
		{
			name: "mismatched tokens",
			body: `{"vector_type":"sparse","data":[{"vector_type":"sparse","sparse_indices":[1],"sparse_values":[0.5],"sparse_tokens":["a","b"]}]}`,
			want: "2 sparse tokens for 1 coordinates",
		},
		{
			name: "duplicate coordinate",
			body: `{"vector_type":"sparse","data":[{"vector_type":"sparse","sparse_indices":[7,7],"sparse_values":[0.5,0.9]}]}`,
			want: "coordinate 7 appears more than once",
		},
		{
			name: "out of vocabulary",
			body: fmt.Sprintf(`{"vector_type":"sparse","data":[{"vector_type":"sparse","sparse_indices":[%d],"sparse_values":[0.5]}]}`, testVocabulary),
			want: "outside the declared vocabulary",
		},
		{
			name: "zero weight",
			body: `{"vector_type":"sparse","data":[{"vector_type":"sparse","sparse_indices":[7],"sparse_values":[0]}]}`,
			want: "is zero",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			encoder := newSparseEncoder(t, respondWith(tc.body))
			_, err := encoder.Encode(context.Background(),
				documentRequest(core.RepresentationSparse, "a"))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

// --------------------------------------------------------------------------
// Verification rather than assumption
// --------------------------------------------------------------------------

// A model reconfigured to return the other vector type would otherwise decode
// into empty representations that pass straight into an index.
func TestEncodeRejectsAMismatchedVectorType(t *testing.T) {
	t.Run("response level", func(t *testing.T) {
		encoder := newTestEncoder(t, respondWith(
			`{"vector_type":"sparse","data":[{"vector_type":"sparse","sparse_indices":[1],"sparse_values":[0.5]}]}`))
		_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense, "a"))
		if err == nil || !strings.Contains(err.Error(), "configured for dense") {
			t.Fatalf("got %v, want a vector-type mismatch", err)
		}
	})

	t.Run("item level", func(t *testing.T) {
		encoder := newTestEncoder(t, respondWith(
			`{"data":[{"vector_type":"sparse","sparse_indices":[1],"sparse_values":[0.5]}]}`))
		_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense, "a"))
		if err == nil || !strings.Contains(err.Error(), "configured for dense") {
			t.Fatalf("got %v, want a vector-type mismatch", err)
		}
	})
}

func TestEncodeRejectsUnsupportedOutputs(t *testing.T) {
	encoder := newTestEncoder(t, respondWith(denseResponse))
	_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationSparse, "a"))
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

// One model returns one vector type, so asking for two is asking for two
// models. The encoder declares a single kind, so the second one is refused as
// unsupported before the multi-output rule is reached — either way the request
// never leaves the process.
func TestEncodeRejectsMultipleOutputs(t *testing.T) {
	encoder := newTestEncoder(t, func(http.ResponseWriter, *http.Request) {
		t.Error("a multi-output request should not reach the provider")
	})
	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense, core.RepresentationSparse},
	})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want the multi-output request refused", err)
	}
	if encoder.Capabilities().SupportsMultiOutput {
		t.Error("capabilities should say multi-output is unsupported")
	}
}

func TestEncodeRejectsMalformedResponses(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"truncated", `{"data": [`, "decode response"},
		{"short batch", `{"vector_type":"dense","data":[{"vector_type":"dense","values":[0.1,0.2]}]}`, "got 1 vectors for 2 inputs"},
		{"ragged widths", `{"vector_type":"dense","data":[{"values":[0.1,0.2]},{"values":[0.3]}]}`, "width 1, space declares 2"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			encoder := newTestEncoder(t, respondWith(tc.body))
			_, err := encoder.Encode(context.Background(),
				documentRequest(core.RepresentationDense, "a", "b"))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestEncodeRejectsOversizedResponses(t *testing.T) {
	encoder := newTestEncoder(t, respondWith(strings.Repeat("x", 4096)),
		pinecone.WithMaxResponseBytes(128))
	_, err := encoder.Encode(context.Background(), documentRequest(core.RepresentationDense, "a"))
	if err == nil || !strings.Contains(err.Error(), "exceeds the 128 byte limit") {
		t.Fatalf("got %v, want an oversized-response error", err)
	}
}

func TestEncodeBatchesLargeRequests(t *testing.T) {
	calls := 0
	encoder := newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body struct {
			Inputs []struct {
				Text string `json:"text"`
			} `json:"inputs"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)

		rows := make([]string, len(body.Inputs))
		for i, in := range body.Inputs {
			rows[i] = fmt.Sprintf(`{"vector_type":"dense","values":[%d,0.5]}`, len(in.Text))
		}
		fmt.Fprintf(w, `{"model":"llama-text-embed-v2","vector_type":"dense","data":[%s],"usage":{"total_tokens":%d}}`,
			strings.Join(rows, ","), len(body.Inputs))
	}, pinecone.WithBatchSize(2))

	resp, err := encoder.Encode(context.Background(),
		documentRequest(core.RepresentationDense, "a", "bb", "ccc", "dddd", "eeeee"))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if calls != 3 {
		t.Fatalf("made %d calls, want 3", calls)
	}
	for i, want := range []float32{1, 2, 3, 4, 5} {
		if resp.Data[i].Dense[0] != want {
			t.Errorf("item %d = %v, want the encoding of input %d", i, resp.Data[i].Dense, i)
		}
	}
	if resp.Usage.RequestCount != 3 || resp.Usage.InputTokens != 5 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

// Chunked sparse output must keep its token diagnostics aligned with the
// merged batch, not just the last chunk.
func TestEncodeWithTokensAcrossChunks(t *testing.T) {
	encoder := newSparseEncoder(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Inputs []struct {
				Text string `json:"text"`
			} `json:"inputs"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)

		rows := make([]string, len(body.Inputs))
		for i, in := range body.Inputs {
			rows[i] = fmt.Sprintf(`{"vector_type":"sparse","sparse_indices":[%d],"sparse_values":[0.5],"sparse_tokens":[%q]}`,
				len(in.Text), in.Text)
		}
		fmt.Fprintf(w, `{"vector_type":"sparse","data":[%s],"usage":{"total_tokens":1}}`,
			strings.Join(rows, ","))
	}, pinecone.WithReturnTokens(true), pinecone.WithBatchSize(2))

	resp, tokens, err := encoder.EncodeWithTokens(context.Background(),
		documentRequest(core.RepresentationSparse, "a", "bb", "ccc"))
	if err != nil {
		t.Fatalf("EncodeWithTokens: %v", err)
	}
	if len(tokens) != 3 {
		t.Fatalf("got %d token slices for 3 inputs", len(tokens))
	}
	for i, want := range []string{"a", "bb", "ccc"} {
		if tokens[i][0] != want {
			t.Errorf("tokens[%d] = %v, want the token for input %d", i, tokens[i], i)
		}
		if resp.Data[i].Sparse.Indices[0] != uint32(len(want)) {
			t.Errorf("item %d does not follow input order", i)
		}
	}
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

// The details array names the offending field and quotes its value, so it is
// deliberately dropped.
func TestAPIErrorsRedactSecretsAndInput(t *testing.T) {
	const document = "the launch code is hunter2"
	encoder := newTestEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, `{"error": {"code": "INVALID_ARGUMENT", "message": "input too long",
			"details": [{"field": "inputs[0].text", "value": %q}]}}`, document)
	})

	_, err := encoder.Encode(context.Background(),
		documentRequest(core.RepresentationDense, document))
	var apiErr *pinecone.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("got %v, want *pinecone.APIError", err)
	}
	if apiErr.Code != "INVALID_ARGUMENT" || apiErr.Detail != "input too long" {
		t.Errorf("error = %+v", apiErr)
	}
	message := err.Error()
	if strings.Contains(message, "hunter2") || strings.Contains(message, "pc-test") {
		t.Errorf("error leaks input or credentials: %q", message)
	}
}

func TestEncodeHonorsCancellation(t *testing.T) {
	encoder := newTestEncoder(t, respondWith(denseResponse))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := encoder.Encode(ctx, documentRequest(core.RepresentationDense, "a"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}

// --------------------------------------------------------------------------
// Conformance
// --------------------------------------------------------------------------

func TestDenseConformance(t *testing.T) {
	conformance.RunRepresentation(t, conformance.RepresentationOptions{
		NewEncoder: func(t *testing.T) core.RepresentationEncoder {
			return newTestEncoder(t, func(w http.ResponseWriter, r *http.Request) {
				writeSynthetic(w, r, false)
			})
		},
		Corpus:        []string{"alpha beta", "gamma", "delta epsilon"},
		Deterministic: true,
	})
}

func TestSparseConformance(t *testing.T) {
	conformance.RunRepresentation(t, conformance.RepresentationOptions{
		NewEncoder: func(t *testing.T) core.RepresentationEncoder {
			return newSparseEncoder(t, func(w http.ResponseWriter, r *http.Request) {
				writeSynthetic(w, r, true)
			})
		},
		Corpus:        []string{"alpha beta", "gamma", "delta epsilon"},
		Deterministic: true,
	})
}

func writeSynthetic(w http.ResponseWriter, r *http.Request, sparse bool) {
	var body struct {
		Inputs []struct {
			Text string `json:"text"`
		} `json:"inputs"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)

	rows := make([]string, len(body.Inputs))
	for i, in := range body.Inputs {
		if sparse {
			rows[i] = fmt.Sprintf(`{"vector_type":"sparse","sparse_indices":[%d],"sparse_values":[0.75]}`,
				len(in.Text)%testVocabulary)
			continue
		}
		rows[i] = fmt.Sprintf(`{"vector_type":"dense","values":[%d,0.5]}`, len(in.Text))
	}

	vectorType := "dense"
	if sparse {
		vectorType = "sparse"
	}
	fmt.Fprintf(w, `{"model":"m","vector_type":%q,"data":[%s],"usage":{"total_tokens":3}}`,
		vectorType, strings.Join(rows, ","))
}

func boolPtr(v bool) *bool { return &v }

func TestMain(m *testing.M) {
	// Keep a developer's real key out of the hermetic tests; every one of them
	// configures its own.
	_ = os.Unsetenv("PINECONE_API_KEY")
	os.Exit(m.Run())
}
