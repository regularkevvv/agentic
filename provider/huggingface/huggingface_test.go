package huggingface_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/huggingface"
	"github.com/regularkevvv/agentic/provider/test/conformance"
)

const goldenResponsePath = "../../deploy/representations/testdata/response.json"

func goldenResponse(t *testing.T) string {
	t.Helper()
	body, err := os.ReadFile(filepath.Clean(goldenResponsePath))
	if err != nil {
		t.Fatalf("read golden response: %v", err)
	}
	return string(body)
}

func newDedicated(t *testing.T, handler http.HandlerFunc, opts ...huggingface.DedicatedOption) *huggingface.DedicatedEncoder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]huggingface.DedicatedOption{
		huggingface.WithToken("hf_test"),
		huggingface.WithModel("BAAI/bge-m3"),
	}, opts...)

	encoder, err := huggingface.NewDedicated(server.URL, opts...)
	if err != nil {
		t.Fatalf("NewDedicated: %v", err)
	}
	return encoder
}

func newShared(t *testing.T, handler http.HandlerFunc, opts ...huggingface.SharedOption) *huggingface.SharedEncoder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]huggingface.SharedOption{
		huggingface.WithSharedToken("hf_test"),
		huggingface.WithRouterURL(server.URL),
	}, opts...)

	encoder, err := huggingface.NewShared("intfloat/multilingual-e5-large-instruct", opts...)
	if err != nil {
		t.Fatalf("NewShared: %v", err)
	}
	return encoder
}

func respondWith(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}
}

func goldenRequest() *core.RepresentationRequest {
	truncate := true
	return &core.RepresentationRequest{
		Input:     []string{"a document", "another document"},
		InputType: core.EmbeddingInputDocument,
		Outputs:   []core.RepresentationKind{core.RepresentationDense, core.RepresentationSparse},
		Truncate:  &truncate,
	}
}

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

func TestNewRequiresToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "")

	if _, err := huggingface.NewDedicated("https://example.endpoints.huggingface.cloud"); err == nil {
		t.Error("NewDedicated should require a token")
	}
	if _, err := huggingface.NewShared("model"); err == nil {
		t.Error("NewShared should require a token")
	}
}

func TestNewReadsTokenFromEnvironment(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "hub-token")

	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		fmt.Fprint(w, goldenResponse(t))
	}))
	t.Cleanup(server.Close)

	encoder, err := huggingface.NewDedicated(server.URL, huggingface.WithModel("BAAI/bge-m3"))
	if err != nil {
		t.Fatalf("NewDedicated: %v", err)
	}
	if _, err := encoder.Encode(context.Background(), goldenRequest()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if authorization != "Bearer hub-token" {
		t.Errorf("Authorization = %q", authorization)
	}
}

func TestNewDedicatedValidatesConfiguration(t *testing.T) {
	tests := []struct {
		name string
		url  string
		opts []huggingface.DedicatedOption
		want string
	}{
		{"empty url", "", nil, "endpoint URL cannot be empty"},
		{"relative url", "example.com", nil, "must be absolute"},
		{"negative retries", "https://e", []huggingface.DedicatedOption{huggingface.WithMaxRetries(-1)}, "max retries"},
		{"zero response limit", "https://e", []huggingface.DedicatedOption{huggingface.WithMaxResponseBytes(0)}, "max response bytes"},
		{"negative batch size", "https://e", []huggingface.DedicatedOption{huggingface.WithBatchSize(-1)}, "batch size"},
		{"unknown output", "https://e", []huggingface.DedicatedOption{huggingface.WithOutputs("colbert")}, "not dense, sparse, or multi_vector"},
		{"invalid pinned space", "https://e", []huggingface.DedicatedOption{
			huggingface.WithVectorSpaces(map[core.RepresentationKind]core.VectorSpace{
				core.RepresentationDense: {Provider: "p"},
			}),
		}, "pinned dense space"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := append([]huggingface.DedicatedOption{huggingface.WithToken("t")}, tc.opts...)
			_, err := huggingface.NewDedicated(tc.url, opts...)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestNewSharedValidatesConfiguration(t *testing.T) {
	if _, err := huggingface.NewShared("", huggingface.WithSharedToken("t")); err == nil {
		t.Error("an empty model should be rejected")
	}
	if _, err := huggingface.NewShared("m", huggingface.WithSharedToken("t"),
		huggingface.WithSharedBatchSize(-1)); err == nil {
		t.Error("a negative batch size should be rejected")
	}
	if _, err := huggingface.NewShared("m", huggingface.WithSharedToken("t"),
		huggingface.WithSharedMaxRetries(-1)); err == nil {
		t.Error("negative retries should be rejected")
	}
	if _, err := huggingface.NewShared("m", huggingface.WithSharedToken("t"),
		huggingface.WithSharedMaxResponseBytes(0)); err == nil {
		t.Error("a zero response limit should be rejected")
	}
}

// --------------------------------------------------------------------------
// Dedicated endpoints
// --------------------------------------------------------------------------

func TestDedicatedSpeaksTheProtocol(t *testing.T) {
	var body map[string]any
	var contentType string

	encoder := newDedicated(t, func(w http.ResponseWriter, r *http.Request) {
		contentType = r.Header.Get("Content-Type")
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, goldenResponse(t))
	})

	resp, err := encoder.Encode(context.Background(), goldenRequest())
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	if contentType != "application/json" {
		t.Errorf("Content-Type = %q", contentType)
	}
	if body["version"] != "agentic.representations.v1" {
		t.Errorf("version = %v", body["version"])
	}
	if body["input_type"] != "document" {
		t.Errorf("input_type = %v", body["input_type"])
	}
	if body["truncate"] != true {
		t.Errorf("truncate = %v", body["truncate"])
	}
	outputs, _ := body["outputs"].([]any)
	if len(outputs) != 2 || outputs[0] != "dense" || outputs[1] != "sparse" {
		t.Errorf("outputs = %v", body["outputs"])
	}

	if len(resp.Data) != 2 {
		t.Fatalf("got %d items", len(resp.Data))
	}
	if resp.Spaces[core.RepresentationDense].ID != "configured-immutable-dense-id" {
		t.Errorf("dense space = %+v", resp.Spaces[core.RepresentationDense])
	}
	if resp.Model != "BAAI/bge-m3" {
		t.Errorf("model = %q", resp.Model)
	}
}

func TestDedicatedNameFallsBackToEndpoint(t *testing.T) {
	encoder, err := huggingface.NewDedicated("https://abc.endpoints.huggingface.cloud/",
		huggingface.WithToken("t"))
	if err != nil {
		t.Fatalf("NewDedicated: %v", err)
	}
	if encoder.Name() != "https://abc.endpoints.huggingface.cloud" {
		t.Errorf("Name() = %q, want the trimmed endpoint", encoder.Name())
	}
}

func TestDedicatedRejectsUnsupportedOutputs(t *testing.T) {
	encoder := newDedicated(t, respondWith(goldenResponse(t)),
		huggingface.WithOutputs(core.RepresentationDense))

	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationMultiVector},
	})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

func TestDedicatedBatchesLargeRequests(t *testing.T) {
	var batches [][]string
	encoder := newDedicated(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Inputs []string `json:"inputs"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		batches = append(batches, body.Inputs)

		items := make([]string, len(body.Inputs))
		for i, text := range body.Inputs {
			items[i] = fmt.Sprintf(`{"dense": [%d, 0.5]}`, len(text))
		}
		fmt.Fprintf(w, `{
			"version": "agentic.representations.v1",
			"model": "BAAI/bge-m3",
			"spaces": {"dense": {"id":"d","provider":"custom","model":"BAAI/bge-m3","kind":"dense","dimensions":2,"metric":"cosine"}},
			"data": [%s],
			"usage": {"input_tokens": %d, "request_count": 1}
		}`, strings.Join(items, ","), len(body.Inputs))
	}, huggingface.WithBatchSize(2))

	resp, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a", "bb", "ccc", "dddd", "eeeee"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(batches) != 3 {
		t.Fatalf("made %d calls, want 3", len(batches))
	}
	for i, want := range []float32{1, 2, 3, 4, 5} {
		if resp.Data[i].Dense[0] != want {
			t.Errorf("item %d = %v, want the encoding of input %d", i, resp.Data[i].Dense, i)
		}
	}
	if resp.Usage.RequestCount != 3 {
		t.Errorf("request count = %d", resp.Usage.RequestCount)
	}
}

func TestDedicatedRejectsMalformedResponses(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"truncated", `{"version": "agentic`, "not valid agentic.representations.v1 JSON"},
		{"wrong major", `{"version":"agentic.representations.v9","model":"m","data":[]}`, "major version 9"},
		{
			name: "short batch",
			body: `{"version":"agentic.representations.v1","model":"m",
				"spaces":{"dense":{"id":"d","provider":"p","model":"m","kind":"dense","dimensions":2,"metric":"cosine"}},
				"data":[{"dense":[0.1,0.2]}]}`,
			want: "returned 1 representations for 2 inputs",
		},
		{
			name: "missing requested kind",
			body: `{"version":"agentic.representations.v1","model":"m",
				"spaces":{"dense":{"id":"d","provider":"p","model":"m","kind":"dense","dimensions":2,"metric":"cosine"}},
				"data":[{"dense":[0.1,0.2]},{"dense":[0.3,0.4]}]}`,
			want: "no vector space for a requested output",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			encoder := newDedicated(t, respondWith(tc.body))
			_, err := encoder.Encode(context.Background(), goldenRequest())
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestDedicatedRejectsOversizedResponses(t *testing.T) {
	encoder := newDedicated(t, respondWith(strings.Repeat("x", 4096)),
		huggingface.WithMaxResponseBytes(128))
	_, err := encoder.Encode(context.Background(), goldenRequest())
	if err == nil || !strings.Contains(err.Error(), "exceeds the 128 byte limit") {
		t.Fatalf("got %v, want an oversized-response error", err)
	}
}

func TestDedicatedEmbedProjectsDense(t *testing.T) {
	var body map[string]any
	encoder := newDedicated(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, `{"version":"agentic.representations.v1","model":"BAAI/bge-m3",
			"spaces":{"dense":{"id":"d","provider":"custom","model":"BAAI/bge-m3","kind":"dense","dimensions":2,"metric":"cosine"}},
			"data":[{"dense":[0.1,0.2]}],"usage":{"input_tokens":3,"request_count":1}}`)
	})

	resp, err := encoder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"a"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 1 || len(resp.Vectors[0]) != 2 {
		t.Fatalf("vectors = %v", resp.Vectors)
	}
	outputs, _ := body["outputs"].([]any)
	if len(outputs) != 1 || outputs[0] != "dense" {
		t.Errorf("the dense projection requested %v", body["outputs"])
	}
}

func TestDedicatedEmbedFailsWhenDenseIsNotServed(t *testing.T) {
	encoder := newDedicated(t, respondWith("{}"), huggingface.WithOutputs(core.RepresentationSparse))
	_, err := encoder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"a"}})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

// A pinned space is what makes a silent redeployment detectable.
func TestDedicatedPinnedSpaceMismatchFails(t *testing.T) {
	encoder := newDedicated(t, respondWith(goldenResponse(t)),
		huggingface.WithVectorSpaces(map[core.RepresentationKind]core.VectorSpace{
			core.RepresentationDense: {
				ID: "my-index-space", Provider: "custom", Model: "BAAI/bge-m3",
				Kind: core.RepresentationDense, Dimensions: 2, Metric: core.SimilarityCosine,
			},
		}))

	_, err := encoder.Encode(context.Background(), goldenRequest())
	if err == nil || !strings.Contains(err.Error(), "configured for my-index-space") {
		t.Fatalf("got %v, want a pinned-space mismatch", err)
	}
}

// --------------------------------------------------------------------------
// Shared router
// --------------------------------------------------------------------------

func TestSharedCallsFeatureExtraction(t *testing.T) {
	var path string
	var body map[string]any

	encoder := newShared(t, func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, `[[0.1, 0.2], [0.3, 0.4]]`)
	})

	resp, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a", "b"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	want := "/hf-inference/models/intfloat/multilingual-e5-large-instruct/pipeline/feature-extraction"
	if path != want {
		t.Errorf("path = %q, want %q", path, want)
	}
	inputs, _ := body["inputs"].([]any)
	if len(inputs) != 2 {
		t.Errorf("inputs = %v", body["inputs"])
	}
	if len(resp.Data) != 2 || resp.Data[1].Dense[1] != 0.4 {
		t.Errorf("data = %v", resp.Data)
	}
	space := resp.Spaces[core.RepresentationDense]
	if space.Dimensions != 2 || space.Metric != core.SimilarityCosine {
		t.Errorf("space = %+v", space)
	}
	if resp.Usage.RequestCount != 1 || resp.Usage.OutputBytes == 0 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	// The router reports no token usage, and this package does not invent one.
	if resp.Usage.InputTokens != 0 {
		t.Errorf("input tokens = %d, want zero rather than a local estimate", resp.Usage.InputTokens)
	}
}

// The shared route returns dense vectors and nothing else, so the encoder must
// not claim otherwise however capable the model is locally.
func TestSharedAdvertisesDenseOnly(t *testing.T) {
	encoder := newShared(t, respondWith(`[[0.1, 0.2]]`))
	caps := encoder.Capabilities()

	if len(caps.Outputs) != 1 || caps.Outputs[0] != core.RepresentationDense {
		t.Fatalf("outputs = %v, want dense only", caps.Outputs)
	}
	if caps.SupportsMultiOutput {
		t.Error("a dense-only encoder cannot support multi-output")
	}
	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationSparse},
	})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

// Roles are only accepted when the model has a prompt configured for them:
// accepting one and sending nothing would let a caller believe an asymmetric
// encoding happened.
func TestSharedInputRolesRequirePromptNames(t *testing.T) {
	t.Run("without prompts", func(t *testing.T) {
		encoder := newShared(t, respondWith(`[[0.1, 0.2]]`))
		caps := encoder.Capabilities()
		if caps.SupportsInputType(core.EmbeddingInputQuery) {
			t.Error("query should not be advertised without a configured prompt")
		}
		_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
			Input:     []string{"a"},
			InputType: core.EmbeddingInputQuery,
			Outputs:   []core.RepresentationKind{core.RepresentationDense},
		})
		if !errors.Is(err, core.ErrInvalidRepresentationRequest) {
			t.Fatalf("got %v, want the unsupported role to be rejected", err)
		}
	})

	t.Run("with prompts", func(t *testing.T) {
		var body map[string]any
		encoder := newShared(t, func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&body)
			fmt.Fprint(w, `[[0.1, 0.2]]`)
		}, huggingface.WithPromptNames("query", "passage"))

		if _, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
			Input:     []string{"a"},
			InputType: core.EmbeddingInputQuery,
			Outputs:   []core.RepresentationKind{core.RepresentationDense},
		}); err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if body["prompt_name"] != "query" {
			t.Errorf("prompt_name = %v, want the configured query prompt", body["prompt_name"])
		}
	})
}

func TestSharedSendsOptions(t *testing.T) {
	var body map[string]any
	encoder := newShared(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, `[[0.1, 0.2]]`)
	}, huggingface.WithNormalize(true), huggingface.WithInferenceProvider("together"))

	truncate := false
	if _, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:    []string{"a"},
		Outputs:  []core.RepresentationKind{core.RepresentationDense},
		Truncate: &truncate,
	}); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if body["normalize"] != true {
		t.Errorf("normalize = %v", body["normalize"])
	}
	if body["truncate"] != false {
		t.Errorf("truncate = %v", body["truncate"])
	}
}

func TestSharedRejectsMalformedResponses(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"token level output", `[[[0.1, 0.2]]]`, "not an array of vectors"},
		{"error object", `{"error": "model not found"}`, "not an array of vectors"},
		{"short batch", `[[0.1, 0.2]]`, "got 1 vectors for 2 inputs"},
		{"ragged widths", `[[0.1, 0.2], [0.3]]`, "width 1, space declares 2"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			encoder := newShared(t, respondWith(tc.body))
			_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
				Input:   []string{"a", "b"},
				Outputs: []core.RepresentationKind{core.RepresentationDense},
			})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestSharedPinnedRevisionChangesSpace(t *testing.T) {
	plain := newShared(t, respondWith(`[[0.1, 0.2]]`))
	pinned := newShared(t, respondWith(`[[0.1, 0.2]]`),
		huggingface.WithSharedVectorSpace(core.VectorSpace{Revision: "rev-1", Tokenizer: "tok-1"}))

	spaceID := func(encoder *huggingface.SharedEncoder) string {
		resp, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
			Input:   []string{"a"},
			Outputs: []core.RepresentationKind{core.RepresentationDense},
		})
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		return resp.Spaces[core.RepresentationDense].ID
	}
	if spaceID(plain) == spaceID(pinned) {
		t.Fatal("recording a revision did not change the space ID")
	}
}

func TestSharedEmbedProjectsDense(t *testing.T) {
	encoder := newShared(t, respondWith(`[[0.1, 0.2], [0.3, 0.4]]`))
	resp, err := encoder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"a", "b"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 2 {
		t.Fatalf("vectors = %v", resp.Vectors)
	}
	if resp.Model != "intfloat/multilingual-e5-large-instruct" {
		t.Errorf("model = %q", resp.Model)
	}
}

func TestSharedBatchesLargeRequests(t *testing.T) {
	calls := 0
	encoder := newShared(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body struct {
			Inputs []string `json:"inputs"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		rows := make([]string, len(body.Inputs))
		for i, text := range body.Inputs {
			rows[i] = fmt.Sprintf("[%d, 0.5]", len(text))
		}
		fmt.Fprint(w, "["+strings.Join(rows, ",")+"]")
	}, huggingface.WithSharedBatchSize(2))

	resp, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a", "bb", "ccc"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if calls != 2 {
		t.Fatalf("made %d calls, want 2", calls)
	}
	if resp.Data[2].Dense[0] != 3 {
		t.Errorf("merged output is not in input order: %v", resp.Data)
	}
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

func TestAPIErrorsRedactSecretsAndInput(t *testing.T) {
	const document = "the launch code is hunter2"
	encoder := newDedicated(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, `{"error": "cannot encode", "echo": %q, "auth": "hf_test"}`, document)
	})

	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{document},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	})
	var apiErr *huggingface.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("got %v, want *huggingface.APIError", err)
	}
	if apiErr.Detail != "cannot encode" {
		t.Errorf("detail = %q", apiErr.Detail)
	}
	message := err.Error()
	if strings.Contains(message, "hunter2") || strings.Contains(message, "hf_test") {
		t.Errorf("error leaks input or credentials: %q", message)
	}
}

func TestEncodeHonorsCancellation(t *testing.T) {
	encoder := newDedicated(t, respondWith(goldenResponse(t)))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := encoder.Encode(ctx, goldenRequest())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}

// --------------------------------------------------------------------------
// Conformance
// --------------------------------------------------------------------------

func TestDedicatedConformance(t *testing.T) {
	conformance.RunRepresentation(t, conformance.RepresentationOptions{
		NewEncoder: func(t *testing.T) core.RepresentationEncoder {
			return newDedicated(t, func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Inputs  []string `json:"inputs"`
					Outputs []string `json:"outputs"`
				}
				_ = json.NewDecoder(r.Body).Decode(&body)
				writeProtocolResponse(w, body.Inputs, body.Outputs)
			})
		},
		Corpus:        []string{"alpha beta", "gamma", "delta epsilon"},
		Deterministic: true,
	})
}

func TestSharedConformance(t *testing.T) {
	conformance.RunRepresentation(t, conformance.RepresentationOptions{
		NewEncoder: func(t *testing.T) core.RepresentationEncoder {
			return newShared(t, func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Inputs []string `json:"inputs"`
				}
				_ = json.NewDecoder(r.Body).Decode(&body)
				rows := make([]string, len(body.Inputs))
				for i, text := range body.Inputs {
					rows[i] = fmt.Sprintf("[%d, 0.5]", len(text))
				}
				fmt.Fprint(w, "["+strings.Join(rows, ",")+"]")
			})
		},
		Corpus:        []string{"alpha beta", "gamma", "delta epsilon"},
		Deterministic: true,
	})
}

// writeProtocolResponse answers with agentic.representations.v1 output derived
// from the inputs, so the conformance suite's order checks have something to
// compare.
func writeProtocolResponse(w http.ResponseWriter, inputs, outputs []string) {
	wants := func(kind string) bool {
		for _, out := range outputs {
			if out == kind {
				return true
			}
		}
		return false
	}

	spaces := []string{}
	if wants("dense") {
		spaces = append(spaces, `"dense":{"id":"d","provider":"custom","model":"BAAI/bge-m3","kind":"dense","dimensions":2,"metric":"cosine"}`)
	}
	if wants("sparse") {
		spaces = append(spaces, `"sparse":{"id":"s","provider":"custom","model":"BAAI/bge-m3","kind":"sparse","dimensions":256,"metric":"dot_product"}`)
	}
	if wants("multi_vector") {
		spaces = append(spaces, `"multi_vector":{"id":"mv","provider":"custom","model":"BAAI/bge-m3","kind":"multi_vector","dimensions":2,"metric":"cosine"}`)
	}

	items := make([]string, len(inputs))
	for i, text := range inputs {
		parts := []string{}
		if wants("dense") {
			parts = append(parts, fmt.Sprintf(`"dense":[%d,0.5]`, len(text)))
		}
		if wants("sparse") {
			parts = append(parts, fmt.Sprintf(`"sparse":{"indices":[%d],"values":[0.75]}`, len(text)%256))
		}
		if wants("multi_vector") {
			parts = append(parts, fmt.Sprintf(`"multi_vector":[[%d,0.25]]`, len(text)))
		}
		items[i] = "{" + strings.Join(parts, ",") + "}"
	}

	fmt.Fprintf(w, `{"version":"agentic.representations.v1","model":"BAAI/bge-m3","spaces":{%s},"data":[%s],"usage":{"input_tokens":3,"request_count":1}}`,
		strings.Join(spaces, ","), strings.Join(items, ","))
}
