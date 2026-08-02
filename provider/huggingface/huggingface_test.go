package huggingface_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/provider/huggingface"
	"github.com/regularkevvv/agentic/provider/test/conformance"
)

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

func denseRequest(inputs ...string) *retrieval.RepresentationRequest {
	return &retrieval.RepresentationRequest{
		Input:   inputs,
		Outputs: []retrieval.RepresentationKind{retrieval.RepresentationDense},
	}
}

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

func TestNewRequiresToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "")

	if _, err := huggingface.NewShared("model"); err == nil {
		t.Error("NewShared should require a token")
	}
}

// Both of Hugging Face's own variable names are honored, in the order the
// hub's tooling settled on.
func TestNewReadsTokenFromEnvironment(t *testing.T) {
	tests := []struct {
		name    string
		hfToken string
		hubName string
		want    string
	}{
		{"HF_TOKEN", "hf-token", "hub-token", "Bearer hf-token"},
		{"HUGGING_FACE_HUB_TOKEN", "", "hub-token", "Bearer hub-token"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("HF_TOKEN", tc.hfToken)
			t.Setenv("HUGGING_FACE_HUB_TOKEN", tc.hubName)

			var authorization string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				authorization = r.Header.Get("Authorization")
				fmt.Fprint(w, `[[0.1, 0.2]]`)
			}))
			t.Cleanup(server.Close)

			encoder, err := huggingface.NewShared("m", huggingface.WithRouterURL(server.URL))
			if err != nil {
				t.Fatalf("NewShared: %v", err)
			}
			if _, err := encoder.Encode(context.Background(), denseRequest("a")); err != nil {
				t.Fatalf("Encode: %v", err)
			}
			if authorization != tc.want {
				t.Errorf("Authorization = %q, want %q", authorization, tc.want)
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

	resp, err := encoder.Encode(context.Background(), denseRequest("a", "b"))
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
	space := resp.Spaces[retrieval.RepresentationDense]
	if space.Dimensions != 2 || space.Metric != retrieval.SimilarityCosine {
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

	if len(caps.Outputs) != 1 || caps.Outputs[0] != retrieval.RepresentationDense {
		t.Fatalf("outputs = %v, want dense only", caps.Outputs)
	}
	if caps.SupportsMultiOutput {
		t.Error("a dense-only encoder cannot support multi-output")
	}
	_, err := encoder.Encode(context.Background(), &retrieval.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []retrieval.RepresentationKind{retrieval.RepresentationSparse},
	})
	if !errors.Is(err, retrieval.ErrUnsupportedRepresentation) {
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
		if caps.SupportsInputType(retrieval.EmbeddingInputQuery) {
			t.Error("query should not be advertised without a configured prompt")
		}
		_, err := encoder.Encode(context.Background(), &retrieval.RepresentationRequest{
			Input:     []string{"a"},
			InputType: retrieval.EmbeddingInputQuery,
			Outputs:   []retrieval.RepresentationKind{retrieval.RepresentationDense},
		})
		if !errors.Is(err, retrieval.ErrInvalidRepresentationRequest) {
			t.Fatalf("got %v, want the unsupported role to be rejected", err)
		}
	})

	t.Run("with prompts", func(t *testing.T) {
		var body map[string]any
		encoder := newShared(t, func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&body)
			fmt.Fprint(w, `[[0.1, 0.2]]`)
		}, huggingface.WithPromptNames("query", "passage"))

		if _, err := encoder.Encode(context.Background(), &retrieval.RepresentationRequest{
			Input:     []string{"a"},
			InputType: retrieval.EmbeddingInputQuery,
			Outputs:   []retrieval.RepresentationKind{retrieval.RepresentationDense},
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
	if _, err := encoder.Encode(context.Background(), &retrieval.RepresentationRequest{
		Input:    []string{"a"},
		Outputs:  []retrieval.RepresentationKind{retrieval.RepresentationDense},
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
			_, err := encoder.Encode(context.Background(), denseRequest("a", "b"))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestSharedRejectsOversizedResponses(t *testing.T) {
	encoder := newShared(t, respondWith(strings.Repeat("x", 4096)),
		huggingface.WithSharedMaxResponseBytes(128))
	_, err := encoder.Encode(context.Background(), denseRequest("a"))
	if err == nil || !strings.Contains(err.Error(), "exceeds the 128 byte limit") {
		t.Fatalf("got %v, want an oversized-response error", err)
	}
}

func TestSharedPinnedRevisionChangesSpace(t *testing.T) {
	plain := newShared(t, respondWith(`[[0.1, 0.2]]`))
	pinned := newShared(t, respondWith(`[[0.1, 0.2]]`),
		huggingface.WithSharedVectorSpace(retrieval.VectorSpace{Revision: "rev-1", Tokenizer: "tok-1"}))

	spaceID := func(encoder *huggingface.SharedEncoder) string {
		resp, err := encoder.Encode(context.Background(), denseRequest("a"))
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		return resp.Spaces[retrieval.RepresentationDense].ID
	}
	if spaceID(plain) == spaceID(pinned) {
		t.Fatal("recording a revision did not change the space ID")
	}
}

func TestSharedEmbedProjectsDense(t *testing.T) {
	encoder := newShared(t, respondWith(`[[0.1, 0.2], [0.3, 0.4]]`))
	resp, err := encoder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a", "b"}})
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

	resp, err := encoder.Encode(context.Background(), denseRequest("a", "bb", "ccc"))
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

func TestSharedConfiguredLimitsAreEnforced(t *testing.T) {
	encoder, err := huggingface.NewShared("m", huggingface.WithSharedToken("t"),
		huggingface.WithSharedLimits(retrieval.RepresentationLimits{MaxInputs: 1}))
	if err != nil {
		t.Fatalf("NewShared: %v", err)
	}
	if _, err := encoder.Encode(context.Background(), denseRequest("a", "b")); !errors.Is(err, retrieval.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want the configured limit to be enforced", err)
	}
}

func TestSharedInjectedHTTPClientIsUsed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `[[0.1, 0.2]]`)
	}))
	t.Cleanup(server.Close)

	var used atomic.Bool
	client := &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		used.Store(true)
		return http.DefaultTransport.RoundTrip(r)
	})}

	encoder, err := huggingface.NewShared("m", huggingface.WithSharedToken("t"),
		huggingface.WithRouterURL(server.URL), huggingface.WithSharedHTTPClient(client))
	if err != nil {
		t.Fatalf("NewShared: %v", err)
	}
	if _, err := encoder.Encode(context.Background(), denseRequest("a")); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !used.Load() {
		t.Error("the injected client was not used")
	}
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

func TestAPIErrorsRedactSecretsAndInput(t *testing.T) {
	const document = "the launch code is hunter2"
	encoder := newShared(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, `{"error": "cannot encode", "echo": %q, "auth": "hf_test"}`, document)
	})

	_, err := encoder.Encode(context.Background(), denseRequest(document))
	var apiErr *huggingface.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("got %v, want *huggingface.APIError", err)
	}
	if apiErr.Provider != "huggingface" || apiErr.Detail != "cannot encode" {
		t.Errorf("error = %+v", apiErr)
	}
	message := err.Error()
	if strings.Contains(message, "hunter2") || strings.Contains(message, "hf_test") {
		t.Errorf("error leaks input or credentials: %q", message)
	}
}

func TestEncodeHonorsCancellation(t *testing.T) {
	encoder := newShared(t, respondWith(`[[0.1, 0.2]]`))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := encoder.Encode(ctx, denseRequest("a"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}

// --------------------------------------------------------------------------
// Conformance
// --------------------------------------------------------------------------

func TestSharedConformance(t *testing.T) {
	conformance.RunRepresentation(t, conformance.RepresentationOptions{
		NewEncoder: func(t *testing.T) retrieval.RepresentationEncoder {
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

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
