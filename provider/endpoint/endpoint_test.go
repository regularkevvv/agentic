package endpoint_test

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
	"sync/atomic"
	"testing"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/provider/endpoint"
	"github.com/regularkevvv/agentic/provider/test/conformance"
)

const goldenResponsePath = "../../internal/retrieval/wire/testdata/response.json"

func goldenResponse(t *testing.T) string {
	t.Helper()
	body, err := os.ReadFile(filepath.Clean(goldenResponsePath))
	if err != nil {
		t.Fatalf("read golden response: %v", err)
	}
	return string(body)
}

func newEncoder(t *testing.T, handler http.HandlerFunc, opts ...endpoint.Option) *endpoint.Encoder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]endpoint.Option{
		endpoint.WithToken("endpoint_test"),
		endpoint.WithModel("BAAI/bge-m3"),
	}, opts...)

	encoder, err := endpoint.New(server.URL, opts...)
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

func goldenRequest() *retrieval.RepresentationRequest {
	truncate := true
	return &retrieval.RepresentationRequest{
		Input:     []string{"a document", "another document"},
		InputType: retrieval.EmbeddingInputDocument,
		Outputs:   []retrieval.RepresentationKind{retrieval.RepresentationDense, retrieval.RepresentationSparse},
		Truncate:  &truncate,
	}
}

// --------------------------------------------------------------------------
// Construction
// --------------------------------------------------------------------------

func TestNewValidatesConfiguration(t *testing.T) {
	tests := []struct {
		name string
		url  string
		opts []endpoint.Option
		want string
	}{
		{"empty url", "", nil, "endpoint URL cannot be empty"},
		{"relative url", "example.com", nil, "must be absolute"},
		{"negative retries", "https://e", []endpoint.Option{endpoint.WithMaxRetries(-1)}, "max retries"},
		{"zero response limit", "https://e", []endpoint.Option{endpoint.WithMaxResponseBytes(0)}, "max response bytes"},
		{"negative batch size", "https://e", []endpoint.Option{endpoint.WithBatchSize(-1)}, "batch size"},
		{"unknown output", "https://e", []endpoint.Option{endpoint.WithOutputs("colbert")}, "not dense, sparse, or multi_vector"},
		{"invalid pinned space", "https://e", []endpoint.Option{
			endpoint.WithVectorSpaces(map[retrieval.RepresentationKind]retrieval.VectorSpace{
				retrieval.RepresentationDense: {Provider: "p"},
			}),
		}, "pinned dense space"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := append([]endpoint.Option{endpoint.WithToken("t")}, tc.opts...)
			_, err := endpoint.New(tc.url, opts...)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

// The credential matrix is the contract this package is most likely to be
// misconfigured against, so every cell of it is asserted rather than the
// working one alone.
func TestTokenResolution(t *testing.T) {
	t.Run("no token anywhere", func(t *testing.T) {
		t.Setenv("AGENTIC_ENDPOINT_TOKEN", "")
		if _, err := endpoint.New("https://e"); err == nil ||
			!strings.Contains(err.Error(), "token not set") {
			t.Fatalf("got %v, want the missing-token error", err)
		}
	})

	t.Run("empty token is not an anonymous request", func(t *testing.T) {
		t.Setenv("AGENTIC_ENDPOINT_TOKEN", "from-the-environment")
		_, err := endpoint.New("https://e", endpoint.WithToken(""))
		if err == nil || !strings.Contains(err.Error(), "empty token") {
			t.Fatalf("got %v, want an empty token to be refused rather than "+
				"replaced by the environment's", err)
		}
	})

	t.Run("mutually exclusive with WithoutAuthentication", func(t *testing.T) {
		for _, token := range []string{"t", ""} {
			_, err := endpoint.New("https://e",
				endpoint.WithToken(token), endpoint.WithoutAuthentication())
			if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
				t.Fatalf("WithToken(%q): got %v, want the exclusion error", token, err)
			}
		}
	})

	t.Run("environment fallback", func(t *testing.T) {
		t.Setenv("AGENTIC_ENDPOINT_TOKEN", "from-the-environment")
		var authorization string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			authorization = r.Header.Get("Authorization")
			fmt.Fprint(w, goldenResponse(t))
		}))
		t.Cleanup(server.Close)

		encoder, err := endpoint.New(server.URL, endpoint.WithModel("BAAI/bge-m3"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := encoder.Encode(context.Background(), goldenRequest()); err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if authorization != "Bearer from-the-environment" {
			t.Errorf("Authorization = %q", authorization)
		}
	})

	// A vendor's variable must not reach a client that talks to any host.
	t.Run("no vendor variable is read", func(t *testing.T) {
		t.Setenv("AGENTIC_ENDPOINT_TOKEN", "")
		t.Setenv("HF_TOKEN", "hf_from_the_environment")
		if _, err := endpoint.New("https://e"); err == nil {
			t.Fatal("HF_TOKEN satisfied a generic endpoint client")
		}
	})

	t.Run("without authentication", func(t *testing.T) {
		t.Setenv("AGENTIC_ENDPOINT_TOKEN", "from-the-environment")
		var present bool
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, present = r.Header["Authorization"]
			fmt.Fprint(w, goldenResponse(t))
		}))
		t.Cleanup(server.Close)

		encoder, err := endpoint.New(server.URL, endpoint.WithoutAuthentication())
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := encoder.Encode(context.Background(), goldenRequest()); err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if present {
			t.Error("an endpoint asked for no authentication was sent a credential")
		}
	})
}

// --------------------------------------------------------------------------
// Encoding
// --------------------------------------------------------------------------

func TestSpeaksTheProtocol(t *testing.T) {
	var body map[string]any
	var contentType string

	encoder := newEncoder(t, func(w http.ResponseWriter, r *http.Request) {
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
	if resp.Spaces[retrieval.RepresentationDense].ID != "configured-immutable-dense-id" {
		t.Errorf("dense space = %+v", resp.Spaces[retrieval.RepresentationDense])
	}
	if resp.Model != "BAAI/bge-m3" {
		t.Errorf("model = %q", resp.Model)
	}
}

func TestNameFallsBackToEndpoint(t *testing.T) {
	encoder, err := endpoint.New("https://abc.endpoints.example.com/", endpoint.WithToken("t"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if encoder.Name() != "https://abc.endpoints.example.com" {
		t.Errorf("Name() = %q, want the trimmed endpoint", encoder.Name())
	}
}

func TestRejectsUnsupportedOutputs(t *testing.T) {
	encoder := newEncoder(t, respondWith(goldenResponse(t)),
		endpoint.WithOutputs(retrieval.RepresentationDense))

	_, err := encoder.Encode(context.Background(), &retrieval.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []retrieval.RepresentationKind{retrieval.RepresentationMultiVector},
	})
	if !errors.Is(err, retrieval.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

func TestBatchesLargeRequests(t *testing.T) {
	var batches [][]string
	encoder := newEncoder(t, func(w http.ResponseWriter, r *http.Request) {
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
	}, endpoint.WithBatchSize(2))

	resp, err := encoder.Encode(context.Background(), &retrieval.RepresentationRequest{
		Input:   []string{"a", "bb", "ccc", "dddd", "eeeee"},
		Outputs: []retrieval.RepresentationKind{retrieval.RepresentationDense},
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

func TestRejectsMalformedResponses(t *testing.T) {
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
			encoder := newEncoder(t, respondWith(tc.body))
			_, err := encoder.Encode(context.Background(), goldenRequest())
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestRejectsOversizedResponses(t *testing.T) {
	encoder := newEncoder(t, respondWith(strings.Repeat("x", 4096)),
		endpoint.WithMaxResponseBytes(128))
	_, err := encoder.Encode(context.Background(), goldenRequest())
	if err == nil || !strings.Contains(err.Error(), "exceeds the 128 byte limit") {
		t.Fatalf("got %v, want an oversized-response error", err)
	}
}

func TestEmbedProjectsDense(t *testing.T) {
	var body map[string]any
	encoder := newEncoder(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, `{"version":"agentic.representations.v1","model":"BAAI/bge-m3",
			"spaces":{"dense":{"id":"d","provider":"custom","model":"BAAI/bge-m3","kind":"dense","dimensions":2,"metric":"cosine"}},
			"data":[{"dense":[0.1,0.2]}],"usage":{"input_tokens":3,"request_count":1}}`)
	})

	resp, err := encoder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a"}})
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

func TestEmbedFailsWhenDenseIsNotServed(t *testing.T) {
	encoder := newEncoder(t, respondWith("{}"), endpoint.WithOutputs(retrieval.RepresentationSparse))
	_, err := encoder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"a"}})
	if !errors.Is(err, retrieval.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

// A pinned space is what makes a silent redeployment detectable.
func TestPinnedSpaceMismatchFails(t *testing.T) {
	encoder := newEncoder(t, respondWith(goldenResponse(t)),
		endpoint.WithVectorSpaces(map[retrieval.RepresentationKind]retrieval.VectorSpace{
			retrieval.RepresentationDense: {
				ID: "my-index-space", Provider: "custom", Model: "BAAI/bge-m3",
				Kind: retrieval.RepresentationDense, Dimensions: 2, Metric: retrieval.SimilarityCosine,
			},
		}))

	_, err := encoder.Encode(context.Background(), goldenRequest())
	if err == nil || !strings.Contains(err.Error(), "configured for my-index-space") {
		t.Fatalf("got %v, want a pinned-space mismatch", err)
	}
}

func TestConfiguredLimitsAreEnforced(t *testing.T) {
	encoder, err := endpoint.New("https://e", endpoint.WithToken("t"),
		endpoint.WithLimits(retrieval.RepresentationLimits{MaxInputs: 1}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = encoder.Encode(context.Background(), &retrieval.RepresentationRequest{
		Input:   []string{"a", "b"},
		Outputs: []retrieval.RepresentationKind{retrieval.RepresentationDense},
	})
	if !errors.Is(err, retrieval.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want the configured limit to be enforced", err)
	}
}

func TestInjectedHTTPClientIsUsed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, goldenResponse(t))
	}))
	t.Cleanup(server.Close)

	var used atomic.Bool
	client := &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		used.Store(true)
		return http.DefaultTransport.RoundTrip(r)
	})}

	encoder, err := endpoint.New(server.URL, endpoint.WithToken("t"),
		endpoint.WithHTTPClient(client), endpoint.WithMaxRetries(0))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := encoder.Encode(context.Background(), goldenRequest()); err != nil {
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
	encoder := newEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, `{"error": "cannot encode", "echo": %q, "auth": "endpoint_test"}`, document)
	})

	_, err := encoder.Encode(context.Background(), &retrieval.RepresentationRequest{
		Input:   []string{document},
		Outputs: []retrieval.RepresentationKind{retrieval.RepresentationDense},
	})
	var apiErr *endpoint.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("got %v, want *endpoint.APIError", err)
	}
	if apiErr.Provider != "endpoint" || apiErr.Detail != "cannot encode" {
		t.Errorf("error = %+v", apiErr)
	}
	message := err.Error()
	if strings.Contains(message, "hunter2") || strings.Contains(message, "endpoint_test") {
		t.Errorf("error leaks input or credentials: %q", message)
	}
}

func TestEncodeHonorsCancellation(t *testing.T) {
	encoder := newEncoder(t, respondWith(goldenResponse(t)))
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

func TestConformance(t *testing.T) {
	conformance.RunRepresentation(t, conformance.RepresentationOptions{
		NewEncoder: func(t *testing.T) retrieval.RepresentationEncoder {
			return newEncoder(t, func(w http.ResponseWriter, r *http.Request) {
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

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
