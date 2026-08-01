package representationwire_test

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/representationwire"
)

// goldenDir holds the fixtures the Python handler tests read too. Sharing them
// is what keeps the two implementations from drifting apart quietly.
const goldenDir = "../../deploy/representations/testdata"

func readGolden(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(goldenDir, name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return body
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

// The request this client sends must be the request the handler tests accept.
func TestNewRequestMatchesGoldenFixture(t *testing.T) {
	encoded, err := json.Marshal(representationwire.NewRequest(goldenRequest()))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got, want map[string]any
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatalf("unmarshal built request: %v", err)
	}
	if err := json.Unmarshal(readGolden(t, "request.json"), &want); err != nil {
		t.Fatalf("unmarshal golden request: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("built request\n  %v\ndiffers from golden\n  %v", got, want)
	}
}

// A request with no input role omits the field rather than sending an empty
// string, so a handler cannot tell "unset" from "explicitly neither".
func TestNewRequestOmitsUnsetFields(t *testing.T) {
	encoded, err := json.Marshal(representationwire.NewRequest(&core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(encoded, &body)

	if _, present := body["input_type"]; present {
		t.Error("input_type should be omitted when unset")
	}
	if _, present := body["truncate"]; present {
		t.Error("truncate should be omitted when unset")
	}
	if body["version"] != representationwire.Version {
		t.Errorf("version = %v", body["version"])
	}
}

func TestDecodeGoldenResponse(t *testing.T) {
	req := goldenRequest()
	payload := readGolden(t, "response.json")

	resp, err := representationwire.Decode(payload, req, representationwire.DecodeOptions{
		Provider:      "huggingface",
		Model:         "BAAI/bge-m3",
		ResponseBytes: len(payload),
	})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}

	if len(resp.Data) != 2 {
		t.Fatalf("got %d items", len(resp.Data))
	}
	if resp.Model != "BAAI/bge-m3" {
		t.Errorf("model = %q", resp.Model)
	}
	if resp.Data[0].Dense[0] != 0.12 || resp.Data[1].Dense[1] != 0.22 {
		t.Errorf("dense output is not in input order: %v", resp.Data)
	}
	if resp.Data[0].Sparse.Indices[0] != 1012 || resp.Data[0].Sparse.Values[0] != 0.91 {
		t.Errorf("sparse output = %+v", resp.Data[0].Sparse)
	}

	dense := resp.Spaces[core.RepresentationDense]
	if dense.ID != "configured-immutable-dense-id" {
		t.Errorf("dense space ID = %q, want the handler's configured value", dense.ID)
	}
	if dense.Revision != "immutable-revision" || dense.Tokenizer != "immutable-tokenizer-revision" {
		t.Errorf("dense space lost its revisions: %+v", dense)
	}
	sparse := resp.Spaces[core.RepresentationSparse]
	if sparse.Dimensions != 250002 || sparse.Metric != core.SimilarityDotProduct {
		t.Errorf("sparse space = %+v", sparse)
	}

	// The fixture reports output_bytes as 0, so the observed payload size
	// stands in rather than reporting a measurement nobody made.
	if resp.Usage.InputTokens != 6 || resp.Usage.RequestCount != 1 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.Usage.OutputBytes != len(payload) {
		t.Errorf("output bytes = %d, want the observed payload size", resp.Usage.OutputBytes)
	}
	if resp.Usage.InputBytes != 27 {
		t.Errorf("input bytes = %d, want the handler's reported value", resp.Usage.InputBytes)
	}

	// The decoded response must satisfy the core contract end to end.
	validator := core.RepresentationValidator{
		Provider: "huggingface",
		Capabilities: core.RepresentationCapabilities{
			Outputs:             []core.RepresentationKind{core.RepresentationDense, core.RepresentationSparse},
			InputTypes:          []core.EmbeddingInputType{core.EmbeddingInputDocument},
			SupportsTruncation:  true,
			SupportsMultiOutput: true,
		},
		Limits: core.DefaultRepresentationLimits(),
	}
	if err := validator.ValidateResponse(req, resp); err != nil {
		t.Fatalf("golden response fails core validation: %v", err)
	}
}

func TestDecodeFillsMissingMeasurements(t *testing.T) {
	payload := []byte(`{
		"version": "agentic.representations.v1",
		"model": "",
		"spaces": {"dense": {"id": "d", "provider": "custom", "model": "m", "kind": "dense", "dimensions": 2, "metric": "cosine"}},
		"data": [{"dense": [0.1, 0.2]}]
	}`)
	req := &core.RepresentationRequest{
		Input:   []string{"a document"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}

	resp, err := representationwire.Decode(payload, req, representationwire.DecodeOptions{
		Provider:      "sagemaker",
		Model:         "configured-model",
		ResponseBytes: len(payload),
	})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if resp.Model != "configured-model" {
		t.Errorf("model = %q, want the configured name when the handler reports none", resp.Model)
	}
	// One invocation happened even if the handler forgot to say so.
	if resp.Usage.RequestCount != 1 {
		t.Errorf("request count = %d, want 1", resp.Usage.RequestCount)
	}
	if resp.Usage.InputBytes != len("a document") {
		t.Errorf("input bytes = %d, want the observed size", resp.Usage.InputBytes)
	}
}

func TestDecodeCompletesPartialSpaceDescriptors(t *testing.T) {
	payload := []byte(`{
		"version": "agentic.representations.v1",
		"model": "m",
		"spaces": {"dense": {"dimensions": 2, "metric": "cosine"}},
		"data": [{"dense": [0.1, 0.2]}]
	}`)
	req := &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}

	resp, err := representationwire.Decode(payload, req, representationwire.DecodeOptions{
		Provider: "sagemaker",
		Model:    "configured-model",
	})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	space := resp.Spaces[core.RepresentationDense]
	if space.Provider != "sagemaker" || space.Model != "configured-model" {
		t.Errorf("space was not completed from configuration: %+v", space)
	}
	if space.Kind != core.RepresentationDense {
		t.Errorf("space kind = %q, want it inferred from the map key", space.Kind)
	}
	if space.ID != space.CanonicalID() {
		t.Error("a handler that reports no ID should get the canonical one")
	}
}

func TestDecodeReconcilesPinnedSpaces(t *testing.T) {
	pinned := core.VectorSpace{
		ID:         "pinned-dense",
		Provider:   "sagemaker",
		Model:      "BAAI/bge-m3",
		Revision:   "rev-1",
		Tokenizer:  "tok-1",
		Kind:       core.RepresentationDense,
		Dimensions: 2,
		Metric:     core.SimilarityCosine,
	}
	req := &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
	opts := representationwire.DecodeOptions{
		Provider: "sagemaker",
		Model:    "BAAI/bge-m3",
		Expected: map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: pinned},
	}

	t.Run("handler describes nothing", func(t *testing.T) {
		payload := []byte(`{"version":"agentic.representations.v1","model":"m","data":[{"dense":[0.1,0.2]}]}`)
		resp, err := representationwire.Decode(payload, req, opts)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if resp.Spaces[core.RepresentationDense].ID != "pinned-dense" {
			t.Error("an undescribed space should be filled from the pin")
		}
	})

	t.Run("handler agrees", func(t *testing.T) {
		payload := []byte(`{
			"version":"agentic.representations.v1","model":"m",
			"spaces":{"dense":{"id":"pinned-dense","provider":"sagemaker","model":"BAAI/bge-m3","revision":"rev-1","tokenizer":"tok-1","kind":"dense","dimensions":2,"metric":"cosine"}},
			"data":[{"dense":[0.1,0.2]}]
		}`)
		if _, err := representationwire.Decode(payload, req, opts); err != nil {
			t.Fatalf("Decode: %v", err)
		}
	})

	// A redeployment onto a different revision must fail loudly rather than
	// mixing two generations of vectors in one index.
	t.Run("handler disagrees", func(t *testing.T) {
		payload := []byte(`{
			"version":"agentic.representations.v1","model":"m",
			"spaces":{"dense":{"id":"pinned-dense","provider":"sagemaker","model":"BAAI/bge-m3","revision":"rev-2","tokenizer":"tok-1","kind":"dense","dimensions":2,"metric":"cosine"}},
			"data":[{"dense":[0.1,0.2]}]
		}`)
		_, err := representationwire.Decode(payload, req, opts)
		if err == nil || !strings.Contains(err.Error(), "configured for") {
			t.Fatalf("got %v, want a pinned-space mismatch", err)
		}
	})

	t.Run("handler reports a different id", func(t *testing.T) {
		payload := []byte(`{
			"version":"agentic.representations.v1","model":"m",
			"spaces":{"dense":{"id":"other","provider":"sagemaker","model":"BAAI/bge-m3","revision":"rev-1","tokenizer":"tok-1","kind":"dense","dimensions":2,"metric":"cosine"}},
			"data":[{"dense":[0.1,0.2]}]
		}`)
		_, err := representationwire.Decode(payload, req, opts)
		if !errors.Is(err, core.ErrInvalidRepresentationResponse) {
			t.Fatalf("got %v, want a pinned-space mismatch", err)
		}
	})
}

// A pinned space for a kind that was not requested must not appear in the
// response, or validation would reject a response the handler got right.
func TestDecodeIgnoresPinsForUnrequestedKinds(t *testing.T) {
	req := &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
	payload := []byte(`{"version":"agentic.representations.v1","model":"m","data":[{"dense":[0.1,0.2]}]}`)

	resp, err := representationwire.Decode(payload, req, representationwire.DecodeOptions{
		Provider: "sagemaker",
		Expected: map[core.RepresentationKind]core.VectorSpace{
			core.RepresentationDense: {
				ID: "d", Provider: "sagemaker", Model: "m",
				Kind: core.RepresentationDense, Dimensions: 2, Metric: core.SimilarityCosine,
			},
			core.RepresentationSparse: {
				ID: "s", Provider: "sagemaker", Model: "m",
				Kind: core.RepresentationSparse, Dimensions: 100, Metric: core.SimilarityDotProduct,
			},
		},
	})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if _, present := resp.Spaces[core.RepresentationSparse]; present {
		t.Error("an unrequested pinned space should not be reported")
	}
}

func TestDecodeVersionHandling(t *testing.T) {
	req := &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
	body := func(version string) []byte {
		return []byte(`{"version":"` + version + `","model":"m","data":[{"dense":[0.1]}]}`)
	}

	t.Run("additive minor is accepted", func(t *testing.T) {
		if _, err := representationwire.Decode(body("agentic.representations.v1.7"), req,
			representationwire.DecodeOptions{Provider: "p"}); err != nil {
			t.Fatalf("Decode: %v", err)
		}
	})

	tests := []struct {
		name    string
		version string
		want    string
	}{
		{"missing", "", "does not declare a protocol version"},
		{"foreign", "openai.embeddings.v1", "want agentic.representations.v1"},
		{"unparsable major", "agentic.representations.vX", "want agentic.representations.v1"},
		{"future major", "agentic.representations.v2", "major version 2"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := representationwire.Decode(body(tc.version), req,
				representationwire.DecodeOptions{Provider: "p"})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
			if !errors.Is(err, core.ErrInvalidRepresentationResponse) {
				t.Error("error does not match ErrInvalidRepresentationResponse")
			}
		})
	}
}

// Additive fields must not break an older client; that is the whole point of
// pinning only the major version.
func TestDecodeIgnoresUnknownFields(t *testing.T) {
	payload := []byte(`{
		"version": "agentic.representations.v1",
		"model": "m",
		"future_top_level": {"anything": true},
		"spaces": {"dense": {"id":"d","provider":"p","model":"m","kind":"dense","dimensions":2,"metric":"cosine","future":1}},
		"data": [{"dense": [0.1, 0.2], "future_kind": [1, 2]}],
		"usage": {"input_tokens": 3, "future_measure": 9}
	}`)
	req := &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
	resp, err := representationwire.Decode(payload, req, representationwire.DecodeOptions{Provider: "p"})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if resp.Usage.InputTokens != 3 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestDecodeRejectsMalformedPayload(t *testing.T) {
	req := &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
	_, err := representationwire.Decode([]byte(`{"version":`), req,
		representationwire.DecodeOptions{Provider: "p"})
	if err == nil || !strings.Contains(err.Error(), "not valid agentic.representations.v1 JSON") {
		t.Fatalf("got %v, want a decode error", err)
	}
}

// A malformed payload may contain the caller's documents, so the error must
// not echo it.
func TestDecodeErrorsDoNotEchoPayload(t *testing.T) {
	req := &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
	payload := []byte(`{"version": "agentic.representations.v1", "echo": "the launch code is hunter2"`)

	_, err := representationwire.Decode(payload, req, representationwire.DecodeOptions{Provider: "p"})
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), "hunter2") {
		t.Fatalf("error echoes the payload: %q", err.Error())
	}
}

func TestCheckVersion(t *testing.T) {
	if err := representationwire.CheckVersion(representationwire.Version, "p"); err != nil {
		t.Fatalf("current version rejected: %v", err)
	}
	if err := representationwire.CheckVersion("agentic.representations.v1.0.3", "p"); err != nil {
		t.Fatalf("additive suffix rejected: %v", err)
	}
	if err := representationwire.CheckVersion("agentic.representations.v", "p"); err == nil {
		t.Fatal("a version with no major number should be rejected")
	}
}

func TestDecodeMapsSparseAndMultiVector(t *testing.T) {
	payload := []byte(`{
		"version": "agentic.representations.v1",
		"model": "m",
		"spaces": {
			"sparse": {"id":"s","provider":"p","model":"m","kind":"sparse","dimensions":100,"metric":"dot_product"},
			"multi_vector": {"id":"mv","provider":"p","model":"m","kind":"multi_vector","dimensions":2,"metric":"cosine"}
		},
		"data": [{"sparse": {"indices": [3, 9], "values": [0.5, 0.25]}, "multi_vector": [[0.1, 0.2],[0.3,0.4]]}]
	}`)
	req := &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationSparse, core.RepresentationMultiVector},
	}

	resp, err := representationwire.Decode(payload, req, representationwire.DecodeOptions{Provider: "p"})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if resp.Data[0].Sparse.Indices[1] != 9 || resp.Data[0].Sparse.Values[1] != 0.25 {
		t.Errorf("sparse = %+v", resp.Data[0].Sparse)
	}
	if len(resp.Data[0].MultiVector) != 2 || resp.Data[0].MultiVector[1][1] != 0.4 {
		t.Errorf("multi-vector = %v", resp.Data[0].MultiVector)
	}
}
