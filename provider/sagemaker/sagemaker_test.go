package sagemaker_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/sagemaker"
	"github.com/regularkevvv/agentic/provider/test/conformance"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/sagemakerruntime"
	"github.com/aws/smithy-go"
)

const goldenResponsePath = "../../deploy/representations/testdata/response.json"

// stubRuntime records invocations and answers them from a scripted function.
// SageMaker has no local emulator, so this is how the transport is driven
// deterministically without an AWS account or a signed request.
type stubRuntime struct {
	mu     sync.Mutex
	inputs []sagemakerruntime.InvokeEndpointInput

	respond func(call int, in *sagemakerruntime.InvokeEndpointInput) (*sagemakerruntime.InvokeEndpointOutput, error)
}

func (s *stubRuntime) InvokeEndpoint(
	_ context.Context,
	params *sagemakerruntime.InvokeEndpointInput,
	_ ...func(*sagemakerruntime.Options),
) (*sagemakerruntime.InvokeEndpointOutput, error) {
	s.mu.Lock()
	call := len(s.inputs)
	s.inputs = append(s.inputs, *params)
	s.mu.Unlock()
	return s.respond(call, params)
}

func (s *stubRuntime) calls() []sagemakerruntime.InvokeEndpointInput {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]sagemakerruntime.InvokeEndpointInput(nil), s.inputs...)
}

func respondWith(body string) *stubRuntime {
	return &stubRuntime{
		respond: func(int, *sagemakerruntime.InvokeEndpointInput) (*sagemakerruntime.InvokeEndpointOutput, error) {
			return &sagemakerruntime.InvokeEndpointOutput{Body: []byte(body)}, nil
		},
	}
}

func goldenResponse(t *testing.T) string {
	t.Helper()
	body, err := os.ReadFile(filepath.Clean(goldenResponsePath))
	if err != nil {
		t.Fatalf("read golden response: %v", err)
	}
	return string(body)
}

func newEncoder(t *testing.T, runtime sagemaker.InvokeAPI, opts ...sagemaker.Option) *sagemaker.Encoder {
	t.Helper()
	opts = append([]sagemaker.Option{
		sagemaker.WithClient(runtime),
		sagemaker.WithModel("BAAI/bge-m3"),
	}, opts...)

	encoder, err := sagemaker.New(context.Background(), "bge-m3-endpoint", opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return encoder
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

func denseRequest(inputs ...string) *core.RepresentationRequest {
	return &core.RepresentationRequest{
		Input:   inputs,
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
}

// --------------------------------------------------------------------------
// Construction
// --------------------------------------------------------------------------

func TestNewValidatesConfiguration(t *testing.T) {
	runtime := respondWith("{}")
	tests := []struct {
		name     string
		endpoint string
		opts     []sagemaker.Option
		want     string
	}{
		{"empty endpoint", "", nil, "endpoint name cannot be empty"},
		{"negative batch size", "e", []sagemaker.Option{sagemaker.WithBatchSize(-1)}, "batch size"},
		{"zero response limit", "e", []sagemaker.Option{sagemaker.WithMaxResponseBytes(0)}, "max response bytes"},
		{"unknown output", "e", []sagemaker.Option{sagemaker.WithOutputs("colbert")}, "not dense, sparse, or multi_vector"},
		{"invalid pinned space", "e", []sagemaker.Option{
			sagemaker.WithVectorSpaces(map[core.RepresentationKind]core.VectorSpace{
				core.RepresentationDense: {Provider: "p"},
			}),
		}, "pinned dense space"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := append([]sagemaker.Option{sagemaker.WithClient(runtime)}, tc.opts...)
			_, err := sagemaker.New(context.Background(), tc.endpoint, opts...)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

// Without an injected client the provider resolves AWS config, which needs a
// region before it needs credentials.
func TestNewRequiresRegionWithoutAnInjectedClient(t *testing.T) {
	t.Setenv("AWS_DEFAULT_REGION", "")
	t.Setenv("AWS_REGION", "")

	_, err := sagemaker.New(context.Background(), "endpoint")
	if err == nil || !strings.Contains(err.Error(), "region not set") {
		t.Fatalf("got %v, want a region error", err)
	}
}

func TestNewReadsRegionFromEnvironment(t *testing.T) {
	t.Setenv("AWS_DEFAULT_REGION", "us-east-1")
	if _, err := sagemaker.New(context.Background(), "endpoint"); err != nil {
		t.Fatalf("New: %v", err)
	}
}

func TestNewAcceptsAnExistingAWSConfig(t *testing.T) {
	t.Setenv("AWS_DEFAULT_REGION", "")
	t.Setenv("AWS_REGION", "")

	encoder, err := sagemaker.New(context.Background(), "endpoint",
		sagemaker.WithAWSConfig(aws.Config{Region: "eu-west-1"}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if encoder.Name() != "endpoint" {
		t.Errorf("Name() = %q, want the endpoint when no model is configured", encoder.Name())
	}
}

func TestNewAcceptsRegionProfileAndCredentials(t *testing.T) {
	encoder, err := sagemaker.New(context.Background(), "endpoint",
		sagemaker.WithRegion("us-west-2"),
		sagemaker.WithProfile("default"),
		sagemaker.WithCredentials("AKIAEXAMPLE", "secret", "session"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if encoder.Name() != "endpoint" {
		t.Errorf("Name() = %q", encoder.Name())
	}
}

func TestMustNew(t *testing.T) {
	encoder := sagemaker.MustNew(context.Background(), "endpoint",
		sagemaker.WithClient(respondWith("{}")), sagemaker.WithModel("m"))
	if encoder.Name() != "m" {
		t.Errorf("Name() = %q", encoder.Name())
	}

	defer func() {
		if recover() == nil {
			t.Error("MustNew should panic on an invalid configuration")
		}
	}()
	sagemaker.MustNew(context.Background(), "")
}

// --------------------------------------------------------------------------
// Invocation
// --------------------------------------------------------------------------

func TestEncodeSendsTheProtocol(t *testing.T) {
	runtime := respondWith(goldenResponse(t))
	encoder := newEncoder(t, runtime)

	resp, err := encoder.Encode(context.Background(), goldenRequest())
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	calls := runtime.calls()
	if len(calls) != 1 {
		t.Fatalf("made %d invocations, want 1", len(calls))
	}
	call := calls[0]
	if aws.ToString(call.EndpointName) != "bge-m3-endpoint" {
		t.Errorf("endpoint = %q", aws.ToString(call.EndpointName))
	}
	if aws.ToString(call.ContentType) != "application/json" || aws.ToString(call.Accept) != "application/json" {
		t.Errorf("content negotiation = %q / %q",
			aws.ToString(call.ContentType), aws.ToString(call.Accept))
	}

	var body map[string]any
	if err := json.Unmarshal(call.Body, &body); err != nil {
		t.Fatalf("request body is not JSON: %v", err)
	}
	if body["version"] != "agentic.representations.v1" {
		t.Errorf("version = %v", body["version"])
	}
	if body["input_type"] != "document" || body["truncate"] != true {
		t.Errorf("body = %v", body)
	}

	if len(resp.Data) != 2 || resp.Data[1].Sparse.Indices[0] != 914 {
		t.Errorf("decoded data = %+v", resp.Data)
	}
	if resp.Spaces[core.RepresentationSparse].Dimensions != 250002 {
		t.Errorf("sparse space = %+v", resp.Spaces[core.RepresentationSparse])
	}
}

func TestEncodeSendsTargetingOptions(t *testing.T) {
	runtime := respondWith(goldenResponse(t))
	encoder := newEncoder(t, runtime,
		sagemaker.WithInferenceComponent("component-1"),
		sagemaker.WithTargetVariant("variant-b"),
		sagemaker.WithTargetModel("model.tar.gz"),
	)

	if _, err := encoder.Encode(context.Background(), goldenRequest()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	call := runtime.calls()[0]
	if aws.ToString(call.InferenceComponentName) != "component-1" {
		t.Errorf("inference component = %q", aws.ToString(call.InferenceComponentName))
	}
	if aws.ToString(call.TargetVariant) != "variant-b" {
		t.Errorf("target variant = %q", aws.ToString(call.TargetVariant))
	}
	if aws.ToString(call.TargetModel) != "model.tar.gz" {
		t.Errorf("target model = %q", aws.ToString(call.TargetModel))
	}
}

func TestEncodeOmitsUnsetTargetingOptions(t *testing.T) {
	runtime := respondWith(goldenResponse(t))
	if _, err := newEncoder(t, runtime).Encode(context.Background(), goldenRequest()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	call := runtime.calls()[0]
	if call.InferenceComponentName != nil || call.TargetVariant != nil || call.TargetModel != nil {
		t.Error("unset targeting options should be omitted, not sent empty")
	}
}

func TestEncodeBatchesLargeRequests(t *testing.T) {
	runtime := &stubRuntime{
		respond: func(_ int, in *sagemakerruntime.InvokeEndpointInput) (*sagemakerruntime.InvokeEndpointOutput, error) {
			var body struct {
				Inputs []string `json:"inputs"`
			}
			_ = json.Unmarshal(in.Body, &body)

			items := make([]string, len(body.Inputs))
			for i, text := range body.Inputs {
				items[i] = fmt.Sprintf(`{"dense": [%d, 0.5]}`, len(text))
			}
			return &sagemakerruntime.InvokeEndpointOutput{Body: []byte(fmt.Sprintf(`{
				"version": "agentic.representations.v1",
				"model": "BAAI/bge-m3",
				"spaces": {"dense": {"id":"d","provider":"sagemaker","model":"BAAI/bge-m3","kind":"dense","dimensions":2,"metric":"cosine"}},
				"data": [%s],
				"usage": {"input_tokens": %d, "request_count": 1}
			}`, strings.Join(items, ","), len(body.Inputs)))}, nil
		},
	}
	encoder := newEncoder(t, runtime, sagemaker.WithBatchSize(2))

	resp, err := encoder.Encode(context.Background(), denseRequest("a", "bb", "ccc", "dddd", "eeeee"))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(runtime.calls()) != 3 {
		t.Fatalf("made %d invocations, want 3", len(runtime.calls()))
	}
	for i, want := range []float32{1, 2, 3, 4, 5} {
		if resp.Data[i].Dense[0] != want {
			t.Errorf("item %d = %v, want the encoding of input %d", i, resp.Data[i].Dense, i)
		}
	}
	if resp.Usage.RequestCount != 3 {
		t.Errorf("request count = %d, want one per invocation", resp.Usage.RequestCount)
	}
}

// --------------------------------------------------------------------------
// Vector spaces
// --------------------------------------------------------------------------

func TestEncodeUsesPinnedSpacesWhenTheHandlerReportsNone(t *testing.T) {
	pinned := core.VectorSpace{
		ID:         "my-index-space",
		Provider:   "sagemaker",
		Model:      "BAAI/bge-m3",
		Revision:   "rev-1",
		Tokenizer:  "tok-1",
		Kind:       core.RepresentationDense,
		Dimensions: 2,
		Metric:     core.SimilarityCosine,
	}
	runtime := respondWith(`{"version":"agentic.representations.v1","model":"BAAI/bge-m3","data":[{"dense":[0.1,0.2]}]}`)
	encoder := newEncoder(t, runtime, sagemaker.WithVectorSpaces(
		map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: pinned}))

	resp, err := encoder.Encode(context.Background(), denseRequest("a"))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if resp.Spaces[core.RepresentationDense].ID != "my-index-space" {
		t.Errorf("space = %+v", resp.Spaces[core.RepresentationDense])
	}
}

// A redeployment onto different weights must fail rather than quietly writing
// incomparable vectors into an existing index.
func TestEncodeRejectsASpaceThatContradictsThePin(t *testing.T) {
	pinned := core.VectorSpace{
		ID: "my-index-space", Provider: "sagemaker", Model: "BAAI/bge-m3",
		Revision: "rev-1", Kind: core.RepresentationDense, Dimensions: 2, Metric: core.SimilarityCosine,
	}
	runtime := respondWith(`{"version":"agentic.representations.v1","model":"BAAI/bge-m3",
		"spaces":{"dense":{"id":"my-index-space","provider":"sagemaker","model":"BAAI/bge-m3","revision":"rev-2","kind":"dense","dimensions":2,"metric":"cosine"}},
		"data":[{"dense":[0.1,0.2]}]}`)
	encoder := newEncoder(t, runtime, sagemaker.WithVectorSpaces(
		map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: pinned}))

	_, err := encoder.Encode(context.Background(), denseRequest("a"))
	if err == nil || !strings.Contains(err.Error(), "configured for my-index-space") {
		t.Fatalf("got %v, want a pinned-space mismatch", err)
	}
}

// With neither a pin nor a handler descriptor there is no identity to store
// the vectors under, so the response is refused.
func TestEncodeRejectsAResponseWithNoSpaceAtAll(t *testing.T) {
	runtime := respondWith(`{"version":"agentic.representations.v1","model":"m","data":[{"dense":[0.1,0.2]}]}`)
	_, err := newEncoder(t, runtime).Encode(context.Background(), denseRequest("a"))
	if err == nil || !strings.Contains(err.Error(), "no vector space for a requested output") {
		t.Fatalf("got %v, want a missing-space error", err)
	}
}

// --------------------------------------------------------------------------
// Failures
// --------------------------------------------------------------------------

// An API error may name the endpoint and the AWS request ID, so a failure can
// be correlated with CloudWatch, but never the payload.
func TestInvocationErrorsCarrySafeMetadataOnly(t *testing.T) {
	const document = "the launch code is hunter2"
	runtime := &stubRuntime{
		respond: func(int, *sagemakerruntime.InvokeEndpointInput) (*sagemakerruntime.InvokeEndpointOutput, error) {
			return nil, &smithy.OperationError{
				ServiceID:     "SageMaker Runtime",
				OperationName: "InvokeEndpoint",
				Err: &awsAPIError{
					code:      "ModelError",
					message:   "model returned status 400",
					requestID: "req-abc-123",
				},
			}
		},
	}
	encoder := newEncoder(t, runtime)

	_, err := encoder.Encode(context.Background(), denseRequest(document))
	var invocation *sagemaker.InvocationError
	if !errors.As(err, &invocation) {
		t.Fatalf("got %v, want *sagemaker.InvocationError", err)
	}
	if invocation.Endpoint != "bge-m3-endpoint" {
		t.Errorf("endpoint = %q", invocation.Endpoint)
	}
	if invocation.RequestID != "req-abc-123" {
		t.Errorf("request id = %q", invocation.RequestID)
	}
	if strings.Contains(err.Error(), "hunter2") {
		t.Errorf("error leaks the payload: %q", err.Error())
	}
	if !strings.Contains(err.Error(), "req-abc-123") {
		t.Errorf("error message drops the request id: %q", err.Error())
	}
}

func TestInvocationErrorWithoutARequestID(t *testing.T) {
	sentinel := errors.New("network unreachable")
	runtime := &stubRuntime{
		respond: func(int, *sagemakerruntime.InvokeEndpointInput) (*sagemakerruntime.InvokeEndpointOutput, error) {
			return nil, sentinel
		},
	}
	_, err := newEncoder(t, runtime).Encode(context.Background(), denseRequest("a"))
	if !errors.Is(err, sentinel) {
		t.Fatalf("got %v, want the SDK error preserved", err)
	}
	var invocation *sagemaker.InvocationError
	if !errors.As(err, &invocation) || invocation.RequestID != "" {
		t.Fatalf("got %v, want an InvocationError with no request id", err)
	}
}

// Cancellation keeps its own cause so a shutdown is not retried as an outage.
func TestCancellationKeepsItsCause(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	runtime := &stubRuntime{
		respond: func(int, *sagemakerruntime.InvokeEndpointInput) (*sagemakerruntime.InvokeEndpointOutput, error) {
			cancel()
			return nil, context.Canceled
		},
	}

	_, err := newEncoder(t, runtime).Encode(ctx, denseRequest("a"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	var invocation *sagemaker.InvocationError
	if errors.As(err, &invocation) {
		t.Error("cancellation should not be wrapped as an endpoint failure")
	}
}

func TestEncodeRejectsEmptyAndOversizedResponses(t *testing.T) {
	t.Run("nil output", func(t *testing.T) {
		runtime := &stubRuntime{
			respond: func(int, *sagemakerruntime.InvokeEndpointInput) (*sagemakerruntime.InvokeEndpointOutput, error) {
				return nil, nil
			},
		}
		_, err := newEncoder(t, runtime).Encode(context.Background(), denseRequest("a"))
		if err == nil || !strings.Contains(err.Error(), "returned no output") {
			t.Fatalf("got %v, want a missing-output error", err)
		}
	})

	t.Run("oversized", func(t *testing.T) {
		runtime := respondWith(strings.Repeat("x", 4096))
		encoder := newEncoder(t, runtime, sagemaker.WithMaxResponseBytes(128))
		_, err := encoder.Encode(context.Background(), denseRequest("a"))
		if err == nil || !strings.Contains(err.Error(), "exceeds the 128 byte limit") {
			t.Fatalf("got %v, want an oversized-response error", err)
		}
	})
}

func TestEncodeRejectsMalformedResponses(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"truncated", `{"version":`, "not valid agentic.representations.v1 JSON"},
		{"wrong major", `{"version":"agentic.representations.v3","model":"m","data":[]}`, "major version 3"},
		{
			name: "short batch",
			body: `{"version":"agentic.representations.v1","model":"m",
				"spaces":{"dense":{"id":"d","provider":"p","model":"m","kind":"dense","dimensions":2,"metric":"cosine"}},
				"data":[]}`,
			want: "returned 0 representations for 1 inputs",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := newEncoder(t, respondWith(tc.body)).Encode(context.Background(), denseRequest("a"))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestEncodeRejectsUnsupportedOutputs(t *testing.T) {
	encoder := newEncoder(t, respondWith("{}"), sagemaker.WithOutputs(core.RepresentationDense))
	_, err := encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationSparse},
	})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

func TestConfiguredLimitsAreEnforced(t *testing.T) {
	encoder := newEncoder(t, respondWith("{}"),
		sagemaker.WithLimits(core.RepresentationLimits{MaxInputs: 1}))
	_, err := encoder.Encode(context.Background(), denseRequest("a", "b"))
	if !errors.Is(err, core.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want the configured limit to be enforced", err)
	}
}

// --------------------------------------------------------------------------
// Embedder compatibility and conformance
// --------------------------------------------------------------------------

func TestEmbedProjectsDenseOutput(t *testing.T) {
	runtime := respondWith(`{"version":"agentic.representations.v1","model":"BAAI/bge-m3",
		"spaces":{"dense":{"id":"d","provider":"sagemaker","model":"BAAI/bge-m3","kind":"dense","dimensions":2,"metric":"cosine"}},
		"data":[{"dense":[0.1,0.2]}],"usage":{"input_tokens":3,"request_count":1}}`)
	encoder := newEncoder(t, runtime)

	resp, err := encoder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"a"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 1 || len(resp.Vectors[0]) != 2 {
		t.Fatalf("vectors = %v", resp.Vectors)
	}

	var body map[string]any
	_ = json.Unmarshal(runtime.calls()[0].Body, &body)
	outputs, _ := body["outputs"].([]any)
	if len(outputs) != 1 || outputs[0] != "dense" {
		t.Errorf("the dense projection requested %v", body["outputs"])
	}
}

func TestEmbedFailsWhenDenseIsNotServed(t *testing.T) {
	encoder := newEncoder(t, respondWith("{}"), sagemaker.WithOutputs(core.RepresentationSparse))
	_, err := encoder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"a"}})
	if !errors.Is(err, core.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want ErrUnsupportedRepresentation", err)
	}
}

func TestConformance(t *testing.T) {
	conformance.RunRepresentation(t, conformance.RepresentationOptions{
		NewEncoder: func(t *testing.T) core.RepresentationEncoder {
			runtime := &stubRuntime{
				respond: func(_ int, in *sagemakerruntime.InvokeEndpointInput) (*sagemakerruntime.InvokeEndpointOutput, error) {
					var body struct {
						Inputs  []string `json:"inputs"`
						Outputs []string `json:"outputs"`
					}
					_ = json.Unmarshal(in.Body, &body)
					return &sagemakerruntime.InvokeEndpointOutput{
						Body: protocolResponse(body.Inputs, body.Outputs),
					}, nil
				},
			}
			return newEncoder(t, runtime)
		},
		Corpus:        []string{"alpha beta", "gamma", "delta epsilon"},
		Deterministic: true,
	})
}

func protocolResponse(inputs, outputs []string) []byte {
	wants := func(kind string) bool {
		for _, out := range outputs {
			if out == kind {
				return true
			}
		}
		return false
	}

	var spaces []string
	if wants("dense") {
		spaces = append(spaces, `"dense":{"id":"d","provider":"sagemaker","model":"BAAI/bge-m3","kind":"dense","dimensions":2,"metric":"cosine"}`)
	}
	if wants("sparse") {
		spaces = append(spaces, `"sparse":{"id":"s","provider":"sagemaker","model":"BAAI/bge-m3","kind":"sparse","dimensions":256,"metric":"dot_product"}`)
	}
	if wants("multi_vector") {
		spaces = append(spaces, `"multi_vector":{"id":"mv","provider":"sagemaker","model":"BAAI/bge-m3","kind":"multi_vector","dimensions":2,"metric":"cosine"}`)
	}

	items := make([]string, len(inputs))
	for i, text := range inputs {
		var parts []string
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

	return []byte(fmt.Sprintf(`{"version":"agentic.representations.v1","model":"BAAI/bge-m3","spaces":{%s},"data":[%s],"usage":{"input_tokens":3,"request_count":1}}`,
		strings.Join(spaces, ","), strings.Join(items, ",")))
}

// awsAPIError satisfies the SDK error interfaces the provider reads the
// request ID from.
type awsAPIError struct {
	code      string
	message   string
	requestID string
}

func (e *awsAPIError) Error() string                 { return e.code + ": " + e.message }
func (e *awsAPIError) ErrorCode() string             { return e.code }
func (e *awsAPIError) ErrorMessage() string          { return e.message }
func (e *awsAPIError) ErrorFault() smithy.ErrorFault { return smithy.FaultServer }
func (e *awsAPIError) ServiceRequestID() string      { return e.requestID }
