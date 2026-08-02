// Package sagemaker provides an Amazon SageMaker implementation of
// core.RepresentationEncoder for endpoints running a handler that speaks
// agentic.representations.v1.
//
// The runtime invocation is the same for a real-time endpoint and a serverless
// one: infrastructure mode is deployment configuration, not a different API.
// SigV4, credentials, regions, and role policy stay with the AWS SDK and the
// deployment.
//
// # Vector spaces
//
// An opaque endpoint cannot prove its own weights revision, so a deployment
// whose output you intend to keep supplies the descriptor with
// [WithVectorSpaces]. A space the handler reports must match what is pinned;
// a kind it leaves undescribed is filled from the pin. Without either, the
// response has no usable space and is rejected rather than stored under an
// identity nobody can verify.
package sagemaker

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/representationbatch"
	"github.com/regularkevvv/agentic/internal/representationwire"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/sagemakerruntime"
	"github.com/aws/smithy-go"
)

// providerName identifies this package in vector spaces and errors.
const providerName = "sagemaker"

// jsonContentType is the only payload type this package speaks.
const jsonContentType = "application/json"

// defaultMaxResponseBytes bounds a decoded response body. A multi-vector
// response is tens of millions of floats for a batch of long documents.
const defaultMaxResponseBytes = 64 << 20

// InvokeAPI is the slice of the SageMaker Runtime client this package uses.
//
// It is an interface so that transport behavior can be driven deterministically
// in tests without an AWS account, a network, or a signed request.
type InvokeAPI interface {
	InvokeEndpoint(
		ctx context.Context,
		params *sagemakerruntime.InvokeEndpointInput,
		optFns ...func(*sagemakerruntime.Options),
	) (*sagemakerruntime.InvokeEndpointOutput, error)
}

// Encoder invokes a SageMaker endpoint that speaks
// agentic.representations.v1.
type Encoder struct {
	client   InvokeAPI
	endpoint string

	model              string
	inferenceComponent string
	targetVariant      string
	targetModel        string
	batchSize          int
	maxResponseBytes   int
	outputs            []core.RepresentationKind
	spaces             map[core.RepresentationKind]core.VectorSpace
	limits             core.RepresentationLimits
	embedder           *core.EncoderEmbedder
}

// Option configures the Encoder.
type Option func(*config)

type config struct {
	client             InvokeAPI
	awsConfig          *aws.Config
	region             string
	profile            string
	accessKeyID        string
	secretAccessKey    string
	sessionToken       string
	model              string
	inferenceComponent string
	targetVariant      string
	targetModel        string
	batchSize          int
	maxResponseBytes   *int
	outputs            []core.RepresentationKind
	spaces             map[core.RepresentationKind]core.VectorSpace
	limits             *core.RepresentationLimits
}

// WithClient injects the SageMaker Runtime client, bypassing AWS config
// resolution. It is how transport tests run without credentials.
func WithClient(client InvokeAPI) Option {
	return func(c *config) { c.client = client }
}

// WithAWSConfig uses an already-resolved AWS config, so an application that
// centralizes credential loading does not resolve it twice.
func WithAWSConfig(cfg aws.Config) Option {
	return func(c *config) { c.awsConfig = &cfg }
}

// WithRegion sets the AWS region. If not set, AWS_DEFAULT_REGION and then
// AWS_REGION are used.
func WithRegion(region string) Option {
	return func(c *config) { c.region = region }
}

// WithProfile selects a shared-configuration profile.
func WithProfile(profile string) Option {
	return func(c *config) { c.profile = profile }
}

// WithCredentials sets static credentials. Prefer the default credential
// chain; this exists for callers who already hold short-lived credentials.
func WithCredentials(accessKeyID, secretAccessKey, sessionToken string) Option {
	return func(c *config) {
		c.accessKeyID = accessKeyID
		c.secretAccessKey = secretAccessKey
		c.sessionToken = sessionToken
	}
}

// WithModel records the model name the endpoint serves, used when the handler
// reports none.
func WithModel(model string) Option {
	return func(c *config) { c.model = model }
}

// WithInferenceComponent targets a specific inference component on a
// multi-model endpoint.
func WithInferenceComponent(name string) Option {
	return func(c *config) { c.inferenceComponent = name }
}

// WithTargetVariant pins the production variant to invoke, for a deliberate
// A/B comparison rather than the endpoint's own traffic split.
func WithTargetVariant(name string) Option {
	return func(c *config) { c.targetVariant = name }
}

// WithTargetModel selects the model artifact on a multi-model endpoint.
func WithTargetModel(name string) Option {
	return func(c *config) { c.targetModel = name }
}

// WithBatchSize splits requests larger than size into that many inputs per
// invocation. Zero, the default, sends the batch as one payload.
//
// SageMaker caps an InvokeEndpoint payload at 6 MB for a real-time endpoint,
// so a large batch of long documents needs a batch size whether or not the
// model would have accepted it.
func WithBatchSize(size int) Option {
	return func(c *config) { c.batchSize = size }
}

// WithMaxResponseBytes caps the response payload this encoder will decode
// (default 64 MiB).
func WithMaxResponseBytes(limit int) Option {
	return func(c *config) { c.maxResponseBytes = &limit }
}

// WithOutputs declares which representation kinds this endpoint serves
// (default all three, matching the reference handler).
func WithOutputs(kinds ...core.RepresentationKind) Option {
	return func(c *config) { c.outputs = append([]core.RepresentationKind(nil), kinds...) }
}

// WithVectorSpaces pins the vector spaces this endpoint encodes into. See the
// package documentation for why an opaque endpoint needs them.
func WithVectorSpaces(spaces map[core.RepresentationKind]core.VectorSpace) Option {
	return func(c *config) {
		c.spaces = make(map[core.RepresentationKind]core.VectorSpace, len(spaces))
		for kind, space := range spaces {
			c.spaces[kind] = space
		}
	}
}

// WithLimits overrides the request and response size ceilings.
func WithLimits(limits core.RepresentationLimits) Option {
	return func(c *config) { c.limits = &limits }
}

// New creates an encoder for a SageMaker endpoint.
//
// Example:
//
//	encoder, err := sagemaker.New(ctx, "bge-m3-endpoint",
//	    sagemaker.WithRegion("us-east-1"),
//	    sagemaker.WithModel("BAAI/bge-m3"),
//	    sagemaker.WithVectorSpaces(spaces),
//	)
func New(ctx context.Context, endpointName string, opts ...Option) (*Encoder, error) {
	if endpointName == "" {
		return nil, errors.New("sagemaker: endpoint name cannot be empty")
	}

	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.batchSize < 0 {
		return nil, errors.New("sagemaker: batch size cannot be negative")
	}

	outputs := cfg.outputs
	if outputs == nil {
		outputs = []core.RepresentationKind{
			core.RepresentationDense,
			core.RepresentationSparse,
			core.RepresentationMultiVector,
		}
	}
	for _, kind := range outputs {
		if !kind.Valid() {
			return nil, errors.New("sagemaker: output kind " + string(kind) +
				" is not dense, sparse, or multi_vector")
		}
	}
	for kind, space := range cfg.spaces {
		if err := space.Validate(); err != nil {
			return nil, errors.New("sagemaker: pinned " + string(kind) + " space: " + err.Error())
		}
	}

	maxResponseBytes := defaultMaxResponseBytes
	if cfg.maxResponseBytes != nil {
		if *cfg.maxResponseBytes <= 0 {
			return nil, errors.New("sagemaker: max response bytes must be positive")
		}
		maxResponseBytes = *cfg.maxResponseBytes
	}

	client := cfg.client
	if client == nil {
		awsCfg, err := resolveAWSConfig(ctx, cfg)
		if err != nil {
			return nil, err
		}
		client = sagemakerruntime.NewFromConfig(awsCfg)
	}

	limits := core.DefaultRepresentationLimits()
	if cfg.limits != nil {
		limits = *cfg.limits
	}

	encoder := &Encoder{
		client:             client,
		endpoint:           endpointName,
		model:              cfg.model,
		inferenceComponent: cfg.inferenceComponent,
		targetVariant:      cfg.targetVariant,
		targetModel:        cfg.targetModel,
		batchSize:          cfg.batchSize,
		maxResponseBytes:   maxResponseBytes,
		outputs:            outputs,
		spaces:             cfg.spaces,
		limits:             limits,
	}

	var err error
	if encoder.Capabilities().Supports(core.RepresentationDense) {
		encoder.embedder, err = core.NewEncoderEmbedder(encoder)
		if err != nil {
			return nil, err
		}
	}
	return encoder, nil
}

// resolveAWSConfig builds the config the runtime client is constructed from.
func resolveAWSConfig(ctx context.Context, cfg *config) (aws.Config, error) {
	if cfg.awsConfig != nil {
		return *cfg.awsConfig, nil
	}

	region := cfg.region
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		region = os.Getenv("AWS_REGION")
	}
	if region == "" {
		return aws.Config{}, errors.New("sagemaker: region not set (use WithRegion or set AWS_DEFAULT_REGION)")
	}

	awsOpts := []func(*awsconfig.LoadOptions) error{awsconfig.WithRegion(region)}
	if cfg.profile != "" {
		awsOpts = append(awsOpts, awsconfig.WithSharedConfigProfile(cfg.profile))
	}
	if cfg.accessKeyID != "" {
		awsOpts = append(awsOpts, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(cfg.accessKeyID, cfg.secretAccessKey, cfg.sessionToken),
		))
	}

	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, awsOpts...)
	if err != nil {
		return aws.Config{}, fmt.Errorf("sagemaker: failed to load AWS config: %w", err)
	}
	return awsCfg, nil
}

// MustNew is like New but panics on error.
func MustNew(ctx context.Context, endpointName string, opts ...Option) *Encoder {
	e, err := New(ctx, endpointName, opts...)
	if err != nil {
		panic(err)
	}
	return e
}

// Name implements core.RepresentationEncoder. It reports the configured model
// name, falling back to the endpoint name, because an endpoint is a deployment
// rather than a model and may not name one.
func (e *Encoder) Name() string {
	if e.model != "" {
		return e.model
	}
	return e.endpoint
}

// Capabilities implements core.RepresentationEncoder.
func (e *Encoder) Capabilities() core.RepresentationCapabilities {
	return core.RepresentationCapabilities{
		Outputs: append([]core.RepresentationKind(nil), e.outputs...),
		InputTypes: []core.EmbeddingInputType{
			core.EmbeddingInputNone,
			core.EmbeddingInputQuery,
			core.EmbeddingInputDocument,
		},
		SupportsTruncation:  true,
		SupportsMultiOutput: true,
	}
}

func (e *Encoder) validator() core.RepresentationValidator {
	return core.RepresentationValidator{
		Provider:     providerName,
		Capabilities: e.Capabilities(),
		Limits:       e.limits,
	}
}

// InvocationError reports a failed InvokeEndpoint call.
//
// It carries the endpoint name and the AWS request ID so a failure can be
// correlated with CloudWatch, and nothing from the payload: an inference
// error message quotes the input that produced it.
type InvocationError struct {
	// Endpoint is the SageMaker endpoint name.
	Endpoint string

	// RequestID is the AWS request ID, when the SDK reported one.
	RequestID string

	// Err is the underlying SDK error.
	Err error
}

func (e *InvocationError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "sagemaker: endpoint %s", e.Endpoint)
	if e.RequestID != "" {
		fmt.Fprintf(&b, " (request %s)", e.RequestID)
	}
	fmt.Fprintf(&b, ": %v", e.Err)
	return b.String()
}

func (e *InvocationError) Unwrap() error { return e.Err }

// Encode implements core.RepresentationEncoder.
func (e *Encoder) Encode(ctx context.Context, req *core.RepresentationRequest) (*core.RepresentationResponse, error) {
	validator := e.validator()
	if err := validator.ValidateRequest(req); err != nil {
		return nil, err
	}

	resp, err := representationbatch.Chunked(ctx, req, e.batchSize, e.encodeChunk)
	if err != nil {
		return nil, err
	}
	if err := validator.ValidateResponse(req, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

func (e *Encoder) encodeChunk(ctx context.Context, req *core.RepresentationRequest) (*core.RepresentationResponse, error) {
	body, err := marshalRequest(req)
	if err != nil {
		return nil, err
	}

	input := &sagemakerruntime.InvokeEndpointInput{
		EndpointName: aws.String(e.endpoint),
		ContentType:  aws.String(jsonContentType),
		Accept:       aws.String(jsonContentType),
		Body:         body,
	}
	if e.inferenceComponent != "" {
		input.InferenceComponentName = aws.String(e.inferenceComponent)
	}
	if e.targetVariant != "" {
		input.TargetVariant = aws.String(e.targetVariant)
	}
	if e.targetModel != "" {
		input.TargetModel = aws.String(e.targetModel)
	}

	out, err := e.client.InvokeEndpoint(ctx, input)
	if err != nil {
		// Cancellation keeps its own cause, so a caller's shutdown is not
		// reported as an endpoint failure and retried as one.
		if ctxErr := ctx.Err(); ctxErr != nil && errors.Is(err, ctxErr) {
			return nil, err
		}
		return nil, &InvocationError{
			Endpoint:  e.endpoint,
			RequestID: awsRequestID(err),
			Err:       err,
		}
	}
	if out == nil {
		return nil, &core.InvalidRepresentationResponseError{
			Provider: providerName,
			Item:     -1,
			Problem:  "endpoint returned no output",
		}
	}
	if len(out.Body) > e.maxResponseBytes {
		return nil, fmt.Errorf("sagemaker: response exceeds the %d byte limit", e.maxResponseBytes)
	}

	return representationwire.Decode(out.Body, req, representationwire.DecodeOptions{
		Provider:      providerName,
		Model:         e.Name(),
		Expected:      e.spaces,
		ResponseBytes: len(out.Body),
	})
}

// Embed implements core.Embedder by requesting dense output only.
func (e *Encoder) Embed(ctx context.Context, req *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	if e.embedder == nil {
		return nil, &core.UnsupportedRepresentationError{
			Provider:  providerName,
			Kind:      core.RepresentationDense,
			Supported: e.outputs,
		}
	}
	return e.embedder.Embed(ctx, req)
}

// awsRequestID extracts the request ID the SDK attaches to an API error, if
// there is one. It is safe to log; the payload is not.
func awsRequestID(err error) string {
	var apiErr interface {
		smithy.APIError
		ServiceRequestID() string
	}
	if errors.As(err, &apiErr) {
		return apiErr.ServiceRequestID()
	}
	return ""
}

// Compile-time checks that Encoder satisfies both contracts.
var (
	_ core.RepresentationEncoder = (*Encoder)(nil)
	_ core.Embedder              = (*Encoder)(nil)
)
