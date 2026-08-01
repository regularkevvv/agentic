// Package huggingface provides Hugging Face implementations of
// core.RepresentationEncoder, in two explicit modes.
//
// [NewDedicated] talks to an Inference Endpoint you operate, running the
// canonical agentic.representations.v1 handler from deploy/representations.
// That is the reliable path for a multi-representation model: a custom handler
// controls which outputs are computed, what the token weights are, and which
// revisions the response declares.
//
// [NewShared] talks to the Inference Providers router's feature-extraction
// task, which returns one dense vector per input and nothing else. It is
// advertised as dense-only for exactly that reason. A model card saying the
// model produces sparse weights locally is not evidence that the hosted route
// returns them, and this package does not claim a capability it has not
// observed in a response.
package huggingface

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/representationbatch"
	"github.com/regularkevvv/agentic/internal/representationwire"
)

// providerName identifies this package in vector spaces and errors.
const providerName = "huggingface"

// DedicatedEncoder speaks agentic.representations.v1 to a Hugging Face
// Inference Endpoint running a custom handler.
type DedicatedEncoder struct {
	*client

	endpoint  string
	model     string
	batchSize int
	outputs   []core.RepresentationKind
	spaces    map[core.RepresentationKind]core.VectorSpace
	limits    core.RepresentationLimits
	embedder  *core.EncoderEmbedder
}

// DedicatedOption configures a DedicatedEncoder.
type DedicatedOption func(*dedicatedConfig)

type dedicatedConfig struct {
	transportConfig

	model     string
	batchSize int
	outputs   []core.RepresentationKind
	spaces    map[core.RepresentationKind]core.VectorSpace
	limits    *core.RepresentationLimits
}

// WithToken sets the Hugging Face access token. If not set, HF_TOKEN and then
// HUGGING_FACE_HUB_TOKEN are used.
func WithToken(token string) DedicatedOption {
	return func(c *dedicatedConfig) { c.token = token }
}

// WithHTTPClient sets a custom HTTP client, for proxies, instrumentation, or
// tests.
func WithHTTPClient(client *http.Client) DedicatedOption {
	return func(c *dedicatedConfig) { c.httpClient = client }
}

// WithMaxRetries sets how many times a request is retried on 429 and transient
// 5xx responses (default 2). A cold endpoint answers 503 while it loads, which
// counts as transient.
func WithMaxRetries(retries int) DedicatedOption {
	return func(c *dedicatedConfig) { c.maxRetries = &retries }
}

// WithMaxResponseBytes caps the response body this encoder will read (default
// 64 MiB).
func WithMaxResponseBytes(limit int64) DedicatedOption {
	return func(c *dedicatedConfig) { c.maxResponseBytes = &limit }
}

// WithModel records the model name the endpoint serves, used when the handler
// reports none.
func WithModel(model string) DedicatedOption {
	return func(c *dedicatedConfig) { c.model = model }
}

// WithBatchSize splits requests larger than size into that many inputs per
// call, preserving order and summing usage. Zero, the default, sends the batch
// as one request.
func WithBatchSize(size int) DedicatedOption {
	return func(c *dedicatedConfig) { c.batchSize = size }
}

// WithOutputs declares which representation kinds this endpoint serves
// (default all three, matching the reference handler).
func WithOutputs(kinds ...core.RepresentationKind) DedicatedOption {
	return func(c *dedicatedConfig) { c.outputs = append([]core.RepresentationKind(nil), kinds...) }
}

// WithVectorSpaces pins the vector spaces this endpoint encodes into.
//
// A pinned space must match what the handler reports, exactly; a kind the
// handler leaves undescribed is filled from here. Pin them for any endpoint
// whose output you intend to keep, so that a redeployment onto different
// weights fails loudly instead of quietly mixing two generations of vectors in
// one index.
func WithVectorSpaces(spaces map[core.RepresentationKind]core.VectorSpace) DedicatedOption {
	return func(c *dedicatedConfig) {
		c.spaces = make(map[core.RepresentationKind]core.VectorSpace, len(spaces))
		for kind, space := range spaces {
			c.spaces[kind] = space
		}
	}
}

// WithLimits overrides the request and response size ceilings.
func WithLimits(limits core.RepresentationLimits) DedicatedOption {
	return func(c *dedicatedConfig) { c.limits = &limits }
}

// NewDedicated creates an encoder for an Inference Endpoint running the
// agentic.representations.v1 handler.
//
// Example:
//
//	encoder, err := huggingface.NewDedicated(
//	    "https://abc123.us-east-1.aws.endpoints.huggingface.cloud",
//	    huggingface.WithModel("BAAI/bge-m3"),
//	)
func NewDedicated(endpointURL string, opts ...DedicatedOption) (*DedicatedEncoder, error) {
	if endpointURL == "" {
		return nil, errors.New("huggingface: endpoint URL cannot be empty")
	}
	if !strings.HasPrefix(endpointURL, "http://") && !strings.HasPrefix(endpointURL, "https://") {
		return nil, errors.New("huggingface: endpoint URL must be absolute")
	}

	cfg := &dedicatedConfig{}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.batchSize < 0 {
		return nil, errors.New("huggingface: batch size cannot be negative")
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
			return nil, errors.New("huggingface: output kind " + string(kind) +
				" is not dense, sparse, or multi_vector")
		}
	}
	for kind, space := range cfg.spaces {
		if err := space.Validate(); err != nil {
			return nil, errors.New("huggingface: pinned " + string(kind) + " space: " + err.Error())
		}
	}

	c, err := newClient(cfg.transportConfig)
	if err != nil {
		return nil, err
	}

	limits := core.DefaultRepresentationLimits()
	if cfg.limits != nil {
		limits = *cfg.limits
	}

	encoder := &DedicatedEncoder{
		client:    c,
		endpoint:  strings.TrimSuffix(endpointURL, "/"),
		model:     cfg.model,
		batchSize: cfg.batchSize,
		outputs:   outputs,
		spaces:    cfg.spaces,
		limits:    limits,
	}
	if encoder.Capabilities().Supports(core.RepresentationDense) {
		encoder.embedder, err = core.NewEncoderEmbedder(encoder)
		if err != nil {
			return nil, err
		}
	}
	return encoder, nil
}

// Name implements core.RepresentationEncoder. It reports the configured model
// name, falling back to the endpoint URL when none was configured, because an
// endpoint is a deployment rather than a model and may not name one.
func (e *DedicatedEncoder) Name() string {
	if e.model != "" {
		return e.model
	}
	return e.endpoint
}

// Capabilities implements core.RepresentationEncoder.
func (e *DedicatedEncoder) Capabilities() core.RepresentationCapabilities {
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

func (e *DedicatedEncoder) validator() core.RepresentationValidator {
	return core.RepresentationValidator{
		Provider:     providerName,
		Capabilities: e.Capabilities(),
		Limits:       e.limits,
	}
}

// Encode implements core.RepresentationEncoder.
func (e *DedicatedEncoder) Encode(ctx context.Context, req *core.RepresentationRequest) (*core.RepresentationResponse, error) {
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

func (e *DedicatedEncoder) encodeChunk(ctx context.Context, req *core.RepresentationRequest) (*core.RepresentationResponse, error) {
	payload, err := e.post(ctx, e.endpoint, representationwire.NewRequest(req))
	if err != nil {
		return nil, err
	}
	return representationwire.Decode(payload, req, representationwire.DecodeOptions{
		Provider:      providerName,
		Model:         e.Name(),
		Expected:      e.spaces,
		ResponseBytes: len(payload),
	})
}

// Embed implements core.Embedder by requesting dense output only.
func (e *DedicatedEncoder) Embed(ctx context.Context, req *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	if e.embedder == nil {
		return nil, &core.UnsupportedRepresentationError{
			Provider:  providerName,
			Kind:      core.RepresentationDense,
			Supported: e.outputs,
		}
	}
	return e.embedder.Embed(ctx, req)
}

// Compile-time checks that DedicatedEncoder satisfies both contracts.
var (
	_ core.RepresentationEncoder = (*DedicatedEncoder)(nil)
	_ core.Embedder              = (*DedicatedEncoder)(nil)
)
