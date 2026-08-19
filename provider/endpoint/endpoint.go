// Package endpoint provides a client for any server speaking
// agentic.representations.v1, the protocol defined by the JSON Schemas in
// internal/retrieval/wire/testdata.
//
// It POSTs protocol JSON to a URL. That URL may be a Hugging Face Inference
// Endpoint, a container on Kubernetes, a Modal or Fly.io deployment, or a
// Python process on a laptop; nothing in this package knows which, and the
// package is named for what it talks to rather than for whoever hosts it. A
// custom handler is the reliable path for a multi-representation model,
// because it controls which outputs are computed, what the token weights are,
// and which revisions the response declares.
//
// # Authentication
//
// The token comes from [WithToken], or from AGENTIC_ENDPOINT_TOKEN when that
// option is not used at all, and [New] fails without one. No vendor's variable
// is read here: a client that works against any server has no business reading
// the credential one hosting provider happens to use.
//
// A handler on a loopback address usually checks no credential at all.
// [WithoutAuthentication] is how to say that, and it must be asked for — an
// empty [WithToken] is an error rather than an anonymous request, and rather
// than a silent fall back to whatever the environment holds.
package endpoint

import (
	"context"
	"errors"
	"os"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/providerhttp"
	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/internal/retrieval/batch"
	"github.com/regularkevvv/agentic/internal/retrieval/wire"
)

// providerName identifies this package in vector spaces and errors. It reaches
// stored data: a space descriptor the handler leaves without a provider is
// completed with this, and the canonical space ID is derived from it.
const providerName = "endpoint"

// tokenEnvVar is the variable a token is read from when none is passed.
const tokenEnvVar = "AGENTIC_ENDPOINT_TOKEN"

// APIError reports a non-200 response from an endpoint.
//
// It is the transport's shared error type. Provider names the client that
// produced it, so a program using more than one HTTP provider reads that field
// rather than distinguishing them by type.
type APIError = providerhttp.APIError

// Encoder speaks agentic.representations.v1 to one endpoint URL.
type Encoder struct {
	client *providerhttp.Client

	endpoint  string
	model     string
	batchSize int
	outputs   []retrieval.RepresentationKind
	spaces    map[retrieval.RepresentationKind]retrieval.VectorSpace
	limits    retrieval.RepresentationLimits
	embedder  *retrieval.EncoderEmbedder
}

// New creates an encoder for a server running the agentic.representations.v1
// handler at endpointURL.
//
// It fails when the URL is not absolute, when an option is out of range, or
// when no token is available and [WithoutAuthentication] was not given.
//
// Example:
//
//	encoder, err := endpoint.New(
//	    "https://abc123.us-east-1.aws.endpoints.huggingface.cloud",
//	    endpoint.WithModel("BAAI/bge-m3"),
//	)
func New(endpointURL string, opts ...Option) (*Encoder, error) {
	if endpointURL == "" {
		return nil, errors.New("endpoint: endpoint URL cannot be empty")
	}
	if !strings.HasPrefix(endpointURL, "http://") && !strings.HasPrefix(endpointURL, "https://") {
		return nil, errors.New("endpoint: endpoint URL must be absolute")
	}

	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.batchSize < 0 {
		return nil, errors.New("endpoint: batch size cannot be negative")
	}
	if err := resolveToken(cfg); err != nil {
		return nil, err
	}

	outputs := cfg.outputs
	if outputs == nil {
		outputs = []retrieval.RepresentationKind{
			retrieval.RepresentationDense,
			retrieval.RepresentationSparse,
			retrieval.RepresentationMultiVector,
		}
	}
	for _, kind := range outputs {
		if !kind.Valid() {
			return nil, errors.New("endpoint: output kind " + string(kind) +
				" is not dense, sparse, or multi_vector")
		}
	}
	for kind, space := range cfg.spaces {
		if err := space.Validate(); err != nil {
			return nil, errors.New("endpoint: pinned " + string(kind) + " space: " + err.Error())
		}
	}

	client, err := providerhttp.New(providerName, cfg.Config)
	if err != nil {
		return nil, err
	}

	limits := retrieval.DefaultRepresentationLimits()
	if cfg.limits != nil {
		limits = *cfg.limits
	}

	encoder := &Encoder{
		client:    client,
		endpoint:  strings.TrimSuffix(endpointURL, "/"),
		model:     cfg.model,
		batchSize: cfg.batchSize,
		outputs:   outputs,
		spaces:    cfg.spaces,
		limits:    limits,
	}
	if encoder.Capabilities().Supports(retrieval.RepresentationDense) {
		encoder.embedder, err = retrieval.NewEncoderEmbedder(encoder)
		if err != nil {
			return nil, err
		}
	}
	return encoder, nil
}

// resolveToken settles what credential, if any, the transport will send.
//
// An unauthenticated client is only ever reached by asking for one, so a
// caller who meant to authenticate and passed an empty string gets an error
// instead of anonymous requests against an endpoint that may not want them.
// For the same reason an empty [WithToken] does not fall through to the
// environment: a program reading its token from a configuration file that
// turned out to be empty would otherwise authenticate as whoever the shell
// happens to be.
func resolveToken(cfg *config) error {
	if cfg.unauthenticated {
		if cfg.tokenSet {
			return errors.New("endpoint: WithToken and WithoutAuthentication are mutually exclusive")
		}
		return nil
	}
	if cfg.tokenSet {
		if cfg.Token == "" {
			return errors.New("endpoint: WithToken was given an empty token (use " +
				"WithoutAuthentication for a handler that checks none)")
		}
		return nil
	}
	cfg.Token = os.Getenv(tokenEnvVar)
	if cfg.Token == "" {
		return errors.New("endpoint: token not set (use WithToken, the " + tokenEnvVar +
			" env var, or WithoutAuthentication for a handler that checks none)")
	}
	return nil
}

// Name implements retrieval.RepresentationEncoder. It reports the configured model
// name, falling back to the endpoint URL when none was configured, because an
// endpoint is a deployment rather than a model and may not name one.
func (e *Encoder) Name() string {
	if e.model != "" {
		return e.model
	}
	return e.endpoint
}

// ModelMetadata reports semantic transport identity for an application-owned
// endpoint whose upstream provider is not known to the client.
func (e *Encoder) ModelMetadata() core.ModelMetadata {
	return core.ModelMetadataForEndpoint("custom", "embeddings", e.endpoint)
}

// Capabilities implements retrieval.RepresentationEncoder.
func (e *Encoder) Capabilities() retrieval.RepresentationCapabilities {
	return retrieval.RepresentationCapabilities{
		Outputs: append([]retrieval.RepresentationKind(nil), e.outputs...),
		InputTypes: []retrieval.EmbeddingInputType{
			retrieval.EmbeddingInputNone,
			retrieval.EmbeddingInputQuery,
			retrieval.EmbeddingInputDocument,
		},
		SupportsTruncation:  true,
		SupportsMultiOutput: true,
	}
}

func (e *Encoder) validator() retrieval.RepresentationValidator {
	return retrieval.RepresentationValidator{
		Provider:     providerName,
		Capabilities: e.Capabilities(),
		Limits:       e.limits,
	}
}

// Encode implements retrieval.RepresentationEncoder.
func (e *Encoder) Encode(ctx context.Context, req *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, error) {
	validator := e.validator()
	if err := validator.ValidateRequest(req); err != nil {
		return nil, err
	}

	resp, err := batch.Chunked(ctx, req, e.batchSize, e.encodeChunk)
	if err != nil {
		return nil, err
	}
	if err := validator.ValidateResponse(req, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

func (e *Encoder) encodeChunk(ctx context.Context, req *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, error) {
	payload, err := e.client.Post(ctx, e.endpoint, wire.NewRequest(req))
	if err != nil {
		return nil, err
	}
	return wire.Decode(payload, req, wire.DecodeOptions{
		Provider:      providerName,
		Model:         e.Name(),
		Expected:      e.spaces,
		ResponseBytes: len(payload),
	})
}

// Embed implements retrieval.Embedder by requesting dense output only.
func (e *Encoder) Embed(ctx context.Context, req *retrieval.EmbeddingRequest) (*retrieval.EmbeddingResponse, error) {
	if e.embedder == nil {
		return nil, &retrieval.UnsupportedRepresentationError{
			Provider:  providerName,
			Kind:      retrieval.RepresentationDense,
			Supported: e.outputs,
		}
	}
	return e.embedder.Embed(ctx, req)
}

// Compile-time checks that Encoder satisfies both contracts.
var (
	_ retrieval.RepresentationEncoder = (*Encoder)(nil)
	_ retrieval.Embedder              = (*Encoder)(nil)
)
