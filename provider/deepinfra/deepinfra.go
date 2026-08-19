// Package deepinfra provides a DeepInfra implementation of
// retrieval.RepresentationEncoder, and of retrieval.Embedder for its dense output.
//
// It targets DeepInfra's native inference API at /v1/inference/{model}, which
// is the only route that exposes a multi-representation model's full output.
// For BAAI/bge-m3-multi that is a dense vector, a learned sparse (lexical
// weights) vector, and a ColBERT-style token multi-vector, all from one
// forward pass over the batch.
//
// DeepInfra also serves an OpenAI-compatible /v1/openai/embeddings route. That
// route returns a dense vector and nothing else, whatever the model can do, so
// this package does not use it and a caller must not read dense-only success
// there as evidence of sparse support. To use that route for plain dense
// embeddings, point provider/openai at DeepInfra's base URL instead.
//
// # Vector spaces
//
// DeepInfra's response carries no model revision, so Revision and Tokenizer
// are empty unless WithModelRevision supplies them. Without a revision, a
// silent model redeployment produces vectors that land in the same space ID as
// the old ones — record a revision for any index you intend to keep.
//
// BGE-M3 is a symmetric encoder: queries and documents go through the same
// weights with no task instruction, so both roles are accepted, neither
// changes the request, and their outputs are directly comparable. The input
// role is therefore not part of the space identity here, as it would have to
// be for an asymmetric model.
package deepinfra

import (
	"net/http"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/retrieval"
)

// providerName identifies this package in vector spaces and errors.
const providerName = "deepinfra"

// BGEM3Model is the DeepInfra model ID for BGE-M3's multi-representation
// endpoint. It is a named constant rather than a default so that the model a
// deployment encodes with is always visible at the call site.
const BGEM3Model = "BAAI/bge-m3-multi"

// BGEM3SparseVocabulary is BGE-M3's tokenizer vocabulary size, the logical
// width of its sparse space. Pass it to WithSparseVocabulary when the response
// shape does not reveal the vocabulary on its own.
const BGEM3SparseVocabulary = 250002

// defaultBatchSize splits requests by default, because this API's responses
// are far larger than its requests.
//
// Measured against BAAI/bge-m3-multi on 2026-08-01, per input:
//
//	dense        39 KB   (a 1024-float vector, sent twice — see encode.go)
//	sparse      977 KB   (the full 250002-wide vocabulary row, mostly zeros)
//	colbert     ~34 KB per token
//	all three  1185 KB
//
// A caller encoding a few hundred documents with sparse output in one request
// would build a response of several hundred megabytes, exceed the response
// ceiling, and fail after the inference had already been done and billed. At
// 32 inputs a sparse batch is around 31 MB, comfortably inside the default
// ceiling with room for the dense and multi-vector kinds alongside it.
//
// Multi-vector output scales with document length rather than count, so a
// batch of long documents needs a smaller size than this: budget roughly
// 34 KB per token per input and set WithBatchSize accordingly.
const defaultBatchSize = 32

// Encoder implements retrieval.RepresentationEncoder and retrieval.Embedder against
// DeepInfra's native inference API.
type Encoder struct {
	*client

	model            string
	normalize        *bool
	batchSize        int
	sparseVocabulary int
	revision         string
	tokenizer        string
	outputs          []retrieval.RepresentationKind
	limits           retrieval.RepresentationLimits

	// embedder projects this encoder onto retrieval.Embedder. It is nil when the
	// encoder does not advertise dense output.
	embedder *retrieval.EncoderEmbedder
}

// Option configures the Encoder.
type Option func(*config)

type config struct {
	transportConfig

	normalize        *bool
	batchSize        *int
	sparseVocabulary int
	revision         string
	tokenizer        string
	outputs          []retrieval.RepresentationKind
	limits           *retrieval.RepresentationLimits
}

// WithAPIToken sets the API token. If not set, the DEEPINFRA_TOKEN env var is
// used.
func WithAPIToken(token string) Option {
	return func(c *config) { c.apiKey = token }
}

// WithBaseURL sets a custom base URL (default https://api.deepinfra.com/v1).
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// WithHTTPClient sets a custom HTTP client, for proxies, instrumentation, or
// tests.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}

// WithMaxRetries sets how many times a request is retried on 429 and transient
// 5xx responses (default 2).
func WithMaxRetries(retries int) Option {
	return func(c *config) { c.maxRetries = &retries }
}

// WithMaxResponseBytes caps the response body this encoder will read (default
// 64 MiB). Lower it when requesting multi-vector output from a deployment with
// a tight memory budget.
func WithMaxResponseBytes(limit int64) Option {
	return func(c *config) { c.maxResponseBytes = &limit }
}

// WithNormalize sets DeepInfra's normalize flag, which scales each returned
// vector to unit length. Leaving it unset uses the API's own default, which is
// not to normalize.
//
// Normalization does not change cosine similarity, which is the metric this
// package declares. It does change dot-product scores, so a store that indexes
// dense vectors under an inner-product metric must keep the setting fixed for
// the life of the index.
func WithNormalize(normalize bool) Option {
	return func(c *config) { c.normalize = &normalize }
}

// WithBatchSize splits requests larger than size into that many inputs per
// provider call, preserving order and summing usage.
//
// The default is 32, chosen from measured response sizes; see
// [defaultBatchSize]. Lower it for multi-vector output over long documents,
// where response size scales with token count rather than input count. Zero
// disables splitting and sends the batch as one request, which is only safe
// when you know the response will fit.
func WithBatchSize(size int) Option {
	return func(c *config) { c.batchSize = &size }
}

// WithSparseVocabulary declares the tokenizer vocabulary size, which is the
// logical width of the sparse space and the bound every sparse index must fall
// within.
//
// It is only needed when the provider returns sparse output as a coordinate
// map, which does not reveal the vocabulary. When the response is a full
// vocabulary-width row the size is observed from it directly, and a value set
// here must agree with what was observed.
func WithSparseVocabulary(size int) Option {
	return func(c *config) { c.sparseVocabulary = size }
}

// WithModelRevision records the immutable model and tokenizer revisions in
// every vector space this encoder reports.
//
// DeepInfra does not return either, and this package will not invent them.
// Supplying them is what lets a consumer detect that the deployment behind a
// model name changed, instead of quietly mixing two generations of vectors in
// one index.
func WithModelRevision(model, tokenizer string) Option {
	return func(c *config) {
		c.revision = model
		c.tokenizer = tokenizer
	}
}

// WithOutputs restricts the representation kinds this encoder advertises.
//
// The default is all three, which is what BAAI/bge-m3-multi produces. Narrow
// it when pointing this package at a DeepInfra model that produces fewer, so
// that Capabilities() describes the deployment rather than the package.
func WithOutputs(kinds ...retrieval.RepresentationKind) Option {
	return func(c *config) { c.outputs = append([]retrieval.RepresentationKind(nil), kinds...) }
}

// WithLimits overrides the request and response size ceilings.
func WithLimits(limits retrieval.RepresentationLimits) Option {
	return func(c *config) { c.limits = &limits }
}

// New creates a DeepInfra encoder for the given native inference model.
//
// Examples:
//
//	encoder, err := deepinfra.New(deepinfra.BGEM3Model)
//	encoder, err := deepinfra.New(
//	    deepinfra.BGEM3Model,
//	    deepinfra.WithAPIToken("di-..."),
//	    deepinfra.WithModelRevision("5617a9f61b028005a4858fdac845db406aefb181", "xlm-roberta-250002"),
//	)
func New(model string, opts ...Option) (*Encoder, error) {
	if model == "" {
		return nil, &retrieval.InvalidRepresentationRequestError{
			Invariant: "model.empty",
			Detail:    "deepinfra: model cannot be empty",
		}
	}

	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}
	batchSize := defaultBatchSize
	if cfg.batchSize != nil {
		if *cfg.batchSize < 0 {
			return nil, &retrieval.InvalidRepresentationRequestError{
				Invariant: "batch_size.negative",
				Detail:    "deepinfra: batch size cannot be negative",
			}
		}
		batchSize = *cfg.batchSize
	}
	if cfg.sparseVocabulary < 0 {
		return nil, &retrieval.InvalidRepresentationRequestError{
			Invariant: "sparse_vocabulary.negative",
			Detail:    "deepinfra: sparse vocabulary cannot be negative",
		}
	}

	c, err := newClient(cfg.transportConfig)
	if err != nil {
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
			return nil, &retrieval.InvalidRepresentationRequestError{
				Invariant: "outputs.unknown",
				Detail:    "deepinfra: output kind " + string(kind) + " is not dense, sparse, or multi_vector",
			}
		}
	}

	limits := retrieval.DefaultRepresentationLimits()
	if cfg.limits != nil {
		limits = *cfg.limits
	}

	encoder := &Encoder{
		client:           c,
		model:            model,
		normalize:        cfg.normalize,
		batchSize:        batchSize,
		sparseVocabulary: cfg.sparseVocabulary,
		revision:         cfg.revision,
		tokenizer:        cfg.tokenizer,
		outputs:          outputs,
		limits:           limits,
	}
	// Built once here rather than per call, and left nil when the encoder was
	// narrowed to non-dense outputs, so Embed fails with the same typed error
	// a dense request would.
	if encoder.Capabilities().Supports(retrieval.RepresentationDense) {
		encoder.embedder, err = retrieval.NewEncoderEmbedder(encoder)
		if err != nil {
			return nil, err
		}
	}
	return encoder, nil
}

// MustNew is like New but panics on error.
func MustNew(model string, opts ...Option) *Encoder {
	e, err := New(model, opts...)
	if err != nil {
		panic(err)
	}
	return e
}

// Name implements retrieval.RepresentationEncoder and retrieval.Embedder.
func (e *Encoder) Name() string { return e.model }

// ModelMetadata reports semantic provider and transport identity.
func (e *Encoder) ModelMetadata() core.ModelMetadata {
	return core.ModelMetadataForEndpoint("deepinfra", "embeddings", e.baseURL)
}

// Capabilities implements retrieval.RepresentationEncoder.
//
// Truncation is not offered: DeepInfra's native inference API has no truncate
// parameter, and accepting the option while ignoring it would let a caller
// believe an over-long document had been rejected when it was silently
// clipped.
func (e *Encoder) Capabilities() retrieval.RepresentationCapabilities {
	return retrieval.RepresentationCapabilities{
		Outputs: append([]retrieval.RepresentationKind(nil), e.outputs...),
		InputTypes: []retrieval.EmbeddingInputType{
			retrieval.EmbeddingInputNone,
			retrieval.EmbeddingInputQuery,
			retrieval.EmbeddingInputDocument,
		},
		SupportsTruncation:  false,
		SupportsMultiOutput: true,
	}
}

// validator returns the contract checker used on both sides of a call.
func (e *Encoder) validator() retrieval.RepresentationValidator {
	return retrieval.RepresentationValidator{
		Provider:     providerName,
		Capabilities: e.Capabilities(),
		Limits:       e.limits,
	}
}

// space builds the descriptor for one kind at the observed width.
func (e *Encoder) space(kind retrieval.RepresentationKind, dimensions int) retrieval.VectorSpace {
	metric := retrieval.SimilarityCosine
	if kind == retrieval.RepresentationSparse {
		metric = retrieval.SimilarityDotProduct
	}
	return retrieval.VectorSpace{
		Provider:   providerName,
		Model:      e.model,
		Revision:   e.revision,
		Tokenizer:  e.tokenizer,
		Kind:       kind,
		Dimensions: dimensions,
		Metric:     metric,
	}.WithCanonicalID()
}

// Compile-time checks that Encoder satisfies both contracts.
var (
	_ retrieval.RepresentationEncoder = (*Encoder)(nil)
	_ retrieval.Embedder              = (*Encoder)(nil)
)
