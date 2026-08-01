// Package pinecone provides a Pinecone Inference implementation of
// core.RepresentationEncoder, and of core.Embedder for its dense models.
//
// The scope is encoding only. Pinecone's standalone /embed endpoint is a
// hosted model API and belongs here; Pinecone indexes, namespaces, and search
// belong in a store adapter on the consumer's side. An integrated database API
// is classified by the operation it performs, not by the vendor name.
//
// # Declaring what a model produces
//
// A Pinecone model returns one vector type, dense or sparse, and the package
// does not guess which. Declare it with [WithOutputs]; the default is dense,
// which is what most of Pinecone's hosted models are. The response carries its
// own vector_type, and a mismatch with what was declared is an error rather
// than a silent reinterpretation.
//
// A sparse model additionally needs [WithSparseVocabulary]. Pinecone does not
// report the vocabulary size, and the bound every sparse index must fall
// within is not something this package will invent — measure it against the
// model you are deploying and pass it.
//
// # Input roles and expansion
//
// Query and document roles are distinct here and map onto Pinecone's query and
// passage input types. They encode into the same vocabulary, which is what
// makes a query comparable to an indexed passage, so the role is not part of
// the space identity.
//
// Pinecone's sparse models expand a query beyond its literal terms. That
// expansion is observable with [WithReturnTokens] and
// [Encoder.EncodeWithTokens], which return the token strings behind each
// coordinate. Use them for diagnostics only: the stable storage identity is
// the vocabulary index and the model revision, not the token string.
package pinecone

import (
	"net/http"

	"github.com/regularkevvv/agentic/internal/core"
)

// providerName identifies this package in vector spaces and errors.
const providerName = "pinecone"

// SparseEnglishModel is Pinecone's hosted learned-sparse English model. It is
// a named constant rather than a default so that the model a deployment
// encodes with is always visible at the call site.
const SparseEnglishModel = "pinecone-sparse-english-v0"

// Encoder implements core.RepresentationEncoder against Pinecone Inference.
type Encoder struct {
	*client

	model            string
	kind             core.RepresentationKind
	dimensions       int
	sparseVocabulary int
	revision         string
	tokenizer        string
	returnTokens     bool
	batchSize        int
	limits           core.RepresentationLimits
	embedder         *core.EncoderEmbedder
}

// Option configures the Encoder.
type Option func(*config)

type config struct {
	transportConfig

	kind             core.RepresentationKind
	dimensions       int
	sparseVocabulary int
	revision         string
	tokenizer        string
	returnTokens     bool
	batchSize        int
	limits           *core.RepresentationLimits
}

// WithAPIKey sets the API key. If not set, the PINECONE_API_KEY env var is
// used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithBaseURL sets a custom base URL (default https://api.pinecone.io).
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// WithAPIVersion overrides the dated API version header (default 2025-04).
func WithAPIVersion(version string) Option {
	return func(c *config) { c.apiVersion = version }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}

// WithMaxRetries sets how many times a request is retried on 429 and transient
// 5xx responses (default 2).
func WithMaxRetries(retries int) Option {
	return func(c *config) { c.maxRetries = &retries }
}

// WithMaxResponseBytes caps the response body this encoder will read.
func WithMaxResponseBytes(limit int64) Option {
	return func(c *config) { c.maxResponseBytes = &limit }
}

// WithOutputs declares the representation kind this model produces. Exactly
// one kind may be declared, because one Pinecone model returns one vector
// type. The default is dense.
func WithOutputs(kinds ...core.RepresentationKind) Option {
	return func(c *config) {
		if len(kinds) == 1 {
			c.kind = kinds[0]
			return
		}
		// Recorded as an invalid kind so New reports it rather than silently
		// keeping the default.
		c.kind = core.RepresentationKind("multiple")
	}
}

// WithDimensions declares a dense model's output width, for models that allow
// one to be chosen. Leaving it zero uses the width observed in the response.
func WithDimensions(dimensions int) Option {
	return func(c *config) { c.dimensions = dimensions }
}

// WithSparseVocabulary declares the vocabulary size of a sparse model, which
// is the logical width of its space and the bound every index must fall
// within. It is required for a sparse model.
func WithSparseVocabulary(size int) Option {
	return func(c *config) { c.sparseVocabulary = size }
}

// WithModelRevision records the immutable model and tokenizer revisions in the
// vector space.
//
// Pinecone does not return either, and this package will not invent them.
// Supplying them is what lets a consumer detect that the model behind a name
// changed, instead of quietly mixing two generations of vectors in one index.
func WithModelRevision(model, tokenizer string) Option {
	return func(c *config) {
		c.revision = model
		c.tokenizer = tokenizer
	}
}

// WithReturnTokens asks Pinecone to return the token string behind each sparse
// coordinate.
//
// The tokens are diagnostics, reachable through [Encoder.EncodeWithTokens].
// They are not the storage identity: a tokenizer change can keep a token
// string and move its coordinate, or keep the coordinate and change the
// string, so an index keyed on strings is keyed on the wrong thing.
func WithReturnTokens(returnTokens bool) Option {
	return func(c *config) { c.returnTokens = returnTokens }
}

// WithBatchSize splits requests larger than size into that many inputs per
// call, preserving order and summing usage.
func WithBatchSize(size int) Option {
	return func(c *config) { c.batchSize = size }
}

// WithLimits overrides the request and response size ceilings.
func WithLimits(limits core.RepresentationLimits) Option {
	return func(c *config) { c.limits = &limits }
}

// New creates a Pinecone Inference encoder for the given hosted model.
//
// Examples:
//
//	encoder, err := pinecone.New("llama-text-embed-v2")
//
//	encoder, err := pinecone.New(pinecone.SparseEnglishModel,
//	    pinecone.WithOutputs(agentic.RepresentationSparse),
//	    pinecone.WithSparseVocabulary(measuredVocabularySize),
//	)
func New(model string, opts ...Option) (*Encoder, error) {
	if model == "" {
		return nil, &core.InvalidRepresentationRequestError{
			Invariant: "model.empty",
			Detail:    "pinecone: model cannot be empty",
		}
	}

	cfg := &config{kind: core.RepresentationDense}
	for _, opt := range opts {
		opt(cfg)
	}

	switch cfg.kind {
	case core.RepresentationDense, core.RepresentationSparse:
	case core.RepresentationMultiVector:
		return nil, &core.UnsupportedRepresentationError{
			Provider: providerName,
			Kind:     core.RepresentationMultiVector,
			Supported: []core.RepresentationKind{
				core.RepresentationDense, core.RepresentationSparse,
			},
		}
	default:
		return nil, &core.InvalidRepresentationRequestError{
			Invariant: "outputs.single",
			Detail: "pinecone: a model returns one vector type; declare exactly one of " +
				"dense or sparse with WithOutputs",
		}
	}

	if cfg.kind == core.RepresentationSparse && cfg.sparseVocabulary <= 0 {
		return nil, &core.InvalidRepresentationRequestError{
			Invariant: "sparse_vocabulary.missing",
			Detail: "pinecone: a sparse model needs its vocabulary size; Pinecone does not " +
				"report one, so measure it and pass WithSparseVocabulary",
		}
	}
	if cfg.dimensions < 0 {
		return nil, &core.InvalidRepresentationRequestError{
			Invariant: "dimensions.negative",
			Detail:    "pinecone: dimensions cannot be negative",
		}
	}
	if cfg.batchSize < 0 {
		return nil, &core.InvalidRepresentationRequestError{
			Invariant: "batch_size.negative",
			Detail:    "pinecone: batch size cannot be negative",
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

	encoder := &Encoder{
		client:           c,
		model:            model,
		kind:             cfg.kind,
		dimensions:       cfg.dimensions,
		sparseVocabulary: cfg.sparseVocabulary,
		revision:         cfg.revision,
		tokenizer:        cfg.tokenizer,
		returnTokens:     cfg.returnTokens,
		batchSize:        cfg.batchSize,
		limits:           limits,
	}
	if cfg.kind == core.RepresentationDense {
		encoder.embedder, err = core.NewEncoderEmbedder(encoder)
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

// Name implements core.RepresentationEncoder and core.Embedder.
func (e *Encoder) Name() string { return e.model }

// Capabilities implements core.RepresentationEncoder.
//
// Multi-output is not supported: one Pinecone model returns one vector type,
// so a caller asking for two kinds is asking for two models.
func (e *Encoder) Capabilities() core.RepresentationCapabilities {
	return core.RepresentationCapabilities{
		Outputs: []core.RepresentationKind{e.kind},
		InputTypes: []core.EmbeddingInputType{
			core.EmbeddingInputNone,
			core.EmbeddingInputQuery,
			core.EmbeddingInputDocument,
		},
		SupportsTruncation:  true,
		SupportsMultiOutput: false,
	}
}

func (e *Encoder) validator() core.RepresentationValidator {
	return core.RepresentationValidator{
		Provider:     providerName,
		Capabilities: e.Capabilities(),
		Limits:       e.limits,
	}
}

// space builds the descriptor at the observed or declared width.
func (e *Encoder) space(dimensions int) core.VectorSpace {
	metric := core.SimilarityCosine
	if e.kind == core.RepresentationSparse {
		metric = core.SimilarityDotProduct
		dimensions = e.sparseVocabulary
	}
	return core.VectorSpace{
		Provider:   providerName,
		Model:      e.model,
		Revision:   e.revision,
		Tokenizer:  e.tokenizer,
		Kind:       e.kind,
		Dimensions: dimensions,
		Metric:     metric,
	}.WithCanonicalID()
}

// Compile-time check that Encoder satisfies the encoder contract. Embedder is
// satisfied too, but only a dense-configured encoder answers it.
var (
	_ core.RepresentationEncoder = (*Encoder)(nil)
	_ core.Embedder              = (*Encoder)(nil)
)
