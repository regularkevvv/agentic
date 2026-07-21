package cohere

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/regularkevvv/agentic/internal/core"
)

// DefaultEmbeddingModel is the model used by the examples in this package and
// the one Cohere currently recommends for new retrieval work.
const DefaultEmbeddingModel = "embed-v4.0"

// InputType is Cohere's native input-type vocabulary. The /v2/embed API
// requires one on every request, so every core.EmbeddingInputType is mapped
// onto one of these before the call is made.
type InputType string

const (
	// InputTypeSearchQuery embeds the inputs as search queries.
	InputTypeSearchQuery InputType = "search_query"

	// InputTypeSearchDocument embeds the inputs as documents to be stored and
	// searched against.
	InputTypeSearchDocument InputType = "search_document"

	// InputTypeClassification embeds the inputs for use as features in a
	// classifier.
	InputTypeClassification InputType = "classification"

	// InputTypeClustering embeds the inputs for use in a clustering algorithm.
	InputTypeClustering InputType = "clustering"
)

// valid reports whether the input type is one Cohere accepts.
func (t InputType) valid() bool {
	switch t {
	case InputTypeSearchQuery, InputTypeSearchDocument, InputTypeClassification, InputTypeClustering:
		return true
	default:
		return false
	}
}

// Embedder implements core.Embedder using the Cohere /v2/embed API.
type Embedder struct {
	model            string
	client           *client
	defaultInputType InputType
	truncate         *bool
}

// Option configures the Cohere Embedder. It is deliberately distinct from
// RerankerOption so that a knob meaningful only to one endpoint cannot be
// passed to the constructor for the other.
type Option func(*embedConfig)

type embedConfig struct {
	apiKey           string
	baseURL          string
	httpClient       *http.Client
	maxRetries       *int
	defaultInputType InputType
	truncate         *bool
}

// WithAPIKey sets the API key. If not set, the CO_API_KEY environment variable
// is used, then COHERE_API_KEY.
func WithAPIKey(apiKey string) Option {
	return func(c *embedConfig) { c.apiKey = apiKey }
}

// WithBaseURL sets a custom base URL (default https://api.cohere.com). The API
// version is part of the request path, so the base URL must not include it.
func WithBaseURL(baseURL string) Option {
	return func(c *embedConfig) { c.baseURL = baseURL }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(httpClient *http.Client) Option {
	return func(c *embedConfig) { c.httpClient = httpClient }
}

// WithMaxRetries sets how many times a request is retried on 429 and 5xx
// responses (default 2, matching the SDK-backed providers).
func WithMaxRetries(retries int) Option {
	return func(c *embedConfig) { c.maxRetries = &retries }
}

// WithDefaultInputType overrides the input type sent for requests that carry
// core.EmbeddingInputNone.
//
// The /v2/embed API requires an input type, but core.EmbeddingInputNone means
// "no task instruction", which has no Cohere equivalent — so a choice has to be
// made on the caller's behalf. The default is InputTypeSearchDocument, on the
// reasoning that an untyped batch is far more often a corpus being indexed than
// a set of queries, and that mistyping a document as a query pushes it away
// from the queries that should retrieve it. Callers embedding untyped text for
// clustering or classification should set this explicitly.
func WithDefaultInputType(inputType InputType) Option {
	return func(c *embedConfig) { c.defaultInputType = inputType }
}

// WithTruncation sets the default for inputs longer than the model's context
// length: true truncates them at the end, false rejects the request with an
// error.
//
// This is only a default. A per-call core.EmbeddingRequest.Truncate takes
// precedence over it. If neither is set the field is omitted and the Cohere
// API's own default applies, which is to truncate — prefer WithTruncation(false)
// when indexing, because storing the vector of a clipped document as if it
// covered the whole document is a quiet, expensive retrieval failure.
func WithTruncation(truncate bool) Option {
	return func(c *embedConfig) { c.truncate = &truncate }
}

// New creates a new Cohere Embedder.
//
// Examples:
//
//	embedder, err := cohere.New("embed-v4.0", cohere.WithAPIKey("..."))
//	embedder, err := cohere.New("embed-multilingual-v3.0") // key from CO_API_KEY
func New(model string, opts ...Option) (*Embedder, error) {
	cfg := &embedConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	if model == "" {
		return nil, fmt.Errorf("cohere: model cannot be empty")
	}

	defaultInputType := cfg.defaultInputType
	if defaultInputType == "" {
		defaultInputType = InputTypeSearchDocument
	}
	if !defaultInputType.valid() {
		return nil, fmt.Errorf("cohere: unknown default input type %q", cfg.defaultInputType)
	}

	c, err := newClient(cfg.apiKey, cfg.baseURL, cfg.httpClient, cfg.maxRetries)
	if err != nil {
		return nil, err
	}

	return &Embedder{
		model:            model,
		client:           c,
		defaultInputType: defaultInputType,
		truncate:         cfg.truncate,
	}, nil
}

// MustNew is like New but panics on error.
func MustNew(model string, opts ...Option) *Embedder {
	e, err := New(model, opts...)
	if err != nil {
		panic(err)
	}
	return e
}

type embedRequest struct {
	Model           string   `json:"model"`
	Texts           []string `json:"texts"`
	InputType       string   `json:"input_type"`
	EmbeddingTypes  []string `json:"embedding_types"`
	OutputDimension int      `json:"output_dimension,omitempty"`
	Truncate        string   `json:"truncate,omitempty"`
}

// embedResponse declares only the fields this package maps. Cohere also
// returns an id and, when asked, the echoed texts; core.EmbeddingResponse has
// nowhere to put either, so they are left undecoded.
type embedResponse struct {
	Embeddings struct {
		Float [][]float32 `json:"float"`
	} `json:"embeddings"`
	Meta meta `json:"meta"`
}

// Embed implements core.Embedder.
//
// The request's InputType is mapped onto Cohere's required input_type:
// EmbeddingInputQuery becomes "search_query", EmbeddingInputDocument becomes
// "search_document", and EmbeddingInputNone becomes the Embedder's default
// (InputTypeSearchDocument unless WithDefaultInputType says otherwise).
//
// No per-call batch cap is enforced here. Cohere documents a limit on texts per
// request, but this package deliberately does not hard-code a figure it cannot
// cite from a source it has verified; guessing one would reject batches the API
// would have accepted. An oversized batch is left to the API's own 400, whose
// message field is included verbatim in the returned error, which makes a limit
// violation self-diagnosing.
func (e *Embedder) Embed(ctx context.Context, req *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	body := embedRequest{
		Model:     e.model,
		Texts:     req.Input,
		InputType: string(e.resolveInputType(req.InputType)),
		// Always request float embeddings. The response is a map keyed by
		// embedding type, and pinning it to a single known key sidesteps the
		// Cohere SDK deserialization bug that pydantic-ai pins around for the
		// same reason (embeddings/cohere.py:187): let the API pick the set of
		// types and a caller can get back a body whose vectors are somewhere
		// this code does not look.
		EmbeddingTypes:  []string{"float"},
		OutputDimension: req.Dimensions,
		Truncate:        e.resolveTruncate(req.Truncate),
	}

	payload, err := e.client.post(ctx, "/v2/embed", body)
	if err != nil {
		return nil, err
	}

	var resp embedResponse
	if err := json.Unmarshal(payload, &resp); err != nil {
		return nil, fmt.Errorf("cohere embeddings: decode response: %w", err)
	}

	// Cohere's embed response carries no per-vector index field: the vectors
	// come back positionally, aligned with the texts that were sent. There is
	// therefore nothing to reorder by, and the index-keyed placement used for
	// OpenAI and Voyage AI would have no input here. The length assertion below
	// is the only guard available, and it matters — a short or long list means
	// the positional alignment is already wrong, and returning it would corrupt
	// the caller's join between vectors and source texts.
	vectors := resp.Embeddings.Float
	if len(vectors) != len(req.Input) {
		return nil, fmt.Errorf("cohere embeddings: got %d vectors for %d inputs", len(vectors), len(req.Input))
	}

	return &core.EmbeddingResponse{
		Vectors: vectors,
		// Cohere does not echo the model back, so the configured name is
		// reported.
		Model: e.model,
		Usage: e.usage(resp.Meta),
	}, nil
}

// Name implements core.Embedder.
func (e *Embedder) Name() string {
	return e.model
}

// usage maps Cohere's meta block onto core.EmbeddingUsage. Billed units are
// preferred because they are what the caller pays for; the tokens block is used
// as a fallback for responses (and proxies) that report only consumption.
func (e *Embedder) usage(m meta) core.EmbeddingUsage {
	tokens := m.BilledUnits.InputTokens
	if tokens == 0 {
		tokens = m.Tokens.InputTokens
	}
	return core.EmbeddingUsage{PromptTokens: tokens, TotalTokens: tokens}
}

// resolveInputType maps our input type onto Cohere's required one, falling back
// to the configured default when the caller expressed no preference.
func (e *Embedder) resolveInputType(requested core.EmbeddingInputType) InputType {
	switch requested {
	case core.EmbeddingInputQuery:
		return InputTypeSearchQuery
	case core.EmbeddingInputDocument:
		return InputTypeSearchDocument
	default:
		return e.defaultInputType
	}
}

// resolveTruncate picks the truncation strategy to send on the wire, returning
// the empty string to omit the field.
//
// Precedence is per-request over constructor: a caller who says "reject this
// document rather than clip it" for one indexing call must win over an Embedder
// built with the opposite default.
func (e *Embedder) resolveTruncate(requested *bool) string {
	truncate := requested
	if truncate == nil {
		truncate = e.truncate
	}
	if truncate == nil {
		return ""
	}
	if *truncate {
		return "END"
	}
	return "NONE"
}

// Compile-time check that Embedder implements core.Embedder.
var _ core.Embedder = (*Embedder)(nil)
