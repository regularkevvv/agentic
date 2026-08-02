package huggingface

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/regularkevvv/agentic/internal/providerhttp"
	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/internal/retrieval/batch"
)

// SharedEncoder calls the Inference Providers router's feature-extraction
// task, which returns one dense vector per input.
//
// It is dense-only and says so. The router normalizes every provider behind it
// onto one response shape — an array of vectors — and there is no field in
// that shape for lexical weights or token vectors. Advertising sparse support
// here because a model can produce it locally would be a capability claim with
// no response behind it.
type SharedEncoder struct {
	client *providerhttp.Client

	model      string
	routerURL  string
	provider   string
	batchSize  int
	normalize  *bool
	promptName map[retrieval.EmbeddingInputType]string
	space      retrieval.VectorSpace
	limits     retrieval.RepresentationLimits
	embedder   *retrieval.EncoderEmbedder
}

// SharedOption configures a SharedEncoder.
type SharedOption func(*sharedConfig)

type sharedConfig struct {
	providerhttp.Config

	routerURL  string
	provider   string
	batchSize  int
	normalize  *bool
	promptName map[retrieval.EmbeddingInputType]string
	space      retrieval.VectorSpace
	limits     *retrieval.RepresentationLimits
}

// WithSharedToken sets the access token. If not set, HF_TOKEN and then
// HUGGING_FACE_HUB_TOKEN are used.
func WithSharedToken(token string) SharedOption {
	return func(c *sharedConfig) { c.Token = token }
}

// WithSharedHTTPClient sets a custom HTTP client.
func WithSharedHTTPClient(client *http.Client) SharedOption {
	return func(c *sharedConfig) { c.HTTPClient = client }
}

// WithSharedMaxRetries sets how many times a request is retried on 429 and
// transient 5xx responses (default 2).
func WithSharedMaxRetries(retries int) SharedOption {
	return func(c *sharedConfig) { c.MaxRetries = &retries }
}

// WithSharedMaxResponseBytes caps the response body this encoder will read.
func WithSharedMaxResponseBytes(limit int64) SharedOption {
	return func(c *sharedConfig) { c.MaxResponseBytes = &limit }
}

// WithRouterURL overrides the Inference Providers router base URL.
func WithRouterURL(url string) SharedOption {
	return func(c *sharedConfig) { c.routerURL = url }
}

// WithInferenceProvider selects which provider behind the router serves the
// model (default "hf-inference").
func WithInferenceProvider(provider string) SharedOption {
	return func(c *sharedConfig) { c.provider = provider }
}

// WithSharedBatchSize splits requests larger than size into that many inputs
// per call.
func WithSharedBatchSize(size int) SharedOption {
	return func(c *sharedConfig) { c.batchSize = size }
}

// WithNormalize sets the router's normalize flag, which scales each vector to
// unit length. Leaving it unset uses the model's own default.
func WithNormalize(normalize bool) SharedOption {
	return func(c *sharedConfig) { c.normalize = &normalize }
}

// WithPromptNames maps the query and document input roles onto the model's
// configured sentence-transformers prompts.
//
// Only roles given a prompt name are advertised as supported. A model with no
// configured prompts has no query/document distinction, and accepting the role
// while sending nothing for it would let a caller believe an asymmetric
// encoding happened when it did not.
func WithPromptNames(query, document string) SharedOption {
	return func(c *sharedConfig) {
		c.promptName = map[retrieval.EmbeddingInputType]string{}
		if query != "" {
			c.promptName[retrieval.EmbeddingInputQuery] = query
		}
		if document != "" {
			c.promptName[retrieval.EmbeddingInputDocument] = document
		}
	}
}

// WithSharedVectorSpace pins the vector space, so that a revision and
// tokenizer can be recorded for a hosted model that reports neither.
//
// Only ID, Revision, and Tokenizer are taken from it; the width and metric are
// observed from the response and fixed by this package respectively.
func WithSharedVectorSpace(space retrieval.VectorSpace) SharedOption {
	return func(c *sharedConfig) { c.space = space }
}

// WithSharedLimits overrides the request and response size ceilings.
func WithSharedLimits(limits retrieval.RepresentationLimits) SharedOption {
	return func(c *sharedConfig) { c.limits = &limits }
}

// NewShared creates a dense-only encoder for a hosted feature-extraction
// model.
//
// Example:
//
//	encoder, err := huggingface.NewShared("intfloat/multilingual-e5-large-instruct")
func NewShared(model string, opts ...SharedOption) (*SharedEncoder, error) {
	if model == "" {
		return nil, errors.New("huggingface: model cannot be empty")
	}

	cfg := &sharedConfig{}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.batchSize < 0 {
		return nil, errors.New("huggingface: batch size cannot be negative")
	}

	c, err := newClient(cfg.Config)
	if err != nil {
		return nil, err
	}

	routerURL := cfg.routerURL
	if routerURL == "" {
		routerURL = defaultRouterURL
	}
	provider := cfg.provider
	if provider == "" {
		provider = "hf-inference"
	}
	limits := retrieval.DefaultRepresentationLimits()
	if cfg.limits != nil {
		limits = *cfg.limits
	}

	encoder := &SharedEncoder{
		client:     c,
		model:      model,
		routerURL:  strings.TrimSuffix(routerURL, "/"),
		provider:   provider,
		batchSize:  cfg.batchSize,
		normalize:  cfg.normalize,
		promptName: cfg.promptName,
		space:      cfg.space,
		limits:     limits,
	}
	encoder.embedder, err = retrieval.NewEncoderEmbedder(encoder)
	if err != nil {
		return nil, err
	}
	return encoder, nil
}

// Name implements retrieval.RepresentationEncoder and retrieval.Embedder.
func (e *SharedEncoder) Name() string { return e.model }

// Capabilities implements retrieval.RepresentationEncoder.
func (e *SharedEncoder) Capabilities() retrieval.RepresentationCapabilities {
	inputTypes := []retrieval.EmbeddingInputType{retrieval.EmbeddingInputNone}
	for _, role := range []retrieval.EmbeddingInputType{retrieval.EmbeddingInputQuery, retrieval.EmbeddingInputDocument} {
		if e.promptName[role] != "" {
			inputTypes = append(inputTypes, role)
		}
	}
	return retrieval.RepresentationCapabilities{
		Outputs:             []retrieval.RepresentationKind{retrieval.RepresentationDense},
		InputTypes:          inputTypes,
		SupportsTruncation:  true,
		SupportsMultiOutput: false,
	}
}

func (e *SharedEncoder) validator() retrieval.RepresentationValidator {
	return retrieval.RepresentationValidator{
		Provider:     providerName,
		Capabilities: e.Capabilities(),
		Limits:       e.limits,
	}
}

// featureExtractionRequest is the router's body for the feature-extraction
// task.
type featureExtractionRequest struct {
	Inputs     []string `json:"inputs"`
	Normalize  *bool    `json:"normalize,omitempty"`
	Truncate   *bool    `json:"truncate,omitempty"`
	PromptName string   `json:"prompt_name,omitempty"`
}

// Encode implements retrieval.RepresentationEncoder.
func (e *SharedEncoder) Encode(ctx context.Context, req *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, error) {
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

func (e *SharedEncoder) encodeChunk(ctx context.Context, req *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, error) {
	url := fmt.Sprintf("%s/%s/models/%s/pipeline/feature-extraction", e.routerURL, e.provider, e.model)

	payload, err := e.client.Post(ctx, url, featureExtractionRequest{
		Inputs:     req.Input,
		Normalize:  e.normalize,
		Truncate:   req.Truncate,
		PromptName: e.promptName[req.InputType],
	})
	if err != nil {
		return nil, err
	}

	var vectors [][]float32
	if err := json.Unmarshal(payload, &vectors); err != nil {
		return nil, &retrieval.InvalidRepresentationResponseError{
			Provider: providerName,
			Item:     -1,
			Kind:     retrieval.RepresentationDense,
			Problem: "feature-extraction response is not an array of vectors; " +
				"a model whose pipeline returns token-level output is not usable here",
		}
	}
	if len(vectors) != len(req.Input) {
		return nil, &retrieval.InvalidRepresentationResponseError{
			Provider: providerName,
			Item:     -1,
			Kind:     retrieval.RepresentationDense,
			Problem:  fmt.Sprintf("got %d vectors for %d inputs", len(vectors), len(req.Input)),
		}
	}

	data := make([]retrieval.Representation, len(vectors))
	inputBytes := 0
	for i, vec := range vectors {
		data[i] = retrieval.Representation{Dense: vec}
	}
	for _, text := range req.Input {
		inputBytes += len(text)
	}

	width := 0
	if len(vectors) > 0 {
		width = len(vectors[0])
	}

	return &retrieval.RepresentationResponse{
		Data:   data,
		Spaces: map[retrieval.RepresentationKind]retrieval.VectorSpace{retrieval.RepresentationDense: e.vectorSpace(width)},
		Model:  e.model,
		Usage: retrieval.RepresentationUsage{
			RequestCount: 1,
			InputBytes:   inputBytes,
			OutputBytes:  len(payload),
		},
	}, nil
}

// vectorSpace builds the descriptor at the observed width.
//
// The router reports no token usage and no model revision, so InputTokens
// stays zero rather than being estimated locally, and the revision is empty
// unless the caller pinned one.
func (e *SharedEncoder) vectorSpace(width int) retrieval.VectorSpace {
	space := retrieval.VectorSpace{
		ID:         e.space.ID,
		Provider:   providerName,
		Model:      e.model,
		Revision:   e.space.Revision,
		Tokenizer:  e.space.Tokenizer,
		Kind:       retrieval.RepresentationDense,
		Dimensions: width,
		Metric:     retrieval.SimilarityCosine,
	}
	return space.WithCanonicalID()
}

// Embed implements retrieval.Embedder.
func (e *SharedEncoder) Embed(ctx context.Context, req *retrieval.EmbeddingRequest) (*retrieval.EmbeddingResponse, error) {
	return e.embedder.Embed(ctx, req)
}

// Compile-time checks that SharedEncoder satisfies both contracts.
var (
	_ retrieval.RepresentationEncoder = (*SharedEncoder)(nil)
	_ retrieval.Embedder              = (*SharedEncoder)(nil)
)
