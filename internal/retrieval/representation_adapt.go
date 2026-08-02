package retrieval

import (
	"context"
	"errors"
	"fmt"
)

// EmbedderEncoder presents an existing dense [Embedder] as a
// [RepresentationEncoder] that produces dense output only.
//
// It exists so that adding the encoder interface is not a flag day. A consumer
// can write one code path against RepresentationEncoder and still pass it an
// OpenAI, Voyage, Cohere, Gemini, Ollama, or Bedrock embedder, without those
// providers gaining a sparse capability they do not have.
type EmbedderEncoder struct {
	embedder Embedder
	space    VectorSpace
}

// NewEmbedderEncoder adapts a dense Embedder to the encoder contract.
//
// The caller supplies the vector space because an Embedder cannot prove one:
// it reports a model name and nothing about weights revision or tokenizer, and
// this package will not manufacture stability it cannot observe. Provider is
// required; Model defaults to the embedder's name, Kind to dense, and Metric
// to cosine. Leaving Dimensions zero defers the width to the first response
// and records the observed value, which is an observation rather than a claim.
// Leaving ID empty derives the canonical ID from the material fields.
func NewEmbedderEncoder(embedder Embedder, space VectorSpace) (*EmbedderEncoder, error) {
	if embedder == nil {
		return nil, errors.New("core: embedder cannot be nil")
	}
	if space.Provider == "" {
		return nil, errors.New("core: vector space provider is required to adapt an embedder")
	}
	if space.Model == "" {
		space.Model = embedder.Name()
	}
	if space.Model == "" {
		return nil, errors.New("core: vector space model is required when the embedder reports no name")
	}
	switch space.Kind {
	case "":
		space.Kind = RepresentationDense
	case RepresentationDense:
	default:
		return nil, fmt.Errorf("core: an embedder produces dense output, not %q", string(space.Kind))
	}
	if space.Metric == "" {
		space.Metric = SimilarityCosine
	}
	if !space.Metric.Valid() {
		return nil, fmt.Errorf("core: vector space metric %q is not a known similarity metric", string(space.Metric))
	}
	if space.Dimensions < 0 {
		return nil, errors.New("core: vector space dimensions cannot be negative")
	}
	return &EmbedderEncoder{embedder: embedder, space: space}, nil
}

// Name implements [RepresentationEncoder].
func (e *EmbedderEncoder) Name() string { return e.embedder.Name() }

// Capabilities implements [RepresentationEncoder]. The adapter reports dense
// output only, whatever the underlying model can do when driven natively.
func (e *EmbedderEncoder) Capabilities() RepresentationCapabilities {
	return RepresentationCapabilities{
		Outputs:            []RepresentationKind{RepresentationDense},
		InputTypes:         []EmbeddingInputType{EmbeddingInputNone, EmbeddingInputQuery, EmbeddingInputDocument},
		SupportsTruncation: true,
	}
}

// Encode implements [RepresentationEncoder].
func (e *EmbedderEncoder) Encode(ctx context.Context, req *RepresentationRequest) (*RepresentationResponse, error) {
	validator := RepresentationValidator{
		Provider:     e.space.Provider,
		Capabilities: e.Capabilities(),
		Limits:       DefaultRepresentationLimits(),
	}
	if err := validator.ValidateRequest(req); err != nil {
		return nil, err
	}

	embedReq := &EmbeddingRequest{
		Input:     req.Input,
		InputType: req.InputType,
		Truncate:  req.Truncate,
	}
	embedResp, err := e.embedder.Embed(ctx, embedReq)
	if err != nil {
		return nil, err
	}
	if embedResp == nil {
		return nil, &InvalidRepresentationResponseError{
			Provider: e.space.Provider,
			Item:     -1,
			Kind:     RepresentationDense,
			Problem:  "embedder returned no response",
		}
	}

	space, err := e.resolveSpace(embedResp.Vectors)
	if err != nil {
		return nil, err
	}

	data := make([]Representation, len(embedResp.Vectors))
	for i, vec := range embedResp.Vectors {
		data[i] = Representation{Dense: vec}
	}

	inputBytes := 0
	for _, text := range req.Input {
		inputBytes += len(text)
	}

	model := embedResp.Model
	if model == "" {
		model = e.embedder.Name()
	}

	resp := &RepresentationResponse{
		Data:   data,
		Spaces: map[RepresentationKind]VectorSpace{RepresentationDense: space},
		Model:  model,
		Usage: RepresentationUsage{
			InputTokens:  embeddingInputTokens(embedResp.Usage),
			RequestCount: 1,
			InputBytes:   inputBytes,
		},
	}
	if err := validator.ValidateResponse(req, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// resolveSpace fills in the width the constructor deferred, then assigns the
// canonical ID. Deferring means the ID depends on the observed width, which is
// the point: a model silently reconfigured to a different output dimension
// lands in a different space rather than corrupting the old one.
func (e *EmbedderEncoder) resolveSpace(vectors [][]float32) (VectorSpace, error) {
	space := e.space
	if space.Dimensions == 0 {
		if len(vectors) == 0 || len(vectors[0]) == 0 {
			return VectorSpace{}, &InvalidRepresentationResponseError{
				Provider: e.space.Provider,
				Item:     -1,
				Kind:     RepresentationDense,
				Problem:  "embedder returned no vector to infer the space width from",
			}
		}
		space.Dimensions = len(vectors[0])
	}
	return space.WithCanonicalID(), nil
}

// embeddingInputTokens picks the token count a provider actually reported.
// Voyage AI fills only TotalTokens; OpenAI fills both.
func embeddingInputTokens(usage EmbeddingUsage) int {
	if usage.PromptTokens > 0 {
		return usage.PromptTokens
	}
	return usage.TotalTokens
}

// EncoderEmbedder projects a [RepresentationEncoder] onto the existing dense
// [Embedder] contract, requesting dense output only.
//
// This is the direction that lets an application already written against
// Embedder adopt a multi-output provider without changing anything, and adopt
// its sparse output later when the index is ready for it.
type EncoderEmbedder struct {
	encoder RepresentationEncoder
}

// NewEncoderEmbedder adapts an encoder to the Embedder contract. It fails when
// the encoder cannot produce dense output, rather than at the first call.
func NewEncoderEmbedder(encoder RepresentationEncoder) (*EncoderEmbedder, error) {
	if encoder == nil {
		return nil, errors.New("core: encoder cannot be nil")
	}
	caps := encoder.Capabilities()
	if !caps.Supports(RepresentationDense) {
		return nil, &UnsupportedRepresentationError{
			Provider:  encoder.Name(),
			Kind:      RepresentationDense,
			Supported: caps.Outputs,
		}
	}
	return &EncoderEmbedder{encoder: encoder}, nil
}

// Name implements [Embedder].
func (e *EncoderEmbedder) Name() string { return e.encoder.Name() }

// Embed implements [Embedder].
//
// EmbeddingRequest.Dimensions is rejected rather than ignored. The encoder
// contract has no dimension override — a space's width is part of its identity
// — and silently returning full-width vectors to a caller who asked for
// truncated ones would produce an index nobody can query.
func (e *EncoderEmbedder) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if req.Dimensions > 0 {
		return nil, &InvalidRepresentationRequestError{
			Invariant: "dimensions.unsupported",
			Detail: fmt.Sprintf("%s encodes into a fixed vector space and cannot honor a %d-dimension override",
				e.encoder.Name(), req.Dimensions),
		}
	}

	resp, err := e.encoder.Encode(ctx, &RepresentationRequest{
		Input:     req.Input,
		InputType: req.InputType,
		Outputs:   []RepresentationKind{RepresentationDense},
		Truncate:  req.Truncate,
	})
	if err != nil {
		return nil, err
	}

	vectors := make([][]float32, len(resp.Data))
	for i, item := range resp.Data {
		vectors[i] = item.Dense
	}
	return &EmbeddingResponse{
		Vectors: vectors,
		Model:   resp.Model,
		Usage: EmbeddingUsage{
			PromptTokens: resp.Usage.InputTokens,
			TotalTokens:  resp.Usage.InputTokens,
		},
	}, nil
}

// Compile-time checks that the adapters satisfy both contracts.
var (
	_ RepresentationEncoder = (*EmbedderEncoder)(nil)
	_ Embedder              = (*EncoderEmbedder)(nil)
)
