package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/embedbatch"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsmiddleware "github.com/aws/aws-sdk-go-v2/aws/middleware"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	smithyhttp "github.com/aws/smithy-go/transport/http"
)

// inputTokenCountHeader is the response header Bedrock uses to report how many
// tokens an InvokeModel call consumed. Embedding model response bodies do not
// carry usage the way the Converse API does, so this header is the only
// consistent source across model families.
const inputTokenCountHeader = "x-amzn-bedrock-input-token-count"

// cohereBatchLimit is the maximum number of texts Cohere's embed models accept
// in one call — a long-standing, documented limit of the Cohere embed API.
// Requests larger than this are split by [embedbatch.Chunked].
//
// The standalone provider/cohere embedder does not chunk, deliberately: an
// oversized batch there returns Cohere's own {message} error body verbatim, so
// letting it 400 is self-diagnosing. On Bedrock the same overflow surfaces as
// an opaque ValidationException wrapped in AWS SDK error machinery, so chunking
// here trades that for requests that simply succeed.
const cohereBatchLimit = 96

// Model ID prefixes that identify an embedding model family, matched against
// the model ID after its geographic prefix has been stripped.
const (
	titanEmbedPrefix  = "amazon.titan-embed"
	cohereEmbedPrefix = "cohere.embed"
	novaEmbedPrefix   = "amazon.nova"
)

// geoPrefixes are the cross-region inference profile prefixes Bedrock accepts
// in front of a base model ID, as in "us.amazon.titan-embed-text-v2:0". They
// are stripped before family detection so a prefixed ID resolves to the same
// family as its bare form.
var geoPrefixes = []string{"us", "eu", "apac", "jp", "au", "ca", "global", "us-gov"}

// versionPattern extracts the major version from a Bedrock model ID, matching
// the "v2" in "amazon.titan-embed-text-v2:0" and the "v4" in "cohere.embed-v4:0".
var versionPattern = regexp.MustCompile(`v(\d+)`)

// embeddingFamily identifies which request and response shape a Bedrock
// embedding model speaks. Bedrock does not normalize these the way the Converse
// API normalizes chat, so each family needs its own encoder and decoder.
type embeddingFamily int

const (
	// familyTitan is the Amazon Titan family, which accepts exactly one text
	// per call and returns a single vector.
	familyTitan embeddingFamily = iota

	// familyCohere is the Cohere family hosted on Bedrock, which accepts a
	// batch of texts and requires an input type.
	familyCohere
)

// Embedder implements core.Embedder using the Bedrock Runtime InvokeModel API.
//
// Bedrock exposes each embedding vendor's native request and response body
// rather than a unified schema, so this type dispatches on the model family
// detected from the model ID. Amazon Titan and Cohere are supported; Amazon
// Nova embeddings are not.
type Embedder struct {
	client      invokeClient
	modelID     string
	family      embeddingFamily
	version     int
	concurrency int
	normalize   *bool
}

// invokeClient is the subset of the Bedrock Runtime client the Embedder uses.
// It exists so tests can supply a stub without reaching for the network.
type invokeClient interface {
	InvokeModel(
		ctx context.Context,
		params *bedrockruntime.InvokeModelInput,
		optFns ...func(*bedrockruntime.Options),
	) (*bedrockruntime.InvokeModelOutput, error)
}

// WithEmbeddingConcurrency caps how many requests run at once when a model
// accepts only one text per call, as Amazon Titan does. Values of zero or less
// leave the default in place.
func WithEmbeddingConcurrency(n int) Option {
	return func(c *config) { c.embedConcurrency = n }
}

// WithTitanNormalize sets the Titan normalize flag, which scales each vector to
// unit length so cosine similarity reduces to a dot product. It defaults to
// true, matching the model's own default, and is ignored by Titan v1, which has
// no such parameter.
func WithTitanNormalize(normalize bool) Option {
	return func(c *config) { c.titanNormalize = &normalize }
}

// NewEmbedder creates a new Bedrock Embedder. It accepts the same connection
// options as New, plus the embedding-specific WithEmbeddingConcurrency and
// WithTitanNormalize.
//
// The modelID should be a Bedrock embedding model ID, optionally carrying a
// cross-region inference prefix:
//   - "amazon.titan-embed-text-v2:0"
//   - "cohere.embed-english-v3"
//   - "us.cohere.embed-v4:0"
//
// Examples:
//
//	embedder, err := bedrock.NewEmbedder("amazon.titan-embed-text-v2:0",
//	    bedrock.WithRegion("us-east-1"),
//	)
//
//	embedder, err := bedrock.NewEmbedder("cohere.embed-english-v3",
//	    bedrock.WithRegion("us-east-1"),
//	    bedrock.WithProfile("my-profile"),
//	)
func NewEmbedder(modelID string, opts ...Option) (*Embedder, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	family, version, err := detectEmbeddingFamily(modelID)
	if err != nil {
		return nil, err
	}

	e := &Embedder{
		modelID:     modelID,
		family:      family,
		version:     version,
		concurrency: cfg.embedConcurrency,
		normalize:   cfg.titanNormalize,
	}

	if cfg.rawClient != nil {
		e.client = cfg.rawClient
		return e, nil
	}

	awsCfg, err := resolveAWSConfig(cfg)
	if err != nil {
		return nil, err
	}
	e.client = bedrockruntime.NewFromConfig(awsCfg)

	return e, nil
}

// MustNewEmbedder is like NewEmbedder but panics on error.
func MustNewEmbedder(modelID string, opts ...Option) *Embedder {
	e, err := NewEmbedder(modelID, opts...)
	if err != nil {
		panic(err)
	}
	return e
}

// stripGeoPrefix removes a cross-region inference profile prefix from a model
// ID, so that "us.cohere.embed-v4:0" resolves the same way "cohere.embed-v4:0"
// does. IDs without a known prefix are returned unchanged.
func stripGeoPrefix(modelID string) string {
	for _, prefix := range geoPrefixes {
		if after, ok := strings.CutPrefix(modelID, prefix+"."); ok {
			return after
		}
	}
	return modelID
}

// modelVersion extracts the major version from a model ID, returning zero when
// the ID carries none. Zero is treated as the oldest version of a family, so an
// unrecognized ID never has a newer version's parameters sent to it.
func modelVersion(modelID string) int {
	match := versionPattern.FindStringSubmatch(modelID)
	if match == nil {
		return 0
	}
	// The capture group is \d+, so it parses unless it overflows an int, in
	// which case zero is the right conservative answer anyway.
	version, err := strconv.Atoi(match[1])
	if err != nil {
		return 0
	}
	return version
}

// detectEmbeddingFamily resolves a model ID to the request and response shape
// it speaks, along with its major version.
//
// It rejects rather than guesses: an unknown ID sent to the wrong encoder would
// fail at the API with a message about a malformed body rather than about an
// unsupported model.
func detectEmbeddingFamily(modelID string) (embeddingFamily, int, error) {
	base := stripGeoPrefix(modelID)

	switch {
	case strings.HasPrefix(base, titanEmbedPrefix):
		return familyTitan, modelVersion(base), nil
	case strings.HasPrefix(base, cohereEmbedPrefix):
		return familyCohere, modelVersion(base), nil
	case strings.HasPrefix(base, novaEmbedPrefix):
		return 0, 0, fmt.Errorf("bedrock embeddings: %q is not supported; Nova embedding models take a different request shape and are out of scope", modelID)
	default:
		return 0, 0, fmt.Errorf("bedrock embeddings: unsupported model %q; supported prefixes are %q and %q", modelID, titanEmbedPrefix, cohereEmbedPrefix)
	}
}

// Embed implements core.Embedder, dispatching on the model family detected at
// construction.
//
// Model is reported as the configured model ID: Bedrock embedding responses do
// not echo one back.
func (e *Embedder) Embed(ctx context.Context, req *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	var (
		vectors [][]float32
		usage   core.EmbeddingUsage
		err     error
	)

	switch e.family {
	case familyTitan:
		vectors, usage, err = e.embedTitan(ctx, req)
	default:
		vectors, usage, err = e.embedCohere(ctx, req)
	}
	if err != nil {
		return nil, err
	}

	// A count mismatch would silently corrupt a caller's join between vectors
	// and source texts, so it fails the request instead.
	if len(vectors) != len(req.Input) {
		return nil, fmt.Errorf("bedrock embeddings: got %d vectors for %d inputs", len(vectors), len(req.Input))
	}

	return &core.EmbeddingResponse{
		Vectors: vectors,
		Model:   e.modelID,
		Usage:   usage,
	}, nil
}

// Name implements core.Embedder.
func (e *Embedder) Name() string {
	return e.modelID
}

// titanRequest is the Amazon Titan embedding request body. Titan accepts
// exactly one text per call.
type titanRequest struct {
	InputText  string `json:"inputText"`
	Dimensions int    `json:"dimensions,omitempty"`
	Normalize  *bool  `json:"normalize,omitempty"`
}

// titanResponse is the Amazon Titan embedding response body.
type titanResponse struct {
	Embedding           []float32 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
}

// embedTitan embeds each input with its own InvokeModel call, since Titan
// accepts a single text per request, and reassembles them in input order.
//
// InputType is ignored: Titan has no notion of query versus document text.
func (e *Embedder) embedTitan(ctx context.Context, req *core.EmbeddingRequest) ([][]float32, core.EmbeddingUsage, error) {
	// Titan has no truncation parameter and rejects over-length input rather
	// than clipping it, so Truncate=false and Truncate=nil both already match
	// the model's behavior. Truncate=true cannot be honored, and silently
	// ignoring it would store a vector covering only part of the input.
	if req.Truncate != nil && *req.Truncate {
		return nil, core.EmbeddingUsage{}, errors.New("bedrock embeddings: Titan does not support truncation; the model rejects over-length input instead of truncating it")
	}
	if req.Dimensions > 0 && e.version < 2 {
		return nil, core.EmbeddingUsage{}, fmt.Errorf("bedrock embeddings: %q has a fixed output dimension and does not accept the dimensions parameter", e.modelID)
	}

	return embedbatch.FanOut(ctx, req.Input, e.concurrency,
		func(ctx context.Context, text string) ([]float32, core.EmbeddingUsage, error) {
			body := titanRequest{InputText: text}
			if e.version >= 2 {
				body.Dimensions = req.Dimensions
				// Titan defaults normalize to true, and this restates it so
				// the request is explicit about a flag that changes what the
				// stored vectors mean.
				normalize := true
				if e.normalize != nil {
					normalize = *e.normalize
				}
				body.Normalize = &normalize
			}

			out, err := e.invoke(ctx, body)
			if err != nil {
				return nil, core.EmbeddingUsage{}, err
			}

			var resp titanResponse
			if err := json.Unmarshal(out.Body, &resp); err != nil {
				return nil, core.EmbeddingUsage{}, fmt.Errorf("bedrock embeddings: decode Titan response: %w", err)
			}
			if len(resp.Embedding) == 0 {
				return nil, core.EmbeddingUsage{}, errors.New("bedrock embeddings: Titan response carried no embedding")
			}

			// Titan is the one family that also reports its token count in the
			// body, which covers a response whose headers were dropped by a
			// proxy in front of Bedrock.
			tokens := inputTokenCount(out)
			if tokens == 0 {
				tokens = resp.InputTextTokenCount
			}

			return resp.Embedding, core.EmbeddingUsage{PromptTokens: tokens, TotalTokens: tokens}, nil
		})
}

// cohereRequest is the Cohere-on-Bedrock embedding request body. InputType is
// required by the model, unlike the standalone Cohere API where it is optional.
type cohereRequest struct {
	Texts           []string `json:"texts"`
	InputType       string   `json:"input_type"`
	Truncate        string   `json:"truncate,omitempty"`
	OutputDimension int      `json:"output_dimension,omitempty"`
}

// cohereResponse is the Cohere-on-Bedrock embedding response body. Embeddings
// is held raw because the model returns it in one of two shapes; see vectors.
type cohereResponse struct {
	Embeddings json.RawMessage `json:"embeddings"`
}

// vectors decodes the two shapes Cohere returns embeddings in: a bare array of
// vectors, or an object keyed by embedding type when the model echoes the
// embeddings-by-type form. Only float vectors are requested, so only the
// "float" key is read.
func (r *cohereResponse) vectors() ([][]float32, error) {
	if len(r.Embeddings) == 0 || string(r.Embeddings) == "null" {
		return nil, errors.New("bedrock embeddings: Cohere response carried no embeddings field")
	}

	var direct [][]float32
	if err := json.Unmarshal(r.Embeddings, &direct); err == nil {
		return direct, nil
	}

	var byType struct {
		Float [][]float32 `json:"float"`
	}
	if err := json.Unmarshal(r.Embeddings, &byType); err != nil {
		return nil, fmt.Errorf("bedrock embeddings: decode Cohere embeddings: %w", err)
	}
	if byType.Float == nil {
		return nil, errors.New("bedrock embeddings: Cohere response carried no float embeddings")
	}
	return byType.Float, nil
}

// embedCohere embeds inputs in batches, splitting anything over the model's
// per-call limit.
func (e *Embedder) embedCohere(ctx context.Context, req *core.EmbeddingRequest) ([][]float32, core.EmbeddingUsage, error) {
	if req.Dimensions > 0 && e.version < 4 {
		return nil, core.EmbeddingUsage{}, fmt.Errorf("bedrock embeddings: %q has a fixed output dimension and does not accept the dimensions parameter", e.modelID)
	}

	inputType := cohereInputType(req.InputType)
	truncate := cohereTruncate(req.Truncate)

	return embedbatch.Chunked(ctx, req.Input, cohereBatchLimit,
		func(ctx context.Context, chunk []string) ([][]float32, core.EmbeddingUsage, error) {
			body := cohereRequest{
				Texts:     chunk,
				InputType: inputType,
				Truncate:  truncate,
			}
			if e.version >= 4 {
				body.OutputDimension = req.Dimensions
			}

			out, err := e.invoke(ctx, body)
			if err != nil {
				return nil, core.EmbeddingUsage{}, err
			}

			var resp cohereResponse
			if err := json.Unmarshal(out.Body, &resp); err != nil {
				return nil, core.EmbeddingUsage{}, fmt.Errorf("bedrock embeddings: decode Cohere response: %w", err)
			}

			vectors, err := resp.vectors()
			if err != nil {
				return nil, core.EmbeddingUsage{}, err
			}
			if len(vectors) != len(chunk) {
				return nil, core.EmbeddingUsage{}, fmt.Errorf("bedrock embeddings: got %d vectors for %d inputs", len(vectors), len(chunk))
			}

			tokens := inputTokenCount(out)
			return vectors, core.EmbeddingUsage{PromptTokens: tokens, TotalTokens: tokens}, nil
		})
}

// cohereInputType maps our input type onto the value Cohere requires. Cohere
// makes the field mandatory, so the no-preference case has to pick something:
// it picks search_document, which is the neutral choice for the indexing side
// of a retrieval task and the more common bulk operation.
func cohereInputType(t core.EmbeddingInputType) string {
	if t == core.EmbeddingInputQuery {
		return "search_query"
	}
	return "search_document"
}

// cohereTruncate maps our truncation flag onto Cohere's three-valued field.
// Nil returns the empty string, which omits the field so the model's own
// default applies.
func cohereTruncate(truncate *bool) string {
	if truncate == nil {
		return ""
	}
	if *truncate {
		return "END"
	}
	return "NONE"
}

// invoke sends body as JSON to InvokeModel and returns the raw output.
func (e *Embedder) invoke(ctx context.Context, body any) (*bedrockruntime.InvokeModelOutput, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("bedrock embeddings: encode request: %w", err)
	}

	out, err := e.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(e.modelID),
		Body:        raw,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("bedrock embeddings: %w", err)
	}
	return out, nil
}

// inputTokenCount reads the token count Bedrock reports in a response header.
//
// InvokeModel installs the middleware that keeps the raw HTTP response on the
// operation metadata, so the header survives as far as here. A response that
// lost the header — through a proxy, or a stub in a test — counts as zero
// rather than as a fabricated number.
func inputTokenCount(out *bedrockruntime.InvokeModelOutput) int {
	return inputTokenCountFrom(awsmiddleware.GetRawResponse(out.ResultMetadata))
}

// inputTokenCountFrom reads the token count from a raw response value.
//
// It is split from inputTokenCount because the metadata key the raw response is
// stored under is unexported by the AWS SDK, so a test cannot build an
// InvokeModelOutput carrying one. Taking the raw value directly keeps every
// branch here reachable.
func inputTokenCountFrom(raw any) int {
	resp, ok := raw.(*smithyhttp.Response)
	if !ok || resp == nil || resp.Response == nil {
		return 0
	}

	count, err := strconv.Atoi(resp.Header.Get(inputTokenCountHeader))
	if err != nil || count < 0 {
		return 0
	}
	return count
}

// Compile-time check that Embedder implements core.Embedder.
var _ core.Embedder = (*Embedder)(nil)
