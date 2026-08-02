package gemini

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/internal/retrieval/embedbatch"

	"google.golang.org/genai"
)

// DefaultEmbeddingModel is the model NewEmbedder uses when the caller passes
// an empty model name.
const DefaultEmbeddingModel = "gemini-embedding-001"

// Gemini task types. Passing one to WithEmbeddingTaskType conditions the
// vectors on that task; the model places a query near the documents that
// answer it rather than near other similarly-worded queries.
//
// [retrieval.EmbeddingRequest.InputType] can only express the retrieval pair, so
// the other six are reachable only through WithEmbeddingTaskType.
const (
	// TaskTypeRetrievalQuery embeds a search query. Mapped from
	// [retrieval.EmbeddingInputQuery].
	TaskTypeRetrievalQuery = "RETRIEVAL_QUERY"

	// TaskTypeRetrievalDocument embeds a document to be indexed and searched
	// against. Mapped from [retrieval.EmbeddingInputDocument].
	TaskTypeRetrievalDocument = "RETRIEVAL_DOCUMENT"

	// TaskTypeSemanticSimilarity embeds text for symmetric similarity
	// comparison, where both sides play the same role.
	TaskTypeSemanticSimilarity = "SEMANTIC_SIMILARITY"

	// TaskTypeClassification embeds text to be assigned to predefined
	// categories.
	TaskTypeClassification = "CLASSIFICATION"

	// TaskTypeClustering embeds text to be grouped by similarity.
	TaskTypeClustering = "CLUSTERING"

	// TaskTypeQuestionAnswering embeds a question to be matched against
	// passages that answer it.
	TaskTypeQuestionAnswering = "QUESTION_ANSWERING"

	// TaskTypeFactVerification embeds a claim to be matched against evidence
	// that supports or refutes it.
	TaskTypeFactVerification = "FACT_VERIFICATION"

	// TaskTypeCodeRetrievalQuery embeds a natural-language query to be
	// matched against code.
	TaskTypeCodeRetrievalQuery = "CODE_RETRIEVAL_QUERY"
)

// WithEmbeddingTaskType pins every request to a specific Gemini task type,
// reaching the task types [retrieval.EmbeddingRequest.InputType] cannot name
// (clustering, classification, fact verification, and the rest).
//
// A request that sets InputType still wins over this option, matching the
// precedence the other providers in this tree use: a per-request setting
// overrides a constructor default rather than being silently dropped. Leave
// InputType empty on every request when you want this task type to apply.
//
// The value is sent verbatim, so a task type this SDK predates still works;
// an unknown one is rejected by the API rather than here.
func WithEmbeddingTaskType(taskType string) Option {
	return func(c *config) { c.taskType = taskType }
}

// Embedder implements [retrieval.Embedder] using the Gemini embeddings API, via
// either the Gemini API or Vertex AI.
type Embedder struct {
	client *genai.Client
	model  string

	// taskType is the constructor-level default from WithEmbeddingTaskType,
	// empty when unset.
	taskType string

	// vertexAI records which backend the client talks to. Several embedding
	// behaviors (token statistics, auto-truncation, the one-input-per-call
	// limit) exist only on Vertex.
	vertexAI bool

	// singleInput is true when the backend accepts exactly one input per
	// call, in which case Embed fans out. See newEmbedderSingleInput.
	singleInput bool
}

// NewEmbedder creates a Gemini Embedder. It accepts the same options as New,
// plus WithEmbeddingTaskType. An empty model name selects
// [DefaultEmbeddingModel].
//
// Examples:
//
//	// Gemini API
//	embedder, err := gemini.NewEmbedder("gemini-embedding-001", gemini.WithAPIKey("..."))
//
//	// Vertex AI, pinned to a task InputType cannot express
//	embedder, err := gemini.NewEmbedder("gemini-embedding-001",
//	    gemini.WithVertexAI("my-project", "us-central1"),
//	    gemini.WithEmbeddingTaskType(gemini.TaskTypeClustering),
//	)
//
// The gemini-embedding-2 family is rejected. Those models are conditioned by
// a task instruction prepended to the input text, not by the taskType field
// this Embedder sets, so sending them a RETRIEVAL_* taskType leaves the text
// unconditioned and degrades retrieval quality — silently, since the API
// accepts the ignored field without complaint. Refusing at construction is
// preferable to returning quietly worse vectors.
func NewEmbedder(model string, opts ...Option) (*Embedder, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	if model == "" {
		model = DefaultEmbeddingModel
	}
	if isTaskPrefixModel(model) {
		return nil, fmt.Errorf("gemini embeddings: model %q is conditioned by a text prefix rather than by a task type, which this Embedder cannot express; use %s", model, DefaultEmbeddingModel)
	}

	client, err := newGenAIClient(cfg)
	if err != nil {
		return nil, err
	}

	return &Embedder{
		client:      client,
		model:       model,
		taskType:    cfg.taskType,
		vertexAI:    cfg.vertexAI,
		singleInput: newEmbedderSingleInput(cfg.vertexAI, model),
	}, nil
}

// MustNewEmbedder is like NewEmbedder but panics on error.
func MustNewEmbedder(model string, opts ...Option) *Embedder {
	e, err := NewEmbedder(model, opts...)
	if err != nil {
		panic(err)
	}
	return e
}

// isTaskPrefixModel reports whether a model conditions on a task through a
// text prefix instead of the taskType field.
//
// This matches the whole gemini-embedding-2 family, which is deliberately
// broader than the upstream reference: pydantic-ai compares against the exact
// string "gemini-embedding-2" and so still sends a taskType to
// "gemini-embedding-2-preview", a model its own name list documents as
// prefix-conditioned. Matching the family avoids reproducing that gap. The
// cost of being too broad is a clear error at construction; the cost of being
// too narrow is silently worse retrieval, so the failure modes are not
// symmetric.
func isTaskPrefixModel(model string) bool {
	return strings.HasPrefix(model, "gemini-embedding-2")
}

// newEmbedderSingleInput reports whether the backend caps a call at one input,
// which forces Embed to fan a batch out into one request per text.
//
// Vertex routes every Gemini embedding model except gemini-embedding-001, and
// every open-source MaaS model, through an embedContent path that rejects more
// than one content outright (genai models.go EmbedContent). This mirrors the
// SDK's own tIsVertexEmbedContentModel so the check stays in step with it.
// The Gemini API has no such limit and batches natively.
//
// In practice this is near unreachable for the default model, since
// gemini-embedding-001 is exempt and the gemini-embedding-2 family is
// rejected at construction. It still matters for any other Gemini embedding
// model a caller names explicitly.
func newEmbedderSingleInput(vertexAI bool, model string) bool {
	if !vertexAI {
		return false
	}
	isGeminiEmbed := strings.Contains(model, "gemini") && model != DefaultEmbeddingModel
	isMaaS := strings.Contains(model, "maas")
	return isGeminiEmbed || isMaaS
}

// Embed implements [retrieval.Embedder].
//
// InputType maps onto Gemini's taskType: query becomes RETRIEVAL_QUERY,
// document becomes RETRIEVAL_DOCUMENT, and none leaves taskType unset so the
// model applies its own default. A task type set with WithEmbeddingTaskType
// applies only when the request leaves InputType empty.
//
// Usage is zero on the Gemini API, which reports no token counts for
// embeddings at all. Only Vertex AI populates per-embedding statistics, and
// only those are summed into TotalTokens; no count is estimated or fabricated
// when the backend does not supply one.
//
// Truncate: auto-truncation is a Vertex-only parameter, and the SDK omits it
// from the payload when false, so a request to reject over-length input
// cannot be expressed on the wire. Both unsupported cases are refused rather
// than ignored — silently storing a vector that covers only the head of a
// long document is the expensive failure the core docs warn about.
//
// Vectors are returned exactly as the API sends them. Requesting fewer
// Dimensions than the model default truncates the vector, and Gemini's
// Matryoshka embeddings are not unit-norm after that truncation, but no
// renormalization is applied here: no other embedder in this tree rescales a
// provider's output, and the upstream reference returns the values verbatim
// too. Renormalize in the caller if your index needs it.
func (e *Embedder) Embed(ctx context.Context, req *retrieval.EmbeddingRequest) (*retrieval.EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	cfg, err := e.buildConfig(req)
	if err != nil {
		return nil, err
	}

	if e.singleInput && len(req.Input) > 1 {
		vectors, usage, err := embedbatch.FanOut(ctx, req.Input, 0,
			func(ctx context.Context, text string) ([]float32, retrieval.EmbeddingUsage, error) {
				vecs, usage, err := e.embedBatch(ctx, []string{text}, cfg)
				if err != nil {
					return nil, retrieval.EmbeddingUsage{}, err
				}
				return vecs[0], usage, nil
			})
		if err != nil {
			return nil, err
		}
		return &retrieval.EmbeddingResponse{Vectors: vectors, Model: e.model, Usage: usage}, nil
	}

	vectors, usage, err := e.embedBatch(ctx, req.Input, cfg)
	if err != nil {
		return nil, err
	}
	return &retrieval.EmbeddingResponse{Vectors: vectors, Model: e.model, Usage: usage}, nil
}

// buildConfig turns a request into the SDK's embedding config, rejecting the
// settings this backend cannot represent.
func (e *Embedder) buildConfig(req *retrieval.EmbeddingRequest) (*genai.EmbedContentConfig, error) {
	cfg := &genai.EmbedContentConfig{TaskType: e.resolveTaskType(req.InputType)}

	if req.Dimensions > 0 {
		dims := int32(req.Dimensions)
		cfg.OutputDimensionality = &dims
	}

	if req.Truncate != nil {
		if !*req.Truncate {
			// AutoTruncate is a plain bool tagged omitempty, so false is
			// dropped from the payload and Vertex falls back to its own
			// default of truncating. There is no way to ask either backend
			// to reject over-length input, so refuse rather than truncate
			// behind the caller's back.
			return nil, errors.New("gemini embeddings: rejecting over-length input is not supported; the API truncates it instead")
		}
		if !e.vertexAI {
			return nil, errors.New("gemini embeddings: truncation control is a Vertex AI parameter and has no effect on the Gemini API")
		}
		cfg.AutoTruncate = true
	}

	return cfg, nil
}

// resolveTaskType maps our input type onto Gemini's taskType, falling back to
// the constructor default when the request does not specify a side of the
// retrieval pair. An empty result leaves taskType unset.
func (e *Embedder) resolveTaskType(inputType retrieval.EmbeddingInputType) string {
	switch inputType {
	case retrieval.EmbeddingInputQuery:
		return TaskTypeRetrievalQuery
	case retrieval.EmbeddingInputDocument:
		return TaskTypeRetrievalDocument
	default:
		return e.taskType
	}
}

// embedBatch issues one EmbedContent call and validates what comes back.
func (e *Embedder) embedBatch(ctx context.Context, inputs []string, cfg *genai.EmbedContentConfig) ([][]float32, retrieval.EmbeddingUsage, error) {
	contents := make([]*genai.Content, len(inputs))
	for i, text := range inputs {
		contents[i] = &genai.Content{Parts: []*genai.Part{genai.NewPartFromText(text)}}
	}

	resp, err := e.client.Models.EmbedContent(ctx, e.model, contents, cfg)
	if err != nil {
		return nil, retrieval.EmbeddingUsage{}, fmt.Errorf("gemini embeddings: %w", err)
	}
	// resp is never nil when err is nil: the SDK allocates the response
	// before decoding into it, so no nil guard is needed here.
	//
	// A count mismatch would silently misalign every vector against its
	// source text, so it is an error rather than a short result.
	if len(resp.Embeddings) != len(inputs) {
		return nil, retrieval.EmbeddingUsage{}, fmt.Errorf("gemini embeddings: got %d vectors for %d inputs", len(resp.Embeddings), len(inputs))
	}

	// The API documents embeddings as coming back in request order and gives
	// no index to reorder by, so position is the only join available.
	vectors := make([][]float32, len(inputs))
	var usage retrieval.EmbeddingUsage
	for i, emb := range resp.Embeddings {
		if emb == nil || emb.Values == nil {
			return nil, retrieval.EmbeddingUsage{}, fmt.Errorf("gemini embeddings: missing vector at index %d", i)
		}
		vectors[i] = emb.Values
		// Statistics are Vertex-only; on the Gemini API this stays zero.
		if emb.Statistics != nil {
			usage.TotalTokens += int(emb.Statistics.TokenCount)
		}
	}
	// Everything an embeddings call bills is input, so the one count Vertex
	// reports is both the prompt total and the request total.
	usage.PromptTokens = usage.TotalTokens

	return vectors, usage, nil
}

// Name implements [retrieval.Embedder].
func (e *Embedder) Name() string {
	return e.model
}

// Compile-time check that Embedder implements retrieval.Embedder.
var _ retrieval.Embedder = (*Embedder)(nil)
