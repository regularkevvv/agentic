package pinecone

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/internal/retrieval/batch"
)

// Pinecone's input type values for the two retrieval roles.
const (
	inputTypeQuery   = "query"
	inputTypePassage = "passage"
)

// Pinecone's truncation strategy values.
const (
	truncateEnd  = "END"
	truncateNone = "NONE"
)

type embedRequest struct {
	Model      string          `json:"model"`
	Parameters embedParameters `json:"parameters,omitempty"`
	Inputs     []embedInput    `json:"inputs"`
}

type embedParameters struct {
	InputType    string `json:"input_type,omitempty"`
	Truncate     string `json:"truncate,omitempty"`
	ReturnTokens bool   `json:"return_tokens,omitempty"`
	Dimension    int    `json:"dimension,omitempty"`
}

type embedInput struct {
	Text string `json:"text"`
}

type embedResponse struct {
	Model      string          `json:"model"`
	VectorType string          `json:"vector_type"`
	Data       []embedVector   `json:"data"`
	Usage      embedUsageBlock `json:"usage"`
}

type embedVector struct {
	VectorType    string    `json:"vector_type"`
	Values        []float32 `json:"values"`
	SparseIndices []uint32  `json:"sparse_indices"`
	SparseValues  []float32 `json:"sparse_values"`
	SparseTokens  []string  `json:"sparse_tokens"`
}

type embedUsageBlock struct {
	TotalTokens int `json:"total_tokens"`
}

// Encode implements retrieval.RepresentationEncoder.
func (e *Encoder) Encode(ctx context.Context, req *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, error) {
	resp, _, err := e.encode(ctx, req)
	return resp, err
}

// EncodeWithTokens encodes and additionally returns the token string behind
// each sparse coordinate, one slice per input, aligned with that input's
// SparseVector.Indices.
//
// It requires [WithReturnTokens]. Use it to see which terms a model actually
// weighted, and whether any of them were absent from the input. Do not store
// the strings as identity: a tokenizer change can keep a string and move its
// coordinate, or the reverse.
//
// Tokens are nil for a dense encoder, which has no coordinates to name.
func (e *Encoder) EncodeWithTokens(ctx context.Context, req *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, [][]string, error) {
	if !e.returnTokens {
		return nil, nil, &retrieval.InvalidRepresentationRequestError{
			Invariant: "return_tokens.disabled",
			Detail:    "pinecone: build the encoder with WithReturnTokens(true) to observe the tokens behind each coordinate",
		}
	}
	return e.encode(ctx, req)
}

func (e *Encoder) encode(ctx context.Context, req *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, [][]string, error) {
	validator := e.validator()
	if err := validator.ValidateRequest(req); err != nil {
		return nil, nil, err
	}

	// Tokens accumulate across chunks in input order, alongside the merged
	// response the batch helper builds.
	var tokens [][]string
	resp, err := batch.Chunked(ctx, req, e.batchSize,
		func(ctx context.Context, chunk *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, error) {
			chunkResp, chunkTokens, err := e.encodeChunk(ctx, chunk)
			if err != nil {
				return nil, err
			}
			tokens = append(tokens, chunkTokens...)
			return chunkResp, nil
		})
	if err != nil {
		return nil, nil, err
	}
	if err := validator.ValidateResponse(req, resp); err != nil {
		return nil, nil, err
	}
	return resp, tokens, nil
}

func (e *Encoder) encodeChunk(ctx context.Context, req *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, [][]string, error) {
	body := embedRequest{
		Model: e.model,
		Parameters: embedParameters{
			InputType:    inputTypeFor(req.InputType),
			Truncate:     truncateFor(req.Truncate),
			ReturnTokens: e.returnTokens,
			Dimension:    e.dimensions,
		},
		Inputs: make([]embedInput, len(req.Input)),
	}
	for i, text := range req.Input {
		body.Inputs[i] = embedInput{Text: text}
	}

	payload, err := e.post(ctx, "/embed", body)
	if err != nil {
		return nil, nil, err
	}

	var wire embedResponse
	if err := json.Unmarshal(payload, &wire); err != nil {
		return nil, nil, fmt.Errorf("pinecone: decode response: %w", err)
	}
	if len(wire.Data) != len(req.Input) {
		return nil, nil, responseError(-1, "", fmt.Sprintf("got %d vectors for %d inputs",
			len(wire.Data), len(req.Input)))
	}

	// Verify rather than assume: a model reconfigured to return the other
	// vector type would otherwise be decoded into empty representations.
	if got := wire.VectorType; got != "" && got != string(e.kind) {
		return nil, nil, responseError(-1, e.kind, fmt.Sprintf(
			"model returned %s vectors, but this encoder is configured for %s", got, e.kind))
	}

	data := make([]retrieval.Representation, len(wire.Data))
	tokens := make([][]string, len(wire.Data))
	width := 0

	for i, vec := range wire.Data {
		if vec.VectorType != "" && vec.VectorType != string(e.kind) {
			return nil, nil, responseError(i, e.kind, fmt.Sprintf(
				"vector is %s, but this encoder is configured for %s", vec.VectorType, e.kind))
		}
		switch e.kind {
		case retrieval.RepresentationDense:
			data[i].Dense = vec.Values
			if i == 0 {
				width = len(vec.Values)
			}
		case retrieval.RepresentationSparse:
			sparse, ordered, err := canonicalSparse(vec, i)
			if err != nil {
				return nil, nil, err
			}
			data[i].Sparse = sparse
			tokens[i] = ordered
		}
	}

	if e.dimensions > 0 {
		width = e.dimensions
	}

	inputBytes := 0
	for _, text := range req.Input {
		inputBytes += len(text)
	}

	return &retrieval.RepresentationResponse{
		Data:   data,
		Spaces: map[retrieval.RepresentationKind]retrieval.VectorSpace{e.kind: e.space(width)},
		Model:  e.model,
		Usage: retrieval.RepresentationUsage{
			InputTokens:  wire.Usage.TotalTokens,
			RequestCount: 1,
			InputBytes:   inputBytes,
			OutputBytes:  len(payload),
		},
	}, tokens, nil
}

// canonicalSparse sorts Pinecone's coordinate arrays into the canonical
// strictly-increasing form, carrying any token strings along with them.
//
// Sorting is safe here only because duplicates are rejected first: reordering
// a list that assigns one coordinate two weights would hide the conflict and
// keep whichever weight happened to land last.
func canonicalSparse(vec embedVector, item int) (*retrieval.SparseVector, []string, error) {
	if len(vec.SparseIndices) != len(vec.SparseValues) {
		return nil, nil, responseError(item, retrieval.RepresentationSparse,
			fmt.Sprintf("got %d sparse indices and %d values",
				len(vec.SparseIndices), len(vec.SparseValues)))
	}
	if len(vec.SparseTokens) > 0 && len(vec.SparseTokens) != len(vec.SparseIndices) {
		return nil, nil, responseError(item, retrieval.RepresentationSparse,
			fmt.Sprintf("got %d sparse tokens for %d coordinates",
				len(vec.SparseTokens), len(vec.SparseIndices)))
	}

	order := make([]int, len(vec.SparseIndices))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return vec.SparseIndices[order[a]] < vec.SparseIndices[order[b]]
	})

	sparse := &retrieval.SparseVector{
		Indices: make([]uint32, len(order)),
		Values:  make([]float32, len(order)),
	}
	var tokens []string
	if len(vec.SparseTokens) > 0 {
		tokens = make([]string, len(order))
	}

	for position, source := range order {
		index := vec.SparseIndices[source]
		if position > 0 && index == sparse.Indices[position-1] {
			return nil, nil, responseError(item, retrieval.RepresentationSparse,
				fmt.Sprintf("sparse coordinate %d appears more than once", index))
		}
		sparse.Indices[position] = index
		sparse.Values[position] = vec.SparseValues[source]
		if tokens != nil {
			tokens[position] = vec.SparseTokens[source]
		}
	}
	return sparse, tokens, nil
}

// Embed implements retrieval.Embedder for a dense-configured encoder.
func (e *Encoder) Embed(ctx context.Context, req *retrieval.EmbeddingRequest) (*retrieval.EmbeddingResponse, error) {
	if e.embedder == nil {
		return nil, &retrieval.UnsupportedRepresentationError{
			Provider:  providerName,
			Kind:      retrieval.RepresentationDense,
			Supported: []retrieval.RepresentationKind{e.kind},
		}
	}
	return e.embedder.Embed(ctx, req)
}

// inputTypeFor maps Agentic's retrieval role onto Pinecone's vocabulary. The
// two roles encode into the same space, which is what makes a query
// comparable to an indexed passage.
func inputTypeFor(inputType retrieval.EmbeddingInputType) string {
	switch inputType {
	case retrieval.EmbeddingInputQuery:
		return inputTypeQuery
	case retrieval.EmbeddingInputDocument:
		return inputTypePassage
	}
	return ""
}

// truncateFor maps the boolean truncate option onto Pinecone's strategy
// values. Nil omits the parameter so the API's own default applies.
func truncateFor(truncate *bool) string {
	if truncate == nil {
		return ""
	}
	if *truncate {
		return truncateEnd
	}
	return truncateNone
}

func responseError(item int, kind retrieval.RepresentationKind, problem string) error {
	return &retrieval.InvalidRepresentationResponseError{
		Provider: providerName,
		Item:     item,
		Kind:     kind,
		Problem:  problem,
	}
}
