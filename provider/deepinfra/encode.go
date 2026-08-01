package deepinfra

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/representationbatch"
)

// inferenceRequest is DeepInfra's native body for a multi-representation
// embedding model.
//
// The output flags are always sent, never omitted. DeepInfra defaults dense to
// true, so omitting the field when dense was not requested would quietly bill
// for and return a representation the caller did not ask for.
type inferenceRequest struct {
	Inputs    []string `json:"inputs"`
	Dense     bool     `json:"dense"`
	Sparse    bool     `json:"sparse"`
	Colbert   bool     `json:"colbert"`
	Normalize *bool    `json:"normalize,omitempty"`
}

// inferenceResponse is DeepInfra's native response. Sparse rows stay raw until
// their shape is known; see decodeSparse.
type inferenceResponse struct {
	Embeddings  [][]float32       `json:"embeddings"`
	Sparse      []json.RawMessage `json:"sparse"`
	Colbert     [][][]float32     `json:"colbert"`
	InputTokens int               `json:"input_tokens"`
	RequestID   string            `json:"request_id"`

	InferenceStatus struct {
		Status      string `json:"status"`
		TokensInput int    `json:"tokens_input"`
	} `json:"inference_status"`
}

// Encode implements core.RepresentationEncoder.
func (e *Encoder) Encode(ctx context.Context, req *core.RepresentationRequest) (*core.RepresentationResponse, error) {
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

// Embed implements core.Embedder by requesting dense output only, so an
// application already written against Embedder can point at this provider and
// adopt its sparse output later.
func (e *Encoder) Embed(ctx context.Context, req *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	if e.embedder == nil {
		return nil, &core.UnsupportedRepresentationError{
			Provider:  providerName,
			Kind:      core.RepresentationDense,
			Supported: e.outputs,
		}
	}
	return e.embedder.Embed(ctx, req)
}

// encodeChunk issues one provider call for the inputs it is given.
func (e *Encoder) encodeChunk(ctx context.Context, req *core.RepresentationRequest) (*core.RepresentationResponse, error) {
	body := inferenceRequest{
		Inputs:    req.Input,
		Dense:     req.Wants(core.RepresentationDense),
		Sparse:    req.Wants(core.RepresentationSparse),
		Colbert:   req.Wants(core.RepresentationMultiVector),
		Normalize: e.normalize,
	}

	payload, err := e.post(ctx, "/inference/"+e.model, body)
	if err != nil {
		return nil, err
	}

	var wire inferenceResponse
	if err := json.Unmarshal(payload, &wire); err != nil {
		return nil, fmt.Errorf("deepinfra: decode response: %w", err)
	}
	if wire.InferenceStatus.Status == "failed" {
		return nil, &APIError{
			Status:    200,
			RequestID: wire.RequestID,
			Detail:    "inference reported status failed",
		}
	}

	return e.buildResponse(req, &wire, len(payload))
}

// buildResponse turns one decoded provider response into the core shape,
// observing each space's width rather than assuming it.
func (e *Encoder) buildResponse(req *core.RepresentationRequest, wire *inferenceResponse, responseBytes int) (*core.RepresentationResponse, error) {
	count := len(req.Input)
	data := make([]core.Representation, count)
	spaces := make(map[core.RepresentationKind]core.VectorSpace, len(req.Outputs))

	if req.Wants(core.RepresentationDense) {
		if len(wire.Embeddings) != count {
			return nil, cardinalityError(core.RepresentationDense, len(wire.Embeddings), count)
		}
		for i, vec := range wire.Embeddings {
			data[i].Dense = vec
		}
		spaces[core.RepresentationDense] = e.space(core.RepresentationDense, width(wire.Embeddings))
	}

	if req.Wants(core.RepresentationSparse) {
		if len(wire.Sparse) != count {
			return nil, cardinalityError(core.RepresentationSparse, len(wire.Sparse), count)
		}
		vocabulary := e.sparseVocabulary
		for i, raw := range wire.Sparse {
			vec, observed, err := decodeSparse(raw, i)
			if err != nil {
				return nil, err
			}
			if observed > 0 {
				if vocabulary > 0 && observed != vocabulary {
					return nil, &core.InvalidRepresentationResponseError{
						Provider: providerName,
						Item:     i,
						Kind:     core.RepresentationSparse,
						Problem: fmt.Sprintf("sparse row is %d wide, but the vocabulary is declared as %d",
							observed, vocabulary),
					}
				}
				vocabulary = observed
			}
			data[i].Sparse = vec
		}
		if vocabulary == 0 {
			return nil, &core.InvalidRepresentationResponseError{
				Provider: providerName,
				Item:     -1,
				Kind:     core.RepresentationSparse,
				Problem: "sparse output arrived as a coordinate map, which does not reveal the " +
					"tokenizer vocabulary size; set it with WithSparseVocabulary",
			}
		}
		spaces[core.RepresentationSparse] = e.space(core.RepresentationSparse, vocabulary)
	}

	if req.Wants(core.RepresentationMultiVector) {
		if len(wire.Colbert) != count {
			return nil, cardinalityError(core.RepresentationMultiVector, len(wire.Colbert), count)
		}
		for i, vectors := range wire.Colbert {
			data[i].MultiVector = vectors
		}
		spaces[core.RepresentationMultiVector] = e.space(core.RepresentationMultiVector, tokenWidth(wire.Colbert))
	}

	inputBytes := 0
	for _, text := range req.Input {
		inputBytes += len(text)
	}
	tokens := wire.InputTokens
	if tokens == 0 {
		tokens = wire.InferenceStatus.TokensInput
	}

	return &core.RepresentationResponse{
		Data:   data,
		Spaces: spaces,
		Model:  e.model,
		Usage: core.RepresentationUsage{
			InputTokens:  tokens,
			RequestCount: 1,
			InputBytes:   inputBytes,
			OutputBytes:  responseBytes,
		},
	}, nil
}

func cardinalityError(kind core.RepresentationKind, got, want int) error {
	return &core.InvalidRepresentationResponseError{
		Provider: providerName,
		Item:     -1,
		Kind:     kind,
		Problem:  fmt.Sprintf("got %d %s representations for %d inputs", got, kind, want),
	}
}

// width returns the first row's width, or zero for an empty batch. The
// validator rejects a batch whose rows disagree, so the first is enough to
// declare the space.
func width(vectors [][]float32) int {
	if len(vectors) == 0 {
		return 0
	}
	return len(vectors[0])
}

func tokenWidth(vectors [][][]float32) int {
	for _, item := range vectors {
		if len(item) > 0 {
			return len(item[0])
		}
	}
	return 0
}

// decodeSparse normalizes DeepInfra's sparse row into coordinate form.
//
// Two shapes occur. A full vocabulary-width row of weights, which is what the
// documented response schema shows, reveals the vocabulary size in its length
// and needs only its zeros dropped. A coordinate map keyed by token ID, which
// is BGE-M3's native lexical-weights shape, needs sorting and reveals no
// vocabulary size.
//
// The second return value is the observed vocabulary size, or zero when the
// shape does not carry one.
func decodeSparse(raw json.RawMessage, item int) (*core.SparseVector, int, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil, 0, sparseError(item, "sparse row is empty")
	}

	switch trimmed[0] {
	case '[':
		return decodeSparseRow(trimmed, item)
	case '{':
		vec, err := decodeSparseMap(trimmed, item)
		return vec, 0, err
	case 'n':
		return nil, 0, sparseError(item, "sparse row is null")
	default:
		return nil, 0, sparseError(item, "sparse row is neither a weight array nor a coordinate map")
	}
}

// decodeSparseRow converts a dense vocabulary-width row into coordinates,
// dropping exact zeros. A non-finite weight is kept so that response
// validation can reject it rather than having it silently disappear.
func decodeSparseRow(raw json.RawMessage, item int) (*core.SparseVector, int, error) {
	var row []float32
	if err := json.Unmarshal(raw, &row); err != nil {
		return nil, 0, sparseError(item, "sparse row is not an array of weights")
	}

	vec := &core.SparseVector{}
	for index, value := range row {
		if value == 0 {
			continue
		}
		vec.Indices = append(vec.Indices, uint32(index))
		vec.Values = append(vec.Values, value)
	}
	return vec, len(row), nil
}

// decodeSparseMap converts a coordinate map into sorted coordinate form.
//
// It walks the JSON tokens rather than unmarshaling into a Go map, because a
// map decode resolves a repeated key by keeping the last value. A response
// that assigns one token two different weights is malformed, and collapsing it
// silently would store whichever weight happened to come second.
func decodeSparseMap(raw json.RawMessage, item int) (*core.SparseVector, error) {
	// The value is already known to be syntactically valid JSON, because the
	// whole response was unmarshaled before this ran. Walking tokens is only
	// about duplicate keys.
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if _, err := decoder.Token(); err != nil {
		return nil, sparseError(item, "sparse coordinate map is malformed")
	}

	weights := make(map[uint32]float32)
	for decoder.More() {
		keyToken, err := decoder.Token()
		key, ok := keyToken.(string)
		if err != nil || !ok {
			return nil, sparseError(item, "sparse coordinate map is malformed")
		}
		index, err := strconv.ParseUint(key, 10, 32)
		if err != nil {
			return nil, sparseError(item, fmt.Sprintf("sparse coordinate key %q is not a token index", truncate(key, 32)))
		}

		var value float32
		if err := decoder.Decode(&value); err != nil {
			return nil, sparseError(item, fmt.Sprintf("sparse weight for coordinate %d is not a number", index))
		}
		if _, duplicate := weights[uint32(index)]; duplicate {
			return nil, sparseError(item, fmt.Sprintf("sparse coordinate %d appears more than once", index))
		}
		weights[uint32(index)] = value
	}

	indices := make([]uint32, 0, len(weights))
	for index := range weights {
		indices = append(indices, index)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })

	vec := &core.SparseVector{
		Indices: indices,
		Values:  make([]float32, len(indices)),
	}
	for i, index := range indices {
		vec.Values[i] = weights[index]
	}
	return vec, nil
}

func sparseError(item int, problem string) error {
	return &core.InvalidRepresentationResponseError{
		Provider: providerName,
		Item:     item,
		Kind:     core.RepresentationSparse,
		Problem:  problem,
	}
}
