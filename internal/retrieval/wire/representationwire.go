// Package representationwire implements agentic.representations.v1, the
// portable JSON contract between Agentic and a self-operated inference
// endpoint.
//
// Endpoints you operate have no shared response format: whatever a custom
// handler returns is the format. Rather than write one decoder per deployment,
// Agentic publishes a small versioned protocol and speaks it from both
// provider/endpoint and provider/sagemaker. The transport differs — HTTPS with
// a bearer token, or InvokeEndpoint with SigV4 — but the payload does not.
//
// testdata holds the golden request and response, and the JSON Schemas that
// define the shape a handler must return. Writing that handler is the
// deployment's job, in whatever language its platform runs.
//
// Unknown fields are ignored so a handler can add data without breaking older
// clients. An unknown major version is refused rather than guessed at, because
// a protocol change large enough to bump the major is one where guessing
// produces vectors that look valid and are not.
package wire

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/regularkevvv/agentic/internal/retrieval"
)

// Version is the protocol version this package speaks.
const Version = "agentic.representations.v1"

// versionPrefix is everything before the major version number.
const versionPrefix = "agentic.representations.v"

// major is the major version this package accepts.
const major = 1

// Request is the JSON body sent to a handler.
type Request struct {
	Version   string   `json:"version"`
	Inputs    []string `json:"inputs"`
	InputType string   `json:"input_type,omitempty"`
	Outputs   []string `json:"outputs"`
	Truncate  *bool    `json:"truncate,omitempty"`
}

// Space is a vector-space descriptor on the wire.
type Space struct {
	ID         string `json:"id"`
	Provider   string `json:"provider"`
	Model      string `json:"model"`
	Revision   string `json:"revision"`
	Tokenizer  string `json:"tokenizer"`
	Kind       string `json:"kind"`
	Dimensions int    `json:"dimensions"`
	Metric     string `json:"metric"`
}

// Sparse is a learned sparse vector on the wire, in coordinate form.
type Sparse struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// Item holds one input's representations. Absent kinds are omitted rather than
// sent as null, so a handler that computes nothing extra costs nothing extra.
type Item struct {
	Dense       []float32   `json:"dense,omitempty"`
	Sparse      *Sparse     `json:"sparse,omitempty"`
	MultiVector [][]float32 `json:"multi_vector,omitempty"`
}

// Usage reports what the handler consumed.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	RequestCount int `json:"request_count"`
	InputBytes   int `json:"input_bytes"`
	OutputBytes  int `json:"output_bytes"`
}

// Response is the JSON body a handler returns.
type Response struct {
	Version string           `json:"version"`
	Model   string           `json:"model"`
	Spaces  map[string]Space `json:"spaces"`
	Data    []Item           `json:"data"`
	Usage   Usage            `json:"usage"`
}

// NewRequest builds the wire request for a core request.
func NewRequest(req *retrieval.RepresentationRequest) Request {
	outputs := make([]string, len(req.Outputs))
	for i, kind := range req.Outputs {
		outputs[i] = string(kind)
	}
	return Request{
		Version:   Version,
		Inputs:    req.Input,
		InputType: string(req.InputType),
		Outputs:   outputs,
		Truncate:  req.Truncate,
	}
}

// DecodeOptions carries what a provider knows that the payload does not.
type DecodeOptions struct {
	// Provider names the caller in errors and in any space descriptor the
	// handler leaves incomplete.
	Provider string

	// Model is the configured model name, used when the handler reports none.
	Model string

	// Expected pins the vector spaces the caller was configured with.
	//
	// A handler behind an opaque endpoint cannot prove its own revision, so a
	// deployment that intends to keep an index supplies the descriptor here.
	// A space the handler does return must match it exactly; a space it omits
	// is filled from here.
	Expected map[retrieval.RepresentationKind]retrieval.VectorSpace

	// ResponseBytes is the size of the payload, recorded as usage when the
	// handler does not report its own.
	ResponseBytes int
}

// Decode converts a handler payload into the core response shape.
//
// It does not run the full contract validation; the caller does that with its
// own retrieval.RepresentationValidator, so that limits and capabilities stay a
// provider concern.
func Decode(payload []byte, req *retrieval.RepresentationRequest, opts DecodeOptions) (*retrieval.RepresentationResponse, error) {
	var wire Response
	if err := json.Unmarshal(payload, &wire); err != nil {
		return nil, wireError(opts.Provider, -1, "", "response is not valid "+Version+" JSON")
	}
	if err := CheckVersion(wire.Version, opts.Provider); err != nil {
		return nil, err
	}

	spaces, err := decodeSpaces(wire.Spaces, req, opts)
	if err != nil {
		return nil, err
	}

	data := make([]retrieval.Representation, len(wire.Data))
	for i, item := range wire.Data {
		data[i] = retrieval.Representation{
			Dense:       item.Dense,
			MultiVector: item.MultiVector,
		}
		if item.Sparse != nil {
			data[i].Sparse = &retrieval.SparseVector{
				Indices: item.Sparse.Indices,
				Values:  item.Sparse.Values,
			}
		}
	}

	model := wire.Model
	if model == "" {
		model = opts.Model
	}

	usage := retrieval.RepresentationUsage{
		InputTokens:  wire.Usage.InputTokens,
		RequestCount: max(wire.Usage.RequestCount, 1),
		InputBytes:   wire.Usage.InputBytes,
		OutputBytes:  wire.Usage.OutputBytes,
	}
	if usage.InputBytes == 0 {
		for _, text := range req.Input {
			usage.InputBytes += len(text)
		}
	}
	if usage.OutputBytes == 0 {
		usage.OutputBytes = opts.ResponseBytes
	}

	return &retrieval.RepresentationResponse{
		Data:   data,
		Spaces: spaces,
		Model:  model,
		Usage:  usage,
	}, nil
}

// CheckVersion accepts an additive minor bump and refuses anything else.
func CheckVersion(version, provider string) error {
	if version == "" {
		return wireError(provider, -1, "", "response does not declare a protocol version")
	}
	if !strings.HasPrefix(version, versionPrefix) {
		return wireError(provider, -1, "",
			fmt.Sprintf("response declares protocol %q, want %s", version, Version))
	}
	// Everything after the major digits is a minor or patch suffix, which is
	// additive by contract and safe to ignore.
	digits := version[len(versionPrefix):]
	end := 0
	for end < len(digits) && digits[end] >= '0' && digits[end] <= '9' {
		end++
	}
	got, err := strconv.Atoi(digits[:end])
	if err != nil {
		return wireError(provider, -1, "",
			fmt.Sprintf("response declares protocol %q, want %s", version, Version))
	}
	if got != major {
		return wireError(provider, -1, "",
			fmt.Sprintf("response speaks protocol major version %d, this client speaks %d", got, major))
	}
	return nil
}

// decodeSpaces converts the wire descriptors and reconciles them with the
// caller's configured spaces.
func decodeSpaces(wire map[string]Space, req *retrieval.RepresentationRequest, opts DecodeOptions) (map[retrieval.RepresentationKind]retrieval.VectorSpace, error) {
	spaces := make(map[retrieval.RepresentationKind]retrieval.VectorSpace, len(req.Outputs))

	for name, space := range wire {
		kind := retrieval.RepresentationKind(name)
		decoded := retrieval.VectorSpace{
			ID:         space.ID,
			Provider:   space.Provider,
			Model:      space.Model,
			Revision:   space.Revision,
			Tokenizer:  space.Tokenizer,
			Kind:       retrieval.RepresentationKind(space.Kind),
			Dimensions: space.Dimensions,
			Metric:     retrieval.SimilarityMetric(space.Metric),
		}
		if decoded.Kind == "" {
			decoded.Kind = kind
		}
		if decoded.Provider == "" {
			decoded.Provider = opts.Provider
		}
		if decoded.Model == "" {
			decoded.Model = opts.Model
		}
		if decoded.ID == "" {
			decoded = decoded.WithCanonicalID()
		}

		if want, pinned := opts.Expected[kind]; pinned {
			if want.ID != decoded.ID || !want.Compatible(decoded) {
				return nil, wireError(opts.Provider, -1, kind,
					fmt.Sprintf("handler reports space %s, but this encoder is configured for %s",
						decoded.ID, want.ID))
			}
		}
		spaces[kind] = decoded
	}

	// A handler that reports no descriptor is trusted only as far as the
	// caller's own configuration: the pinned space is used, and a kind with
	// neither is left absent for response validation to reject.
	for kind, want := range opts.Expected {
		if _, ok := spaces[kind]; !ok && req.Wants(kind) {
			spaces[kind] = want
		}
	}
	return spaces, nil
}

func wireError(provider string, item int, kind retrieval.RepresentationKind, problem string) error {
	return &retrieval.InvalidRepresentationResponseError{
		Provider: provider,
		Item:     item,
		Kind:     kind,
		Problem:  problem,
	}
}
