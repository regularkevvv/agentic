package agentic

import (
	"context"

	"github.com/regularkevvv/agentic/internal/retrieval"
)

// EncodeQueries encodes texts as search queries, returning the requested
// representations for each.
//
// It is the multi-output counterpart of [EmbedQuery]. Outputs are explicit
// because they differ in cost by orders of magnitude; nothing is selected for
// you. Batching and splitting stay with the encoder, so what you pass is what
// it receives.
func EncodeQueries(
	ctx context.Context,
	encoder RepresentationEncoder,
	texts []string,
	outputs ...RepresentationKind,
) (*RepresentationResponse, error) {
	return encode(ctx, encoder, texts, EmbeddingInputQuery, outputs)
}

// EncodeDocuments encodes texts as documents to be stored and searched
// against, returning the requested representations for each.
//
// Persist each output beside the ID of the space it came from — see
// [RepresentationResponse.Spaces]. Vectors without their space identity cannot
// be checked for compatibility later, and the failure mode of querying an
// index with a differently-encoded vector is quietly worse recall rather than
// an error.
func EncodeDocuments(
	ctx context.Context,
	encoder RepresentationEncoder,
	texts []string,
	outputs ...RepresentationKind,
) (*RepresentationResponse, error) {
	return encode(ctx, encoder, texts, EmbeddingInputDocument, outputs)
}

func encode(
	ctx context.Context,
	encoder RepresentationEncoder,
	texts []string,
	inputType EmbeddingInputType,
	outputs []RepresentationKind,
) (*RepresentationResponse, error) {
	if encoder == nil {
		return nil, &InvalidRepresentationRequestError{
			Invariant: "encoder.nil",
			Detail:    "encoder cannot be nil",
		}
	}
	return encoder.Encode(ctx, &RepresentationRequest{
		Input:     texts,
		InputType: inputType,
		Outputs:   outputs,
	})
}

// DefaultRepresentationLimits returns the request and response ceilings every
// encoder in this repository applies: generous enough for normal retrieval
// workloads, tight enough that a malformed or hostile response cannot exhaust
// memory before validation runs.
//
// It is exported because [RepresentationValidator] is. An encoder written
// outside this module cannot reach the same defaults otherwise, and would
// either invent its own numbers or run with every bound disabled — which would
// make a consumer's error behavior depend on where the encoder was compiled.
func DefaultRepresentationLimits() RepresentationLimits {
	return retrieval.DefaultRepresentationLimits()
}

// EmbedderAsRepresentationEncoder presents an existing dense [Embedder] as a
// dense-only [RepresentationEncoder], so one code path can drive both the
// single-vector providers and the multi-output ones.
//
// The vector space is supplied by the caller because an Embedder cannot prove
// one: it reports a model name and nothing about its weights revision or
// tokenizer. Only space.Provider is required. Model defaults to the embedder's
// name, Kind to dense, Metric to cosine, Dimensions to the width observed in
// the first response, and ID to the canonical hash of those fields.
func EmbedderAsRepresentationEncoder(embedder Embedder, space VectorSpace) (RepresentationEncoder, error) {
	return retrieval.NewEmbedderEncoder(embedder, space)
}

// RepresentationEncoderAsEmbedder projects a [RepresentationEncoder] onto the
// dense [Embedder] contract, requesting dense output only.
//
// Use it to adopt a multi-output provider in code already written against
// Embedder, and to turn on its sparse output later when the index is ready.
// It fails at construction, not at first call, when the encoder produces no
// dense output.
func RepresentationEncoderAsEmbedder(encoder RepresentationEncoder) (Embedder, error) {
	return retrieval.NewEncoderEmbedder(encoder)
}
