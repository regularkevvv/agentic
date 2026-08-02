// Package batch splits an oversized encoding request into chunks a provider
// will accept, and reassembles the chunks into a single response that still
// satisfies the core contract.
//
// It is the multi-output counterpart of retrieval/embedbatch. The extra work
// over the dense case is agreement: a reassembled response has one set of
// vector spaces and one model name, so every chunk must have encoded into the
// same spaces. A provider that silently rolls its deployment mid-batch would
// otherwise hand back a response whose first half and second half cannot be
// compared, and nothing downstream would notice.
//
// There is deliberately no fan-out helper here. Fan-out exists in embedbatch
// for models that accept exactly one input per call; no multi-representation
// service in scope has that shape, and an untested concurrency helper with no
// caller is a liability rather than symmetry.
package batch

import (
	"context"
	"errors"
	"fmt"

	"github.com/regularkevvv/agentic/internal/retrieval"
)

// Chunked encodes inputs in batches of at most size, preserving input order
// and merging the chunk responses into one.
//
// Chunks run sequentially. A provider that caps its batch size is usually
// signaling a rate limit rather than a payload limit, so issuing the chunks
// concurrently tends to trade 429s for nothing.
//
// The first failing chunk aborts the whole call and no partial result is
// returned. Successful earlier chunks are not retried or replayed: re-issuing
// them would double their cost, and the caller is better placed to decide
// whether to restart the batch at all.
func Chunked(
	ctx context.Context,
	req *retrieval.RepresentationRequest,
	size int,
	fn func(context.Context, *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, error),
) (*retrieval.RepresentationResponse, error) {
	if req == nil {
		return nil, errors.New("retrieval/batch: request cannot be nil")
	}
	if size <= 0 || size >= len(req.Input) {
		return single(ctx, req, fn)
	}

	merged := &retrieval.RepresentationResponse{
		Data: make([]retrieval.Representation, 0, len(req.Input)),
	}
	first := true

	for start := 0; start < len(req.Input); start += size {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+size, len(req.Input))

		chunkReq := req.Clone()
		chunkReq.Input = req.Input[start:end]

		chunk, err := fn(ctx, chunkReq)
		if err != nil {
			return nil, err
		}
		if chunk == nil {
			return nil, &CountMismatchError{Want: end - start, Got: 0}
		}
		if len(chunk.Data) != end-start {
			return nil, &CountMismatchError{Want: end - start, Got: len(chunk.Data)}
		}

		if first {
			merged.Spaces = cloneSpaces(chunk.Spaces)
			merged.Model = chunk.Model
			first = false
		} else {
			if err := sameSpaces(merged.Spaces, chunk.Spaces); err != nil {
				return nil, err
			}
			if chunk.Model != merged.Model {
				return nil, &ModelMismatchError{Want: merged.Model, Got: chunk.Model}
			}
		}

		merged.Data = append(merged.Data, chunk.Data...)
		merged.Usage.Add(chunk.Usage)
	}
	return merged, nil
}

// single issues the request unsplit, still checking cancellation and
// cardinality so that the chunked and unchunked paths behave the same way.
//
// The context check is here rather than left to the transport because not
// every transport observes it: an SDK client or a stub can return a result for
// a context that was already canceled, and a caller who has given up must not
// be billed for work they cannot use.
func single(
	ctx context.Context,
	req *retrieval.RepresentationRequest,
	fn func(context.Context, *retrieval.RepresentationRequest) (*retrieval.RepresentationResponse, error),
) (*retrieval.RepresentationResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	resp, err := fn(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, &CountMismatchError{Want: len(req.Input), Got: 0}
	}
	if len(resp.Data) != len(req.Input) {
		return nil, &CountMismatchError{Want: len(req.Input), Got: len(resp.Data)}
	}
	return resp, nil
}

// CountMismatchError reports a provider returning a number of representations
// that does not match the number of inputs it was given.
type CountMismatchError struct {
	Want int
	Got  int
}

func (e *CountMismatchError) Error() string {
	return fmt.Sprintf("representation batch returned %d representations for %d inputs", e.Got, e.Want)
}

// SpaceMismatchError reports two chunks of one logical request encoding into
// different vector spaces, which makes their outputs incomparable.
type SpaceMismatchError struct {
	Kind retrieval.RepresentationKind
	Want string
	Got  string
}

func (e *SpaceMismatchError) Error() string {
	return fmt.Sprintf("representation batch chunk changed the %s vector space from %s to %s",
		e.Kind, e.Want, e.Got)
}

// ModelMismatchError reports two chunks of one logical request being answered
// by different models.
type ModelMismatchError struct {
	Want string
	Got  string
}

func (e *ModelMismatchError) Error() string {
	return fmt.Sprintf("representation batch chunk changed the model from %q to %q", e.Want, e.Got)
}

// sameSpaces checks that a chunk encoded into exactly the spaces the batch
// started with. Both the set of kinds and each descriptor must match.
func sameSpaces(want, got map[retrieval.RepresentationKind]retrieval.VectorSpace) error {
	for kind, wantSpace := range want {
		gotSpace, ok := got[kind]
		if !ok {
			return &SpaceMismatchError{Kind: kind, Want: wantSpace.ID, Got: ""}
		}
		if gotSpace.ID != wantSpace.ID || !gotSpace.Compatible(wantSpace) {
			return &SpaceMismatchError{Kind: kind, Want: wantSpace.ID, Got: gotSpace.ID}
		}
	}
	for kind, gotSpace := range got {
		if _, ok := want[kind]; !ok {
			return &SpaceMismatchError{Kind: kind, Want: "", Got: gotSpace.ID}
		}
	}
	return nil
}

func cloneSpaces(spaces map[retrieval.RepresentationKind]retrieval.VectorSpace) map[retrieval.RepresentationKind]retrieval.VectorSpace {
	if spaces == nil {
		return nil
	}
	out := make(map[retrieval.RepresentationKind]retrieval.VectorSpace, len(spaces))
	for kind, space := range spaces {
		out[kind] = space
	}
	return out
}
