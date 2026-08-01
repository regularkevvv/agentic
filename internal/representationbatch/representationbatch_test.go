package representationbatch_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/representationbatch"
)

func denseSpace(dims int) core.VectorSpace {
	return core.VectorSpace{
		Provider:   "test",
		Model:      "model",
		Revision:   "rev-1",
		Tokenizer:  "tok-1",
		Kind:       core.RepresentationDense,
		Dimensions: dims,
		Metric:     core.SimilarityCosine,
	}.WithCanonicalID()
}

func sparseSpace() core.VectorSpace {
	return core.VectorSpace{
		Provider:   "test",
		Model:      "model",
		Revision:   "rev-1",
		Tokenizer:  "tok-1",
		Kind:       core.RepresentationSparse,
		Dimensions: 1000,
		Metric:     core.SimilarityDotProduct,
	}.WithCanonicalID()
}

// echo returns one dense representation per input, encoding the input's text
// length so a test can tell which input produced which output.
func echo(_ context.Context, req *core.RepresentationRequest) (*core.RepresentationResponse, error) {
	data := make([]core.Representation, len(req.Input))
	bytes := 0
	for i, text := range req.Input {
		data[i] = core.Representation{Dense: []float32{float32(len(text))}}
		bytes += len(text)
	}
	return &core.RepresentationResponse{
		Data:   data,
		Spaces: map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: denseSpace(1)},
		Model:  "model",
		Usage:  core.RepresentationUsage{InputTokens: bytes, RequestCount: 1, InputBytes: bytes},
	}, nil
}

func request(inputs ...string) *core.RepresentationRequest {
	return &core.RepresentationRequest{
		Input:     inputs,
		InputType: core.EmbeddingInputDocument,
		Outputs:   []core.RepresentationKind{core.RepresentationDense},
	}
}

func TestChunkedPreservesOrderAndSumsUsage(t *testing.T) {
	req := request("a", "bb", "ccc", "dddd", "eeeee")

	var chunks [][]string
	resp, err := representationbatch.Chunked(context.Background(), req, 2,
		func(ctx context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			chunks = append(chunks, append([]string(nil), chunk.Input...))
			return echo(ctx, chunk)
		})
	if err != nil {
		t.Fatalf("Chunked: %v", err)
	}

	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want 3", len(chunks))
	}
	if len(resp.Data) != 5 {
		t.Fatalf("got %d representations for 5 inputs", len(resp.Data))
	}
	for i, want := range []float32{1, 2, 3, 4, 5} {
		if resp.Data[i].Dense[0] != want {
			t.Errorf("Data[%d] = %v, want the encoding of input %d", i, resp.Data[i].Dense, i)
		}
	}
	if resp.Usage.RequestCount != 3 {
		t.Errorf("request count = %d, want one per chunk", resp.Usage.RequestCount)
	}
	if resp.Usage.InputBytes != 15 {
		t.Errorf("input bytes = %d, want the summed 15", resp.Usage.InputBytes)
	}
	if resp.Model != "model" {
		t.Errorf("model = %q", resp.Model)
	}
	if _, ok := resp.Spaces[core.RepresentationDense]; !ok {
		t.Error("merged response lost its vector space")
	}
}

// The chunk request must carry the whole request's settings, or a chunked
// batch would silently encode its tail with different options.
func TestChunkedPropagatesRequestSettings(t *testing.T) {
	truncate := true
	req := &core.RepresentationRequest{
		Input:     []string{"a", "b", "c"},
		InputType: core.EmbeddingInputQuery,
		Outputs:   []core.RepresentationKind{core.RepresentationDense},
		Truncate:  &truncate,
	}

	var seen []core.RepresentationRequest
	if _, err := representationbatch.Chunked(context.Background(), req, 2,
		func(ctx context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			seen = append(seen, *chunk)
			return echo(ctx, chunk)
		}); err != nil {
		t.Fatalf("Chunked: %v", err)
	}

	for i, chunk := range seen {
		if chunk.InputType != core.EmbeddingInputQuery {
			t.Errorf("chunk %d lost its input type", i)
		}
		if chunk.Truncate == nil || !*chunk.Truncate {
			t.Errorf("chunk %d lost its truncate setting", i)
		}
		if len(chunk.Outputs) != 1 || chunk.Outputs[0] != core.RepresentationDense {
			t.Errorf("chunk %d lost its outputs", i)
		}
	}
}

func TestChunkedSingleCallWhenUnderSize(t *testing.T) {
	for _, size := range []int{0, -1, 5, 99} {
		t.Run(fmt.Sprintf("size %d", size), func(t *testing.T) {
			calls := 0
			resp, err := representationbatch.Chunked(context.Background(), request("a", "bb", "ccc", "dddd", "eeeee"), size,
				func(ctx context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
					calls++
					return echo(ctx, chunk)
				})
			if err != nil {
				t.Fatalf("Chunked: %v", err)
			}
			if calls != 1 {
				t.Errorf("made %d calls, want 1", calls)
			}
			if resp.Usage.RequestCount != 1 {
				t.Errorf("request count = %d", resp.Usage.RequestCount)
			}
		})
	}
}

func TestChunkedRejectsNilRequest(t *testing.T) {
	_, err := representationbatch.Chunked(context.Background(), nil, 2,
		func(context.Context, *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			t.Fatal("provider should not be called for a nil request")
			return nil, nil
		})
	if err == nil || !strings.Contains(err.Error(), "cannot be nil") {
		t.Fatalf("got %v, want a nil-request error", err)
	}
}

func TestChunkedRejectsWrongChunkCardinality(t *testing.T) {
	tests := []struct {
		name string
		fn   func(context.Context, *core.RepresentationRequest) (*core.RepresentationResponse, error)
		want string
	}{
		{
			name: "short chunk",
			fn: func(_ context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
				return &core.RepresentationResponse{
					Data:   []core.Representation{{Dense: []float32{1}}},
					Spaces: map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: denseSpace(1)},
				}, nil
			},
			want: "returned 1 representations for 2 inputs",
		},
		{
			name: "long chunk",
			fn: func(_ context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
				return &core.RepresentationResponse{
					Data:   []core.Representation{{}, {}, {}},
					Spaces: map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: denseSpace(1)},
				}, nil
			},
			want: "returned 3 representations for 2 inputs",
		},
		{
			name: "nil chunk",
			fn: func(context.Context, *core.RepresentationRequest) (*core.RepresentationResponse, error) {
				return nil, nil
			},
			want: "returned 0 representations for 2 inputs",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := representationbatch.Chunked(context.Background(), request("a", "b", "c"), 2, tc.fn)
			var mismatch *representationbatch.CountMismatchError
			if !errors.As(err, &mismatch) {
				t.Fatalf("got %v, want *CountMismatchError", err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.want)
			}
		})
	}
}

func TestChunkedUnsplitRejectsWrongCardinality(t *testing.T) {
	t.Run("nil response", func(t *testing.T) {
		_, err := representationbatch.Chunked(context.Background(), request("a"), 0,
			func(context.Context, *core.RepresentationRequest) (*core.RepresentationResponse, error) {
				return nil, nil
			})
		var mismatch *representationbatch.CountMismatchError
		if !errors.As(err, &mismatch) {
			t.Fatalf("got %v, want *CountMismatchError", err)
		}
	})

	t.Run("provider error", func(t *testing.T) {
		sentinel := errors.New("unsplit call failed")
		_, err := representationbatch.Chunked(context.Background(), request("a"), 0,
			func(context.Context, *core.RepresentationRequest) (*core.RepresentationResponse, error) {
				return nil, sentinel
			})
		if !errors.Is(err, sentinel) {
			t.Fatalf("got %v, want the provider's own error", err)
		}
	})

	t.Run("wrong count", func(t *testing.T) {
		_, err := representationbatch.Chunked(context.Background(), request("a", "b"), 0,
			func(context.Context, *core.RepresentationRequest) (*core.RepresentationResponse, error) {
				return &core.RepresentationResponse{Data: []core.Representation{{}}}, nil
			})
		var mismatch *representationbatch.CountMismatchError
		if !errors.As(err, &mismatch) {
			t.Fatalf("got %v, want *CountMismatchError", err)
		}
	})
}

// A provider that changes its space or model between chunks has produced two
// incomparable halves of one logical batch.
func TestChunkedRejectsSpaceOrModelDrift(t *testing.T) {
	tests := []struct {
		name    string
		second  func() *core.RepresentationResponse
		wantErr func(error) bool
		want    string
	}{
		{
			name: "different space id",
			second: func() *core.RepresentationResponse {
				return &core.RepresentationResponse{
					Data:   []core.Representation{{Dense: []float32{1}}},
					Spaces: map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: denseSpace(768)},
					Model:  "model",
				}
			},
			want: "changed the dense vector space",
		},
		{
			name: "missing space",
			second: func() *core.RepresentationResponse {
				return &core.RepresentationResponse{
					Data:   []core.Representation{{Dense: []float32{1}}},
					Spaces: map[core.RepresentationKind]core.VectorSpace{},
					Model:  "model",
				}
			},
			want: "changed the dense vector space",
		},
		{
			name: "extra space",
			second: func() *core.RepresentationResponse {
				return &core.RepresentationResponse{
					Data: []core.Representation{{Dense: []float32{1}}},
					Spaces: map[core.RepresentationKind]core.VectorSpace{
						core.RepresentationDense:  denseSpace(1),
						core.RepresentationSparse: sparseSpace(),
					},
					Model: "model",
				}
			},
			want: "changed the sparse vector space",
		},
		{
			name: "different model",
			second: func() *core.RepresentationResponse {
				return &core.RepresentationResponse{
					Data:   []core.Representation{{Dense: []float32{1}}},
					Spaces: map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: denseSpace(1)},
					Model:  "model-v2",
				}
			},
			want: "changed the model",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			call := 0
			_, err := representationbatch.Chunked(context.Background(), request("a", "b"), 1,
				func(ctx context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
					call++
					if call == 1 {
						return echo(ctx, chunk)
					}
					return tc.second(), nil
				})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

// A space whose ID matches but whose material fields differ is the dangerous
// case: an operator who reuses an ID across a tokenizer change would otherwise
// see two incompatible halves merged without complaint.
func TestChunkedRejectsRelabeledSpace(t *testing.T) {
	original := denseSpace(1)
	relabeled := original
	relabeled.Tokenizer = "tok-2"

	call := 0
	_, err := representationbatch.Chunked(context.Background(), request("a", "b"), 1,
		func(ctx context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			call++
			space := original
			if call == 2 {
				space = relabeled
			}
			return &core.RepresentationResponse{
				Data:   []core.Representation{{Dense: []float32{1}}},
				Spaces: map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: space},
				Model:  "model",
			}, nil
		})
	var mismatch *representationbatch.SpaceMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("got %v, want *SpaceMismatchError", err)
	}
}

func TestChunkedStopsAtFirstFailure(t *testing.T) {
	sentinel := errors.New("chunk failed")
	calls := 0
	_, err := representationbatch.Chunked(context.Background(), request("a", "b", "c", "d", "e", "f"), 2,
		func(ctx context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			calls++
			if calls == 2 {
				return nil, sentinel
			}
			return echo(ctx, chunk)
		})
	if !errors.Is(err, sentinel) {
		t.Fatalf("got %v, want the chunk's own error", err)
	}
	if calls != 2 {
		t.Errorf("made %d calls; a failed chunk must not be followed by more", calls)
	}
}

func TestChunkedStopsOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	_, err := representationbatch.Chunked(ctx, request("a", "b", "c", "d"), 1,
		func(ctx context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			calls++
			cancel()
			return echo(ctx, chunk)
		})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if calls != 1 {
		t.Errorf("made %d calls after cancellation, want 1", calls)
	}
}

func TestChunkedCancellationBeforeFirstChunk(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := representationbatch.Chunked(ctx, request("a", "b"), 1,
		func(context.Context, *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			t.Fatal("provider should not be called with an already-canceled context")
			return nil, nil
		})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}

// The unsplit path must refuse an already-canceled context too, because not
// every transport observes one: an SDK client or a stub can answer regardless.
func TestChunkedUnsplitStopsOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := representationbatch.Chunked(ctx, request("a", "b"), 0,
		func(context.Context, *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			t.Fatal("provider should not be called with an already-canceled context")
			return nil, nil
		})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}

func TestChunkedMergesEveryRequestedKind(t *testing.T) {
	req := &core.RepresentationRequest{
		Input:   []string{"a", "b", "c"},
		Outputs: []core.RepresentationKind{core.RepresentationDense, core.RepresentationSparse},
	}
	resp, err := representationbatch.Chunked(context.Background(), req, 2,
		func(_ context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			data := make([]core.Representation, len(chunk.Input))
			for i, text := range chunk.Input {
				data[i] = core.Representation{
					Dense:  []float32{float32(len(text))},
					Sparse: &core.SparseVector{Indices: []uint32{uint32(text[0])}, Values: []float32{1}},
				}
			}
			return &core.RepresentationResponse{
				Data: data,
				Spaces: map[core.RepresentationKind]core.VectorSpace{
					core.RepresentationDense:  denseSpace(1),
					core.RepresentationSparse: sparseSpace(),
				},
				Model: "model",
				Usage: core.RepresentationUsage{RequestCount: 1},
			}, nil
		})
	if err != nil {
		t.Fatalf("Chunked: %v", err)
	}
	if len(resp.Spaces) != 2 {
		t.Fatalf("merged response has %d spaces, want 2", len(resp.Spaces))
	}
	for i, item := range resp.Data {
		if item.Sparse == nil || len(item.Dense) == 0 {
			t.Errorf("item %d lost a representation kind in the merge", i)
		}
	}
	if resp.Data[2].Sparse.Indices[0] != uint32('c') {
		t.Error("merged sparse output does not follow input order")
	}
}

// Mutating the merged response's space map must not reach back into the chunk
// response the provider still holds.
func TestChunkedClonesSpaces(t *testing.T) {
	spaces := map[core.RepresentationKind]core.VectorSpace{core.RepresentationDense: denseSpace(1)}
	resp, err := representationbatch.Chunked(context.Background(), request("a", "b"), 1,
		func(_ context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			return &core.RepresentationResponse{
				Data:   []core.Representation{{Dense: []float32{1}}},
				Spaces: spaces,
				Model:  "model",
			}, nil
		})
	if err != nil {
		t.Fatalf("Chunked: %v", err)
	}
	delete(resp.Spaces, core.RepresentationDense)
	if _, ok := spaces[core.RepresentationDense]; !ok {
		t.Error("merged response shares the provider's space map")
	}
}

func TestChunkedHandlesNilSpaceMap(t *testing.T) {
	resp, err := representationbatch.Chunked(context.Background(), request("a", "b"), 1,
		func(_ context.Context, chunk *core.RepresentationRequest) (*core.RepresentationResponse, error) {
			return &core.RepresentationResponse{
				Data:  []core.Representation{{Dense: []float32{1}}},
				Model: "model",
			}, nil
		})
	if err != nil {
		t.Fatalf("Chunked: %v", err)
	}
	if resp.Spaces != nil {
		t.Errorf("spaces = %v, want nil preserved", resp.Spaces)
	}
}

func TestErrorMessages(t *testing.T) {
	count := &representationbatch.CountMismatchError{Want: 3, Got: 1}
	if !strings.Contains(count.Error(), "1 representations for 3 inputs") {
		t.Errorf("CountMismatchError message = %q", count.Error())
	}
	space := &representationbatch.SpaceMismatchError{Kind: core.RepresentationSparse, Want: "a", Got: "b"}
	if !strings.Contains(space.Error(), "sparse vector space from a to b") {
		t.Errorf("SpaceMismatchError message = %q", space.Error())
	}
	model := &representationbatch.ModelMismatchError{Want: "a", Got: "b"}
	if !strings.Contains(model.Error(), `from "a" to "b"`) {
		t.Errorf("ModelMismatchError message = %q", model.Error())
	}
}
