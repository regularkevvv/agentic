package embedbatch_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/embedbatch"
)

// vectorFor derives a distinguishable vector from a text so order errors are
// visible rather than silently plausible.
func vectorFor(text string) []float32 {
	return []float32{float32(len(text)), float32(text[0])}
}

func TestFanOutPreservesOrder(t *testing.T) {
	inputs := []string{"a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"}

	vectors, usage, err := embedbatch.FanOut(context.Background(), inputs, 3,
		func(_ context.Context, text string) ([]float32, core.EmbeddingUsage, error) {
			return vectorFor(text), core.EmbeddingUsage{PromptTokens: len(text), TotalTokens: len(text)}, nil
		})
	if err != nil {
		t.Fatalf("FanOut() = %v, want nil", err)
	}

	if len(vectors) != len(inputs) {
		t.Fatalf("got %d vectors, want %d", len(vectors), len(inputs))
	}
	for i, text := range inputs {
		want := vectorFor(text)
		if vectors[i][0] != want[0] || vectors[i][1] != want[1] {
			t.Errorf("vectors[%d] = %v, want %v (order not preserved)", i, vectors[i], want)
		}
	}

	wantTokens := 0
	for _, text := range inputs {
		wantTokens += len(text)
	}
	if usage.TotalTokens != wantTokens || usage.PromptTokens != wantTokens {
		t.Errorf("usage = %+v, want %d tokens in both fields", usage, wantTokens)
	}
}

func TestFanOutRespectsConcurrencyLimit(t *testing.T) {
	const limit = 3
	var (
		mu       sync.Mutex
		inFlight int
		peak     int
	)

	inputs := make([]string, 25)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("input-%02d", i)
	}

	_, _, err := embedbatch.FanOut(context.Background(), inputs, limit,
		func(_ context.Context, text string) ([]float32, core.EmbeddingUsage, error) {
			mu.Lock()
			inFlight++
			if inFlight > peak {
				peak = inFlight
			}
			mu.Unlock()

			defer func() {
				mu.Lock()
				inFlight--
				mu.Unlock()
			}()
			return vectorFor(text), core.EmbeddingUsage{}, nil
		})
	if err != nil {
		t.Fatalf("FanOut() = %v, want nil", err)
	}
	if peak > limit {
		t.Errorf("peak concurrency was %d, want at most %d", peak, limit)
	}
}

func TestFanOutFirstErrorCancelsAndReturns(t *testing.T) {
	sentinel := errors.New("provider rejected input")

	inputs := make([]string, 40)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("input-%02d", i)
	}

	var calls atomic.Int64
	vectors, usage, err := embedbatch.FanOut(context.Background(), inputs, 4,
		func(ctx context.Context, text string) ([]float32, core.EmbeddingUsage, error) {
			if calls.Add(1) == 1 {
				return nil, core.EmbeddingUsage{}, sentinel
			}
			// Honor cancellation so the fan-out can actually wind down.
			select {
			case <-ctx.Done():
				return nil, core.EmbeddingUsage{}, ctx.Err()
			default:
			}
			return vectorFor(text), core.EmbeddingUsage{}, nil
		})

	if !errors.Is(err, sentinel) {
		t.Fatalf("FanOut() = %v, want %v", err, sentinel)
	}
	// A partial result would break len(Vectors) == len(Input) for the caller.
	if vectors != nil {
		t.Errorf("vectors = %v, want nil on error", vectors)
	}
	if (usage != core.EmbeddingUsage{}) {
		t.Errorf("usage = %+v, want zero on error", usage)
	}
}

func TestFanOutDefaultsConcurrencyWhenNonPositive(t *testing.T) {
	inputs := []string{"a", "b", "c"}
	vectors, _, err := embedbatch.FanOut(context.Background(), inputs, 0,
		func(_ context.Context, text string) ([]float32, core.EmbeddingUsage, error) {
			return vectorFor(text), core.EmbeddingUsage{}, nil
		})
	if err != nil {
		t.Fatalf("FanOut() = %v, want nil", err)
	}
	if len(vectors) != len(inputs) {
		t.Fatalf("got %d vectors, want %d", len(vectors), len(inputs))
	}
}

func TestFanOutStopsOnCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	inputs := make([]string, 20)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("input-%02d", i)
	}

	var calls atomic.Int64
	_, _, err := embedbatch.FanOut(ctx, inputs, 2,
		func(ctx context.Context, text string) ([]float32, core.EmbeddingUsage, error) {
			calls.Add(1)
			return nil, core.EmbeddingUsage{}, ctx.Err()
		})
	if err == nil {
		t.Fatal("FanOut() = nil, want an error for a cancelled context")
	}
}

func TestChunkedSplitsAndPreservesOrder(t *testing.T) {
	inputs := []string{"a", "bb", "ccc", "dddd", "eeeee"}
	var batchSizes []int

	vectors, usage, err := embedbatch.Chunked(context.Background(), inputs, 2,
		func(_ context.Context, batch []string) ([][]float32, core.EmbeddingUsage, error) {
			batchSizes = append(batchSizes, len(batch))
			out := make([][]float32, len(batch))
			tokens := 0
			for i, text := range batch {
				out[i] = vectorFor(text)
				tokens += len(text)
			}
			return out, core.EmbeddingUsage{PromptTokens: tokens, TotalTokens: tokens}, nil
		})
	if err != nil {
		t.Fatalf("Chunked() = %v, want nil", err)
	}

	if want := []int{2, 2, 1}; fmt.Sprint(batchSizes) != fmt.Sprint(want) {
		t.Errorf("batch sizes = %v, want %v", batchSizes, want)
	}
	for i, text := range inputs {
		want := vectorFor(text)
		if vectors[i][0] != want[0] || vectors[i][1] != want[1] {
			t.Errorf("vectors[%d] = %v, want %v (order not preserved across chunks)", i, vectors[i], want)
		}
	}
	if usage.TotalTokens != 15 {
		t.Errorf("usage.TotalTokens = %d, want 15 (summed across chunks)", usage.TotalTokens)
	}
}

func TestChunkedSendsOneBatchWhenSizeCoversInput(t *testing.T) {
	inputs := []string{"a", "bb", "ccc"}

	for _, size := range []int{0, -1, 3, 10} {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			calls := 0
			vectors, _, err := embedbatch.Chunked(context.Background(), inputs, size,
				func(_ context.Context, batch []string) ([][]float32, core.EmbeddingUsage, error) {
					calls++
					out := make([][]float32, len(batch))
					for i, text := range batch {
						out[i] = vectorFor(text)
					}
					return out, core.EmbeddingUsage{}, nil
				})
			if err != nil {
				t.Fatalf("Chunked() = %v, want nil", err)
			}
			if calls != 1 {
				t.Errorf("made %d calls, want 1", calls)
			}
			if len(vectors) != len(inputs) {
				t.Errorf("got %d vectors, want %d", len(vectors), len(inputs))
			}
		})
	}
}

func TestChunkedPropagatesError(t *testing.T) {
	sentinel := errors.New("rate limited")

	_, _, err := embedbatch.Chunked(context.Background(), []string{"a", "b", "c"}, 2,
		func(_ context.Context, batch []string) ([][]float32, core.EmbeddingUsage, error) {
			return nil, core.EmbeddingUsage{}, sentinel
		})
	if !errors.Is(err, sentinel) {
		t.Fatalf("Chunked() = %v, want %v", err, sentinel)
	}
}

func TestChunkedRejectsCountMismatch(t *testing.T) {
	_, _, err := embedbatch.Chunked(context.Background(), []string{"a", "b", "c"}, 2,
		func(_ context.Context, batch []string) ([][]float32, core.EmbeddingUsage, error) {
			// Return one vector short — a silent truncation if unchecked.
			return [][]float32{vectorFor(batch[0])}, core.EmbeddingUsage{}, nil
		})

	var mismatch *embedbatch.CountMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("Chunked() = %v, want a *CountMismatchError", err)
	}
	if mismatch.Want != 2 || mismatch.Got != 1 {
		t.Errorf("mismatch = %+v, want Want=2 Got=1", mismatch)
	}
	if mismatch.Error() == "" {
		t.Error("CountMismatchError.Error() is empty")
	}
}
