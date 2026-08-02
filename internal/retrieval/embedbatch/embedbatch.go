// Package embedbatch adapts providers whose native batch shape does not match
// the batch-first contract of retrieval.Embedder.
//
// Two mismatches occur in practice. Some models accept exactly one input per
// call (Amazon Titan; Gemini's embedding-2 family on Vertex), which needs
// FanOut. Others accept batches but cap them below what a caller may pass,
// which needs Chunked. Both preserve input order and sum usage, so the
// provider keeps the invariant that len(Vectors) == len(Input).
package embedbatch

import (
	"context"
	"fmt"
	"sync"

	"github.com/regularkevvv/agentic/internal/retrieval"
)

// defaultConcurrency bounds in-flight requests when a caller does not choose.
// Embedding endpoints are rate-limited per account, so fanning out an entire
// batch at once trades a burst of 429s for no useful latency win.
const defaultConcurrency = 8

// FanOut embeds inputs one at a time, running up to concurrency requests
// together, and reassembles the vectors in input order.
//
// The first error cancels the remaining requests and is returned: a partial
// result would break the len(Vectors) == len(Input) invariant that callers
// rely on to join vectors back to their source texts.
func FanOut(
	ctx context.Context,
	inputs []string,
	concurrency int,
	fn func(context.Context, string) ([]float32, retrieval.EmbeddingUsage, error),
) ([][]float32, retrieval.EmbeddingUsage, error) {
	if concurrency <= 0 {
		concurrency = defaultConcurrency
	}
	if concurrency > len(inputs) {
		concurrency = len(inputs)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	vectors := make([][]float32, len(inputs))
	usages := make([]retrieval.EmbeddingUsage, len(inputs))

	var (
		wg       sync.WaitGroup
		mu       sync.Mutex
		firstErr error
	)
	sem := make(chan struct{}, concurrency)

	for i, text := range inputs {
		wg.Add(1)
		go func(i int, text string) {
			defer wg.Done()

			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				return
			}

			vec, usage, err := fn(ctx, text)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
					cancel()
				}
				mu.Unlock()
				return
			}
			// Disjoint indices, so these writes need no lock.
			vectors[i] = vec
			usages[i] = usage
		}(i, text)
	}
	wg.Wait()

	if firstErr != nil {
		return nil, retrieval.EmbeddingUsage{}, firstErr
	}
	return vectors, sumUsage(usages), nil
}

// Chunked embeds inputs in batches of at most size, preserving input order.
//
// Chunks run sequentially. A provider that caps its batch size is usually
// signaling a rate limit rather than a payload limit, so issuing the chunks
// concurrently tends to trade 429s for nothing.
func Chunked(
	ctx context.Context,
	inputs []string,
	size int,
	fn func(context.Context, []string) ([][]float32, retrieval.EmbeddingUsage, error),
) ([][]float32, retrieval.EmbeddingUsage, error) {
	if size <= 0 || size >= len(inputs) {
		vectors, usage, err := fn(ctx, inputs)
		if err != nil {
			return nil, retrieval.EmbeddingUsage{}, err
		}
		return vectors, usage, nil
	}

	vectors := make([][]float32, 0, len(inputs))
	usages := make([]retrieval.EmbeddingUsage, 0, (len(inputs)+size-1)/size)

	for start := 0; start < len(inputs); start += size {
		end := min(start+size, len(inputs))

		chunk, usage, err := fn(ctx, inputs[start:end])
		if err != nil {
			return nil, retrieval.EmbeddingUsage{}, err
		}
		if len(chunk) != end-start {
			return nil, retrieval.EmbeddingUsage{}, &CountMismatchError{Want: end - start, Got: len(chunk)}
		}
		vectors = append(vectors, chunk...)
		usages = append(usages, usage)
	}
	return vectors, sumUsage(usages), nil
}

// CountMismatchError reports a provider returning a number of vectors that
// does not match the number of inputs it was given.
type CountMismatchError struct {
	Want int
	Got  int
}

func (e *CountMismatchError) Error() string {
	return fmt.Sprintf("embedding batch returned %d vectors for %d inputs", e.Got, e.Want)
}

func sumUsage(usages []retrieval.EmbeddingUsage) retrieval.EmbeddingUsage {
	var total retrieval.EmbeddingUsage
	for _, u := range usages {
		total.PromptTokens += u.PromptTokens
		total.TotalTokens += u.TotalTokens
	}
	return total
}
