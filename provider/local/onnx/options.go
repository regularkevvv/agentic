package onnx

import (
	"errors"

	agentic "github.com/regularkevvv/agentic"
)

// defaultMaxTokens is the sequence limit BERT and RoBERTa positional embeddings
// impose, and the maximum the reference export declares. Beyond it the graph
// does not produce a worse answer; it indexes past its position table.
const defaultMaxTokens = 512

// defaultBatchBytes caps the logits tensor one forward pass allocates.
//
// That tensor is batch × width × vocabulary × 4 bytes, and the vocabulary term
// is what makes it enormous: a single 512-token input against Granite's 50,265
// entries is already 103 MB of logits for a vector with a few hundred nonzero
// coordinates. Because the vocabulary is fixed for a given model, this ceiling
// is really a cap on total token positions per pass — about 1,300 at that
// vocabulary, whether that is one long document or a hundred short queries.
//
// A row that exceeds the budget on its own is still run, alone. Refusing to
// encode an input because a memory ceiling would rather it did not is not a
// useful failure.
const defaultBatchBytes = 256 << 20

// Option configures an [Encoder].
type Option func(*config)

type config struct {
	libraryPath string
	maxTokens   int
	padTokenID  int64
	batchBytes  int
	limits      agentic.RepresentationLimits
}

func (c config) validate() error {
	if c.maxTokens <= 0 {
		return errors.New("onnx: maximum token count must be positive")
	}
	if c.padTokenID < 0 {
		return errors.New("onnx: padding token id cannot be negative")
	}
	if c.batchBytes <= 0 {
		return errors.New("onnx: maximum batch bytes must be positive")
	}
	return nil
}

// WithLibraryPath points ONNX Runtime at its shared library —
// libonnxruntime.dylib, libonnxruntime.so, or onnxruntime.dll.
//
// Without it the AGENTIC_ONNX_LIBRARY environment variable is used, and without
// that the binding falls back to the platform's default name, which resolves
// only if the library is already on the loader's search path.
//
// The runtime environment is global to the process: whichever encoder is
// constructed first settles the path, and a later different one is ignored.
func WithLibraryPath(path string) Option {
	return func(c *config) { c.libraryPath = path }
}

// WithMaxTokens sets the longest tokenized input the encoder will accept,
// defaulting to 512.
//
// Inputs above it are rejected with their token count rather than truncated.
// Raise it only for a model whose positional embeddings actually extend that
// far; the limit exists because exceeding it is a runtime fault inside the
// graph rather than a degraded result.
func WithMaxTokens(tokens int) Option {
	return func(c *config) { c.maxTokens = tokens }
}

// WithPadTokenID sets the id written into padded positions, defaulting to 0.
//
// The value does not affect the result: padded positions are masked out of
// attention and out of pooling, which the live tests assert by encoding one
// padded row under two padding ids and comparing every coordinate. It still has
// to be a token the model has, since it indexes the embedding table, and zero
// is the only id every vocabulary contains. Set the model's real padding id if
// you would rather the tensors read the way its own tooling writes them.
func WithPadTokenID(id int64) Option {
	return func(c *config) { c.padTokenID = id }
}

// WithMaxBatchBytes caps the logits tensor a single forward pass allocates,
// defaulting to 256 MiB.
//
// See [defaultBatchBytes] for what that buys. Lower it on a memory-constrained
// machine; raising it does not make encoding faster, because the encoder groups
// by token length and a group is closed by padding waste before it is closed by
// this ceiling for any realistic mix of inputs.
func WithMaxBatchBytes(bytes int) Option {
	return func(c *config) { c.batchBytes = bytes }
}

// WithLimits replaces the request-size ceilings, which default to
// [agentic.DefaultRepresentationLimits].
//
// The response-side ceilings still apply. MaxSparseNonZero is the one that
// matters here: a SPLADE head is sparse by training rather than by
// construction, so a long document can carry thousands of nonzero coordinates
// and nothing in the graph bounds that.
func WithLimits(limits agentic.RepresentationLimits) Option {
	return func(c *config) { c.limits = limits }
}
