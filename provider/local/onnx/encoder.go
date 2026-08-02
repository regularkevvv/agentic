// Package onnx implements [agentic.RepresentationEncoder] in this process,
// through ONNX Runtime, with no server and no network. This directory is a
// separate Go module and deliberately not part of
// github.com/regularkevvv/agentic, because it requires CGO, a native ONNX
// Runtime shared library, and a statically linked tokenizer — none of which can
// be a condition of using the library. That is why `go get
// github.com/regularkevvv/agentic` never pulls this package, and why a
// directory under provider/ is absent from the root module's build. Working on
// it means `cd provider/local/onnx`, with the setup in README.md done first.
//
// # What it produces
//
// Learned sparse vectors, and nothing else. Dense and multi-vector requests
// return [agentic.UnsupportedRepresentationError] rather than an answer derived
// from the wrong reduction. The target is the SPLADE family — a masked-language
// -model head pooled into vocabulary weights — which is what makes a document
// about an "automobile" carry weight on "car", a word it never contained.
//
// # Nothing is downloaded
//
// [New] takes filesystem paths and reads them. There is no model cache, no
// registry lookup, and no HTTP client in this package; a model you have not
// already exported is an error rather than a fetch. Producing the graph is a
// documented one-time step — see provider/local/onnx/export_onnx.py — and
// keeping it a step is the point: a 117 MiB artifact that appears by surprise
// during a test run is not a dependency anyone agreed to.
//
// # Batch width, not batch size, is the cost
//
// Every row in one forward pass is padded to the widest row in it, and padding
// buys compute nobody asked for. Measured on 2026-08-01, three short inputs
// padded to a common width of 18 took 20 ms as one call against 13 ms as three.
// The encoder therefore orders inputs by token length and groups neighbors, so
// a long document never drags short ones up to its width and no row is ever
// padded past twice its own length. The rule, and the bound it buys, are in
// batch.go.
package onnx

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	agentic "github.com/regularkevvv/agentic"
)

// providerName identifies this package in vector spaces and errors. It reaches
// stored data: a space left without a provider is completed with this, and the
// canonical space ID is derived from it.
const providerName = "onnx"

// The graph's input and output names. They are fixed rather than configurable
// because they are half of the contract with the export step: a graph that does
// not use these names was not produced by export_onnx.py, and guessing which of
// its tensors is the attention mask is how a caller ends up with plausible
// wrong numbers.
const (
	inputIDsName      = "input_ids"
	attentionMaskName = "attention_mask"
	logitsName        = "logits"
)

// libraryEnvVar names the ONNX Runtime shared library when [WithLibraryPath]
// is not used.
//
// It is read from the environment for the same reason a compiler reads
// LD_LIBRARY_PATH and unlike a credential: ONNX Runtime has no standard install
// location on any platform this runs on, so the path is a property of the
// machine rather than of the program. The option always wins over it.
const libraryEnvVar = "AGENTIC_ONNX_LIBRARY"

// Encoder runs an exported masked-language-model graph and pools its logits
// into a learned sparse vector.
//
// It owns memory outside Go's heap and must be closed; see [Encoder.Close].
type Encoder struct {
	tokenizer *tokenizer

	space      agentic.VectorSpace
	vocabulary int
	maxTokens  int
	padTokenID int64
	batchBytes int
	limits     agentic.RepresentationLimits

	// mu serializes Encode and guards the fields below it.
	//
	// ONNX Runtime already runs one graph across every core it is configured
	// for, so concurrent calls would contend for the same threads rather than
	// scale. Serializing them costs nothing measurable, keeps one pooling
	// accumulator reusable across the whole batch, and makes Close on a
	// concurrently-encoding encoder an error instead of a use-after-free.
	mu      sync.Mutex
	session *ort.DynamicAdvancedSession
	pool    pooler
	closed  bool
}

// New loads an exported SPLADE-family graph and its tokenizer and encodes with
// them in this process.
//
// modelPath is an ONNX file whose graph takes int64 input_ids and
// attention_mask and returns float32 logits of shape
// [batch, sequence, vocabulary], with the batch and sequence axes dynamic.
// provider/local/onnx/export_onnx.py produces exactly that. Pooling is
// deliberately outside the graph, so what the model contributes and what this
// package contributes stay separable.
//
// tokenizerPath is a Hugging Face tokenizer.json — the portable fast-tokenizer
// format, not a directory and not a sentencepiece model.
//
// space is the identity that will be persisted beside every vector. Provider
// defaults to "onnx", Kind must be sparse, Metric defaults to dot product
// because a learned sparse weight is part of the score rather than a direction,
// Dimensions is filled from the graph's own output width when left zero, and ID
// is derived from the rest when left empty. Model is required: a file name is
// not an identity, since two exports of different weights can share one.
//
// Nothing is downloaded, then or later. Both paths are read from the local
// filesystem and this package contains no HTTP client at all.
//
// The graph is loaded twice here — once to read the vocabulary from its output
// shape, once for the session that will run it — which is what makes the
// vocabulary observed rather than assumed. Measured on darwin/arm64 against the
// 117 MiB Granite export, that costs 0.12 s of a 0.30 s construction. An encoder
// is meant to outlive the request that needed it.
func New(modelPath, tokenizerPath string, space agentic.VectorSpace, opts ...Option) (encoder *Encoder, err error) {
	if modelPath == "" {
		return nil, errors.New("onnx: model path cannot be empty")
	}
	if tokenizerPath == "" {
		return nil, errors.New("onnx: tokenizer path cannot be empty")
	}

	cfg := config{
		maxTokens:  defaultMaxTokens,
		batchBytes: defaultBatchBytes,
		limits:     agentic.DefaultRepresentationLimits(),
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	// The space is refused here, before the runtime is brought up and the graph
	// is read, so declaring the wrong kind costs nothing.
	if space.Kind != "" && space.Kind != agentic.RepresentationSparse {
		return nil, &agentic.UnsupportedRepresentationError{
			Provider:  providerName,
			Kind:      space.Kind,
			Supported: []agentic.RepresentationKind{agentic.RepresentationSparse},
		}
	}
	if space.Model == "" {
		return nil, errors.New("onnx: vector space model cannot be empty " +
			"(the export's file name is not a model identity: two exports of " +
			"different weights can share one)")
	}

	if err := initializeRuntime(cfg.libraryPath); err != nil {
		return nil, err
	}
	vocabulary, err := readVocabulary(modelPath)
	if err != nil {
		return nil, err
	}
	space, err = resolveSpace(space, vocabulary)
	if err != nil {
		return nil, err
	}

	tk, err := newTokenizer(tokenizerPath)
	if err != nil {
		return nil, err
	}
	// From here on the tokenizer is live, so every later failure has to release
	// it: it holds a Rust-side allocation that no garbage collector reclaims.
	defer func() {
		if err != nil {
			_ = tk.close()
		}
	}()
	if size := tk.vocabularySize(); size > vocabulary {
		return nil, fmt.Errorf("onnx: tokenizer has %d tokens but the graph's "+
			"output is %d wide; this tokenizer does not belong to this model, "+
			"and its ids would index past the embedding table", size, vocabulary)
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{inputIDsName, attentionMaskName},
		[]string{logitsName},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("onnx: load %s: %w", modelPath, err)
	}

	return &Encoder{
		tokenizer:  tk,
		space:      space,
		vocabulary: vocabulary,
		maxTokens:  cfg.maxTokens,
		padTokenID: cfg.padTokenID,
		batchBytes: cfg.batchBytes,
		limits:     cfg.limits,
		session:    session,
	}, nil
}

// runtimeMu serializes bringing up the process-global ONNX Runtime environment.
var runtimeMu sync.Mutex

// initializeRuntime brings up ONNX Runtime on first use.
//
// The environment belongs to the onnxruntime_go package rather than to any one
// encoder, so it is created once and never torn down here: a second encoder in
// the same process must not destroy the runtime the first is still using. A
// program that wants it released calls onnxruntime_go.DestroyEnvironment itself,
// after closing every encoder.
//
// One consequence is worth stating: the library path of whichever encoder is
// constructed first is the one the process uses. A later, different path is
// ignored rather than honored, because there is nothing to honor it with.
func initializeRuntime(libraryPath string) error {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	if ort.IsInitialized() {
		return nil
	}
	if libraryPath == "" {
		libraryPath = os.Getenv(libraryEnvVar)
	}
	if libraryPath != "" {
		ort.SetSharedLibraryPath(libraryPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("onnx: initialize onnx runtime (point WithLibraryPath "+
			"or %s at libonnxruntime.dylib or libonnxruntime.so): %w", libraryEnvVar, err)
	}
	return nil
}

// readVocabulary reports the logical width of the sparse space by reading the
// graph's own output shape.
//
// A constant would be wrong the first time somebody exports a different
// checkpoint, and wrong silently: a vocabulary that is too small truncates
// every vector, and one that is too large reads past the logits tensor. The
// shape is the only description of the width that ships with the weights.
func readVocabulary(modelPath string) (int, error) {
	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return 0, fmt.Errorf("onnx: read %s: %w", modelPath, err)
	}
	if err := requireInputs(inputs); err != nil {
		return 0, err
	}

	for _, output := range outputs {
		if output.Name != logitsName {
			continue
		}
		if output.DataType != ort.TensorElementDataTypeFloat {
			return 0, fmt.Errorf("onnx: graph output %q is %s, not float32",
				logitsName, output.DataType)
		}
		if len(output.Dimensions) != 3 {
			return 0, fmt.Errorf("onnx: graph output %q has shape %s, expected "+
				"[batch, sequence, vocabulary]", logitsName, output.Dimensions)
		}
		width := output.Dimensions[2]
		if width <= 0 {
			return 0, fmt.Errorf("onnx: graph output %q declares a dynamic "+
				"vocabulary axis (shape %s), so the width cannot be observed; "+
				"export it with a fixed vocabulary or set the vector space's "+
				"Dimensions", logitsName, output.Dimensions)
		}
		return int(width), nil
	}
	return 0, fmt.Errorf("onnx: graph has no output named %q", logitsName)
}

// requireInputs checks the graph takes the two tensors this package feeds it,
// so a mismatched export fails at construction with the names in the message
// rather than inside ONNX Runtime on the first Encode.
func requireInputs(inputs []ort.InputOutputInfo) error {
	found := make(map[string]bool, len(inputs))
	for _, input := range inputs {
		found[input.Name] = true
	}
	for _, name := range []string{inputIDsName, attentionMaskName} {
		if !found[name] {
			return fmt.Errorf("onnx: graph has no input named %q; it takes %s",
				name, inputNames(inputs))
		}
	}
	return nil
}

func inputNames(inputs []ort.InputOutputInfo) string {
	names := make([]string, len(inputs))
	for i, input := range inputs {
		names[i] = input.Name
	}
	return fmt.Sprintf("%q", names)
}

// resolveSpace reconciles the caller's declared space with the graph.
//
// The graph wins on width, because it is the thing that produces the
// coordinates. A caller who declared a different width is told rather than
// quietly corrected: the mismatch usually means the space ID they are about to
// key an index on belongs to another model.
func resolveSpace(space agentic.VectorSpace, vocabulary int) (agentic.VectorSpace, error) {
	if space.Kind == "" {
		space.Kind = agentic.RepresentationSparse
	}
	if space.Provider == "" {
		space.Provider = providerName
	}
	if space.Metric == "" {
		space.Metric = agentic.SimilarityDotProduct
	}
	switch {
	case space.Dimensions == 0:
		space.Dimensions = vocabulary
	case space.Dimensions != vocabulary:
		return space, fmt.Errorf("onnx: vector space declares a vocabulary of %d "+
			"but the graph's output is %d wide", space.Dimensions, vocabulary)
	}
	space = space.WithCanonicalID()
	if err := space.Validate(); err != nil {
		return space, fmt.Errorf("onnx: vector space: %w", err)
	}
	return space, nil
}

// Name implements [agentic.RepresentationEncoder]. It reports the model
// identifier from the vector space, which is the only model name this package
// has: an ONNX file declares no identity of its own.
func (e *Encoder) Name() string { return e.space.Model }

// Capabilities implements [agentic.RepresentationEncoder].
//
// Truncation is not advertised. The exported graph accepts sequences up to the
// model's positional limit and nothing beyond it, and an over-long input is
// rejected with its token count rather than clipped — silently dropping the end
// of a document produces a vector for text the caller never asked about.
//
// MaximumBatchSize is zero because this encoder splits a request into forward
// passes itself; the caller's batch is a request shape, not a hardware one.
//
// An empty sparse vector is not allowed. A SPLADE head that predicts no
// vocabulary entry at all for a non-empty input is a broken graph rather than a
// short document, and storing the empty vector would put an unmatchable row in
// an index instead of failing.
func (e *Encoder) Capabilities() agentic.RepresentationCapabilities {
	return agentic.RepresentationCapabilities{
		Outputs: []agentic.RepresentationKind{agentic.RepresentationSparse},
		// A SPLADE head is symmetric: queries and documents go through the same
		// weights with no task instruction, so both roles are accepted, neither
		// changes the request, and the role is not part of the space identity.
		InputTypes: []agentic.EmbeddingInputType{
			agentic.EmbeddingInputNone,
			agentic.EmbeddingInputQuery,
			agentic.EmbeddingInputDocument,
		},
		SupportsTruncation:  false,
		SupportsMultiOutput: false,
	}
}

func (e *Encoder) validator() agentic.RepresentationValidator {
	return agentic.RepresentationValidator{
		Provider:     providerName,
		Capabilities: e.Capabilities(),
		Limits:       e.limits,
	}
}

// Encode implements [agentic.RepresentationEncoder].
//
// Cancellation is observed between forward passes rather than inside one. ONNX
// Runtime's Run has no cancellation, so a pass that has already started runs to
// completion; ctx bounds how many more begin.
func (e *Encoder) Encode(ctx context.Context, req *agentic.RepresentationRequest) (*agentic.RepresentationResponse, error) {
	validator := e.validator()
	if err := validator.ValidateRequest(req); err != nil {
		return nil, err
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("onnx: encoder is closed")
	}

	rows, tokens, err := e.tokenize(req.Input)
	if err != nil {
		return nil, err
	}

	lengths := make([]int, len(rows))
	for i, row := range rows {
		lengths[i] = len(row)
	}
	groups := groupByWidth(lengths, e.vocabulary, e.batchBytes)

	data := make([]agentic.Representation, len(rows))
	for _, group := range groups {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := e.encodeGroup(rows, group, data); err != nil {
			return nil, err
		}
	}

	inputBytes := 0
	for _, text := range req.Input {
		inputBytes += len(text)
	}
	resp := &agentic.RepresentationResponse{
		Data:   data,
		Spaces: map[agentic.RepresentationKind]agentic.VectorSpace{agentic.RepresentationSparse: e.space},
		Model:  e.space.Model,
		Usage: agentic.RepresentationUsage{
			// The token count is counted, not estimated: this encoder holds the
			// tokenizer that produced the ids. RequestCount is forward passes,
			// which is the in-process analog of a provider call and what makes
			// this comparable to a hosted API. OutputBytes stays zero because
			// there is no payload — the vectors are already in the caller's
			// address space.
			InputTokens:  tokens,
			RequestCount: len(groups),
			InputBytes:   inputBytes,
		},
	}
	if err := validator.ValidateResponse(req, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// tokenize turns every input into model ids, and returns the total token count
// alongside them.
//
// An input longer than the sequence limit is rejected here, before any
// inference is done, and the error names the position and both counts without
// echoing the text.
func (e *Encoder) tokenize(inputs []string) ([][]int64, int, error) {
	rows := make([][]int64, len(inputs))
	total := 0
	for i, text := range inputs {
		ids, err := e.tokenizer.encode(text)
		if err != nil {
			return nil, 0, fmt.Errorf("onnx: tokenize input %d: %w", i, err)
		}
		if len(ids) > e.maxTokens {
			return nil, 0, &agentic.InvalidRepresentationRequestError{
				Invariant: "input.sequence_length",
				Detail: fmt.Sprintf("input %d is %d tokens, the model accepts %d",
					i, len(ids), e.maxTokens),
			}
		}
		rows[i] = ids
		total += len(ids)
	}
	return rows, total, nil
}

// encodeGroup runs one forward pass over group and writes its pooled vectors
// into out at each row's original position.
//
// Every tensor is destroyed before it returns. ONNX Runtime allocations live
// outside Go's heap and are never collected, so a leaked tensor is a leak for
// the life of the process; the destroy errors join the result rather than being
// dropped, because a failure to release is the one thing that compounds.
func (e *Encoder) encodeGroup(rows [][]int64, group []int, out []agentic.Representation) (err error) {
	batch := len(group)
	width := 0
	for _, i := range group {
		if len(rows[i]) > width {
			width = len(rows[i])
		}
	}

	// The padding id is irrelevant to the result — the mask, not the id, is what
	// silences a position — but it still indexes the embedding table, so it has
	// to be a token the model has. Zero is the only id every vocabulary has.
	ids := make([]int64, batch*width)
	mask := make([]int64, batch*width)
	for r, i := range group {
		row := rows[i]
		base := r * width
		copy(ids[base:], row)
		for j := range row {
			mask[base+j] = 1
		}
		for j := len(row); j < width; j++ {
			ids[base+j] = e.padTokenID
		}
	}

	shape := ort.NewShape(int64(batch), int64(width))
	idTensor, err := ort.NewTensor(shape, ids)
	if err != nil {
		return fmt.Errorf("onnx: input_ids tensor: %w", err)
	}
	defer func() { err = errors.Join(err, idTensor.Destroy()) }()

	maskTensor, err := ort.NewTensor(shape, mask)
	if err != nil {
		return fmt.Errorf("onnx: attention_mask tensor: %w", err)
	}
	defer func() { err = errors.Join(err, maskTensor.Destroy()) }()

	logits, err := ort.NewEmptyTensor[float32](
		ort.NewShape(int64(batch), int64(width), int64(e.vocabulary)))
	if err != nil {
		return fmt.Errorf("onnx: logits tensor: %w", err)
	}
	defer func() { err = errors.Join(err, logits.Destroy()) }()

	if err := e.session.Run([]ort.Value{idTensor, maskTensor}, []ort.Value{logits}); err != nil {
		return fmt.Errorf("onnx: run: %w", err)
	}

	// GetData aliases the tensor's own memory, so pooling has to finish before
	// the deferred Destroy above runs. The pooled vector is a Go allocation and
	// outlives it.
	raw := logits.GetData()
	span := width * e.vocabulary
	for r, i := range group {
		out[i] = agentic.Representation{
			Sparse: e.pool.reduce(raw[r*span:(r+1)*span], mask[r*width:(r+1)*width], e.vocabulary),
		}
	}
	return nil
}

// Close releases the session and the tokenizer, and is safe to call twice.
//
// Both hold memory outside Go's heap that no garbage collector reclaims, so an
// encoder that is never closed leaks a loaded model for the life of the
// process. Encode after Close returns an error rather than entering a destroyed
// session, which would be a crash.
//
// The process-global ONNX Runtime environment is deliberately left up: another
// encoder may still be using it. A program that wants it down calls
// onnxruntime_go.DestroyEnvironment after closing every encoder.
func (e *Encoder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true

	var errs []error
	if e.session != nil {
		errs = append(errs, e.session.Destroy())
		e.session = nil
	}
	if e.tokenizer != nil {
		errs = append(errs, e.tokenizer.close())
		e.tokenizer = nil
	}
	return errors.Join(errs...)
}

// Compile-time check that Encoder satisfies the contract it claims. There is no
// Embedder here on purpose: this encoder produces no dense output, and adapting
// a sparse vector into one would be an invented answer.
var _ agentic.RepresentationEncoder = (*Encoder)(nil)
