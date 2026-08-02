# Changelog

All notable changes to this project are documented here.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, breaking changes may appear in minor releases.

## [Unreleased]

Adds multi-representation inference: a batch-first `RepresentationEncoder`
beside the existing dense-only `Embedder`, so a model that produces dense,
learned sparse, and token-level output from one forward pass can return all
three without flattening them into `[][]float32` or inventing a vendor
interface per application.

Agentic normalizes inference and stops there. Indexing, BM25, candidate fusion,
and final ranking belong to the consuming retrieval system, which is why
nothing added here returns a score.

This is additive. Existing `Embedder` users compile and behave as before.

### Added

- **`RepresentationEncoder`** — returns one `Representation` per input, in
  input order, carrying exactly the requested kinds: `RepresentationDense`,
  `RepresentationSparse`, and `RepresentationMultiVector`. Root helpers
  `EncodeQueries` and `EncodeDocuments` set the retrieval role and nothing
  else.
- **Vector-space identity** — every output carries a `VectorSpace` whose ID is
  a deterministic, inspectable hash of the fields that make values comparable.
  A revision, tokenizer, dimension, or metric change lands in a new space
  instead of silently corrupting an existing index.
- **Typed errors** — `ErrUnsupportedRepresentation`,
  `ErrInvalidRepresentationRequest`, and `ErrInvalidRepresentationResponse`,
  matched by `errors.Is` from the concrete error types. They carry positions,
  shapes, and provider names, never credentials, raw bodies, or input text.
- **Compatibility adapters** — `EmbedderAsRepresentationEncoder` and
  `RepresentationEncoderAsEmbedder`, so adopting the new interface is not a
  flag day and sparse output can be turned on when an index is ready for it.
- **`provider/deepinfra`** — native BGE-M3, the reference multi-output
  provider: dense, learned sparse, and ColBERT token vectors from one call,
  plus dense `Embedder` compatibility.
- **`provider/endpoint`** — the versioned protocol against any host running the
  handler: a Hugging Face Inference Endpoint, a container, or a Python process
  on a laptop. The token comes from `WithToken` or `AGENTIC_ENDPOINT_TOKEN`,
  never from a hosting provider's own variable, and `WithoutAuthentication`
  covers a handler that checks no credential without inventing one.
- **`provider/huggingface`** — `NewShared` calls the Inference Providers
  router's feature-extraction task and advertises dense only, because that
  route returns dense vectors and nothing else.
- **`provider/sagemaker`** — the same protocol through SageMaker Runtime,
  behind a one-method interface so transport is testable without an AWS
  account.
- **`provider/pinecone`** — the standalone Inference API, dense or learned
  sparse. `EncodeWithTokens` makes a sparse model's query expansion observable
  as diagnostics.
- **`agentic.representations.v1`** — a versioned JSON contract for endpoints
  you operate, with JSON Schemas and golden fixtures in
  `internal/representationwire/testdata`.
- **Test doubles** — `provider/test.NewTestRepresentationEncoder`, a
  deterministic fake, and `provider/test/conformance.RunRepresentation`, the
  shared contract suite that providers here and downstream retrieval systems
  are both checked against.
- **`provider/local/onnx`** — learned sparse encoding in the calling process, through
  ONNX Runtime, with no server and no network. It is a **nested module** and the
  one directory under `provider/` the root module does not contain, because it
  needs CGO, a native ONNX Runtime, and a statically linked tokenizer; `go get
  github.com/regularkevvv/agentic` does not pull it and the root build does not
  reach it. Sparse only — dense and multi-vector requests return
  `ErrUnsupportedRepresentation` rather than an answer from the wrong reduction.
  Measured against the PyTorch reference on 2026-08-01, it reproduces every
  coordinate index and every weight to 4.56e-06.
- **`DefaultRepresentationLimits`** — the ceilings every encoder here applies,
  now reachable from the public package. `RepresentationValidator` was already
  exported; without this an encoder compiled outside the root module had to
  invent its own numbers or run with every bound disabled, which would make a
  consumer's error behavior depend on where its provider was built.
- **`provider/local/onnx/export_onnx.py`** — the one-time export that
  produces the ONNX graph `provider/local/onnx` runs, and the PyTorch reference its
  tests assert against. Pooling is deliberately left outside the graph, and the
  script refuses to write a graph whose batch axis `torch.export` specialized.

### Changed

- **The examples moved to `e2e/examples/`** and the live provider tests to
  `e2e/providers/`, both inside a new `e2e` module. Nothing they import can
  reach an application that imports Agentic. Run them from the repository root,
  where the committed `go.work` resolves the module:

  ```bash
  go run ./e2e/examples/basic      # was ./examples/basic
  go run ./e2e/examples/retrieval  # was ./examples/retrieval
  go run ./e2e/examples/sparse     # was ./examples/sparse
  ```

  `make test-e2e` is unchanged. No import path a consumer uses moved.

### Safety

- Requests are validated before transport and responses after decoding; a
  response that fails is discarded whole, because a partially valid batch
  written into an index is worse than an error.
- Sparse output is canonicalized to strictly increasing coordinates with
  duplicates rejected rather than resolved to whichever weight came last.
- Request items, bytes, response bytes, sparse nonzero counts, and token-vector
  counts are all bounded, so a large or malformed response cannot exhaust
  memory before validation runs.
- Cancellation keeps its own cause through every provider, so a caller's
  shutdown is not indistinguishable from a transient outage.

## [0.5.1] — 2026-08-01

### Fixed

- **Nested tool invocation context** — composite tool hosts can use
  `WithToolCallContext` to install the nested call ID, name, and attempt while
  clearing handler-resume metadata inherited from the outer tool. This prevents
  nested handlers from observing the outer call's idempotency identity or
  opaque resume decisions.

## [0.5.0] — 2026-08-01

This release adds a small, capability-neutral extension to the resumable
execution driver for tools that discover deferred work only after their handler
has started. It is the root prerequisite for durable nested execution without
introducing harness or code-execution concepts into the root module.

### Added

- **Suspendable tool handlers** — handlers may explicitly implement
  `SuspendableToolHandler` and return `ToolHandlerSuspension` with an ordinary
  `ToolDeferral`. The driver returns the existing `ExecutionSuspended` state
  without committing a duplicate or synthetic tool result.
- **Handler resume context** — `ToolResumeDecision.Payload` carries opaque,
  caller-validated resume data. The re-entered handler retrieves defensive
  copies of that payload and its persisted deferral through
  `CurrentToolResume`.

### Safety

- A suspendable handler must be the only regular tool call in its batch, which
  is validated before any handler starts.
- Undeclared, malformed, mismatched, and stale handler suspensions fail closed.
- Handler-owned resume data is never exposed to ordinary gate-suspended tools.
- A resumed handler may suspend again through the same durable driver contract.
- Existing v0.4 gate-suspension payloads remain valid and resume without handler
  context.

## [0.4.0] — 2026-07-21

This release makes an agent execution an explicit, resumable capability. It is
an observable transcript change for consumers that relied on unmatched tool
calls under early output handling: every completed assistant tool call now has
one paired result.

### Added

- **`Driver[O]`** — `Drive` starts or continues an explicit transcript and
  `Resume` completes a suspended tool frontier. `RequireDriver` lets an
  orchestrator opt into the capability without widening the existing
  `Runner[O]` interface.
- **Durable suspension metadata** — suspension state carries its frontier hash,
  identity, usage/retry state, and execution configuration fingerprint. Resume
  validates all decisions and override inputs before handlers run.
- **Per-run controls** — turn hooks, canonical event sinks, immutable toolset
  overlays, batch gates, result processors, streaming transport selection, and
  tool-cancellation grace are available as `RunOption`s.
- **Canonical execution events** — typed preview, authoritative, and lifecycle
  events report commits from one execution fold.
- **`StreamResult.Snapshot`** — agent-owned streams expose a defensive final or
  partial execution snapshot after completion; provider-owned streams remain
  snapshot-free.
- **`CurrentToolCall`** — handlers can retrieve the admitted call ID, name, and
  retry attempt from their context.

### Changed

- Blocking, streaming, text, and typed runs share one fold for limits, tool
  pairing, retries, validation, hooks, and terminal arbitration.
- Typed output is parsed and validated before a turn hook observes a completion
  candidate. Invalid output retries within the same fold.
- Early output and failed result projection now emit deterministic paired tool
  results for every skipped or failed call instead of leaving an invalid
  transcript frontier.
- Truncated model turns fail before their partial assistant message or tool
  calls can enter the transcript or execute.

## [0.3.0] — 2026-07-20

This is a **breaking release**. It corrects a set of wire-protocol defects that
could not be fixed without changing public types, and it takes the opportunity
to remove the OpenAI-shaped leftovers in the response envelope while callers are
already migrating.

Every provider was audited against a reference implementation. Several of the
bugs below failed on the ordinary path — Azure did not work at all — so
upgrading is strongly recommended despite the migration cost.

### Breaking changes

**`ChatResponse` no longer has `Choices`.** The library never requested more
than one completion, so the slice was always length one and every caller
indexed `[0]`.

```go
// before
msg := resp.Choices[0].Message
why := resp.Choices[0].FinishReason

// after
msg := resp.Message
why := resp.FinishReason
```

The `Choice` type is removed. `ChatResponse.SystemFingerprint` is removed — it
was written by one provider and read by none.

**`ChatRequest.FrequencyPenalty` and `.PresencePenalty` are removed.** They were
unreachable: no agent option exposed them and `buildRequest` never populated
them. Send them through `ProviderOptions` instead.

**Embedding vectors are now `[][]float32`.** No provider transmits more than
float32 of precision, and at 3072 dimensions the narrower element type halves
what an index costs to hold.

```go
// before: var v [][]float64 = resp.Vectors
// after:  var v [][]float32 = resp.Vectors
```

**Unrecognized finish reasons no longer report as `FinishReasonStop`.** Every
provider's converter previously ended in `default: return FinishReasonStop`, so
an Anthropic refusal, a Gemini `RECITATION` block, a Bedrock context-window
overflow, and an OpenRouter in-band error all looked like clean successful
completions. They now map to their closest meaning, or to `FinishReasonUnknown`.
Code branching on `FinishReasonStop` should be reviewed.

**The Azure provider now speaks only the Azure OpenAI v1 API.** The older
deployment-path API (`/openai/deployments/{deployment}?api-version=...`) is no
longer supported, and `WithDeployment`, `WithAPIVersion`, and
`DefaultAPIVersion` are removed.

Most callers need no change: the v1 API is served by the same Azure resource, so
an endpoint like `https://my-resource.openai.azure.com` is resolved to its v1
path automatically. Pass your deployment name as the model argument:

```go
// before
azure.New("gpt-4o", azure.WithEndpoint(ep), azure.WithDeployment("my-deploy"), azure.WithAPIKey(k))

// after
azure.New("my-deploy", azure.WithEndpoint(ep), azure.WithAPIKey(k))
```

An endpoint containing `/deployments/` is rejected with an explanatory error
rather than silently rewritten.

### Added

- **`Reranker`** — a new core capability. Rerankers are cross-encoders that
  score a query against each document directly, which is markedly more accurate
  than comparing independently-computed embeddings and markedly more expensive.
  The usual arrangement is two stages: retrieve a shortlist with an `Embedder`,
  narrow it with a `Reranker`.

  ```go
  resp, err := agentic.Rerank(ctx, reranker, query, docs, 10)
  ```

  Scores are ordinal within a single response. They are not comparable across
  providers, models, or calls — rank by them, never threshold on them.

- **New embedding providers:** Gemini, Ollama, Cohere, Bedrock.
- **New reranking providers:** Voyage AI, Cohere.
- **`ChatRequest.ProviderOptions`** — provider-specific settings keyed by
  provider name, so a provider-only knob no longer requires forking the
  provider. Each provider reads only its own key.
- **`ChatRequest.StopSequences`** — supported by every provider here, and
  previously unreachable.
- **`ChatResponse.RawFinishReason`** — the provider's original stop reason,
  passed through losslessly.
- **`StreamEvent.Signature`, `.ProviderName`, `.ThinkingID`, `.FinishReason`** —
  see the streaming reasoning fix below.
- **`ToolResult.Name` and `NewToolResultMessageFor`** — records which tool
  produced a result. Prefer the new constructor; `NewToolResultMessage` still
  works and leaves the name empty.
- **`EmbeddingRequest.Truncate`** — controls whether an over-length input is
  truncated or rejected. Prefer rejection when indexing: silently embedding the
  first N tokens of a long document and storing the vector as if it represented
  the whole document is a quiet retrieval failure.
- **`ImageURL.MediaType` and `VideoURL.MediaType`** — providers that must
  declare a type when referencing a URL no longer have to assume one.
- **`ProviderError`** and **`IsProviderError`** — a provider that completed the
  transport but produced no usable turn.
- **`Int`, `Float64`, `Bool`, `String`** — pointer helpers for the optional
  request fields.

### Fixed

- **Azure: every request was missing `api-version` and therefore failed.** The
  parameter was encoded into the base URL, where the SDK's path resolution
  discarded it per RFC 3986. `WithAPIVersion` and `OPENAI_API_VERSION` were dead
  code. Azure also received two competing credentials (`api-key` *and*
  `Authorization: Bearer`).
- **Streaming reported zero token usage on six providers.**
  `stream_options.include_usage` was never sent, so every streamed response
  reported no usage at all.
- **Streaming and multi-turn reasoning were mutually exclusive.** A streamed
  thinking block was reconstructed without its provider signature, so replaying
  it in the next request was rejected by Anthropic and Bedrock.
- **Enabling thinking on Anthropic always failed.** The default `max_tokens` of
  1024 was below the default thinking budget of 10000, which the API rejects.
- **Anthropic tool schemas lost `$defs`** while still forwarding `$ref`,
  producing a dangling reference for any nested schema; a zero-argument tool
  emitted a phantom argument named `type`; and `additionalProperties: false` was
  force-injected over the caller's own schema.
- **Gemini could not correlate tool results.** The tool-use *id* was being
  passed where the API expects the function *name*, and that id embedded a
  wall-clock timestamp, so it was not even stable across conversions.
- **Structured output was broken on OpenAI's own default path.** Strict
  `json_schema` was not normalized, despite the normalizer already existing in
  the package and being wired only into tool conversion. The normalizer also
  skipped the `definitions`/`$defs` pools and the `anyOf`/`oneOf`/`allOf`
  branches, so any output struct with a **nested struct field** — which a
  reflected schema emits as a definition plus a `$ref` — still produced a
  definition object without `additionalProperties: false` and was rejected.
- **OpenAI refusals were dropped**, yielding an empty message with a `stop`
  finish reason.
- **OpenRouter silently ignored `MaxTokens`** (wrong field name) and surfaced
  in-band HTTP 200 error bodies as an unrelated error.
- **Gemini, Bedrock, and the OpenAI Responses API kept only the last system
  message** when given several.
- **Bedrock dropped cache points** despite the documentation promising support,
  and its TTL support was documented as nonexistent.
- **Ollama rejected its own documented `OLLAMA_HOST` format.** A schemeless host
  either errored confusingly or silently lost its port.
- **`SummarizeHistory` could permanently discard messages** if the summarizing
  model returned no content.
- Reasoning content is now extracted for OpenAI-compatible providers
  (Ollama, OpenRouter, Together, Grok), which previously discarded it.
- Sampling parameters are no longer sent to reasoning models that reject them.

## [0.2.0]

- Provider-agnostic embeddings support.

## [0.1.0]

- Initial release.
