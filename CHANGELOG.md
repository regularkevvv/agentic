# Changelog

All notable changes to this project are documented here.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, breaking changes may appear in minor releases.

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
