# Multi-Representation Inference Plan

Status: implemented, then amended. The plan is left as written so what was
predicted stays legible, but two things it describes no longer exist: the
reference Python handler and its `deploy/representations` directory were removed
in favour of a Go-only repository, and the ONNX provider ships as the nested
nested `provider/local/onnx` module. The protocol itself is
unchanged; its JSON Schemas and golden fixtures live in
`internal/representationwire/testdata`, and writing a handler is now the
deployment's job in whatever language its platform runs.

Date: 2026-08-01

Scope: Agentic core plus provider packages, tests, examples, CI, and release proof

## Executive decision

Agentic will add a batch-first `RepresentationEncoder` abstraction alongside the existing dense-only `Embedder`. An encoder may return one or more of these representations for every input:

- one dense vector;
- one learned sparse vector;
- one token-level multi-vector representation, for models such as BGE-M3's ColBERT output.

Agentic will normalize inference. It will not own a search index, BM25, candidate retrieval, score fusion, or final ranking. Those responsibilities belong to the consuming retrieval framework, such as Chacla.

The first fully supported multi-output path will be DeepInfra's native BGE-M3 API. Hugging Face dedicated endpoints and SageMaker will share a documented, provider-neutral JSON endpoint contract. Pinecone's standalone Inference API can be added as a learned-sparse provider. Watsonx is useful only as an optional dense provider unless its public service exposes sparse output in the future.

The existing `Embedder`, provider implementations, and public helper functions remain source-compatible. This is additive work, not an embedding API rewrite.

## Why this boundary

Three things are easy to conflate:

1. A model converts text into a representation.
2. A database indexes that representation and returns candidates.
3. A retrieval system combines or reranks candidates.

Agentic should solve only the first problem, plus the transport concerns around it. This keeps the library useful to applications backed by Postgres, SQLite, Qdrant, Pinecone, Vespa, or another index without embedding any database policy in the provider layer.

The ownership split is therefore:

| Concern | Owner |
| --- | --- |
| Dense, sparse, and token-vector inference | Agentic |
| Provider authentication, transport, retry, batching, usage | Agentic |
| Normalized representation and model-space metadata | Agentic |
| BM25, FTS, and database-native lexical search | Consumer/store adapter |
| Vector persistence and ANN/exact indexes | Consumer/store adapter |
| Scope, tenant, temporal, and status filters | Consumer/store adapter |
| Candidate fusion, reranking, and learned ranking | Consumer |
| Cost policy and provider selection | Application |

An integrated database API is classified by the operation it performs, not by the vendor name. Pinecone Inference can be an Agentic encoder provider; Pinecone database search belongs in a consumer store adapter. Qdrant Cloud Inference coupled to a Qdrant collection similarly belongs with the Qdrant store integration, not in Agentic core.

## Current baseline

The repository already has the pieces that this work should extend rather than replace:

- `internal/core/embedding.go` defines a batch-first dense `Embedder`, typed input roles, dimensions, truncation, vectors, and token usage.
- `embedding.go` and `aliases.go` expose the root facade and convenience helpers.
- `internal/embedbatch` provides fan-out and chunked dense batching while preserving input order and aggregating usage.
- OpenAI, Voyage, Cohere, Gemini, Ollama, and Bedrock implement dense embeddings.
- `internal/core/reranking.go` and `reranking.go` already define a separate document reranking contract.
- `provider/test` has deterministic embedding and reranking doubles.
- `e2e/retrieval_test.go` uses opt-in live provider tests to prove semantic behavior, not merely response decoding.
- Bedrock embedding model-family dispatch demonstrates how provider-specific payloads can be normalized behind one public contract.
- CI and release verification already exercise root-module and nested-module views, including `GOWORK=off`.

The missing capability is a typed way to request and receive learned sparse or token-level representations without forcing them into `[][]float32` or inventing a vendor-specific interface in every application.

## Goals

1. Represent dense, learned sparse, and token multi-vector outputs without loss.
2. Permit a model to return several outputs for the same input in one provider call.
3. Make batching the default public shape and preserve exact input/output order.
4. Identify the vector space strongly enough for consumers to reject incompatible indexes.
5. Keep the existing dense embedding surface compatible.
6. Provide consistent validation, errors, retry behavior, usage accounting, and test doubles.
7. Support hosted APIs, dedicated endpoints, and self-managed SageMaker endpoints through one conceptual API.
8. Test wire behavior hermetically and semantic behavior against small live corpora.
9. Make multi-vector output available without requiring consumers to store or use it.

## Non-goals

- Implement BM25, SPLADE retrieval, ANN, MaxSim search, fusion, or reranking inside Agentic.
- Pick a universal score normalization across databases or retrieval legs.
- Hide model or tokenizer changes behind a stable-looking vector-space identifier.
- Make costs part of the core API or hardcode provider prices.
- Require all providers to support all representation kinds.
- Turn the existing `Embedder` into an untyped union.
- Add Pinecone or Qdrant collection management.
- Ship a training pipeline for sparse or ranking models.

## Public contract

### Additive core types

The exact names may receive normal Go API review, but the semantic shape is fixed:

```go
type RepresentationKind string

const (
    RepresentationDense       RepresentationKind = "dense"
    RepresentationSparse      RepresentationKind = "sparse"
    RepresentationMultiVector RepresentationKind = "multi_vector"
)

type SimilarityMetric string

const (
    SimilarityCosine     SimilarityMetric = "cosine"
    SimilarityDotProduct SimilarityMetric = "dot_product"
)

type SparseVector struct {
    Indices []uint32
    Values  []float32
}

type RepresentationRequest struct {
    Input     []string
    InputType EmbeddingInputType
    Outputs   []RepresentationKind
    Truncate  *bool
}

type Representation struct {
    Dense       []float32
    Sparse      *SparseVector
    MultiVector [][]float32
}

type VectorSpace struct {
    ID         string
    Provider   string
    Model      string
    Revision   string
    Tokenizer  string
    Kind       RepresentationKind
    Dimensions int
    Metric     SimilarityMetric
}

type RepresentationUsage struct {
    InputTokens  int
    RequestCount int
    InputBytes   int
    OutputBytes  int
}

type RepresentationResponse struct {
    Data   []Representation
    Spaces map[RepresentationKind]VectorSpace
    Model  string
    Usage  RepresentationUsage
}

type RepresentationCapabilities struct {
    Outputs             []RepresentationKind
    InputTypes          []EmbeddingInputType
    MaximumBatchSize    int
    SupportsTruncation  bool
    SupportsMultiOutput bool
}

type RepresentationEncoder interface {
    Encode(context.Context, *RepresentationRequest) (*RepresentationResponse, error)
    Name() string
    Capabilities() RepresentationCapabilities
}
```

`Data` has exactly one element for each input, in the same order. Within an element, only requested and supported outputs may be populated. `Spaces` describes each populated output.

`MultiVector` means a sequence of token-level vectors to which a consumer may apply MaxSim or another late-interaction algorithm. It is not a flattened dense vector and must never be silently averaged.

### Why sparse vectors use indices and values

A learned sparse representation has a large logical vocabulary but only a small number of nonzero weights. The canonical form stores parallel arrays:

```text
indices = [1012, 8271, 22091]
values  = [0.91, 0.37, 0.12]
```

The pair at each position identifies one vocabulary coordinate and its weight. This form maps cleanly to Postgres `sparsevec`, compressed blobs, and most provider wire formats. Token strings may be exposed separately in optional diagnostics, but they are not the stable storage identity; the tokenizer vocabulary and revision are.

### Vector-space identity

The response must identify the space, not just the marketing model name. Dense vectors from two revisions, sparse weights from different tokenizers, or query/document modes with incompatible encoders cannot safely share an index.

`VectorSpace.ID` is a stable, opaque identifier computed or configured from the material compatibility fields. The provider must not invent stability it cannot prove. For opaque custom endpoints, the caller supplies the revision and tokenizer identifiers during construction. Consumers persist the descriptor and refuse to query an incompatible index.

Changing any of these normally creates a new space:

- provider and model;
- immutable model revision or endpoint deployment revision;
- tokenizer/vocabulary revision;
- representation kind;
- logical dimensions or vocabulary size;
- similarity metric;
- provider mode when query and document encoders are not compatible by default.

### Query and document helpers

The root facade should expose helpers equivalent to the existing dense helpers:

```go
func EncodeQueries(
    ctx context.Context,
    encoder RepresentationEncoder,
    texts []string,
    outputs ...RepresentationKind,
) (*RepresentationResponse, error)

func EncodeDocuments(
    ctx context.Context,
    encoder RepresentationEncoder,
    texts []string,
    outputs ...RepresentationKind,
) (*RepresentationResponse, error)
```

Helpers set `InputType`; they do not split batches or choose outputs implicitly.

### Compatibility adapters

Two adapters prevent a flag day:

- `EmbedderAsRepresentationEncoder` exposes an existing dense `Embedder` as dense-only.
- `RepresentationEncoderAsEmbedder` requests only dense output and projects it into the existing `EmbeddingResponse`.

A multi-output provider may implement both interfaces directly. Its `Embed` method requests only dense output and preserves all current dense contracts. Existing provider constructors and helper behavior remain unchanged.

The next retrieval release should be additive. The exact semantic version is chosen when implementation begins rather than being promised in this plan.

## Contract invariants and validation

Core validation is provider-independent and must run for test providers too.

### Request validation

- The request and encoder are non-nil.
- `Input` is nonempty and contains no empty item unless the provider explicitly supports it.
- `Outputs` is nonempty, contains known values, and has no duplicates.
- Every requested output appears in `Capabilities().Outputs`.
- The requested input type is supported.
- Provider and configured batch limits are checked before transport.
- Unsupported output errors are typed and discoverable with `errors.Is` or `errors.As`.

### Response validation

- The response is non-nil and has exactly one `Data` entry per input.
- Every requested output is present for every item; partial batches fail atomically.
- Unrequested outputs are rejected unless a provider has a documented always-returned field that the adapter deliberately drops before validation.
- All floats are finite.
- Dense vector widths are nonzero, uniform, and consistent with `VectorSpace.Dimensions`.
- Sparse `Indices` and `Values` lengths match.
- Sparse indices are strictly increasing, unique, within the declared vocabulary, and have nonzero finite weights.
- Empty sparse output is allowed only when explicitly documented for an empty or fully filtered input; otherwise it is malformed.
- Token-vector widths are nonzero and uniform across tokens and inputs.
- Model and space metadata do not contradict configured immutable values.
- Multi-vector response size is capped before allocation and after decoding.

Provider adapters may sort an unordered sparse response only when doing so cannot hide duplicate indices. Duplicates must either be combined according to a documented provider rule before canonical validation or rejected.

### Score semantics

Agentic does not return a retrieval similarity score from encoding. It returns representation values and the metric needed to compare compatible values. Any database distance-to-similarity conversion remains the consumer's job.

## Error model

Add typed errors without leaking credentials or full input text:

- `UnsupportedRepresentationError` identifies provider and requested kind.
- `InvalidRepresentationRequestError` identifies the violated invariant.
- `InvalidRepresentationResponseError` identifies item, kind, and shape problem.
- Existing provider status and retry errors continue to carry safe status/request metadata.

Errors must never include authorization headers, API keys, AWS signed headers, arbitrary response bodies, or complete user documents. Provider response excerpts should be bounded and sanitized.

Cancellation and deadlines are returned with their original cause. A retry wrapper must not turn `context.Canceled` into a generic provider error.

## Batching, retry, and usage

`internal/embedbatch` should either be generalized or mirrored by `internal/representationbatch`. The new helper must:

- preserve one output record per input and exact order;
- preserve every requested representation kind;
- reject providers that return short or long chunks;
- sum usage and request counts across chunks;
- stop promptly on cancellation;
- avoid retrying successful chunks after a later chunk fails unless the caller explicitly restarts;
- never expose partial output as a successful response.

Retry behavior should follow existing provider conventions:

- retry bounded 429 and transient 5xx responses;
- honor a valid bounded `Retry-After`;
- do not retry ordinary 4xx contract/authentication failures;
- close every response body;
- make the retry policy injectable in transport tests;
- never sleep past the request context deadline.

`Usage` reports measurements, not money. Providers populate token counts when returned by the service and byte counts where cheaply observable. `RequestCount` makes endpoint-per-minute and compute-backed deployments measurable. Applications combine usage with a dated external price catalog.

## Canonical hosted-endpoint wire contract

Hugging Face dedicated endpoints and SageMaker deployments need a predictable handler contract. Agentic will publish a tiny versioned JSON protocol, implemented by an example Python handler and consumed by both Go providers.

Request:

```json
{
  "version": "agentic.representations.v1",
  "inputs": ["a document", "another document"],
  "input_type": "document",
  "outputs": ["dense", "sparse"],
  "truncate": true
}
```

Response:

```json
{
  "version": "agentic.representations.v1",
  "model": "BAAI/bge-m3",
  "spaces": {
    "dense": {
      "id": "configured-immutable-id",
      "provider": "custom",
      "model": "BAAI/bge-m3",
      "revision": "immutable-revision",
      "tokenizer": "immutable-tokenizer-revision",
      "kind": "dense",
      "dimensions": 1024,
      "metric": "cosine"
    },
    "sparse": {
      "id": "configured-immutable-sparse-id",
      "provider": "custom",
      "model": "BAAI/bge-m3",
      "revision": "immutable-revision",
      "tokenizer": "immutable-tokenizer-revision",
      "kind": "sparse",
      "dimensions": 250002,
      "metric": "dot_product"
    }
  },
  "data": [
    {
      "dense": [0.12, -0.03],
      "sparse": {"indices": [1012, 8271], "values": [0.91, 0.37]}
    },
    {
      "dense": [0.08, 0.22],
      "sparse": {"indices": [914, 1012], "values": [0.63, 0.11]}
    }
  ],
  "usage": {
    "input_tokens": 6,
    "request_count": 1,
    "input_bytes": 27,
    "output_bytes": 0
  }
}
```

The illustrative arrays are shortened; declared dimensions and actual dense widths must match in real responses.

The protocol must have JSON Schema fixtures and golden request/response tests in Go and Python. Unknown additive fields are ignored; an unknown major protocol version fails. The handler accepts explicit requested outputs so costly multi-vector data is never returned accidentally.

## Provider delivery plan

### 1. DeepInfra native BGE-M3

Package: `provider/deepinfra`

This is the reference multi-output provider because the native BGE-M3 API exposes dense, sparse, and ColBERT-style representations. Its OpenAI-compatible embeddings endpoint is dense-only and must not be used to claim sparse support.

Constructor/options should follow existing provider style:

- API token or `DEEPINFRA_TOKEN`;
- explicit model, with BGE-M3 as an example rather than an invisible permanent default;
- base URL override;
- injected `http.Client`;
- retry policy;
- bounded maximum response bytes;
- model/revision descriptor override when the service response is insufficient.

The provider maps query/document input modes to the native API, requests only requested outputs, normalizes its sparse map/list shape, and implements dense `Embedder` compatibility.

Required proof:

- a single request can return dense and sparse outputs for every input;
- sparse dimensions and tokenizer identity are stable and recorded;
- query and document encodings are compatible for the declared space;
- multi-vector output is opt-in and bounded;
- the live semantic test exercises native rather than OpenAI-compatible routing.

### 2. Hugging Face

Package: `provider/huggingface`

Support two explicit modes rather than pretending every Hugging Face endpoint has one response format:

1. A known shared Inference Provider task/model with a contract proven by fixtures and live tests.
2. A dedicated Inference Endpoint running the canonical `agentic.representations.v1` handler.

The dedicated endpoint is the reliable path for BGE-M3 or IBM Granite sparse models because custom handlers control token weights, revisions, and multi-output behavior. Deployment cost is compute-time based; this fact belongs in documentation, not the runtime types.

Do not mark a shared model as sparse-capable merely because its model card supports sparse inference locally. Capability is granted only when the hosted response returns and documents the sparse representation.

The repository should include:

- the minimal custom `handler.py` example;
- pinned dependencies and model revision variables;
- health and sample payload documentation;
- CPU/GPU and autoscaling guidance without promising fixed prices;
- a contract test that runs the handler locally when Python dependencies are explicitly enabled.

### 3. Amazon SageMaker

Package: `provider/sagemaker`

Use AWS SDK v2's SageMaker Runtime client. Adding that client is incremental because Agentic already uses AWS SDK v2 for Bedrock. The runtime invocation is the same Agentic contract for real-time and serverless endpoints; infrastructure mode is deployment configuration, not a different encoder API.

The provider accepts:

- a small internal interface wrapping `InvokeEndpoint` for deterministic tests;
- endpoint name and optional inference component/variant settings;
- content type and bounded response size;
- a required caller-supplied immutable `VectorSpace` descriptor unless the handler returns an exact matching descriptor;
- normal AWS configuration injection patterns.

It sends and receives `agentic.representations.v1`. SigV4, credentials, regions, and role policy remain AWS SDK/deployment concerns. Errors may include endpoint and AWS request IDs but never signed headers or payload text.

The example deployment can use the same handler artifacts as Hugging Face with a SageMaker-compatible entrypoint wrapper.

### 4. Pinecone standalone Inference

Package: `provider/pinecone`

This provider covers encoding/reranking APIs only, not collections or search. Initial scope is the hosted learned-sparse model plus any dense models that naturally fit the existing `Embedder` contract.

The implementation must verify rather than assume:

- whether query and document input roles are distinct;
- whether token expansion is observable and stable enough for the declared model;
- sparse index ordering and vocabulary bounds;
- provider batch and usage behavior.

An empirical contract fixture should include a synonym/expansion case. Marketing terminology is not sufficient evidence of the returned structure.

### 5. Watsonx dense bridge, optional

Package: `provider/watsonx`

Current public Watsonx embedding APIs are a managed dense option, not the managed Granite sparse option desired here. This package is therefore optional and does not block the sparse milestone. If implemented, it should first satisfy the existing dense `Embedder`, then gain representation compatibility through the dense adapter.

If IBM later exposes Granite sparse weights through a stable managed API, add that capability based on the live response contract without changing the core types.

### Explicitly excluded from Agentic provider scope

- Qdrant collection inference/search.
- Pinecone collection/index APIs.
- Postgres extensions and SQLite codecs.
- BM25 implementations.
- Retrieval-result score fusion.

## Provider capability matrix

The implementation docs should maintain a tested matrix. The proposed starting state is:

| Provider path | Dense | Sparse | Multi-vector | Billing shape | Agentic wave |
| --- | --- | --- | --- | --- | --- |
| DeepInfra native BGE-M3 | Yes | Yes | Yes | Input tokens | 1 |
| DeepInfra OpenAI-compatible | Yes | No | No | Input tokens | Existing-style dense only |
| HF shared provider | Contract-dependent | Contract-dependent | Contract-dependent | Shared provider usage | 2, only when proven |
| HF dedicated custom handler | Yes | Yes | Optional | Endpoint compute time | 2 |
| SageMaker custom endpoint | Yes | Yes | Optional | Endpoint/serverless compute | 2 |
| Pinecone Inference | Model-dependent | Yes for sparse model | No initially | Inference tokens/units | 3 |
| Watsonx | Yes | No currently | No currently | Input tokens | Optional |

The table is documentation, not a substitute for runtime `Capabilities()` and live contract tests. Prices change; link to official pricing pages and record the observation date instead of embedding dollar amounts in code.

## Security and operational requirements

- Accept secrets through explicit configuration or standard provider environment variables.
- Redact secrets from errors, logs, golden fixtures, and HTTP dumps.
- Never log input documents by default.
- Bound request item count, per-item bytes, total request bytes, response bytes, sparse nonzero entries, tokens per item, and token-vector count.
- Use the caller's context for transport, retry, and SDK invocation.
- Permit injected clients for proxies, observability, and tests.
- Expose provider/model/space identifiers and usage for tracing, but not raw vectors by default.
- Document data residency and self-hosted considerations as application decisions.

## Test strategy

### Core conformance suite

Create a reusable conformance suite accepted by `provider/test` implementations and real provider fixtures. It checks:

- exact input/output cardinality and order;
- dense-only, sparse-only, multi-vector-only, and supported combined requests;
- query/document input-role propagation;
- capabilities agree with behavior;
- finite values and shape validation;
- deterministic typed errors;
- no mutation of request slices;
- cancellation and deadlines;
- usage aggregation under chunking;
- compatibility projection to dense `Embedder`.

Add a deterministic `TestRepresentationEncoder` whose outputs are simple, inspectable functions of input text. It must support configurable capabilities and injected failures, making it useful to Chacla and other consumers.

### Provider unit and transport tests

Every HTTP provider gets `httptest.Server` coverage; SageMaker uses a stub runtime client. For each provider test:

- constructor and environment precedence;
- authentication and endpoint path;
- exact request body and content type;
- input-role and output-selection mapping;
- response decoding and input order;
- model/revision fallback behavior;
- usage decoding;
- 429, `Retry-After`, bounded 5xx retry, and nonretryable 4xx;
- malformed/truncated JSON;
- response body closure;
- context cancellation during transport and backoff;
- short/long response batches;
- missing requested output;
- duplicate, unsorted, out-of-range, or mismatched sparse values;
- NaN/Inf and inconsistent dense widths;
- inconsistent token-vector widths;
- oversized response rejection;
- unsupported representation kind;
- secret and input-text redaction in errors.

Golden fixtures should be derived from sanitized real responses and annotated with the API/model revision observed.

### Canonical handler tests

- JSON Schema validates all request and response examples.
- Go client and Python handler share golden vectors and protocol versions.
- The handler returns one item per input and only requested outputs.
- Model loading is once per process, not once per request.
- Batch tokenization preserves order.
- Sparse arrays are canonicalized and bounded.
- Multi-vector output is opt-in.
- Model and tokenizer revisions are immutable configuration.
- Health checks do not trigger full inference or leak configuration.

### Live end-to-end tests

Live tests remain behind the existing `e2e` build tag and skip cleanly without credentials. Suggested gates:

| Provider | Environment gate |
| --- | --- |
| DeepInfra | `DEEPINFRA_TOKEN` |
| Hugging Face shared | `HF_TOKEN` plus explicit model |
| Hugging Face dedicated | `HF_TOKEN` plus endpoint URL and expected space ID |
| SageMaker | normal AWS credentials plus endpoint, region, and expected space ID |
| Pinecone | `PINECONE_API_KEY` plus explicit model |
| Watsonx | service API key, project/space, URL, and model |

Use a tiny fixed multilingual corpus to control cost. Do not only assert HTTP 200. Prove:

1. Dense paraphrase similarity puts the intended document first.
2. Sparse lexical matching puts the intended rare-term document first.
3. A selected expansion/synonym example behaves as documented for models that claim expansion.
4. A combined dense+sparse request returns both spaces for each input in one provider request where supported.
5. Query and document modes are distinguishable when the provider requires them.
6. Multi-vector dimensions are stable and a local MaxSim sanity calculation prefers the intended document.
7. Usage is nonnegative and request count is correct.

Tests should tolerate small floating-point/model drift by asserting ordering and structural ranges, not exact live values.

### Compatibility and release tests

- Existing embedding and reranking suites pass unchanged.
- Root facade aliases and examples compile.
- `GOWORK=off go test ./...` passes from the root module.
- Nested modules still pass in their release view.
- A temporary fresh consumer imports the tagged module and implements/uses the new interface without the workspace.
- The public API snapshot/facade contract test confirms no accidental internal type leakage.
- `go vet`, race tests, format, tidy, and `git diff --check` pass.

## Proposed repository changes

Names are indicative; implementation should minimize file churn while preserving the boundaries.

```text
internal/core/representation.go
internal/core/representation_test.go
internal/representationbatch/...
representation.go
representation_test.go
aliases.go
provider/test/representation.go
provider/test/representation_test.go
provider/deepinfra/...
provider/huggingface/...
provider/sagemaker/...
provider/pinecone/...
provider/watsonx/...              # optional
deploy/representations/handler.py
deploy/representations/schema/...
e2e/representations_test.go
docs/multi-representation-inference.md
.env.example
README.md
CHANGELOG.md
```

Provider packages should remain independent. Do not create a single provider package with mode switches for unrelated APIs.

## Delivery sequence

Each commit must build and pass its relevant tests. Preserve one abstraction commit followed by three implementation waves.

### Commit 1: core abstraction and compatibility

- Add representation kinds, vectors, spaces, capabilities, requests, responses, usage, validation, and typed errors.
- Add root aliases/helpers.
- Add dense compatibility adapters.
- Add deterministic test encoder and conformance suite.
- Generalize batch helpers.
- Prove all existing embedding providers and tests remain unchanged.

Exit gate: a consumer can request sparse output from a fake encoder and can adapt every existing dense embedder without provider-specific code.

### Commit 2: DeepInfra reference implementation

- Add native BGE-M3 provider.
- Add sanitized fixtures and full transport tests.
- Add dense `Embedder` compatibility.
- Add opt-in live dense, sparse, combined, and multi-vector semantic tests.
- Document native versus OpenAI-compatible endpoint behavior.

Exit gate: one live request produces validated dense+sparse output, and optional multi-vector output passes the MaxSim sanity check.

### Commit 3: portable hosted-endpoint contract

- Freeze `agentic.representations.v1` schema.
- Add Python handler, dependency pins, and golden protocol tests.
- Add Hugging Face dedicated/shared modes with honest capabilities.
- Add SageMaker Runtime provider and stub tests.
- Add deployment and cost-shape documentation.

Exit gate: the same logical request succeeds through local handler contract tests and opt-in HF/SageMaker live endpoints.

### Commit 4: provider breadth and release proof

- Add Pinecone standalone Inference.
- Add Watsonx dense support only if it is ready; it does not block release.
- Complete provider capability and live-test matrix.
- Run root, nested-module, `GOWORK=off`, race, fresh-consumer, and documentation gates.
- Update README, examples, `.env.example`, and changelog.

Exit gate: the tagged API is consumable outside the monorepo/workspace and every claimed capability has a hermetic contract test plus an opt-in live proof.

## CI shape

Default CI must not require paid credentials:

1. Core and provider unit tests.
2. Race tests for concurrency/batching code.
3. Handler schema/golden tests with lightweight dependencies or a separate explicitly pinned job.
4. Root and nested release-view tests with `GOWORK=off`.
5. Secret-scan assurance for fixtures.

Live provider CI should be a manually dispatched or protected scheduled workflow with per-provider jobs. One provider outage must be attributable and must not obscure other results. Record provider model IDs and protocol revisions in the job summary. Set strict request and cost ceilings.

The existing `make test-e2e` path remains the developer entrypoint unless repository conventions change during implementation.

## Observability

Provider spans/metrics should make these fields available without recording text or vector bodies:

- provider and operation;
- model and vector-space ID;
- requested representation kinds;
- batch item count and byte count;
- latency, attempt count, status, and cancellation;
- input tokens, output bytes, and request count;
- sparse nonzero count and token-vector count as distributions;
- validation failure category.

Agentic reports measurements only. A consumer can apply its own price table, budgets, sampling, and data policy.

## Documentation examples

The primary example should show a consumer persisting space identity beside output:

```go
resp, err := agentic.EncodeDocuments(
    ctx,
    encoder,
    []string{"PostgreSQL supports sparse vectors"},
    agentic.RepresentationDense,
    agentic.RepresentationSparse,
)
if err != nil {
    return err
}

item := resp.Data[0]
denseSpace := resp.Spaces[agentic.RepresentationDense]
sparseSpace := resp.Spaces[agentic.RepresentationSparse]

// The application stores item.Dense and item.Sparse in its chosen index,
// together with denseSpace.ID and sparseSpace.ID.
```

A second example should show dense compatibility with an existing provider. A third should show how to deploy the canonical handler without implying that Agentic owns the resulting index.

## Risks and required spikes

These questions require measured work, but do not change the architecture:

- Confirm the exact DeepInfra BGE-M3 native response and output flags at implementation time.
- Measure worst-case multi-vector response memory and choose conservative default limits.
- Verify which Hugging Face shared provider/model combinations expose raw sparse weights; advertise only proven combinations.
- Validate IBM Granite sparse custom-handler outputs, vocabulary metadata, and CPU/GPU throughput.
- Validate Pinecone's observed expansion behavior and stable vocabulary semantics.
- Determine whether provider token usage can be normalized without fabricating unavailable values.
- Decide whether space IDs should use a canonical hash helper or require provider/caller-supplied opaque IDs. Either choice must remain deterministic and inspectable.

Provider APIs, prices, and model availability are time-sensitive. Recheck official documentation and live schemas immediately before implementation and release.

## Definition of done

The work is complete only when all of the following are true:

- The additive public API represents dense, sparse, and multi-vector outputs without ambiguity.
- Existing `Embedder` users compile and behave as before.
- Every output carries a validated vector-space descriptor.
- DeepInfra native BGE-M3 passes hermetic and live multi-output tests.
- The versioned custom endpoint contract passes Go/Python golden tests.
- Hugging Face dedicated and SageMaker providers pass hermetic tests; live tests pass when configured.
- Pinecone claims only capabilities demonstrated by its tests.
- All malformed-shape, retry, cancellation, order, and secret-redaction cases are covered.
- Paid tests skip cleanly without credentials and use a bounded corpus when enabled.
- Root, nested modules, `GOWORK=off`, race, vet, tidy, facade, and fresh-consumer gates are green.
- Documentation clearly assigns storage, BM25, fusion, and final ranking to the consumer.
- Chacla can implement its companion retrieval plan using only Agentic's public API.

## Primary references to revalidate during implementation

- DeepInfra BGE-M3 model and native API: <https://deepinfra.com/BAAI/bge-m3-multi>
- BGE-M3 model card: <https://huggingface.co/BAAI/bge-m3>
- IBM Granite sparse model card: <https://huggingface.co/ibm-granite/granite-embedding-30m-sparse>
- Hugging Face Inference pricing: <https://huggingface.co/docs/inference-providers/en/pricing>
- Hugging Face Endpoint pricing and custom handlers: <https://huggingface.co/docs/inference-endpoints/en/support/pricing>
- SageMaker Runtime `InvokeEndpoint`: <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html>
- Pinecone inference models: <https://docs.pinecone.io/models/overview>
