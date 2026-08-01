# Multi-representation inference

Status: implemented

Date: 2026-08-01

This documents what shipped. The design rationale and the alternatives
considered are in [multi-representation-inference-plan.md](multi-representation-inference-plan.md).

## What this is

Some retrieval models produce more than one view of a text from a single
forward pass. BGE-M3 produces a dense vector, a learned sparse vector, and a
per-token multi-vector at once. `agentic.Embedder` can carry the first of those
and nothing else, so `agentic.RepresentationEncoder` sits beside it and carries
all three without flattening them.

Agentic normalizes inference. It does not own an index.

| Concern | Owner |
| --- | --- |
| Dense, sparse, and token-vector inference | Agentic |
| Provider authentication, transport, retry, batching, usage | Agentic |
| Normalized representation and vector-space identity | Agentic |
| BM25, FTS, and database-native lexical search | Consumer / store adapter |
| Vector persistence and ANN or exact indexes | Consumer / store adapter |
| Scope, tenant, temporal, and status filters | Consumer / store adapter |
| Candidate fusion, reranking, and learned ranking | Consumer |
| Cost policy and provider selection | Application |

Nothing in this package returns a similarity score. It returns values and the
metric needed to compare compatible values; converting a database distance into
a score is the consumer's job, because only the consumer knows which database.

An integrated database API is classified by the operation it performs, not by
the vendor name: Pinecone Inference is an encoder provider here, and Pinecone
index search is not.

## Using it

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

`Data` has exactly one entry per input, in input order. Within an entry, only
the requested and supported kinds are populated. `Spaces` describes each one.

`EncodeQueries` is the same call with the query role. Both helpers set the
input role and nothing else: they do not split batches and do not choose
outputs, because the kinds differ in cost by orders of magnitude and picking
one for you would be picking a bill for you.

### Vector-space identity

`VectorSpace` is the part most easily skipped and most expensive to skip. Two
vectors from the same *named* model but different weights revisions, or two
sparse vectors from different tokenizer vocabularies, are not comparable — and
the failure is silent, in the form of quietly worse recall rather than an
error.

Persist `space.ID` beside every value and refuse to query an index whose ID does
not match. Changing any of these normally creates a new space: provider, model,
weights revision, tokenizer revision, representation kind, dimensions or
vocabulary size, metric, or a provider mode whose query and document encoders
are not compatible.

The ID is derived from those fields with a deterministic hash, so two processes
agree on it without coordinating, and `space.CanonicalKey()` shows the readable
string it was derived from when two IDs differ and you want to know why. A
caller-supplied ID is honored and never overwritten, but it cannot assert a
compatibility the fields deny: `Compatible` compares the fields, not the ID.

Providers do not invent revisions they cannot observe. DeepInfra, Pinecone, and
the Hugging Face router all report a model name and nothing about weights, so
`WithModelRevision` is how a deployment records one. Leaving it empty is
allowed and is itself information: an index keyed on an unrevisioned space
cannot detect a silent model swap.

### Sparse vectors

A learned sparse vector is stored in coordinate form: parallel `Indices` and
`Values`, strictly increasing, no duplicates, no zero weights, every index
inside the space's declared vocabulary.

```text
indices = [1012, 8271, 22091]
values  = [0.91, 0.37, 0.12]
```

That form maps cleanly onto Postgres `sparsevec`, a compressed blob, and most
provider wire formats. Token strings, where a provider exposes them, are
diagnostics — a tokenizer change can keep a string and move its coordinate, or
the reverse, so an index keyed on strings is keyed on the wrong thing.

### Multi-vector

`MultiVector` is one vector per token, for MaxSim or another late-interaction
algorithm. It is never averaged into a single vector by anything here, because
averaging discards exactly what late interaction exists to use. It is also
large: a 512-token document at 1024 dimensions is half a million floats, so
every provider bounds its response and none of them request it unless asked.

### Compatibility with `Embedder`

Existing `Embedder` code keeps working, and adopting an encoder is not a flag
day:

```go
// An existing dense embedder, driven through the encoder interface.
encoder, _ := agentic.EmbedderAsRepresentationEncoder(
    voyageEmbedder,
    agentic.VectorSpace{Provider: "voyageai", Revision: "2025-01"},
)

// A multi-output encoder, driven through the embedder interface.
embedder, _ := agentic.RepresentationEncoderAsEmbedder(bgeEncoder)
```

The first adapter advertises dense only, whatever the underlying model can do
natively. The second requests dense only and rejects `EmbeddingRequest.Dimensions`
rather than ignoring it, because a space's width is part of its identity.

The multi-output providers implement both interfaces directly, so
`deepinfra.New(...)` is usable anywhere an `agentic.Embedder` is expected.

## Providers

| Provider path | Dense | Sparse | Multi-vector | Billing shape |
| --- | --- | --- | --- | --- |
| `deepinfra` native BGE-M3 | Yes | Yes | Yes | Input tokens |
| `deepinfra` OpenAI-compatible route | Not used — see below | No | No | Input tokens |
| `huggingface` dedicated endpoint | Yes | Yes | Optional | Endpoint compute time |
| `huggingface` shared router | Yes | No | No | Shared provider usage |
| `sagemaker` custom endpoint | Yes | Yes | Optional | Endpoint or serverless compute |
| `pinecone` Inference | Model-dependent | Yes, for a sparse model | No | Inference tokens or units |

This table is documentation. `Capabilities()` is the runtime contract, and the
hermetic contract tests plus the opt-in live tests are the evidence. Prices
change; the official pricing pages are linked below rather than embedded in
code.

### DeepInfra

`provider/deepinfra` targets the native `/v1/inference/{model}` route, the only
one that exposes a multi-representation model's full output.

DeepInfra also serves an OpenAI-compatible `/v1/openai/embeddings` route. That
route returns a dense vector and nothing else, whatever the model can do, so
this package does not use it — and a caller must not read dense-only success
there as evidence of sparse support. For plain dense embeddings over that
route, point `provider/openai` at DeepInfra's base URL.

Output flags are sent explicitly on every request, never omitted: DeepInfra
defaults `dense` to true, so an omitted flag would return and bill for a
representation nobody asked for.

Truncation is not advertised. The native API has no truncate parameter, and
accepting the option while ignoring it would let a caller believe an over-long
document had been rejected when it was silently clipped.

BGE-M3 is symmetric — queries and documents go through the same weights with no
task instruction — so both roles are accepted, neither changes the request, and
their outputs are directly comparable. The role is therefore not part of the
space identity here, as it would have to be for an asymmetric model.

Sparse output is normalized from either observed shape: a full vocabulary-width
row, whose length reveals the vocabulary size, or a coordinate map, which does
not and therefore needs `WithSparseVocabulary`. Verified live on 2026-08-01,
`BAAI/bge-m3-multi` returns the full row, so the vocabulary (250002) is
observed and `WithSparseVocabulary` is unnecessary against that model.

#### Response size and why requests are split by default

This API's responses are far larger than its requests. Measured against
`BAAI/bge-m3-multi` on 2026-08-01:

| Requested outputs | Decompressed per input | On the wire |
| --- | --- | --- |
| dense | 39 KB | — |
| sparse | 977 KB | **1.3 KB** |
| multi-vector | ~34 KB per token | — |
| all three | 1185 KB | — |

Sparse decompresses to a megabyte because the row is the entire 250002-entry
vocabulary, almost all of it zeros — a short input leaves about **seven**
coordinates nonzero, 0.003% of the payload. Dense is roughly twice the size the
numbers need, because the response also carries `embedding_jsons`, the same
vectors again as JSON strings, which this package ignores.

Bandwidth is not the problem: Go's transport requests gzip automatically, and a
megabyte of repeated zeros compresses about 760:1, so 1.3 KB actually crosses
the network. The cost is decode-side — parsing a megabyte of JSON and holding a
250002-entry `float32` row to extract seven numbers.

Two things follow.

`provider/deepinfra` splits at **32 inputs per request** by default. Without it,
encoding a few hundred documents with sparse output would decompress to
hundreds of megabytes, exceed the response ceiling, and fail *after* the
inference had been done and billed. Multi-vector scales with document length
rather than count, so lower `WithBatchSize` for long documents;
`WithBatchSize(0)` disables splitting when you know the response will fit.

The sparse row is decoded into a **scratch buffer reused across the batch**,
which costs 464 B/op instead of 5.2 MB/op at the same speed. The tempting
alternative — streaming the row token by token so the full width is never held
— measures three times slower and allocates 1.5 million times per row, because
`encoding/json` boxes every number into an interface. `BenchmarkDecodeSparseRow`
keeps that measurement so the decision is not re-litigated from intuition.

### Hugging Face

`provider/huggingface` has two explicit modes rather than pretending every
Hugging Face endpoint has one response format.

`NewDedicated` speaks `agentic.representations.v1` to an Inference Endpoint you
operate, running the handler in [`deploy/representations`](../deploy/representations).
That is the reliable path for BGE-M3 or a Granite sparse model, because a
custom handler controls which outputs are computed, what the token weights are,
and which revisions the response declares.

`NewShared` calls the Inference Providers router's feature-extraction task,
which returns one dense vector per input. It advertises dense only for exactly
that reason. A model card saying the model produces sparse weights locally is
not evidence that the hosted route returns them.

Input roles are advertised only when `WithPromptNames` maps them onto the
model's configured sentence-transformers prompts. A model with no configured
prompts has no query/document distinction, and accepting the role while sending
nothing for it would let a caller believe an asymmetric encoding happened.

### SageMaker

`provider/sagemaker` invokes the same protocol through the AWS SDK v2 SageMaker
Runtime client. The invocation is identical for a real-time endpoint and a
serverless one: infrastructure mode is deployment configuration, not a
different API. SigV4, credentials, regions, and role policy stay with the SDK
and the deployment.

`InvokeAPI` is a one-method interface, so transport behavior is testable
without an AWS account, a network, or a signed request.

Pin the spaces with `WithVectorSpaces` for any endpoint whose output you intend
to keep. A pinned space must match what the handler reports, so a redeployment
onto different weights fails loudly instead of quietly mixing two generations of
vectors in one index.

### Pinecone

`provider/pinecone` covers the standalone `/embed` endpoint only — encoding, not
indexes, namespaces, or search.

One Pinecone model returns one vector type. Declare it with `WithOutputs`; the
default is dense. The response carries its own `vector_type`, and a mismatch
with what was declared is an error rather than a silent reinterpretation.

A sparse model needs `WithSparseVocabulary`. Pinecone does not report a
vocabulary size, and the bound every index must fall within is not something
this package will invent.

Query and document roles map onto Pinecone's `query` and `passage` input types.
They encode into the same vocabulary, which is what makes a query comparable to
an indexed passage, so the role is not part of the space identity.

Pinecone's sparse models expand a query beyond its literal terms.
`WithReturnTokens` plus `EncodeWithTokens` make that observable, for diagnostics
only.

### Not in scope

Qdrant collection inference and search, Pinecone index APIs, Postgres
extensions, SQLite codecs, BM25 implementations, and retrieval-result fusion
all belong to a store adapter or to the consumer.

Watsonx is not implemented. Its current public embedding API is a managed dense
option, not the managed Granite sparse option that would justify a package
here, and the six existing dense providers already cover that shape. If IBM
exposes Granite sparse weights through a stable managed API, it can be added
against the live response contract without changing any core type.

## The portable endpoint contract

`agentic.representations.v1` is a small versioned JSON protocol for endpoints
you operate yourself. Hugging Face dedicated endpoints and SageMaker endpoints
have no shared response format — whatever your handler returns *is* the format —
so Agentic publishes one, implements it in a reference Python handler, and
speaks it from both Go providers.

Schemas, the handler, deployment instructions, and the cost shape of
compute-billed endpoints are in
[`deploy/representations/README.md`](../deploy/representations/README.md). The
golden request and response fixtures under `deploy/representations/testdata` are
read by both the Go tests and the Python tests, so the two implementations
cannot drift apart without a test failing.

Additive fields are ignored. An unknown *major* version fails rather than being
parsed optimistically, because a change that large is one where guessing
produces vectors that look valid and are not.

## Validation

Every encoder runs the same provider-independent checks, including the
deterministic test double, so a consumer sees identical error behavior from a
fake and from a live provider.

Requests are checked before transport: non-empty inputs, no empty item unless
the provider documents one, non-empty and duplicate-free outputs, every kind
supported, the input role supported, truncation supported if requested, and the
batch and byte ceilings.

Responses are checked after decoding, and a failure discards the whole batch: a
partially valid batch written into an index is worse than an error, because the
damage is silent. One entry per input; every requested kind present and no
other; all floats finite; dense widths uniform and equal to the declared
dimensions; sparse indices strictly increasing, unique, in range, with finite
nonzero weights; token-vector widths uniform; response sizes bounded.

Errors are typed and match `errors.Is`:

```go
switch {
case errors.Is(err, agentic.ErrUnsupportedRepresentation):     // wrong provider for this kind
case errors.Is(err, agentic.ErrInvalidRepresentationRequest):  // caller's request is malformed
case errors.Is(err, agentic.ErrInvalidRepresentationResponse): // provider returned something unusable
}
```

They carry positions, shapes, and provider names. They never carry API keys,
signed headers, raw response bodies, or input text — a provider's validation
error quotes the offending input back at you, so an error that may contain a
user's document cannot be logged freely.

Cancellation keeps its own cause. A retry wrapper never turns `context.Canceled`
into a generic provider error, because a caller's shutdown must not be
indistinguishable from a transient outage.

## Testing

`provider/test.NewTestRepresentationEncoder` is a deterministic fake whose
outputs are pure functions of the input text, the input role, and the
configured space. Equal inputs encode identically, different inputs encode
differently, and two inputs sharing a word share a sparse coordinate — enough
structure to test a retrieval pipeline end to end without a live model.

`provider/test/conformance.RunRepresentation` is the shared contract suite. It
is exported so provider packages here and downstream retrieval systems assert
the same behavior: cardinality and order, each supported output alone and
combined, input roles, capabilities matching behavior, finite values and shapes,
deterministic typed errors, no mutation of request slices, batch limits,
cancellation, and projection to `Embedder`.

Live tests sit behind the `e2e` build tag and skip cleanly without credentials.
They prove retrieval rather than status codes: dense finds a paraphrase, sparse
finds a coined rare term, and a local MaxSim over the token vectors prefers the
intended document. Gates are listed at the top of
[`e2e/representations_test.go`](../e2e/representations_test.go).

## Observability

Provider spans and metrics can carry the provider and operation, the model and
vector-space ID, the requested kinds, batch item and byte counts, latency,
attempt count, status, cancellation, input tokens, output bytes, request count,
sparse nonzero counts, token-vector counts, and the validation failure category.

Agentic reports measurements only — `RequestCount` in particular is what makes
an endpoint-hour deployment comparable to a per-token one. A consumer applies
its own price table, budgets, sampling, and data policy. Do not record raw
vectors or input text by default.

## References to revalidate

Provider APIs, prices, and model availability are time-sensitive. Recheck these
before a release.

- DeepInfra BGE-M3 native API: <https://deepinfra.com/BAAI/bge-m3-multi>
- DeepInfra native inference API: <https://docs.deepinfra.com/apis/deepinfra-native>
- BGE-M3 model card: <https://huggingface.co/BAAI/bge-m3>
- IBM Granite sparse model card: <https://huggingface.co/ibm-granite/granite-embedding-30m-sparse>
- Hugging Face feature-extraction task: <https://huggingface.co/docs/inference-providers/en/tasks/feature-extraction>
- Hugging Face Inference pricing: <https://huggingface.co/docs/inference-providers/en/pricing>
- Hugging Face Endpoint pricing and custom handlers: <https://huggingface.co/docs/inference-endpoints/en/support/pricing>
- SageMaker Runtime `InvokeEndpoint`: <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html>
- Pinecone embed API: <https://docs.pinecone.io/reference/api/2025-04/inference/generate-embeddings>
- Pinecone inference models: <https://docs.pinecone.io/models/overview>
