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
| `endpoint`, any host running the handler | Yes | Yes | Optional | Endpoint compute time |
| `huggingface` shared router | Yes | No | No | Shared provider usage |
| `sagemaker` custom endpoint | Yes | Yes | Optional | Endpoint or serverless compute |
| `pinecone` Inference | Model-dependent | Yes, for a sparse model | No | Inference tokens or units |
| `onnx`, in your own process | No | Yes | No | Your own CPU |

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

### Endpoints you operate

`provider/endpoint` speaks `agentic.representations.v1` to a URL running a
handler that implements it. That is the reliable path for BGE-M3 or a Granite
sparse model, because a custom handler
controls which outputs are computed, what the token weights are, and which
revisions the response declares.

The URL may be a Hugging Face Inference Endpoint, a container on Kubernetes, a
Modal or Fly.io deployment, or a Python process on a laptop. Nothing in the
package knows which, which is why it is named for what it talks to rather than
for whoever hosts it, and why it reads `AGENTIC_ENDPOINT_TOKEN` rather than any
one vendor's variable.

A handler on a loopback address usually checks no credential at all.
`WithoutAuthentication` is how to say that, and it has to be asked for: an
empty `WithToken` is an error rather than an anonymous request. The
alternative — inventing a token — writes a credential that does not exist into
whatever configuration file carries it, and hides the fact that nothing is
being authenticated.

### Hugging Face

`provider/huggingface` covers the Inference Providers router only. An endpoint
you operate is your handler on a URL Hugging Face happens to host, so it is
served by `provider/endpoint` above rather than by a Hugging Face-specific
client.

`NewShared` calls the router's feature-extraction task, which returns one dense
vector per input. It advertises dense only for exactly that reason. A model
card saying the model produces sparse weights locally is not evidence that the
hosted route returns them.

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

A sparse model needs `WithSparseIndexSpace`. Pinecone reports no bound on its
sparse indices, and this package will not invent one.

Its indices are **32-bit hashes of the token, not vocabulary positions**.
Sampling ordinary English on 2026-08-01 produced indices up to 4,209,819,644 —
98% of the way through the unsigned 32-bit range — so the bound is the whole
range, exported as `SparseEnglishIndexSpace`. An earlier version of this
repository's own test guessed 2^31 and would have rejected about half of all
valid vectors.

Query and document roles map onto Pinecone's `query` and `passage` input types,
and they are **not interchangeable**. Both return the same coordinates for the
same text, but the query side weights every term at exactly 1.0 while the
passage side carries the saliency:

| token | query | passage |
| --- | --- | --- |
| `quensel` | 1.0000 | 5.8281 |
| `actuator` | 1.0000 | 4.3438 |
| `the` | 1.0000 | 0.6924 |

A document encoded with the query role loses every distinction between a rare
term and a stopword, and nothing reports an error. They share a coordinate
space, which is what makes a query comparable to an indexed passage, so the
role is not part of the space identity.

**This model does not expand, and it matches exactly.** Learned sparse
encoders are often described as weighting synonyms the text never contained.
Measured against `pinecone-sparse-english-v0`, this one does not:
`"automobile"` returns exactly one coordinate, `automobile`, and a ten-word
sentence returns ten coordinates with no term that was not typed.

Nor does it normalize beyond case. Two words share a coordinate only when they
are the same word:

| pair | same coordinate |
| --- | --- |
| `Automobile` / `automobile` | yes |
| `car` / `cars` | no |
| `run` / `running` | no |
| `recalibrated` / `recalibration` | no |
| `organize` / `organise` | no |
| `car` / `automobile` | no |

This is worth internalizing before choosing the model. In this repository's own
e2e test the query `"quensel actuator recalibration"` scores 10.9102 against a
document reading `"The quensel actuator must be recalibrated..."` — and that
score is exactly `quensel` (5.9883) plus `actuator` (4.9219). The term
`recalibration` contributed nothing, because the document says `recalibrated`.

What the model adds over plain keyword search is learned per-term weights, not
generalization: closer to BM25 with better weights than to a semantic matcher.
Generalization is the dense encoder's job, which is the whole argument for
running both.

`WithReturnTokens` plus `EncodeWithTokens` expose the token behind each
coordinate so you can check that against whatever model you deploy rather than
trusting a model card. Treat the strings as diagnostics only.

### In-process, with no server

`provider/local/onnx` runs a SPLADE-family sparse model through ONNX Runtime in the
calling process. There is no HTTP, no handler, and no Python at runtime.

It is a **nested module**, and the only directory under `provider/` that the
root module does not contain. `go get github.com/regularkevvv/agentic` does not
pull it, `go build ./...` at the root does not reach it, and it is deliberately
absent from `go.work`. That is not tidiness: it needs CGO, a native ONNX Runtime
shared library, and a statically linked tokenizer, and none of those can be a
condition of importing the library. Working on it means `cd provider/local/onnx`.

```go
encoder, err := onnx.New(modelPath, tokenizerPath, agentic.VectorSpace{
    Model:    "ibm-granite/granite-embedding-30m-sparse",
    Revision: "ad82b1fd09541c998c8d45045d601c51fdb8a9b7",
})
defer encoder.Close()
```

Sparse only. Dense and multi-vector requests return
`ErrUnsupportedRepresentation` rather than an answer from the wrong reduction.
The pooling is the same `log1p(relu(logits))`, masked, max over positions that
`handler_splade.py` performs — restated in Go rather than baked into the graph,
so a disagreement with PyTorch says which half moved.

#### The one-time export

Granite ships `tokenizer.json`, which is what a Go tokenizer needs, but no ONNX
export. Producing one is a documented step rather than a download, because the
result is a 117 MiB artifact and an artifact that appears by surprise during a
test run is not a dependency anyone agreed to:

```sh
uv run provider/local/onnx/export_onnx.py --out /some/writable/directory
```

The script carries its dependencies in a PEP 723 header, so that one command
fetches Python 3.11 if the machine has none, builds an environment pinned by
`export_onnx.py.lock`, runs the export, and throws the environment away. Nothing
is installed system-wide. Without uv, `pip install torch==2.13.0 transformers
onnx onnxscript` does the same thing unpinned.

Python appears here and nowhere else in this path. `provider/local/onnx` opens a file;
it imports no Python, spawns none, and does not care whether any is installed —
the container the Linux figures below were measured in had none.

That writes two things. The graph — the raw masked-language model,
`(input_ids, attention_mask) -> logits`, with both leading axes dynamic — and
`reference.json`, a small file recording PyTorch's own input ids, nonzero count,
top terms, and every coordinate for two sample inputs. The reference is checked
in as `provider/local/onnx/testdata/granite_reference.json` and is what makes the Go
implementation falsifiable.

Two details in that script are load-bearing and were both learned the expensive
way. Pooling is left *out* of the graph, so the Go side has to implement it and
can be shown to have implemented it correctly. And the export sample has two
rows, because `torch.export` specializes an axis whose example extent is 1 —
declaring the batch axis dynamic does not prevent it — which produces a graph
that silently accepts one input at a time and fails inside ONNX Runtime the
first time anyone batches. The script now checks its own output for this and
refuses to write a graph with a fixed leading axis.

#### Runtime setup

Two artifacts have to be on disk before the module compiles, and both are
otherwise discovered at link time: `libtokenizers.a`, fetched per platform from
the `daulet/tokenizers` releases, and ONNX Runtime, a separate download.
[`provider/local/onnx/README.md`](../provider/local/onnx/README.md) has the exact commands,
the sizes, and the environment variables the live tests are gated on.

#### What was measured

On darwin/arm64, 2026-08-01, against `granite-embedding-30m-sparse`:

| Claim | Result |
| --- | --- |
| Go tokenizer reproduces the `transformers` ids | identical, both inputs |
| Go reproduces PyTorch's coordinate *set* | identical, 41 and 112 of 50,265 |
| Go reproduces PyTorch's coordinate *weights* | largest difference **4.56e-06** |
| A row padded to twice its length matches the same row alone | same coordinates, weights within **1.46e-06** |
| The padding id reaches a padded row's result | no: identical to the bit under two padding ids |
| Constructing an encoder | 0.30 s, of which 0.12 s reads the vocabulary |
| Root module stays CGO-free with the nested module present | `go.mod` and `go.sum` byte-identical after `go mod tidy` |

The top terms for `"automobile"` are `auto`, `Motor`, `car`, `aut`, `Autom` —
expansion, in Go, from a model that never saw the word "car".

darwin/arm64 and linux/amd64 have both been verified against the real graph:
every coordinate index matched, with a largest weight difference of 4.56e-06 and
4.38e-06 respectively. No other platform has been run.

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
you operate yourself. A dedicated endpoint and a SageMaker endpoint have no
shared response format — whatever your handler returns *is* the format — so
Agentic publishes one, implements it in a reference Python handler, and speaks
it from `provider/endpoint` and `provider/sagemaker`. The transport differs;
the payload does not.

The JSON Schemas that define the protocol, and the golden request and response
the Go tests are checked against, are in
`internal/retrieval/wire/testdata`. Writing the handler itself belongs to
the deployment, in whatever language its platform runs.

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
[`e2e/providers/representations_test.go`](../e2e/providers/representations_test.go).

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
