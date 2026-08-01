# agentic.representations.v1

A small versioned JSON contract between Agentic and an inference endpoint you
operate yourself, plus a reference handler that implements it for BGE-M3.

Hugging Face dedicated endpoints and SageMaker endpoints have no shared
response format — whatever your handler returns *is* the format. Rather than
write a decoder per deployment, Agentic publishes one protocol, implements it
here, and speaks it from both `provider/huggingface` and `provider/sagemaker`.
The transport differs between them; the payload does not.

## The protocol

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
      "id": "configured-immutable-dense-id",
      "provider": "custom",
      "model": "BAAI/bge-m3",
      "revision": "immutable-revision",
      "tokenizer": "immutable-tokenizer-revision",
      "kind": "dense",
      "dimensions": 1024,
      "metric": "cosine"
    }
  },
  "data": [{"dense": [0.12, -0.03]}],
  "usage": {"input_tokens": 6, "request_count": 1, "input_bytes": 27, "output_bytes": 0}
}
```

Rules, enforced on both sides:

- Exactly one `data` entry per input, in request order.
- Exactly the requested representations, no more and no fewer.
- Sparse vectors are coordinate pairs with strictly increasing indices inside
  the declared vocabulary, and finite nonzero weights.
- `multi_vector` is one vector per token. It is never pooled into one vector.
- Additive fields are ignored by older clients. An unknown **major** version
  fails rather than being parsed optimistically, because a change that large
  is one where guessing produces vectors that look valid and are not.

`schema/request.schema.json` and `schema/response.schema.json` are the
normative shapes; `testdata/` holds the golden pair that the Go client and the
Python handler are both tested against.

## Vector-space identity is your configuration, not the runtime's

The handler does not derive `spaces[*].id`. It reads it from the environment,
and refuses to start without it.

An endpoint cannot prove its own weights revision. If the identity a consumer
keys an index on were computed at runtime, a redeployment onto different
weights would produce vectors that land in the same space as the old ones and
retrieve slightly worse forever, with nothing failing. Declaring the ID — and
changing it whenever the model, tokenizer, dimensions, or metric change — is
what makes that detectable.

## Configuration

| Variable | Required | Meaning |
| --- | --- | --- |
| `AGENTIC_MODEL` | no | Model to load (default `BAAI/bge-m3`, or the SageMaker model directory) |
| `AGENTIC_OUTPUTS` | no | Comma-separated kinds this endpoint serves (default all three) |
| `AGENTIC_SPACE_ID_DENSE` | for `dense` | Immutable dense space ID |
| `AGENTIC_SPACE_ID_SPARSE` | for `sparse` | Immutable sparse space ID |
| `AGENTIC_SPACE_ID_MULTI_VECTOR` | for `multi_vector` | Immutable multi-vector space ID |
| `AGENTIC_MODEL_REVISION` | no | Immutable weights revision, recorded in every space |
| `AGENTIC_TOKENIZER_REVISION` | no | Immutable tokenizer/vocabulary revision |
| `AGENTIC_DENSE_DIMENSIONS` | no | Dense width (default 1024) |
| `AGENTIC_SPARSE_VOCABULARY` | no | Tokenizer vocabulary size (default 250002) |
| `AGENTIC_PROVIDER` | no | Provider label in the space descriptor (default `custom`) |
| `AGENTIC_FP16` | no | `0` disables half precision |

Leaving the revisions empty is allowed and is itself information: an index
keyed on an unrevisioned space cannot detect a silent model swap.

## Deploying to a Hugging Face Inference Endpoint

1. Create a model repository containing `handler.py`, `protocol.py`, and
   `requirements.txt`.
2. Pin the model revision in `AGENTIC_MODEL` (`BAAI/bge-m3@<commit>`) or in
   `AGENTIC_MODEL_REVISION`, and set the space IDs.
3. Create an Inference Endpoint from that repository with the *Custom* task.
4. Point Agentic at it:

```go
encoder, err := huggingface.NewDedicated(
    "https://<endpoint>.endpoints.huggingface.cloud",
    huggingface.WithModel("BAAI/bge-m3"),
)
```

A custom handler is the reliable path for BGE-M3 or IBM Granite sparse models.
The shared Inference Providers route returns dense vectors and nothing else,
whatever a model card says it can do locally, so `provider/huggingface`'s
shared mode advertises dense only.

## Deploying to SageMaker

Use the same artifacts with `sagemaker_entrypoint.py` as the inference script.
The runtime invocation is identical for a real-time endpoint and a serverless
one — infrastructure mode is deployment configuration, not a different API.

```go
encoder, err := sagemaker.New(ctx, "my-bge-m3-endpoint",
    sagemaker.WithVectorSpaces(map[agentic.RepresentationKind]agentic.VectorSpace{...}),
)
```

## Cost shape

These endpoints bill for **compute time**, not for tokens: you pay for the
instance while it is up, whether or not it is encoding anything. That makes
`usage.request_count` the measurement that matters here, and token counts much
less meaningful than they are on a hosted per-token API.

Practical consequences, not prices — prices change, so check the official
pages and record the date you checked:

- Scale-to-zero cuts idle cost and adds a cold start to the first request
  after it; a latency-sensitive query path usually cannot afford one.
- A CPU instance is enough for dense output on short inputs. Sparse and
  especially multi-vector output over long documents want a GPU.
- Multi-vector responses are large. Keep `outputs` narrow, and prefer a
  smaller batch over a larger one when requesting it.

- Hugging Face endpoint pricing: <https://huggingface.co/docs/inference-endpoints/en/support/pricing>
- SageMaker pricing: <https://aws.amazon.com/sagemaker/pricing/>

## Testing

Contract tests need no GPU and no model download; the model is faked.

```bash
pip install -r requirements-dev.txt
pytest
```

The Go side reads the same `testdata/` fixtures, so the two implementations
cannot drift apart without a test failing.
