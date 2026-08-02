# Module Boundaries Plan

Status: implemented, 2026-08-01, in the three commits below. The plan is left as
it was written — including the parts the implementation contradicted — so what
was predicted stays legible next to what was built. "Open decisions, resolved"
and "Deviations" are the only sections added afterwards, and they are where the
differences are recorded.

Date: 2026-08-01

Scope: Repository module layout, the generic endpoint provider, and in-process
ONNX inference. No change to the public retrieval API.

## Executive decision

Agentic will enforce one dependency rule — **nothing the root module ships may
depend on anything outside it** — and will express that rule structurally
rather than by convention:

- Extract the transport half of `huggingface.NewDedicated` into
  `provider/endpoint`, a generic client for any server speaking
  `agentic.representations.v1`. Nothing in it is Hugging Face-specific.
- Move `e2e/` and `examples/` into one nested `e2e` module, so live tests and
  runnable programs may take presentation and harness dependencies that the
  library never carries.
- Add `localinference`, a nested module implementing `RepresentationEncoder`
  in-process via ONNX Runtime, so a caller can encode with no server at all.
  CGO lives there and nowhere else.

The root module stays pure Go, CGO-free, and minimal. The public API does not
change.

## The three problems

### 1. A provider is branded for a vendor it does not require

`huggingface.NewDedicated` does exactly this:

```go
payload, err := e.post(ctx, e.endpoint, representationwire.NewRequest(req))
```

POST protocol JSON to a URL with a bearer token. That works against localhost,
Kubernetes, Modal, Fly.io, or a machine under a desk. The Hugging Face name in
the type is inaccurate, and it hides the provider from everyone not deploying
to Hugging Face.

### 2. Test and example dependencies land in the library

`e2e/` and `examples/` are packages inside the root module. Anything they
import becomes a root dependency, and build tags do not help: `go mod tidy`
considers all build configurations, so a `//go:build e2e` import still lands in
the root `go.mod` and in every consumer's module graph.

A styling library for a readable e2e table would today be inherited by every
application that imports Agentic. For a repository that runs a supply-chain
audit on every pull request, that is the wrong default.

### 3. In-process inference needs CGO, which the library must not require

Running an encoder locally — no HTTP, no server, no Python — means ONNX Runtime
or libtorch through CGO. CGO breaks cross-compilation and requires a native
library at build and run time. It cannot be a condition of `go get agentic`.

## Current baseline

Verified in the repository, not assumed:

- `go.mod` and `harness/go.mod` are the only modules. `e2e/` and `examples/`
  are packages inside the root module.
- `harness/go.mod` requires `github.com/regularkevvv/agentic v0.5.1` — a
  released version, deliberately, so it tests the published library.
- The root `go.mod` does not mention `harness`. The arrow points one way.
- `harness/` tests itself; root `e2e/` never imports it.
- `go.work` declares `use ( . ./harness )`.
- CI runs: lint, test, harness, harness-lint, workspace, coverage, vet,
  handler (Python), fixtures (secret scan).
- `make test-e2e` runs `go test -tags=e2e ./e2e/...` from the root.
- `COVERAGE_PACKAGES` excludes `e2e`, `examples`, `internal/testutil`, and
  `provider/test/conformance`.

The `harness` module is the precedent this plan follows. It already
demonstrates the rule: a nested module may require the root, tests itself, and
is invisible to the root's build.

## The dependency rule

One invariant, from which the structure follows:

> A nested module may require the root module. The root module may never
> require a nested module.

Everything below is a consequence.

## Proposed structure

```text
agentic/
├── go.mod                          root — pure Go, CGO-free, minimal deps
├── go.work                         use ( . ./harness ./e2e )
├── representation.go               public interfaces and helpers
├── internal/core/                  types, validation, adapters
├── internal/representationwire/    agentic.representations.v1 codec
├── provider/
│   ├── endpoint/                   NEW: generic protocol client
│   ├── huggingface/                shared router only, after extraction
│   ├── sagemaker/                  AWS-specific transport
│   ├── deepinfra/  pinecone/  …    unchanged
│   └── onnx/                       ── NESTED MODULE ──
│       ├── go.mod                  requires agentic; CGO + ONNX Runtime
│       ├── encoder.go              implements RepresentationEncoder
│       ├── tokenizer.go            reads tokenizer.json
│       └── encoder_live_test.go    tests itself
│
├── harness/                        ── NESTED MODULE ── unchanged
│
├── deploy/representations/         schema + Python handlers (no Go)
│
└── e2e/                            ── NESTED MODULE ──
    ├── go.mod                      replace agentic => ..
    ├── internal/corpus/            shared fixtures and scoring helpers
    ├── providers/                  live provider tests
    └── examples/                   basic, tools, structured, retrieval, sparse
```

### Dependency graph

```text
                  ┌──────────┐
                  │ agentic  │   depends on nothing internal
                  └────┬─────┘
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
    harness/         e2e/      localinference/    consumers
                                     ▲
                                     └── e2e does NOT require this
```

## Design decisions and their reasons

### `localinference` tests itself; `e2e` does not require it

If `e2e` required `localinference`, running any e2e test would need ONNX Runtime
installed. Following the `harness` precedent — each nested module owns its own
tests — keeps CGO strictly opt-in. `e2e` stays about HTTP and SDK providers.

`localinference` is therefore **not** added to `go.work`. Working on it means
`cd provider/local/onnx`. That keeps the default contributor loop CGO-free.

### `localinference` rather than a top-level `localinference/`

It is a provider; it belongs beside its siblings, and consumers get the natural
`go get github.com/regularkevvv/agentic/provider/local/onnx`. Its nested `go.mod` is
what excludes it from the root module.

The one surprise is a directory under `provider/` that is not part of the root
module. Its package documentation must say so in the first paragraph.

### `e2e` uses `replace`; `harness` keeps its pinned version

They answer different questions and must not be unified:

| module | requires | proves |
| --- | --- | --- |
| `harness` | `agentic v0.5.1` | the *published* library works |
| `e2e` | `replace => ..` | the code you just wrote works |

### `examples` moves into the `e2e` module

They share a corpus and scoring helpers — `cosine`, `dotProduct`, and `maxSim`
are currently duplicated between the e2e tests and the retrieval example. They
need the same credentials, and neither belongs in the library. `internal/corpus`
removes the duplication.

The cost is discoverability: `go run ./examples/basic` becomes
`go run ./e2e/examples/basic`. It still works from the repository root because
`go.work` is committed here. See Open decisions.

## Goals

1. The root module's dependency list contains only what the library needs at
   runtime.
2. `provider/endpoint` works against any server speaking the protocol, with no
   vendor in its name.
3. Live tests and examples may take any dependency without consulting the
   library's supply-chain budget.
4. In-process encoding is available to callers who want it and invisible to
   callers who do not.
5. Existing public API, import paths for current providers, and behavior are
   unchanged.
6. Every module builds and tests in isolation with `GOWORK=off`.

## Non-goals

- Changing `RepresentationEncoder` or any core type.
- Making CGO a requirement of any existing import path.
- Vendoring ONNX Runtime, or shipping model weights in the repository.
- Reimplementing tokenization in pure Go where a maintained binding exists.
- Removing the Python handlers; they remain the reference server
  implementation.
- Adding an ONNX export pipeline to CI. Export is a documented one-time step.

## Delivery sequence

Each commit builds and passes its own gates.

### Commit 1: extract `provider/endpoint`

- Move the dedicated-endpoint client from `provider/huggingface` to
  `provider/endpoint`, dropping the Hugging Face naming.
- `huggingface` keeps the shared feature-extraction router only.
- Options rename accordingly: `WithToken`, `WithHTTPClient`, `WithBatchSize`,
  `WithOutputs`, `WithVectorSpaces`, `WithLimits`.
- `HF_TOKEN` stops being an implicit fallback; a generic client should not read
  a vendor's variable. Callers pass the token, or set `AGENTIC_ENDPOINT_TOKEN`.
- Move the transport tests with it; keep the conformance run.
- Update `docs/multi-representation-inference.md`, the README table, and
  `deploy/representations/README.md`.

Exit gate: `provider/endpoint` talks to the local Python handler, and both
leakage checks exit 1 — no match, rather than no output:

```sh
grep -rnE "representations\.v1|NewDedicated" provider/huggingface/
grep -rnE "HF_TOKEN|HUGGING_FACE_HUB_TOKEN" provider/endpoint/ --exclude='*_test.go'
```

`-E` and the escaped dot are load-bearing. Without them `grep` reads the
pattern as a basic regular expression, where `|` is a literal character and `.`
is a wildcard, so the check searches for one 30-character string that appears
nowhere and passes over a package that plainly still contains both terms. The
second check skips tests because the test proving the vendor variable is
ignored has to name it.

### Commit 2: `e2e` becomes a module and absorbs `examples`

- Add `e2e/go.mod` with `replace github.com/regularkevvv/agentic => ..`.
- Move `examples/` to `e2e/examples/`, including `internal/envutil`.
- Add `e2e/internal/corpus` and delete the duplicated scoring helpers.
- `go.work` gains `./e2e`.
- `make test-e2e` runs in `e2e/`.
- CI: the e2e job changes directory; **a new job compiles the e2e module**, so
  a breaking API change still fails the build.
- `COVERAGE_PACKAGES` drops its `e2e` and `examples` exclusions.
- README paths updated.

Exit gate: `GOWORK=off go build ./...` in `e2e/` succeeds; the root `go.mod`
gains no dependency; `go mod tidy` at root removes nothing that the library
needs.

### Commit 3: `localinference`

- Nested module with its own `go.mod`, requiring the released `agentic`.
- `Encoder` implements `core.RepresentationEncoder`, sparse first.
- SPLADE-family pooling in Go: `log1p(relu(logits))`, masked, max over
  positions — the same four lines the Python handler uses.
- Tokenization reads `tokenizer.json`, the portable fast-tokenizer format that
  both target models ship.
- Model path and vector space are constructor arguments. No download at
  runtime, no implicit network.
- Own live test, gated on a model path being set.
- Own CI job, matching the `harness` job's shape.
- Deployment note in `docs/` covering the one-time ONNX export.

Exit gate: `localinference` encodes the same corpus as the Python handler and
agrees on **every coordinate index, and on weights to within 1e-4** — the spike
below reached 4.6e-06, so a looser gate would be accepting a regression. The
root module still builds without ONNX Runtime installed.

## Why ONNX, and what it targets

Measured on 2026-08-01, four models called "learned sparse" behave differently
enough that the target matters:

| model | expands? | size | architecture |
| --- | --- | --- | --- |
| BGE-M3 | no | 2.3 GB | XLM-RoBERTa large |
| `pinecone-sparse-english-v0` | no | hosted | hashed, exact-match only |
| SPLADE cocondenser | yes | 1.1 GB | `BertForMaskedLM` |
| **`granite-embedding-30m-sparse`** | **yes** | **60 MB** | `RobertaForMaskedLM` |

Granite is the target: it expands, it is thirty million parameters, and it
loads as a stock `transformers` architecture with no custom modelling code. The
same Python handler ran SPLADE and Granite unchanged, which is evidence the
pooling generalizes across the family rather than fitting one checkpoint.

It ships no ONNX export, so export is a one-time documented step. It does ship
`tokenizer.json`, which is what a Go tokenizer needs.

## What the spike measured

Commit 3 rests on claims that would be expensive to discover false halfway
through, so they were run first, on 2026-08-01, on darwin/arm64.

Granite was exported to ONNX with `torch.onnx.export(dynamo=True)` — the raw
`RobertaForMaskedLM` graph, `(input_ids, attention_mask) -> logits`, with both
the batch and sequence axes dynamic. Pooling was deliberately left out of the
graph so the Go side had to implement it. A throwaway Go program then ran that
graph through `yalue/onnxruntime_go` v1.31.0 against ONNX Runtime 1.28.0,
tokenized with `daulet/tokenizers` v1.27.0, and pooled in Go.

| claim | result |
| --- | --- |
| Granite exports to ONNX at all | yes, clean, no custom ops |
| Go tokenizer reproduces the `transformers` ids | **identical**, both inputs |
| Go reproduces PyTorch's coordinate *set* | **identical**, 41 and 112 of 50,265 |
| Go reproduces PyTorch's coordinate *weights* | largest difference **4.6e-06** |
| Padded rows in a batch match the same row alone | yes, within 1.5e-06 |
| Root module stays CGO-free with the nested module present | yes, `go.mod` and `go.sum` byte-identical after `go mod tidy` |

The top terms for `"automobile"` were `auto`, `Motor`, `car`, `aut`, `Autom` —
expansion, in Go, from a model that never saw the word "car".

Costs, which the plan previously did not state: the exported graph is **117 MB**
fp32; `libtokenizers.a` links statically and adds ~20 MB to a binary; ONNX
Runtime is a separate ~32 MB download. Loading the model and running six
inferences took **0.40 s** wall clock total, so this is cheap enough for an
ordinary test.

One result contradicted an assumption worth recording: batching is not free.
Three short inputs padded to a common width of 18 took 20 ms as one call versus
13 ms as three, because padding buys compute nobody asked for. Batch width, not
batch size, is what costs.

## Risks

- ~~**Only darwin/arm64 was verified.**~~ Retired. linux/amd64 was verified after
  the fact by running the live tests in a `golang:1.25-bookworm` container
  against the real graph: every coordinate index matched, largest weight
  difference 4.38e-06 against darwin/arm64's 4.56e-06. The CI job gates on
  itself rather than being marked `continue-on-error`. Platforms beyond those
  two remain untested.
- ~~**Neither binding's licence or maintenance was assessed.**~~ Retired. Both
  are MIT and pinned exactly: `daulet/tokenizers` v1.27.0 and
  `yalue/onnxruntime_go` v1.31.0.
- **The 117 MB graph has no home.** It cannot go in the repository, and Commit 3
  assumes a documented export step rather than a download. If the test needs it
  in CI, that is an unsolved distribution problem — which is an argument for the
  test staying gated on a local path.
- **`libtokenizers.a` must be fetched per platform** before `localinference`
  builds. That is a real barrier to entry for a contributor and belongs in the
  module's README, not discovered at link time.
- **Examples can rot silently** once they leave the root module's `go build
  ./...`. Mitigated by the new CI job in Commit 2; without it this is a
  regression.
- **`go.work` divergence.** Contributors who build with `GOWORK=off` see
  different resolution from those who do not. CI already runs both views and
  must continue to.
- **CGO in CI.** The `localinference` job needs ONNX Runtime installed on the
  runner. That job must not gate the others.

## Open decisions, resolved

1. **Example discoverability** — the path change was accepted, as recommended.
   `go run ./e2e/examples/basic` from the repository root is what the committed
   `go.work` resolves. The old and new paths sit side by side in the CHANGELOG
   so a reader who types the old one finds out why it moved rather than only
   that it is gone.
2. **`AGENTIC_ENDPOINT_TOKEN`** — confirmed as the variable name, and the
   decision grew a third case the plan did not anticipate. A handler on a
   loopback address usually checks no credential at all, so `endpoint.New`
   accepts `WithoutAuthentication()`; an empty `WithToken("")` is an error
   rather than a quiet anonymous request, because inventing a token writes a
   credential that does not exist into whatever configuration file carries it.
   The two are mutually exclusive. `HF_TOKEN` is never read, and a test sets it
   to prove the constructor still fails.
3. **`Embedder` on `localinference`** — not implemented. Sparse is the reason
   the module exists; dense and multi-vector requests return
   `ErrUnsupportedRepresentation` rather than an answer from the wrong
   reduction. Widening it later needs no change here, since the constructor
   already takes the vector space as an argument.

## Deviations

Where the implementation departed from the text above, and why.

1. **`localinference` uses `replace ../..`, not a released `agentic`.** Commit 3
   above says "requiring the released `agentic`". No release carries
   `RepresentationEncoder` — it is unreleased work in this same branch — so
   there is no version the module could require and still compile. The `replace`
   is therefore not a stale pin, and its `go.mod` comment says when to remove
   it: once a release carries the interface, this module should pin that tag the
   way `harness` pins its own, so the two nested modules keep proving different
   things.

2. **Both nested modules import only the root's *public* packages.** Commit 3
   says `Encoder` implements `core.RepresentationEncoder`; a module outside the
   root cannot import `internal/core` at all. Both nested modules therefore
   follow `harness/architecture_test.go`, and each carries a parser-based test
   of its own — `e2e/architecture_test.go` and
   `localinference/architecture_test.go` — that fails on any
   `agentic/internal/...` import. The parser rather than `go list` because the
   provider tests are behind `//go:build e2e`, and a build-configuration-aware
   walk would skip the files most likely to break the rule. One existing test,
   `e2e/providers/v030_wire_test.go`, imported `internal/core` and was rewritten
   against the root package; every type and constant it used has an exact alias
   there, so no assertion was weakened.

3. **The root's public API gained `DefaultRepresentationLimits`.** Goal 5 says
   the public API does not change. This one addition was unavoidable:
   `RepresentationValidator` was already exported but the ceilings it needs were
   not, so an encoder compiled outside the root module had to invent its own
   numbers or run with every bound disabled — which would make a consumer's
   error behavior depend on where its provider was built. Additive, and no
   existing signature moved.

4. **The export script hardens the spike's recipe; the recipe itself was
   sound.** An earlier draft of this section claimed the spike's one-row sample
   specialized the batch axis and produced a graph that could not batch. That is
   wrong, and was retracted after measurement: the spike's graph declares
   `['batch', 'seq']`, it encoded a batch of three, and the shipped live test
   groups four inputs into two forward passes against that same file. Exporting
   one-row and two-row samples side by side on torch 2.13 keeps the batch axis
   symbolic both times, because the explicit `torch.export.Dim` is the
   constraint that prevents 0/1 specialization.

   `provider/local/onnx/export_onnx.py` still uses a two-row sample and still
   refuses to write a graph whose leading axis is fixed. Both are insurance
   rather than fixes for an observed failure — specialization is a real
   `torch.export` behaviour, it is silent, and it surfaces only much later
   inside ONNX Runtime. Two corrections that *are* load-bearing:
   `from_pretrained(..., dtype=)` fails on transformers 4.48.0, so the script
   calls `.float()`, which says the same thing across releases; and the export
   needs `onnx` and `onnxscript`, which the plan's install line omitted.

5. **The rename breaks no released consumer.** The Definition of done treats
   `huggingface.NewDedicated` to `endpoint.New` as the one migration a caller
   faces. In fact `NewDedicated` appears in no tag — it was added on this same
   unreleased branch — so the CHANGELOG lists `provider/endpoint` under Added
   rather than as a breaking change. Anyone tracking `main` between the two
   commits is the only affected reader.

6. **There was no e2e CI job to move.** Commit 2 says "the e2e job changes
   directory". CI never ran the live tests — they need credentials — it only
   excluded them from the root test job. That exclusion was dropped and the new
   `e2e-module` job added. It vets with `-tags=e2e`, because the provider tests
   have no untagged files and a plain `go build ./...` in that module skips the
   package silently rather than failing.

7. **One e2e corpus was deliberately not merged.** The plan says
   `internal/corpus` "removes the duplication". `cosine`, `dotProduct`,
   `maxSim`, and the four-document representation corpus did move there and are
   now single-sourced. The retrieval test's own corpus stayed where it is: it
   shares wording between query and answer to exercise a reranker's
   cross-encoding, where `corpus.Documents` deliberately withholds it, so
   merging them would quietly weaken both. A comment at the site records that.

8. ~~**Neither nested module is linted by CI or by `make lint`.**~~ Resolved
   after the fact. `make lint` now runs the root, `harness`, and `e2e` in one
   pass, and CI gained an `e2e-lint` job plus a lint step inside the existing
   `onnx` job. `e2e` is linted with `--build-tags=e2e` for the same reason it is
   vetted with it: `e2e/providers` has no untagged files, and golangci-lint
   without the tag exits cleanly having read nothing — confirmed by running both
   forms, which report `no go files to analyze` and `0 issues` respectively.

   `localinference` is deliberately absent from `make lint` and lives in
   `make lint-onnx`: linting cgo compiles it, and requiring ONNX Runtime for the
   ordinary lint loop would reintroduce the dependency this layout exists to
   keep optional. `make lint-all` runs everything; CI runs everything regardless.

10. **The Python handler was removed, and `provider/onnx` became
    `localinference`.** The plan placed the ONNX encoder under `provider/` and
    kept `deploy/representations` — a directory of Python handlers, schemas, and
    fixtures — as the protocol's reference implementation. Both were reversed
    afterwards on the repository owner's instruction: this is a Go library, and
    a directory of Python servers in it is a maintenance surface nobody asked
    for. The handlers, their tests, their requirements files, and the CI job
    that ran them are gone. The JSON Schemas and the golden request/response
    moved to `internal/representationwire/testdata`, where three Go test
    packages read them. Exactly one Python file survives, `export_onnx.py`,
    beside the only thing that needs it; it runs once per model and nothing at
    runtime invokes it. The encoder briefly moved to a top-level
    `localinference/` and then back to `provider/local/onnx/`, which is where it
    stays: it implements `RepresentationEncoder` exactly as five sibling
    providers do, so leaving `provider/` cost more than the name gained. `local`
    marks where the compute happens — the one property that is single-valued,
    unlike capability, which most providers hold several of at once — and the
    nested `go.mod` still keeps CGO out of everything else.

9. **The graph is 117 MiB, which is 124 MB.** The plan's "117 MB" was MiB. The
   two READMEs state both units, since the ~6% gap is otherwise read as two
   different artifacts.

11. **`e2e/localinference` exists so `e2e` can stay CGO-free.** The plan gave
    the ONNX encoder its own tests and stopped there, which left it with no
    consumer: nothing imported it, and no example showed it working. Importing
    it from `e2e` directly would have been simpler, and Go compiles only what an
    entry point reaches, so a single example would still run — but
    `go build ./...`, `go vet ./...`, and `golangci-lint run ./...` compile every
    package in their module, so both e2e CI jobs and any contributor running
    those commands would have needed a native ONNX Runtime and
    `libtokenizers.a`. A fourth module costs one `go.mod` and keeps every other
    example buildable with nothing but a Go toolchain. It holds the only example
    in the repository that needs no credentials.

## Definition of done

- The root `go.mod` requires nothing added by tests, examples, or ONNX.
- `provider/endpoint` is documented and tested as vendor-neutral; the
  huggingface package no longer implements the protocol.
- `e2e` builds and runs as its own module, contains the examples, and shares
  one corpus package.
- CI compiles the e2e module on every push.
- `localinference` encodes locally, agrees with the Python handler on term
  ranking, and is absent from the root build.
- `GOWORK=off` passes in root, `harness`, `e2e`, and `localinference`.
- Existing consumers compile unchanged, apart from the documented
  `huggingface.NewDedicated` to `endpoint.New` rename.
