# provider/local/onnx

Learned sparse encoding in your own process, through ONNX Runtime. No server,
no network, no Python at runtime.

This directory is a **separate Go module**. `go get github.com/regularkevvv/agentic`
does not pull it, nothing in the root module's build reaches it, and it is
deliberately absent from `go.work` — working on it means `cd provider/local/onnx`.
That is what keeps CGO out of the default contributor loop and out of every
application that imports Agentic.

## Two things must be on disk before anything compiles

Both are discovered at link time otherwise, as `library 'tokenizers' not found`
or as a runtime failure to load `libonnxruntime`. Neither is a Go dependency and
neither is fetched by `go build`.

### 1. `libtokenizers.a`, per platform

`github.com/daulet/tokenizers` is a CGO wrapper around the Rust crate
`transformers` uses. The Go side is on proxy.golang.org; the compiled static
library is a release asset you fetch yourself.

```sh
cd provider/local/onnx
mkdir -p lib
# darwin/arm64 — the only platform this has been verified on
curl -L https://github.com/daulet/tokenizers/releases/download/v1.27.0/libtokenizers.darwin-arm64.tar.gz | tar xz -C lib
# linux/amd64
curl -L https://github.com/daulet/tokenizers/releases/download/v1.27.0/libtokenizers.linux-amd64.tar.gz | tar xz -C lib

export CGO_LDFLAGS="-L$PWD/lib"
```

`-L` is all that is needed. The binding's own `#cgo LDFLAGS` already supply
`-ltokenizers -ldl -lm -lstdc++`, and repeating them only produces
`ld: warning: ignoring duplicate libraries`.

The release publishes `darwin-{arm64,x86_64}`, `linux-{amd64,arm64,aarch64,
ppc64le,s390x}`, and musl variants of each. Pick the one matching where the
binary will run, not where it is built: this links statically.

### 2. ONNX Runtime, a separate download

```sh
# darwin/arm64
curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-osx-arm64-1.28.0.tgz
tar xzf onnxruntime-osx-arm64-1.28.0.tgz
export AGENTIC_ONNX_LIBRARY=$PWD/onnxruntime-osx-arm64-1.28.0/lib/libonnxruntime.dylib

# linux/amd64
curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz
tar xzf onnxruntime-linux-x64-1.28.0.tgz
export AGENTIC_ONNX_LIBRARY=$PWD/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime.so
```

This one is loaded at runtime rather than linked, so it can also be pointed at
with `onnx.WithLibraryPath`. The environment variable exists because ONNX
Runtime has no standard install location on any platform here; the option always
wins over it.

### What all of that costs

Measured 2026-08-01 on darwin/arm64.

| Artifact | Size | Where it goes |
| --- | --- | --- |
| ONNX Runtime, macOS arm64 | 32 MB download, 39 MB `.dylib` | loaded at runtime |
| ONNX Runtime, Linux x64 | 9 MB download | loaded at runtime |
| `libtokenizers.a` | 14 MB download, 39 MB on disk | ~18 MB linked into your binary |
| Exported Granite graph | ~117 MiB (124 MB on disk) | read at construction |

The 18 MB is the difference between this package's test binary (29 MB) and a
comparable pure-Go binary that imports Agentic (12 MB). It is static: the
resulting binary needs no `libtokenizers` at runtime.

## Getting the model

Granite ships no ONNX export, so producing one is a documented one-time step
rather than a download. `provider/local/onnx/export_onnx.py` does it and
writes the PyTorch reference this package's tests assert against:

```sh
uv run provider/local/onnx/export_onnx.py --out /some/writable/directory
```

One command, and it leaves nothing behind: the script declares its dependencies
in a PEP 723 header, so uv builds a throwaway environment pinned by
`export_onnx.py.lock` and discards it afterwards. Python is a build tool here,
in the same sense a compiler is — this package links against no Python and
executes none. Without uv, `pip install torch==2.13.0 transformers onnx
onnxscript` does the same thing unpinned.

The 117 MiB graph cannot live in the repository, which is why the live tests are
gated on a local path. See
[docs/multi-representation-inference.md](../../../docs/multi-representation-inference.md#in-process-with-no-server)
for the export step and the runtime setup in one place.

## Using it

```go
import (
    agentic "github.com/regularkevvv/agentic"
    "github.com/regularkevvv/agentic/provider/local/onnx"
)

encoder, err := onnx.New(
    "/models/granite-sparse.onnx",
    "/models/tokenizer.json",
    agentic.VectorSpace{
        Model:    "ibm-granite/granite-embedding-30m-sparse",
        Revision: "ad82b1fd09541c998c8d45045d601c51fdb8a9b7",
    },
)
if err != nil {
    return err
}
defer encoder.Close()

resp, err := agentic.EncodeDocuments(ctx, encoder, docs, agentic.RepresentationSparse)
```

`Close` is not optional. The session and the tokenizer hold memory outside Go's
heap that no garbage collector reclaims.

The vector space is yours to declare, and the parts it can prove are filled in:
`Dimensions` comes from the graph's own output width, `Provider` defaults to
`onnx`, `Metric` to dot product, and the ID is the canonical hash of the rest.
Record `Revision` — an index keyed on an unrevisioned space cannot detect a
silent re-export.

## What it does and does not do

**Sparse only.** Dense and multi-vector requests return
`agentic.ErrUnsupportedRepresentation` rather than an answer from the wrong
reduction. The pooling is SPLADE's — `log1p(relu(logits))`, masked, max over
positions — the reduction the model's own sentence-transformers pooling config
declares, and the reason a document about an "automobile" carries weight on
"car".

**Nothing is downloaded.** Both constructor arguments are filesystem paths.
There is no HTTP client in this package.

**Over-long inputs are rejected, not truncated.** The limit is the model's
positional bound, 512 tokens by default; past it the graph indexes off the end
of its position table rather than degrading. Silently dropping the tail of a
document would produce a vector for text nobody asked about.

**Batch width is the cost, not batch size.** Every row in a forward pass is
padded to the widest one. Inputs are therefore ordered by token length and
grouped with neighbors, so no row is padded past twice its own length and a
long document never drags short ones up to its width. A second ceiling caps the
logits tensor a pass may allocate — at Granite's vocabulary a single 512-token
input is already 103 MB of logits.

**Calls are serialized.** ONNX Runtime already spreads one graph across every
core it is configured for, so concurrent `Encode` calls would contend rather
than scale.

## Running the tests

The unit tests need the tokenizer library on the link line and nothing else:

```sh
CGO_LDFLAGS="-L$PWD/lib" GOWORK=off go test ./...
```

The live tests are the gate for this package. They skip cleanly without their
three variables and assert against `testdata/granite_reference.json`:

| Variable | Meaning |
| --- | --- |
| `AGENTIC_ONNX_MODEL` | the exported `.onnx` graph |
| `AGENTIC_ONNX_TOKENIZER` | the matching `tokenizer.json` |
| `AGENTIC_ONNX_LIBRARY` | `libonnxruntime.dylib` or `libonnxruntime.so` |

```sh
export AGENTIC_ONNX_MODEL=/models/granite-sparse.onnx
export AGENTIC_ONNX_TOKENIZER=/models/tokenizer.json
export AGENTIC_ONNX_LIBRARY=/opt/onnxruntime/lib/libonnxruntime.dylib
CGO_LDFLAGS="-L$PWD/lib" GOWORK=off go test -v -run Live ./...
```

They check four separable claims, so a failure says which one broke:

- the Go tokenizer reproduces `transformers`' input ids from the raw text;
- every coordinate index matches PyTorch's and every weight is within 1e-4 —
  **measured 4.56e-06** on 2026-08-01, which is why the gate is 1e-4 and not
  looser;
- a row padded to twice its own length carries the same coordinate indices as
  the same row alone, and weights differing by at most **1.46e-06** — the wider
  forward pass's arithmetic rather than the padding;
- the padding id does not reach the result: that same padded row is identical to
  the bit under two padding ids, which is what says the mask is doing the
  excluding.

## Platform support and licenses

**darwin/arm64** and **linux/amd64** have both been verified end to end against
the real exported graph, on 2026-08-01. Every coordinate index matched the
PyTorch golden on both, and the largest weight difference was 4.56e-06 on
darwin/arm64 and 4.38e-06 on linux/amd64 — the two platforms disagree with
PyTorch by about as much as they disagree with each other, and both sit far
inside the 1e-4 the live test gates on.

No other platform has been run. The bindings publish artifacts for linux/arm64,
musl, and darwin/x86_64 among others; those are untested here.

Both bindings are MIT: `github.com/daulet/tokenizers` v1.27.0 and
`github.com/yalue/onnxruntime_go` v1.31.0. Both are pinned exactly. ONNX Runtime
itself is MIT and is not vendored here.
