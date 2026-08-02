# e2e/localinference

One example, and the only one in this repository that needs no API key: it
encodes with a model running in the calling process.

```sh
uv run ../../provider/local/onnx/export_onnx.py --out ./models   # once, per model

AGENTIC_ONNX_MODEL=./models/granite-embedding-30m-sparse.onnx \
AGENTIC_ONNX_TOKENIZER=<snapshot>/tokenizer.json \
AGENTIC_ONNX_LIBRARY=/path/to/libonnxruntime.dylib \
CGO_LDFLAGS=-L/path/to/libtokenizers-dir \
go run ./examples/sparse
```

## Why this is a module and not a directory in e2e

Importing `provider/local/onnx` means CGO, a native ONNX Runtime, and a
statically linked tokenizer. Go compiles only what an entry point reaches, so a
single example would still run — but `go build ./...`, `go vet ./...`, and
`golangci-lint run ./...` compile every package in their module. Putting this
example directly in `e2e` would therefore make both e2e CI jobs, and any
contributor running those commands, depend on two external release artifacts.

Its own `go.mod` costs one file and keeps every other example buildable with
nothing but a Go toolchain. It is not in `go.work` for the same reason
`provider/local/onnx` is not.

Setup for the two native libraries is in
[`provider/local/onnx/README.md`](../../provider/local/onnx/README.md).
