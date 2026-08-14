// The example in this module encodes with a model running in its own process,
// which means CGO, a native ONNX Runtime, and a statically linked tokenizer.
//
// It is a module of its own so that none of that reaches the e2e module beside
// it. Whole-module commands there — go build ./..., go vet ./..., golangci-lint
// run ./... — would otherwise need both native libraries present, and the two
// e2e CI jobs would depend on downloading them. Every other example stays
// buildable with nothing but a Go toolchain, which is the point.
module github.com/regularkevvv/agentic/e2e/localinference

go 1.25.4

require (
	github.com/regularkevvv/agentic v0.6.0
	github.com/regularkevvv/agentic/provider/local/onnx v0.1.0
)

require (
	github.com/daulet/tokenizers v1.27.0 // indirect
	github.com/gabriel-vasile/mimetype v1.4.12 // indirect
	github.com/go-playground/locales v0.14.1 // indirect
	github.com/go-playground/universal-translator v0.18.1 // indirect
	github.com/go-playground/validator/v10 v10.30.1 // indirect
	github.com/leodido/go-urn v1.4.0 // indirect
	github.com/swaggest/jsonschema-go v0.3.79 // indirect
	github.com/swaggest/refl v1.4.0 // indirect
	github.com/yalue/onnxruntime_go v1.31.0 // indirect
	golang.org/x/crypto v0.50.0 // indirect
	golang.org/x/sys v0.46.0 // indirect
	golang.org/x/text v0.36.0 // indirect
)

// Both resolve to the checkout, for the reason e2e/go.mod gives: this module
// exists to demonstrate the code in this tree, not the last released tag.
replace github.com/regularkevvv/agentic => ../..

replace github.com/regularkevvv/agentic/provider/local/onnx => ../../provider/local/onnx
