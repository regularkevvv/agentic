// Command main is the fresh no-replace consumer proof for the native ONNX
// provider release view. It compiles and links the public provider module
// against its published root Agentic dependency without loading a model.
package main

import (
	"fmt"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/local/onnx"
)

func main() {
	fmt.Printf("compatible_encoder=%T\n", (*onnx.Encoder)(nil))
}

var _ agentic.RepresentationEncoder = (*onnx.Encoder)(nil)
