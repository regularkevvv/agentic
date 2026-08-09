// Command main is the fresh no-replace consumer proof for the optional
// GoMonty adapter. Construction must remain source-only: importing and
// assembling the adapter cannot download or load a native runtime.
package main

import (
	"fmt"

	"github.com/regularkevvv/agentic/harness/codemode"
	montyexecutor "github.com/regularkevvv/agentic/harness/codemode/gomonty"
)

func main() {
	executor := montyexecutor.New(montyexecutor.Config{})
	if executor == nil {
		panic("gomonty adapter returned a nil executor")
	}
	var compatible codemode.Executor = executor
	fmt.Printf("compatible_executor=%T\n", compatible)
}
