package main

import (
	"github.com/regularkevvv/agentic"
	agenticotel "github.com/regularkevvv/agentic/otel"
)

func main() {
	instrumentation, err := agenticotel.New()
	if err != nil {
		panic(err)
	}
	_ = agentic.WithInstrumentation(instrumentation)
}
