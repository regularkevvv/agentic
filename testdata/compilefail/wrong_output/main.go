package wrongoutput

import (
	"context"

	agentic "github.com/regularkevvv/agentic"
)

type answer struct{ Value string }

func doesNotCompile(model agentic.Model) {
	agent := agentic.NewTypedAgent[answer]("system", model, "answer")
	result, _ := agent.Run(context.Background(), "prompt")
	var _ string = result.Output
}
