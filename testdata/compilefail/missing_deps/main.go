package missingdeps

import (
	"context"

	agentic "github.com/regularkevvv/agentic"
)

type deps struct{}

func doesNotCompile(model agentic.Model) {
	agent := agentic.NewAgentWithDeps[*deps]("system", model)
	_, _ = agent.Run(context.Background(), "prompt")
}
