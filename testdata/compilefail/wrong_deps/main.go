package wrongdeps

import (
	"context"

	agentic "github.com/regularkevvv/agentic"
)

type expected struct{}
type wrong struct{}

func doesNotCompile(model agentic.Model) {
	agent := agentic.NewAgentWithDeps[*expected]("system", model)
	_, _ = agent.Run(context.Background(), "prompt", &wrong{})
}
