package wrongvalidatoroutput

import (
	"context"

	agentic "github.com/regularkevvv/agentic"
)

type deps struct{}
type answer struct{ Value string }

func doesNotCompile(model agentic.Model) {
	agent := agentic.NewTypedAgentWithDeps[answer, *deps]("system", model, "answer")
	agent.AddOutputValidatorWithDeps(agentic.TypedOutputValidatorWithDepsFunc[*deps, string](
		func(agentic.RunContext[*deps], string) error { return nil },
	))
	_, _ = agent.Run(context.Background(), "prompt", &deps{})
}
