package unboundrunner

import agentic "github.com/regularkevvv/agentic"

type deps struct{}

func doesNotCompile(model agentic.Model) {
	var _ agentic.Runner[string] = agentic.NewAgentWithDeps[*deps]("system", model)
}
