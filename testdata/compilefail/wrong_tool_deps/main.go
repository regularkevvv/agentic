package wrongtooldeps

import agentic "github.com/regularkevvv/agentic"

type agentDeps struct{}
type toolDeps struct{}
type input struct{}
type output struct{}

func doesNotCompile(model agentic.Model) {
	agent := agentic.NewAgentWithDeps[*agentDeps]("system", model)
	agentic.AddToolWithDeps(agent, func(agentic.RunContext[*toolDeps], input) (output, error) {
		return output{}, nil
	})
}
