// Example: typed agent with structured output and validation.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/examples/internal/envutil"
	"github.com/regularkevvv/agentic/provider/openai"
)

// RepoSummary defines the structured output schema.
// Validation tags are enforced automatically — invalid output is
// sent back to the model for retry.
type RepoSummary struct {
	Name     string   `json:"name"      description:"Repository or project name"  validate:"required,min=1"`
	UseCases []string `json:"use_cases" description:"Main use cases"              validate:"required,min=1"`
	Strength string   `json:"strength"  description:"Most important strength"     validate:"required,min=5"`
}

func main() {
	if err := envutil.LoadDotEnv(); err != nil {
		log.Fatal(err)
	}

	model, err := openai.New("gpt-4o")
	if err != nil {
		log.Fatal(err)
	}

	agent := agentic.NewTypedAgent[RepoSummary](
		"You summarize repositories as structured data.",
		model,
		"Return the final answer as a structured repository summary.",
	)

	result, err := agent.Run(
		context.Background(),
		envutil.PromptFromArgs("Summarize the Go standard library."),
	)
	if err != nil {
		log.Fatal(err)
	}

	data, _ := json.MarshalIndent(result.Output, "", "  ")
	fmt.Println(string(data))
}
