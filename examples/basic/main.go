// Example: basic agent with no tools.
package main

import (
	"context"
	"fmt"
	"log"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/examples/internal/envutil"
	"github.com/regularkevvv/agentic/provider/openai"
)

func main() {
	if err := envutil.LoadDotEnv(); err != nil {
		log.Fatal(err)
	}

	model, err := openai.New("gpt-4o")
	if err != nil {
		log.Fatal(err)
	}

	agent := agentic.NewAgent(
		"You are a concise assistant. Keep answers short.",
		model,
	)

	result, err := agent.Run(
		context.Background(),
		envutil.PromptFromArgs("What is the capital of France?"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Output)
}
