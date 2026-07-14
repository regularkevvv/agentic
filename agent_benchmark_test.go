package agentic_test

import (
	"context"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

type benchmarkModel struct{}

func (benchmarkModel) Name() string { return "benchmark:model" }

func (benchmarkModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	return &agentic.ChatResponse{
		ID:    "benchmark-response",
		Model: "benchmark:model",
		Choices: []agentic.Choice{{
			Message:      agentic.NewTextMessage(agentic.RoleAssistant, "ok"),
			FinishReason: agentic.FinishReasonStop,
		}},
	}, nil
}

type benchmarkDeps struct {
	Value string
}

var benchmarkResult *agentic.Result[string]

func BenchmarkAgentRunModes(b *testing.B) {
	ctx := context.Background()
	model := benchmarkModel{}
	deps := &benchmarkDeps{Value: "value"}

	b.Run("no_deps", func(b *testing.B) {
		agent := agentic.NewAgent("system", model)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			result, err := agent.Run(ctx, "prompt")
			if err != nil {
				b.Fatal(err)
			}
			benchmarkResult = result
		}
	})

	b.Run("direct_deps", func(b *testing.B) {
		agent := agentic.NewAgentWithDeps[*benchmarkDeps]("system", model)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			result, err := agent.Run(ctx, "prompt", deps)
			if err != nil {
				b.Fatal(err)
			}
			benchmarkResult = result
		}
	})

	b.Run("bound_deps", func(b *testing.B) {
		runner := agentic.NewAgentWithDeps[*benchmarkDeps]("system", model).Bind(deps)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			result, err := runner.Run(ctx, "prompt")
			if err != nil {
				b.Fatal(err)
			}
			benchmarkResult = result
		}
	})
}
