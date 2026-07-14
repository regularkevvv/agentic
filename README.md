# Agentic

[![CI](https://github.com/regularkevvv/agentic/actions/workflows/ci.yml/badge.svg)](https://github.com/regularkevvv/agentic/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/regularkevvv/agentic.svg)](https://pkg.go.dev/github.com/regularkevvv/agentic)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A lightweight, type-safe Go framework for building AI agents with tool use, structured output, and multi-agent orchestration.

Current release: **v0.1.0**.

## Features

- **Provider-agnostic** -- pluggable `Model` interface with 9 built-in providers
- **Type-safe tools** -- generic tool builders with automatic JSON Schema generation from Go structs
- **Structured output** -- `TypedAgent[OutputT]` returns validated typed results
- **Streaming** -- first-class streaming with channel-based event delivery
- **Dependency injection** -- opt-in, exact dependency types shared by runs, prompts, tools, validators, and handoffs
- **Multi-agent** -- delegate tasks between agents with `Handoff`
- **Context-aware tools** -- cancellation, deadlines, tracing, and request-scoped values
- **Approval and channel tools** -- human approval gates and channel-backed results
- **History processors** -- truncation, sliding window, and LLM-based summarization
- **Multi-modal** -- images, audio, video, and document inputs
- **MCP support** -- use tools from Model Context Protocol servers
- **Thinking tokens** -- extended reasoning for Anthropic, OpenAI o-series, and Gemini
- **Output validation** -- struct tag validation with automatic retry

## Installation

```bash
go get github.com/regularkevvv/agentic
```

Requires **Go 1.25.4** or later.

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    agentic "github.com/regularkevvv/agentic"
    "github.com/regularkevvv/agentic/provider/openai"
)

func main() {
    model, _ := openai.New("gpt-4o")
    agent := agentic.NewAgent("You are a helpful assistant.", model)

    result, err := agent.Run(context.Background(), "Hello!")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result.Output)
}
```

## Tools

Define tools with automatic schema inference from Go types:

```go
type GetWeatherInput struct {
    _        struct{} `tool:"Look up the current weather for a city"`
    Location string   `json:"location" description:"City name"`
    Unit     string   `json:"unit,omitempty" enum:"celsius,fahrenheit"`
}

agentic.AddTool(agent, func(ctx context.Context, input GetWeatherInput) (WeatherOutput, error) {
    return WeatherOutput{Temperature: 24, Condition: "sunny"}, nil
})
```

`AddTool` is context-aware by default so cancellation, deadlines, and tracing
propagate from the agent run. Use `AddToolPlain` only for a small function that
does not need context. `AddToolWithContext` is an explicit alias for `AddTool`.

Tools with dependencies:

```go
type MyDeps struct {
    DB *sql.DB
}

agent := agentic.NewAgentWithDeps[*MyDeps]("Answer database questions.", model)
agentic.AddToolWithDeps(agent, func(ctx agentic.RunContext[*MyDeps], input QueryInput) (QueryOutput, error) {
    rows, _ := ctx.Deps.DB.QueryContext(ctx.Ctx, input.SQL)
    return QueryOutput{Rows: rows}, nil
})
```

## Structured Output

```go
type Summary struct {
    Title    string   `json:"title"    validate:"required"`
    Points   []string `json:"points"   validate:"required,min=1"`
}

agent := agentic.NewTypedAgent[Summary](
    "Summarize the input as structured data.",
    model,
    "Return a structured summary.",
)

result, _ := agent.Run(ctx, "Summarize Go's strengths.")
fmt.Println(result.Output.Title)  // typed access
```

## Providers

| Provider | Import | Constructor |
|----------|--------|-------------|
| OpenAI | `provider/openai` | `openai.New("gpt-4o")` |
| Anthropic | `provider/anthropic` | `anthropic.New("claude-sonnet-4-6")` |
| Google Gemini | `provider/gemini` | `gemini.New("gemini-2.0-flash")` |
| Azure OpenAI | `provider/azure` | `azure.New(endpoint, deployment, apiKey)` |
| AWS Bedrock | `provider/bedrock` | `bedrock.New("anthropic.claude-sonnet-4-6")` |
| Together AI | `provider/together` | `together.New("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")` |
| OpenRouter | `provider/openrouter` | `openrouter.New("anthropic/claude-sonnet-4")` |
| Ollama | `provider/ollama` | `ollama.New("llama3.1")` |
| Grok | `provider/grok` | `grok.New("grok-3")` |

Implement the `Model` interface to add your own.

## Agent-to-Agent

Delegate tasks to specialized sub-agents:

```go
researcher := agentic.NewAgentWithDeps[*MyDeps]("You are a research assistant.", model)
writer := agentic.NewAgentWithDeps[*MyDeps]("You write clear reports.", model)

// The identity handoff passes the writer's exact dependency value to the child.
writer.AddHandoffWithDeps(agentic.NewIdentityTextHandoff(
    "research", "Delegate research tasks", researcher,
))

result, err := writer.Run(ctx, "Research and report on Go.", deps)
```

## Approval and Channel-Backed Tools

Tools that require human approval before execution:

```go
tool, handler, _ := agentic.ApprovalTool(
    "delete_record", "Delete a database record",
    func(ctx context.Context, input DeleteInput) (string, error) {
        return db.Delete(input.ID)
    },
    func(ctx context.Context, call agentic.ToolUse) (bool, error) {
        return promptUser("Approve deletion of %s?", call.Input["id"]), nil
    },
)
```

`ApprovalTool` runs the approval callback and, if approved, the handler during
the current agent run. `ChannelTool` similarly waits for one channel result,
cancellation, or an optional timeout. Neither API suspends and later resumes a
durable workflow.

## Streaming

```go
stream, _ := agent.RunStream(ctx, "Tell me a story.")
for event := range stream.Events {
    if event.Type == agentic.StreamEventTextDelta {
        fmt.Print(event.Delta)
    }
}
```

## History Processors

Manage context window with built-in processors:

```go
agent := agentic.NewAgent("You are helpful.", model,
    agentic.WithHistoryProcessor(agentic.TruncateHistory(20)),
)

// Or chain multiple processors
agentic.WithHistoryProcessor(agentic.ChainProcessors(
    agentic.SlidingWindowHistory(4000, tokenCounter),
    agentic.TruncateHistory(50),
))
```

## Choosing a dependency pattern

- Use closures for a fixed service that is the same for every run.
- Use `NewAgentWithDeps[D]` or `NewTypedAgentWithDeps[O, D]` when application
  state varies by run and should be checked across prompts, tools, validators,
  and handoffs.
- Use `Bind(deps)` when an orchestrator needs a ready `Runner[O]` for one fixed
  dependency value.
- Use `BindProvider(provider)` when dependencies must be resolved once at the
  beginning of every run. Provider and dependency-validation failures happen
  before prompt, model, or tool side effects.

```go
agent := agentic.NewAgentWithDeps[*MyDeps]("Be helpful.", model)

var fixed agentic.Runner[string] = agent.Bind(deps)
result, err := fixed.Run(ctx, prompt)

perRun := agent.BindProvider(func(ctx context.Context) (*MyDeps, error) {
    return dependenciesForRequest(ctx)
})
```

## Project Layout

```
agentic/
  agent.go            # Core agent orchestration
  agent_options.go    # Configuration options
  stream.go           # Streaming support
  typed_agent.go      # TypedAgent for structured output
  handoff.go          # Agent-to-agent delegation
  tool/channel.go     # Channel-backed and approval-gated tools
  history_processor.go # Message history transforms
  tool/               # Tool builders, registry, toolsets
  provider/           # LLM provider implementations
  internal/core/      # Shared types (Message, Tool, Model, etc.)
  mcp/                # Model Context Protocol integration
  examples/           # Runnable example programs
```

## Examples

```bash
cp .env.example .env   # fill in your API key
go run ./examples/basic
go run ./examples/tools
go run ./examples/structured
```

See [`examples/`](examples/README.md) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR guidelines.

## License

[MIT](LICENSE)
