# Agentic

[![CI](https://github.com/regularkevvv/agentic/actions/workflows/ci.yml/badge.svg)](https://github.com/regularkevvv/agentic/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/regularkevvv/agentic.svg)](https://pkg.go.dev/github.com/regularkevvv/agentic)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A lightweight, type-safe Go framework for building AI agents with tool use, structured output, and multi-agent orchestration.

Current release: **v0.5.1**.

## Features

- **Provider-agnostic** -- pluggable `Model` interface with 9 built-in providers
- **Type-safe tools** -- generic tool builders with automatic JSON Schema generation from Go structs
- **Structured output** -- `TypedAgent[OutputT]` returns validated typed results
- **Streaming** -- first-class streaming with channel-based event delivery
- **Resumable execution** -- explicit `Driver` control for start, continue, gate or handler suspension, and resume
- **Execution events** -- canonical lifecycle, transcript, tool, and preview events for one run
- **Dependency injection** -- opt-in, exact dependency types shared by runs, prompts, tools, validators, and handoffs
- **Multi-agent** -- delegate tasks between agents with `Handoff`
- **Context-aware tools** -- cancellation, deadlines, tracing, and request-scoped values
- **Approval and channel tools** -- human approval gates and channel-backed results
- **History processors** -- truncation, sliding window, and LLM-based summarization
- **Multi-modal** -- images, audio, video, and document inputs
- **MCP support** -- use tools from Model Context Protocol servers
- **Embeddings** -- provider-agnostic `Embedder` interface with OpenAI, Voyage AI, Cohere, Gemini, Ollama, and Bedrock implementations and retrieval-tuned query/document input types
- **Multi-representation encoding** -- `RepresentationEncoder` returns dense, learned sparse, and token multi-vectors from one call, with validated vector-space identity, via DeepInfra, an endpoint you operate, Hugging Face, SageMaker, Pinecone, or ONNX Runtime in your own process
- **Reranking** -- `Reranker` cross-encoder interface with Voyage AI and Cohere implementations for two-stage retrieval
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

`provider/` is flat because capability is not single-valued: Bedrock is a model
*and* an embedder, Cohere an embedder *and* a reranker, DeepInfra an embedder
*and* an encoder. Any arrangement by role would have to file those packages in
two places, so the table is the index instead — and a test enforces that every
provider declares its capabilities in code, so this cannot silently go stale.

| Provider | Import | Chat | Embed | Encode | Rerank |
|---|---|:-:|:-:|:-:|:-:|
| OpenAI | `provider/openai` | ✓ | ✓ | | |
| Anthropic | `provider/anthropic` | ✓ | | | |
| Google Gemini | `provider/gemini` | ✓ | ✓ | | |
| Azure OpenAI | `provider/azure` | ✓ | | | |
| AWS Bedrock | `provider/bedrock` | ✓ | ✓ | | |
| Together AI | `provider/together` | ✓ | | | |
| OpenRouter | `provider/openrouter` | ✓ | | | |
| Ollama | `provider/ollama` | ✓ | ✓ | | |
| Grok | `provider/grok` | ✓ | | | |
| Voyage AI | `provider/voyageai` | | ✓ | | ✓ |
| Cohere | `provider/cohere` | | ✓ | | ✓ |
| DeepInfra | `provider/deepinfra` | | ✓ | ✓ | |
| Endpoint you operate | `provider/endpoint` | | ✓ | ✓ | |
| Hugging Face router | `provider/huggingface` | | ✓ | ✓ | |
| SageMaker | `provider/sagemaker` | | ✓ | ✓ | |
| Pinecone Inference | `provider/pinecone` | | ✓ | ✓ | |
| In-process ONNX | `provider/local/onnx` | | | ✓ | |

**Chat** is `Model`/`StreamModel`, **Embed** is `Embedder`, **Encode** is
`RepresentationEncoder` (dense, learned sparse, and multi-vector in one pass),
**Rerank** is `Reranker`. Implement the matching interface to add your own.

`provider/local/onnx` is the one entry that is not part of this module: it needs
CGO and a native ONNX Runtime, so it carries its own `go.mod` and `go get
agentic` never pulls it. Constructors:

```go
openai.New("gpt-4o")                          anthropic.New("claude-sonnet-4-6")
gemini.New("gemini-2.0-flash")                bedrock.New("anthropic.claude-sonnet-4-6")
azure.New(deployment, azure.WithEndpoint(u), azure.WithAPIKey(k))
together.New("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
openrouter.New("anthropic/claude-sonnet-4")   ollama.New("llama3.1")
grok.New("grok-3")
```

## Embeddings

The same interface-plus-providers pattern covers embeddings: `agentic.Embedder`
is the interface, with implementations in `provider/openai`, `provider/voyageai`,
`provider/cohere`, `provider/gemini`, `provider/ollama`, and `provider/bedrock`.

```go
import "github.com/regularkevvv/agentic/provider/voyageai"

embedder, _ := voyageai.New("voyage-3.5") // key from VOYAGE_API_KEY
// or: openai.NewEmbedder("text-embedding-3-small")
// or: cohere.New("embed-v4.0"), gemini.NewEmbedder("gemini-embedding-001"),
//     ollama.NewEmbedder("nomic-embed-text"), bedrock.NewEmbedder("amazon.titan-embed-text-v2:0")

// Index documents, then embed queries against them:
docs, _ := agentic.EmbedDocuments(ctx, embedder, "The monthly budget is $2,000.")
query, _ := agentic.EmbedQuery(ctx, embedder, "what is the budget?")
```

`EmbedQuery` and `EmbedDocuments` set the request's `InputType`.
Retrieval-tuned providers (Voyage AI, Cohere, Gemini) use it to prepend a task
instruction before vectorizing, which embeds queries near their answering
documents; providers without the concept (OpenAI) ignore it. For similarity,
clustering, or classification, call `Embed` directly and leave `InputType` unset.

Requests are batch-first (`Input []string`), support dimension control via
`Dimensions` on models that allow it (OpenAI `text-embedding-3` and later;
Voyage AI `voyage-3.5` and later), and report token usage. Vectors are returned
in input order as `[][]float32`. `provider/test.NewTestEmbedder` provides a
deterministic in-memory fake for tests.

## Multi-representation encoding

Some retrieval models produce more than one view of a text from a single
forward pass. BGE-M3 produces a dense vector, a learned sparse vector, and a
per-token multi-vector at once. `agentic.Embedder` carries the first and
nothing else, so `agentic.RepresentationEncoder` sits beside it and carries all
three without flattening them.

```go
import "github.com/regularkevvv/agentic/provider/deepinfra"

encoder, _ := deepinfra.New(deepinfra.BGEM3Model) // token from DEEPINFRA_TOKEN

resp, _ := agentic.EncodeDocuments(ctx, encoder,
    []string{"PostgreSQL supports sparse vectors"},
    agentic.RepresentationDense,
    agentic.RepresentationSparse,
)

item := resp.Data[0]
denseSpace := resp.Spaces[agentic.RepresentationDense]
sparseSpace := resp.Spaces[agentic.RepresentationSparse]

// Store item.Dense and item.Sparse in your index, each beside its space ID.
```

Outputs are explicit because they differ in cost by orders of magnitude. `Data`
has one entry per input, in input order, carrying exactly the kinds you asked
for.

Store `space.ID` beside every value. Two vectors from the same *named* model
but different weights revisions, or two sparse vectors from different tokenizer
vocabularies, are not comparable — and the failure is silent, in the form of
quietly worse recall rather than an error. The ID is a deterministic hash of
the fields that make values comparable, so two processes agree on it without
coordinating.

Which representations each encoder returns — the Encode column above, in
detail:

| Import | Constructor | Dense | Sparse | Multi-vector |
|---|---|---|---|---|
| `provider/deepinfra` | `deepinfra.New(deepinfra.BGEM3Model)` | Yes | Yes | Yes |
| `provider/endpoint` | `endpoint.New(endpointURL)` | Yes | Yes | Optional |
| `provider/huggingface` | `huggingface.NewShared(model)` | Yes | No | No |
| `provider/sagemaker` | `sagemaker.New(ctx, endpointName)` | Yes | Yes | Optional |
| `provider/pinecone` | `pinecone.New(model)` | Yes | Yes | No |
| `provider/local/onnx` | `onnx.New(model, tokenizer, space)` | No | Yes | No |

Agentic normalizes inference and stops there. Indexing, BM25, candidate fusion,
and final ranking belong to the retrieval system you build on top, which is why
nothing here returns a score.

Existing `Embedder` code is unaffected. `EmbedderAsRepresentationEncoder` drives
any dense embedder through the new interface, and
`RepresentationEncoderAsEmbedder` drives a multi-output encoder through the old
one, so sparse output can be adopted when your index is ready for it.
`provider/test.NewTestRepresentationEncoder` is a deterministic fake, and
`provider/test/conformance.RunRepresentation` is the shared contract suite that
every provider — and your own — is checked against.

`provider/endpoint` speaks the versioned protocol to any host running a
handler that implements it — a Hugging Face Inference Endpoint, a container, or
a process on a laptop — and takes its token from `AGENTIC_ENDPOINT_TOKEN`, or none at all
with `endpoint.WithoutAuthentication()`. `provider/sagemaker` speaks the same
protocol through SageMaker Runtime. See
[docs/multi-representation-inference.md](docs/multi-representation-inference.md)
for the full contract.

`provider/local/onnx` encodes sparse vectors in your own process with no server at
all. It is a **nested module** — the one directory under `provider/` that
`go get github.com/regularkevvv/agentic` does not pull — because it needs CGO, a
native ONNX Runtime, and a statically linked tokenizer, none of which can be a
condition of importing the library. See
[`provider/local/onnx/README.md`](provider/local/onnx/README.md) for the setup, which is a
real barrier rather than a formality.

## Reranking

A reranker is a cross-encoder: it scores the query against each document
directly, which is markedly more accurate than comparing independently computed
embeddings, and markedly more expensive. The usual arrangement is two stages —
retrieve a shortlist with an `Embedder`, then reorder it with a `Reranker`.
`agentic.Reranker` is the interface, with implementations in `provider/voyageai`
and `provider/cohere`.

```go
import "github.com/regularkevvv/agentic/provider/voyageai"

reranker, _ := voyageai.NewReranker("rerank-2.5") // key from VOYAGE_API_KEY
// or: cohere.NewReranker("rerank-v3.5")

resp, _ := agentic.Rerank(ctx, reranker, "what is the budget?", shortlist, 5)
for _, r := range resp.Results { // ordered by descending score
    fmt.Printf("%.3f  %s\n", r.Score, r.Document)
}
```

`RerankResult.Index` maps each result back to its position in the input slice,
so metadata kept in a parallel slice rejoins cleanly. Scores are ordinal within
a single response only — rank by them, never threshold across providers, models,
or calls. `provider/test.NewTestReranker` provides a deterministic fake for
tests.

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

## Controlled and Resumable Runs

`Run` remains the concise API for ordinary one-shot requests. When an
orchestrator needs to continue a transcript, install per-run controls, or resume
a deferred tool batch, opt into the explicit `Driver` capability:

```go
driver, err := agentic.RequireDriver[string](agent)
if err != nil {
    log.Fatal(err)
}

events := agentic.EventSinkFunc(func(context.Context, agentic.Event) error {
    return nil // persist or observe canonical execution events here
})
prompt := agentic.NewTextMessage(agentic.RoleUser, "Prepare the report.")
execution, err := driver.Drive(ctx, agentic.DriveInput{
    Mode:   agentic.DriveStart,
    Prompt: &prompt,
}, agentic.WithRunEventSink(events))
if err != nil {
    log.Fatal(err)
}
fmt.Println(execution.Status)
```

`DriveContinue` accepts an already paired transcript without appending a
synthetic user prompt. A `ToolGate` may suspend an admitted batch before any
handler runs; persist the returned `Execution.Result.Messages` and
`Execution.Suspension`, then call `Resume` with exactly one
`ToolResumeDecision` for each suspended executable call. `Bind` and
`BindProvider` still return `Runner`, while their concrete values also implement
`Driver` and can be checked with `RequireDriver`.

A handler that can discover deferred work only after it starts may opt into
`SuspendableToolHandler` and return `ToolHandlerSuspension`. The driver isolates
that handler from other regular calls before execution and leaves its tool
frontier open. On `Resume`, the same handler reads its persisted deferral and
the caller-validated decision payload with `CurrentToolResume`. Ordinary
handlers and gate-originated resumptions never receive that resume context.
Composite tool hosts use `WithToolCallContext` before invoking a nested handler
so it receives its own call identity without inheriting the outer handler's
resume payload.

## Experimental Durable Harness

The repository also contains the nested experimental module
`github.com/regularkevvv/agentic/harness`. Its Phase 4 surface adds durable
single-process sessions, steering/follow-up/next-turn queues, interruption and
recovery, bounded cursor subscriptions, transcript repair, Local and Memory
environments, session-scoped artifact spill, an immutable capability DAG,
context compaction, default-deny permissions, exact deferred resume, and
capture-restricted in-process child agents with separate durable sessions,
addressed inboxes, tagged events, recursion limits, cancellation, and shared
budget accounting.

The low-level constructor is `harness.NewRuntime`; it depends on generic
journal, codec, event, environment, clock/ID, and result-processing ports.
Concrete memory, JSONL, local, file, and in-process adapters are selected only
at the application composition root. `harness.New(...).Build()` composes public
capabilities without changing those dependency directions, while
`harness.Default` is an explicit local convenience assembly. Subagents remain
optional capabilities rather than part of that default. Out-of-process
subagents, topology presets, codemode, long-term memory, skills, and evals
remain later-phase work. See
[`harness/README.md`](harness/README.md) for the exact experimental boundary.

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
  go.work             # Local workspace: root + nested harness and e2e
  agent.go            # Core agent orchestration
  agent_options.go    # Configuration options
  driver.go           # Start/continue/resume execution capability
  execution_loop.go   # Shared blocking and streaming execution fold
  events.go           # Canonical execution events
  stream.go           # Streaming support
  typed_agent.go      # TypedAgent for structured output
  handoff.go          # Agent-to-agent delegation
  tool/channel.go     # Channel-backed and approval-gated tools
  history_processor.go # Message history transforms
  tool/               # Tool builders, registry, toolsets
  representation.go   # Multi-representation encoding helpers and adapters
  provider/           # LLM, embedding, reranking, and encoding providers
  internal/core/      # Shared types (Message, Tool, Model, etc.)
  mcp/                # Model Context Protocol integration
  harness/            # Nested module: experimental durable sessions
  e2e/                # Nested module: live provider tests and examples
  provider/local/onnx/ # Nested module: in-process sparse encoding (CGO)
```

## Examples

```bash
cp .env.example .env   # fill in your API key
go run ./e2e/examples/basic
go run ./e2e/examples/tools
go run ./e2e/examples/structured
```

The examples live in the nested `e2e` module so that what they need to be
readable — a table writer, a live API key — never reaches an application that
imports the library. The committed `go.work` is what keeps them runnable from
the repository root.

See [`e2e/examples/`](e2e/examples/README.md) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR guidelines.

## License

[MIT](LICENSE)
