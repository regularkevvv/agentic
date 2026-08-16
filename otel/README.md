# Agentic OpenTelemetry

`github.com/regularkevvv/agentic/otel` is the optional OpenTelemetry adapter for
Agentic. It emits GenAI traces, metrics, and structured OTel log records without
making the root `agentic` source or public instrumentation API import
OpenTelemetry.

The dependency direction is:

```text
your application
  ├── agentic
  ├── agentic/otel ──> agentic + OpenTelemetry APIs
  └── OpenTelemetry SDK + exporter chosen by the application
```

The adapter does not create an SDK, exporter, batch processor, or shutdown
policy. Configure those at the application composition root. `New` uses the
global OTel providers by default, or accepts explicit providers for isolated
services and tests.

## Install

```bash
go get github.com/regularkevvv/agentic/otel
```

```go
import (
    "github.com/regularkevvv/agentic"
    agenticotel "github.com/regularkevvv/agentic/otel"
)

instrumentation, err := agenticotel.New()
if err != nil {
    return err
}

agent := agentic.NewAgent(
    "You are helpful.",
    model,
    agentic.WithInstrumentation(instrumentation),
    agentic.WithAgentIdentity(agentic.AgentIdentity{
        Name:        "support-agent",
        Description: "Helps customers resolve invoice questions",
    }),
)

result, err := agent.Run(ctx, "Help me with my invoice.",
    agentic.WithRunMetadata(agentic.RunMetadata{
        ConversationID: conversationID,
        RunID:          runID,
    }),
)
```

Harness `v0.5.0` and later supplies its real session and durable run IDs
automatically. Plain Agentic callers should pass IDs only when their
application already owns them; the framework deliberately does not invent
correlation IDs.

## What is emitted

One agent invocation is a trace subtree:

```text
invoke_agent support-agent                 INTERNAL span
  ├── chat gpt-4.1                         CLIENT span
  ├── execute_tool lookup_invoice          INTERNAL span
  └── chat gpt-4.1                         CLIENT span
```

Each `Drive`, `Resume`, or `RunStream` invocation closes its own
`invoke_agent` span. A durable suspension is a successful terminal outcome for
that invocation; a later `Resume` creates another invocation span. Handoffs are
ordinary tools. Nested child agents inherit the parent observer and correlation
IDs through `context`, so each child appears below the handoff tool span and
owns its own inference/tool call counts. An explicit child-level or run-level
instrumentation option, including `nil`, overrides that inheritance.

The adapter emits:

- Spans: `invoke_agent`, provider inference (`chat`, `generate_content`, or a
  configured operation), `execute_tool`, and `embeddings`.
- Metrics: client token usage, client duration, streaming time-to-first-chunk
  and time-per-output-chunk, agent duration, per-agent inference/tool call
  counts, and tool duration.
- Log records: `gen_ai.client.operation.exception` by default,
  `gen_ai.client.inference.operation.details` when enabled, and explicit
  `gen_ai.evaluation.result` records through `RecordEvaluation`.

GenAI named events are OTel LogRecords, not trace span events. The SDK
correlates them to the current trace/span through the context passed to the
adapter.

## Collector-backed proof

The repository includes a credential-free runnable example and a Docker
Compose smoke test under [`e2e/otel`](../e2e/otel/README.md). It exercises
nested agents, tools, suspension/resume, streaming, provider failure,
embeddings, inference details, and evaluation records against a pinned real
OpenTelemetry Collector.

```bash
make otel-e2e
```

The gate parses the Collector's exported OTLP JSON using the official protobuf
types. It checks exact span topology and counts, all metric instruments,
trace-correlated log records, durable IDs, semantic attributes, and a privacy
tripwire that fails if deliberately sensitive raw text appears anywhere.

Error dimensions are deliberately low-cardinality. They are `canceled`,
`deadline_exceeded`, the canonical leaf Go error type, `provider_error` for an
in-band model failure, `tool_error` for an unsuccessful tool result, or the
agent outcomes `failed` and `interrupted` when no concrete error is available.

## Privacy

Prompts, responses, system instructions, reasoning, tool definitions,
arguments, results, inline binary data, and exception messages are not exported
by default. Default exception records contain only `exception.type`.

Opt in deliberately:

```go
instrumentation, err := agenticotel.New(
    agenticotel.WithMessageContent(),
    agenticotel.WithToolContent(),
    agenticotel.WithInferenceDetails(),
    agenticotel.WithExceptionContent(),
    agenticotel.WithMaxContentBytes(64*1024),
    agenticotel.WithContentFilter(func(kind agenticotel.ContentKind, value string) (string, bool) {
        return redact(value), true
    }),
)
```

`WithMaxContentBytes` omits a complete structured attribute when its valid JSON
representation exceeds the limit; it never emits truncated invalid JSON.
Inline binary payloads, provider signatures, vendor metadata, and cache markers
are never exported by this adapter. A rejected or panicking content filter
omits that value, and telemetry callback failures never change agent behavior.
Tool-call arguments and successful tool results are emitted on tool spans only
when they are JSON objects, as required by the pinned schemas; non-object tool
returns remain available in the opt-in message history but are not mislabeled as
`gen_ai.tool.call.result`.

## Explicit providers

```go
instrumentation, err := agenticotel.New(
    agenticotel.WithTracerProvider(tracerProvider),
    agenticotel.WithMeterProvider(meterProvider),
    agenticotel.WithLoggerProvider(loggerProvider),
)
```

All built-in chat providers report their canonical provider identity and every
HTTP-configured client reports its endpoint. Every embedder in the root module
does the same when that information is exposed by its configuration, including
OpenAI, Cohere, Gemini, Voyage AI, Ollama, Bedrock, DeepInfra, generic
endpoints, Hugging Face, Pinecone, and SageMaker. SDK-owned endpoint resolution
is left unknown rather than fabricated. The separately released CGO ONNX
module can be identified with `WithEmbedderMetadata` until it adopts the root
v0.7 metadata capability. For a custom model or compatible endpoint:

```go
agentic.WithModelMetadata(agentic.ModelMetadata{
    Provider:      "my_provider",
    Operation:     "chat",
    ServerAddress: "models.example.com",
    ServerPort:    443,
})
```

For embeddings:

```go
instrumented, err := instrumentation.WrapEmbedder(
    embedder,
    agenticotel.WithEmbedderMetadata(agentic.ModelMetadata{
        Provider:      "my_provider",
        ServerAddress: "embeddings.example.com",
    }),
)
```

## Evaluations

Evaluation is normally performed after Agentic returns, so it is explicit:

```go
score := 0.96
err := instrumentation.RecordEvaluation(ctx, agenticotel.EvaluationResult{
    Name:        "correctness",
    ScoreValue:  &score,
    ScoreLabel:  "pass",
    Explanation: "Matched the reference answer.",
    ResponseID:  responseID,
})
```

Pass a context containing the evaluated operation span when possible. Otherwise
provide the response ID when it is available.

## Semantic-convention version

The implementation is pinned to the standalone OpenTelemetry GenAI semantic
conventions commit
`a685613a207a580163353b8e48a7ad88967e7b42` (2026-08-15), exposed as
`SemanticConventionsRevision`.

Those GenAI conventions are currently marked Development and the standalone
repository does not publish a stable schema URL. The adapter therefore
centralizes the exact attribute/instrument names and intentionally sets no
schema URL. Upgrading the pinned revision requires updating the shape tests and
privacy projection together.

`AgentIdentity.Version` remains available to dependency-neutral observers, but
the pinned conventions explicitly exclude `gen_ai.agent.version` from an
in-process `invoke_agent` span. This adapter preserves it as the namespaced
`agentic.agent.version` attribute instead. Durable run IDs, execution mode and
outcome, and resumed/suspended handler lifecycle use the same `agentic.*`
extension namespace.

The runtime package imports only Agentic and OTel API modules. OTel SDK modules
appear in this module's `go.mod` solely for its in-memory trace/log/metric tests;
the application still chooses which SDK and exporters it constructs.

The root module's provider SDKs already bring some OTel modules into its
transitive Go module graph. This boundary therefore promises source/API
decoupling and no Agentic OTel adapter at the root—not that an ordinary
`go get agentic` downloads no transitive OTel code.
