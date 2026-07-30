# Agentic Harness (experimental Phase 4)

This nested module contains the experimental `v0.1` harness surface for
Agentic `v0.4.0`. It currently provides:

- write-ahead `Harness`/`Session` execution with steering, follow-up,
  next-turn queues, interruption, snapshots, budgets, and crash recovery;
- generic journal/codec/event/environment/result-processing ports;
- memory and JSONL journal adapters with durable cursors and exclusive leases;
- a bounded nonblocking in-process event adapter;
- deterministic terminal transcript repair;
- Local and Memory environments plus `ToolRuntime` context plumbing;
- session-scoped artifact storage and oversized tool-result spill;
- an immutable capability DAG and public registry extension points;
- durable/ephemeral context policy with structured and optional LLM
  compactors;
- most-specific, default-deny permission policy with `ReadOnly` and
  `WorkspaceWrite` presets;
- exact, write-ahead deferred resume through the released root `Driver`;
- an explicit local `Default` assembly;
- capture-restricted, in-process child agents with separate durable sessions
  and addressed inbox routing;
- shared or narrowed history, environment, tools, permissions, dependencies,
  and cumulative budget policies; and
- tagged descendant events, explicit recursion limits, parent cancellation,
  and bounded UTF-8 child summaries.

The runtime and session core import only ports. `harness.NewRuntime` remains
the policy-neutral constructor; `harness.New(...).Build()` layers an immutable
capability plan over the same explicit substrate:

```go
import (
    "github.com/regularkevvv/agentic/harness"
    memoryartifact "github.com/regularkevvv/agentic/harness/artifact/memory"
    "github.com/regularkevvv/agentic/harness/artifact/spill"
    jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
    memoryenv "github.com/regularkevvv/agentic/harness/env/memory"
    "github.com/regularkevvv/agentic/harness/event/inproc"
    "github.com/regularkevvv/agentic/harness/runtime/system"
    memoryjournal "github.com/regularkevvv/agentic/harness/store/memory"
)

sessions := memoryjournal.New()
events := inproc.NewFactory()
environments, _ := memoryenv.NewFactory(memoryenv.Config{Cwd: "/workspace"})
artifacts := memoryartifact.New()
results, _ := spill.NewFactory(artifacts, spill.Config{})

runtimeConfig := harness.RuntimeConfig{
    Sessions:         sessions,
    Codec:            jsoncodec.New(),
    Events:           events,
    Environments:     environments,
    ResultProcessors: results,
    Clock:            system.NewClock(),
    IDs:              system.NewIDs(),
}

runtime, err := harness.NewRuntime(runner, runtimeConfig)
// Or:
runtime, err = harness.New(
    runner,
    harness.WithRuntime(runtimeConfig),
    harness.WithCapabilities(applicationCapabilities...),
).Build()
```

The same core accepts other conforming adapters without modification. Reusable
conformance suites live in `store/storetest`, `event/eventtest`, `env/envtest`,
and `artifact/artifacttest`. The `env/local` adapter constrains its filesystem
methods to a root, but its shell runs as the host user and is **not an OS
sandbox**.

`harness.Default` is a convenience composition, not a dependency direction:
it explicitly requires absolute, non-overlapping workspace/session paths and
assembles the same public capabilities over Local, JSONL, file-artifact, and
in-process adapters. Local permission policy is governance around ordinary host
execution; it does not create an OS sandbox.

Subagents are ordinary optional capabilities and are not installed by
`harness.Default`:

```go
import "github.com/regularkevvv/agentic/harness/subagent"

worker := agentic.NewAgent("You are a focused researcher.", workerModel)
researcher, err := subagent.New(worker, subagent.Config{
    Name:        "researcher",
    Description: "Research one bounded task.",
    Runtime:     runtimeConfig,
    Capture: subagent.Capture{
        Environment: subagent.ModeNarrow,
        Tools:       subagent.ModeNarrow,
    },
    EnvironmentRoot: "research",
    ToolFilter: func(name string) bool {
        return name == "read_file"
    },
})

runtime, err = harness.New(
    parentRunner,
    harness.WithRuntime(runtimeConfig),
    harness.WithCapabilities(researcher),
).Build()
```

The zero-value capture policy isolates history and tools, shares dependencies,
environment, and budget, and imports parent permissions as a non-broadening
gate. Every field can instead select `ModeIsolate`, `ModeShare`, or
`ModeNarrow`; narrowing requires the corresponding projector, environment
root, tool filter, or usage limit. `NewWithDeps` gives its binder the resolved
dependency mode so application code can derive the exact bound child runner
without reflection or a harness-owned dependency container.

Capture governs resources contributed by the parent harness. The child
runner's own system prompt, dependencies, and intrinsic tools are
child-native, so callers should construct that runner with only the resources
the child is meant to own. `Config.Capabilities` adds the child's explicit
harness-native tools and policies.

Child sessions use distinct durable journals, inboxes, environment leases, and
artifact scopes. Their parent ID, agent name, and depth are persisted with
session creation and remain authoritative after reopen. The process-local
address router is available while a delegation call is executing. Completed
children remain recoverable by their opaque session IDs. In-process synchronous
delegation is the Phase 4 boundary: out-of-process workers, topology presets,
automatic suspended-child orchestration, codemode, long-term memory, skills,
and evals remain deferred to the later work described in
[`../docs/spike-harness-framework.md`](../docs/spike-harness-framework.md).

The module requires the released root module:

```text
github.com/regularkevvv/agentic v0.4.0
```

Local development uses the committed repository `go.work`; the module does not
contain a `replace` directive. To verify both release and workspace views:

```sh
GOWORK=off go test -race -count=1 ./...
(cd harness && GOWORK=off go test -race -count=1 ./...)
go test -race -count=1 ./... ./harness/...
(cd harness && GOWORK=off make coverage-check)
```
