# Agentic Harness (experimental Phase 3)

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
- exact, write-ahead deferred resume through the released root `Driver`; and
- an explicit local `Default` assembly.

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

Subagents, codemode, long-term memory, skills, and evals are not part of Phase
3. They remain planned for later phases in
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
