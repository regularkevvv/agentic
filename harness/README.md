# Agentic Harness (experimental Phase 2)

This nested module contains the durable, single-process session runtime for
Agentic `v0.4.0`. It currently provides:

- write-ahead `Harness`/`Session` execution with steering, follow-up,
  next-turn queues, interruption, snapshots, budgets, and crash recovery;
- generic journal/codec/event/environment/result-processing ports;
- memory and JSONL journal adapters with durable cursors and exclusive leases;
- a bounded nonblocking in-process event adapter;
- deterministic terminal transcript repair;
- Local and Memory environments plus `ToolRuntime` context plumbing; and
- session-scoped artifact storage and oversized tool-result spill.

Construct the low-level runtime with `harness.NewRuntime`. The runtime and
session core import only ports; concrete implementations are selected at the
application composition root:

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

runtime, err := harness.NewRuntime(runner, harness.RuntimeConfig{
    Sessions:         sessions,
    Codec:            jsoncodec.New(),
    Events:           events,
    Environments:     environments,
    ResultProcessors: results,
    Clock:            system.NewClock(),
    IDs:              system.NewIDs(),
})
```

The same core accepts other conforming adapters without modification. Reusable
conformance suites live in `store/storetest`, `event/eventtest`, `env/envtest`,
and `artifact/artifacttest`. The `env/local` adapter constrains its filesystem
methods to a root, but its shell runs as the host user and is **not an OS
sandbox**.

Capability graphs, context policy, permissions, policy-driven deferred resume,
`Default`, subagents, codemode, memory, skills, and evals are not part of Phase
2. They remain planned for later phases in
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
```
