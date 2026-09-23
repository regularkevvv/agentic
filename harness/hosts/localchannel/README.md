# Local channel

This is the concrete session-loop implementation used by the interactive CLI.

```text
TUI -> sessionloop.Session.Dispatch -> local mailbox
                                           |
bootstrap -> Host.Run -> shared actor Worker
                          acquire -> open fenced native Harness
                          journal acceptance -> mailbox acknowledgment
                          run / steer / resolve / interrupt
                          quiescence -> close -> release

TUI <- snapshots / replay / previews <- owned native Harness
```

`New(Config)` only assembles. Bootstrap independently calls `Run(serviceCtx)`
and joins it at shutdown. A request never starts or runs a worker.

`Config.Build` receives the journal repository it must install in the native
Harness. The flavor supplies `store.WithAuthority` for worker-owned sessions.
Model, tools, permissions, environments and storage remain application choices.

`OpenSession` asks the worker to restore the same journal. A missing/corrupt
journal is an error, never permission to create a replacement. New sessions
are explicitly created and closed before their identity becomes available.

The local-channel adapter holds commands in memory; the channel only signals
state changes. It is not process-crash durable. `Dispatch` waits beyond mailbox
submission for the worker's verified journal receipt before returning the
journal's acceptance guarantee. Canceling the wait does not withdraw work.

The client view survives ordinary worker retirement. Snapshot and finite replay
come from the worker-owned journal, without opening a competing writer. The
in-memory event cache is disposable UI projection, not an additional durable
execution log. Slow streams fail with `ErrLagged`; previews may be dropped.

Client `Close` detaches observation. Accepted work remains owned by the worker.
To interrupt execution, dispatch an interrupt; to stop the service, cancel Run
and join its cleanup. Keep the Host when replacing a local worker.

Verification: shared actor contract tests, full native session-loop conformance,
disk-journal mailbox e2e, and the CLI/TUI e2e with streaming, approval, tools,
interruption, child progress and resume. Live-provider tests remain opt-in.
