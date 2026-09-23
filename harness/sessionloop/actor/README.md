# Mailbox and Worker

Applications use `Mailbox.Submit(ctx, command)` and `Worker.Run(ctx)`.
Submission only retains work. Service bootstrap independently runs the worker.
No required Doorbell, broker, repair cron or database appears in these ports.

```text
request ──> Mailbox.Submit ──> retained input
                                   │
bootstrap ──> Worker.Run            │
                ├─ Adapter.Receive <┘  pending OR unfinished execution
                ├─ Ownership.Acquire   grant + execution marker
                ├─ SessionOpener.Open(lease)
                │       └─ same session + authoritative harness journal
                ├─ Acceptance lookup / Dispatch
                ├─ Acknowledge         remove mailbox after journal receipt
                ├─ continue receiving  steering during active execution
                └─ quiescence → Close → Ownership.Release
```

## Files and boundaries

| File | Responsibility |
|---|---|
| `mailbox.go` | Application ports, immutable commands and submission receipts |
| `adapter.go` | Runtime ports: Adapter, Ownership, SessionOpener |
| `worker.go` | Shared worker; independent discovery and active-session delivery |
| `localchannel/` | Process-local receive flavor; no process-crash durability claim |
| `memory/` | Deprecated import-path alias of localchannel |
| `conformance/` | Shared tests for concrete adapters |
| `spec/` | Lean model, proofs, adapter obligations and receive flavor |

The existing `harness/store.Repository` and `Journal` remain the journal
abstraction. Mailbox entries do not become another execution log.
`Session.Acceptance` reads existing acceptance facts without dispatching.
This recovers a lost Start receipt even when the session is already running.
Command identity must outlive mailbox cleanup.

NewWorker is one implementation of Worker, not mandatory scheduling technology.
Its Adapter blocks in Receive until pending input OR unfinished execution is
claimable. Active sessions check input continuously; paginated busy Starts
cannot hide later steering. Event buffering is independent of mailbox paging.

## Ownership reaches the journal

SessionOpener receives the lease. It restores the persistently bound session
and binds authority to ALL journal commits, including asynchronous execution
and recovery. A check followed by an unguarded write is insufficient.

For the local flavor, runtime assembly can use:

```go
ownedRepository := store.WithAuthority(journals, mailbox.Authority(lease))
// Assemble the harness with ownedRepository; open the persisted journal ID.
// Return its sessionloop view, which also implements AcceptanceReader.
```

The local authority holds the same mutex through validation and the bounded
repository primitive, never through a model/tool call. PostgreSQL adapters
can bind their journal directly, or use WithAuthority when the repository
honors the authority's transaction context. The wrapper cannot manufacture
atomicity across unrelated stores. No port requires a held database connection.

Creation and actor-to-journal binding belong to assembly. Failed restoration
never permits silently creating a replacement session. Openers clean up
partially constructed resources before returning an error.

## Failure and observation semantics

- Submission declares its guarantee. The memory flavor returns `accepted`:
  state survives worker restart, not process destruction.
- Acknowledgment follows lookup of the exact journal receipt. Ambiguous errors
  retain the input; retries preserve dispatch kind, target, ID and semantics.
- Failed opening, execution, acknowledgment or Close preserves the execution
  marker. Expiry and ordinary Receive make it claimable again.
- Only the owning worker, after harness quiescence and successful Close, calls
  Release. Concurrent pending submissions survive retirement.
- Suspensions and next-turn inputs are intentionally parked awaiting external
  input. The worker invents neither a resolution nor another Start.
- Unsupported, invalid or stale-run errors become explicit journaled rejection
  receipts through RejectionRecorder before acknowledgment. Busy/suspended work
  is deferred. Conflicts, storage and unknown errors retain the input for retry;
  they are never guessed to be durable resolution. Receipt.Rejection exposes the
  recorded reason through AcceptanceReader without adding conversation messages.
- EventSink is a projection, not the journal or an exactly-once output service.
  SnapshotSink reconciles after recovery and at retirement; complete downstream
  projections require replay and idempotent writes.
- Adapter I/O and callbacks must cooperate with cancellation. Go cannot force
  arbitrary callbacks or external tools to stop.

## Verification and migration

The [contract](spec/CONTRACT.md) has checked Lean proofs. Go has conformance,
race, failure-injection and real-harness/journal integration tests. This is
NOT a formal proof of Go or production storage. See the local flavor's
[mapping and assumptions](localchannel/README.md).

The [local end-to-end scenario](../../../e2e/sessionloop/README.md) assembles
the public memory adapter, independent workers, default Harness, real file tool
and disk-backed JSONL journal. Run `just sessionloop-e2e` at the repository root.

This intentionally replaces the old API:

| Old | New |
|---|---|
| Supervisor.Submit | Mailbox.Submit on the adapter |
| New(Config) / Supervisor.Run | NewWorker(Config) / Worker.Run |
| CommandStore, LeaseStore, required Doorbell | Adapter with Ownership and reliable Receive |
| Open(ctx, actorID) | Open(ctx, lease), binding journal authority |
| MarkDispatched/MarkSettled/MarkFailed queue history | Journal receipt, then mailbox Acknowledge |
| Private memory test infrastructure | Public actor/localchannel flavor (memory remains an alias) |

Consumers must migrate their assembly/adapters separately. This package does
not prescribe or migrate application storage schemas.
