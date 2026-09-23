# PostgreSQL session-loop flavor

An executable reference implementation **inside e2e only**. It uses the existing
`actor.Adapter`, shared `actor.Worker`, `actor.SessionOpener`, and `store.Repository`
ports. No production module imports pgx; no Harness/SessionLoop API changes are
needed to select this flavor.

```text
caller                 independent shared Worker
  | Submit                  | Receive -> Acquire
  v                         v
commands (pending) <----- Pending
                            | Open native Harness with Repository(lease)
                            v
                       journal acceptance -> Acknowledge
                            |                  |
                       execute/replay     payload removed
                            |             digest retained
                       settle -> Close -> Release

sessions = owner + expiry + fence + needs_execution + journal leaf/handle
Receive  = poll pending input OR unfinished execution; no repair cron/broker
```

## Read the implementation

| File | Responsibility |
|---|---|
| [schema.sql](schema.sql) | Three tables in a caller-selected schema |
| [store.go](store.go) | Short READ COMMITTED transactions; row lock and database-time grant validation |
| [mailbox.go](mailbox.go) | Submit, ordered Pending, acceptance-backed payload cleanup |
| [ownership.go](ownership.go) | Polling Receive, fenced Acquire/Renew/Release |
| [journal.go](journal.go) | Native opaque-byte journal; atomic expected-leaf append under the same fence |
| [codec.go](codec.go) | Versioned pending-envelope codec; preserves absent data and exact structured bytes |
| [fixture_test.go](fixture_test.go) | Actual native Harness + independent shared worker assembly |
| [worker_test.go](worker_test.go) | Conversation, real file tool, steering, restart, request-prefix continuity and pooling |
| [crash_test.go](crash_test.go) | Kill real worker processes at storage/execution boundaries |

The important assembly is small:

```go
db, err := postgres.Connect(ctx, dsn, schema)
// Provision a new schema once with db.Migrate(ctx), not per worker startup.

// Creation only: assemble a native Harness with db.Bootstrap(), NewSession,
// Close it, then publish its ID. The actor ID is exactly that journal ID.

opener := actor.SessionOpenerFunc(func(ctx context.Context, lease actor.Lease) (actor.Session, error) {
    // Assemble the native Harness with Runtime.Sessions = db.Repository(lease).
    // OpenSession restores the same ID; missing/corrupt history is an error.
    return openNativeSession(ctx, lease.ActorID, db.Repository(lease))
})
worker, err := actor.NewWorker(actor.Config{
    Owner: workerID, Adapter: db, SessionOpener: opener,
})

// Independently started by service bootstrap:
go worker.Run(workerContext)

// Request handler: retain input only. It does not run the worker.
submission, err := db.Submit(requestContext, command)
```

`openNativeSession` above denotes application assembly, not another library API;
the compiling implementation is `fixture.worker` in the fixture file.

## Run

From the repository root, with Docker running:

```sh
bash e2e/sessionloop/postgres/run.sh
# Optional repeated run or one scenario:
AGENTIC_POSTGRES_COUNT=3 bash e2e/sessionloop/postgres/run.sh
bash e2e/sessionloop/postgres/run.sh -run TestProcessCrashRecovery
```

The runner uses pinned PostgreSQL/PgBouncer images, random loopback ports and a
unique Compose project. It runs the suite with the race detector directly, then
through `pool_mode=transaction`, **two backend connections**, and no prepared
statement tracking. Six simultaneously blocked model calls must still leave SQL
usable. It removes its containers, network and disposable database volumes on exit.

From `e2e`, ordinary `go test ./sessionloop/postgres` runs codec tests and skips database tests
unless `AGENTIC_POSTGRES_DSN` is set. The dedicated CI job always uses the runner.
For an existing disposable database, set that variable; tests create and remove
only their own random schemas. Never point this example at production data.

## Guarantees and limits

- Submission returns only after commit. Same-ID retry uses normalized command
  semantics; a SHA-256 tombstone survives removal of the pending envelope.
  Hash collision resistance is an explicit assumption, not a Lean theorem.
- The shared worker verifies exact native acceptance before acknowledgment.
  SQL also validates the current lease and durable receipt's journal reference.
  It never parses private Harness journal payloads or duplicates the transcript.
- `Repository(lease)` is a worker-bound restoration repository. Its handle is
  guarded on every Load/Append and never holds a backend between calls. A SQL
  handle token prevents two simultaneous opens under the same grant; stale Close
  cannot revoke a successor. Child-session creation/forking is not implemented.
- `Bootstrap()` is deliberately creation-only: its handle permits Load/Close,
  not execution. Initial entries commit atomically with a 30-second creation
  grant. A creator crash leaves the valid initial journal discoverable at expiry.
  Publish the ID only after successful creation and Close; ambiguous creation
  results require retaining/reconciling the same ID, not creating a replacement.
- The worker's renewal and quiescence rules are unchanged. Failed workers leave
  `needs_execution` set. Even an empty mailbox is rediscovered after lease expiry.
  Successful Release never deletes concurrently submitted commands.
- The crash matrix kills processes after Submit, Acquire, Open, native acceptance,
  payload cleanup, settlement/Close and Release, and before a journal transaction
  commits. Fresh workers recover without new input. An interrupted execution can
  settle as failed; recovery does not promise a successful model response or
  exactly-once external tool effects.
- Sync commits assume PostgreSQL durable storage (`fsync`/WAL), appropriate
  failover policy, and eventually available workers/database. No HA/failover,
  disk-loss, live-provider or external-side-effect qualification is claimed.
  This is a readable reference, not a production-operated database service.

The [Lean proof](../../../harness/hosts/postgres/PROOF.md) proves the transaction
algorithm. These tests check its implementation bridge; **the Go/SQL program,
driver and database engine are not themselves formally proved**.

Storage references: [PostgreSQL row locks](https://www.postgresql.org/docs/current/explicit-locking.html),
[READ COMMITTED](https://www.postgresql.org/docs/current/transaction-iso.html),
[PgBouncer pooling](https://www.pgbouncer.org/features.html).
