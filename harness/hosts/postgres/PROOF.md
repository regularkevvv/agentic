# PostgreSQL transaction proof

Status: **the Lean transaction algorithm is proved; an executable reference
adapter and real-database tests now live only in [e2e](../../../e2e/sessionloop/postgres/README.md).**
The Go/SQL implementation is tested, not formally proved.
The proof is in
[`SessionContract/Flavors/Postgres.lean`](../../sessionloop/actor/spec/SessionContract/Flavors/Postgres.lean).
The existing verification script builds it, checks its transitive axioms, runs
negative audit fixtures and replays the proof terms through `leanchecker`.

## What is modeled

```text
any number of clients, one pre-existing session/journal binding

enqueue -> lock/read session row -> validate -> write commands privately
        -> write journal privately -> finish private writes -> COMMIT -> reply
            |               |                |                  |
         rollback / process failure / server failure / lost reply
```

The mailbox is the pending part of `commands`. `journal` contains authoritative
acceptance/resolution facts. `sessions` contains ownership, generation and the
execution-needed flag. A channel, broker and repair cron are not required.

Each client can fail independently. Other clients can enqueue and wait while a
transaction holds the row lock. Neither partial writes nor a success response
become visible before commit. Server failure discards uncommitted transaction
state but retains committed rows. A lost reply never reverses a commit.

## Source and checked guarantees

| File | Responsibility |
|---|---|
| `Postgres/Rows.lean` | Independent table-column algorithm; state AND reply correspondence to the contract |
| `Postgres/Transactions.lean` | Row exclusion, stable locked snapshot, private write phases, atomic commit, rollback and crashes |
| `Postgres/Recovery.lean` | Every microstep history linearizes; committed input/history survive; concrete post-crash recovery transactions; conditional progress |
| `Postgres/Receipts.lean` | Every successful visible submission/acceptance response has durable backing; later cleanup/crashes cannot invalidate identity |
| `Postgres/Expiry.lean` | Database-deadline guards, expired/stale renewal rejection, expiry-plus-operation correspondence |
| `Postgres/Examples.lean` | Both retirement races, post-cleanup recovery, stale owners, and counterexamples for missing locks/early replies |

The principal theorem `crash_at_every_sql_phase_recoverable` quantifies over
arbitrary permitted transaction histories, not a finite set of test scenarios.
It constructs a continuation using the same enqueue/lock/private-write/commit
steps, starting with expiry of the old authority. It does not merely assert
that an abstract worker could somehow recover.

`history_linearizes` is the bridge: every committed row update matches the
contract, and all other transport/private-write steps leave its state unchanged.
`LockedSnapshot` is DERIVED from the operational row-lock rules. It is not an
assumption added to the commit operation to make the proof succeed.

`fair_committed_worker_progresses` is a separate conditional claim: a fair
schedule of successfully completed worker primitives eventually resolves work
and cleans up its mailbox entry. It does not prove fairness of a future SQL
polling query, progress during perpetual outages, or service of unbounded load.

## Representation and trust boundary

This is a proof of the algorithm under stated storage semantics, **not** of SQL
text, the illustrative DDL, PostgreSQL internals, a driver, PgBouncer or Go.

- Atomic durable commit, rollback and session-row exclusion are the modeled
  PostgreSQL guarantees. They must match the deployment's WAL, replication and
  failover configuration. Permanent loss of committed storage is not covered.
- `Rows` uses table columns as mathematical functions keyed by command identity.
  It is not an executable database. Pending `true` means a retained payload;
  pending `false` means its disposable payload has been removed. Identity is an
  exact semantic value. A digest-only SQL tombstone introduces a collision-
  resistance assumption; the Lean proof does not prove a hash injective.
- Journal acceptance and resolution are projections of native journal facts,
  not another transcript or a mandatory extra status table. Their extraction,
  byte codec, expected-leaf check, replay and the native harness itself need
  implementation-level validation. A settlement can be completion, explicit
  failure, or safe suspension awaiting external input; it is not invented by SQL.
- The Lean `owner` is a generation-bearing authority token. Its physical
  representation must bind SQL owner identity AND fence. Tokens cannot be forged
  or reused; machine-sized fences must fail before overflow.
- `handles` and `emptyObservation` are proof witnesses, not SQL columns. Local
  open/close and native quiescence must respect the worker contract. Crashes may
  retain old handle witnesses deliberately: even a paused old process must be
  fenced. The implementation must establish their linearization with the
  guarded journal operations; a Go interface does not enforce that law.
- Deadline checks represent database time read after locking. `Expiry.lean`
  proves the guard and compound-action correspondence; the microstep model
  represents serialized expiry as an explicit action. It is not a model of
  timestamp encodings or real clock behavior. Recovery requires eventual
  reclaimability. Never renew already-expired authority.
- Stable session/journal binding exists before this model begins. Bootstrap,
  migrations, tenant authorization, external tool effects and journal compaction
  are not proved. Two processes may overlap in CPU time; only current authority
  may commit. Exactly-once external side effects need their own contract.

## Obligations for implementations

1. Use a fresh transaction per metadata/journal mutation; acquire the session
   row lock BEFORE reading dependent command/journal data. At READ COMMITTED,
   use subsequent statements after acquiring the lock so dependent reads cannot
   retain a snapshot taken before waiting. All writers obey this same rule.
2. Evaluate expiry, fence, command identity and any journal leaf precondition
   inside that critical section. Never validate on one connection and write on
   another. An expiry check and following action may be one atomic batch, as
   proved by `deadline_checked_operation_refines`.
3. Preserve transaction atomicity for all related writes. Return success only
   after durable commit. Retry ambiguous results using unchanged identities.
4. Discover pending payloads OR unfinished execution. Cleanup must retain
   identity/acceptance; release must never delete concurrent pending input.
5. Establish the native journal/witness mapping above. Keep all database
   connections and transactions outside model/tool execution.
6. Exercise the actual mapping with real PostgreSQL and transaction-pooled
   PgBouncer, concurrent clients, killed processes and lost replies. Those tests
   are evidence for the implementation bridge, not a proof of the Go program.

The proof should guide those transaction definitions; it must not become a
separately maintained idealized algorithm that production quietly bypasses.

### Reference implementation mapping

`e2e/sessionloop/postgres` implements the row algorithm through `Store.transaction` and
the session-row `lock`/`owned` helpers. `Submit`, `Acquire`, `Renew`, `Acknowledge`,
`Release` and journal `Append` use that same critical section. `Receive` scans
pending OR unfinished state; `last_claimed_at` orders successful claims.

The SQL journal-handle token is an additional exclusivity check, not a new
application concept. It is cleared on takeover; old handles remain fenced by
owner/fence checks. Native acceptance and settlement are still journal facts,
with `actor.Worker` supplying the acceptance and quiescence witnesses. Bootstrap
is separately constrained to initial creation and remains outside this proof.

The e2e matrix checks the shared actor laws, real native replay/steering, opaque
journal bytes, stale writers/handles, atomic batch rollback, expiry after a lock
wait, killed-process recovery and transaction-pool connection reuse. This is
implementation evidence, not a machine-checked Go-to-Lean correspondence.

## Verify

```sh
cd harness/sessionloop/actor/spec
bash verify.sh
```

Primary storage references: [row-lock semantics](https://www.postgresql.org/docs/current/explicit-locking.html),
[READ COMMITTED snapshots](https://www.postgresql.org/docs/current/transaction-iso.html),
[commit durability](https://www.postgresql.org/docs/current/wal-async-commit.html),
[PgBouncer transaction pooling](https://www.pgbouncer.org/features.html).
