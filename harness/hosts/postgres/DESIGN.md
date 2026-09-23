# PostgreSQL flavor: transaction algorithm and e2e implementation

The [Lean transaction proof](PROOF.md) now covers the row algorithm, interleaved
private writes/commit, crashes, durable replies and conditional recovery/progress.
The executable reference adapter is [e2e/sessionloop/postgres](../../../e2e/sessionloop/postgres/README.md).
Its tests exercise real PostgreSQL and transaction-pooled PgBouncer. The Lean
proof is not a proof of the Go code, SQL text, driver or database engine.

Use the same Mailbox, Worker, Ownership, SessionOpener and Journal contracts.
The first durable implementation should use ordinary worker polling and short
database transactions. No broker, required doorbell, repair cron, session-level
advisory lock or held connection is necessary.

## Storage owned by this adapter

Three tables suffice. Names/schema below are illustrative and configurable;
they are not a library-wide application schema requirement.

```sql
CREATE TABLE sessions (
    id text PRIMARY KEY,
    owner text,
    fence bigint NOT NULL DEFAULT 0 CHECK (fence >= 0),
    expires_at timestamptz,
    needs_execution boolean NOT NULL DEFAULT false,
    next_command_seq bigint NOT NULL DEFAULT 0,
    journal_seq bigint NOT NULL DEFAULT 0,
    journal_entry_id text,
    journal_handle text,         -- one open handle under the current grant
    last_claimed_at timestamptz,
    CHECK ((owner IS NULL) = (expires_at IS NULL))
);

CREATE TABLE commands (
    session_id text NOT NULL REFERENCES sessions(id),
    id text NOT NULL,
    sequence bigint NOT NULL,
    digest bytea NOT NULL,
    payload bytea,                -- immutable envelope while pending
    accepted_at_seq bigint,       -- points into the existing journal
    PRIMARY KEY (session_id, id),
    UNIQUE (session_id, sequence),
    CHECK ((payload IS NULL) = (accepted_at_seq IS NOT NULL))
);
CREATE INDEX pending_commands ON commands(session_id, sequence)
    WHERE accepted_at_seq IS NULL;

CREATE TABLE journal (
    session_id text NOT NULL REFERENCES sessions(id),
    sequence bigint NOT NULL,
    entry_id text NOT NULL,
    parent_id text NOT NULL,
    schema_version smallint NOT NULL,
    kind text NOT NULL,
    payload bytea NOT NULL,       -- codec-owned bytes, never normalized JSON
    durability smallint NOT NULL,
    PRIMARY KEY (session_id, sequence),
    UNIQUE (session_id, entry_id)
);
```

The logical mailbox is the pending subset of `commands`. Acknowledgment clears
its payload and records the journal reference. Only a small identity tombstone
survives, so deduplication still works after delivery cleanup. There is no second
persisted conversation transcript. Exact envelope comparison/digest rules must
match command normalization; digest collision resistance is an explicit trust
assumption if the original envelope is discarded.

## Transaction boundaries

Every mutation first locks that session's row with `SELECT ... FOR UPDATE`.
Use one consistent lock order: session, command, journal. The row lock lasts
only through the bounded transaction, never through a model/tool call.

| Operation | Atomic work before COMMIT |
|---|---|
| Submit | Validate identity/digest, allocate sequence, retain immutable envelope |
| Acquire | Check absent/expired owner using database time; increment fence; set owner, expiry and needs_execution |
| Renew | Verify owner/fence and unexpired lease; extend expiry |
| Journal append | Verify owner/fence/expiry AND expected journal leaf; append the entire batch and advance leaf |
| Acknowledge | Verify authority and exact acceptance receipt; remove pending payload, retain identity/reference |
| Release | After worker-established quiescence and Close, verify authority; clear owner and needs_execution, NEVER pending commands |

Check expiry after obtaining the row lock, using current database time.
At READ COMMITTED, read dependent command/journal rows in subsequent statements
after acquiring that lock, not a snapshot captured before waiting for the lock.
Never permit expired-owner renewal. Fence overflow is a hard error.
Journal `Open` returns a lightweight authority-bound handle, not a pinned SQL
connection. Journal `Close` closes that handle; it cannot revoke a successor.
`store.WithAuthority` is usable only if the journal joins its SAME SQL
transaction; a wrapper around an unrelated connection does not fence writes.

Each committed synchronous journal append must meet the configured durable
acknowledgment requirement. Document database/replication failure assumptions.

## Discovery and the retirement race

`Receive` queries for sessions with pending commands OR needs_execution whose
ownership is absent/expired. It fairly rechecks on a bounded interval, including
after transient errors. Use last-claim ordering or an equivalent starvation-free
policy. `Acquire` resolves competing discoveries. A combined internal claim may
use `FOR UPDATE SKIP LOCKED`; this is not required by the public interface.

```text
Submit wins the row lock first:
    pending input commits -> Release leaves it untouched -> Receive finds it

Release wins first:
    ownership clears -> Submit commits pending input -> Receive finds it

Crash after acceptance and mailbox cleanup:
    needs_execution remains -> expiry -> Receive -> replay same journal
```

Polling here IS the worker's receive implementation, not a separate repair job.
Progress assumes eventual healthy database access and fair worker scheduling.

## Integration acceptance

The reference runner executes the shared actor suite plus native Harness e2e against PostgreSQL,
then repeats through PgBouncer configured with `pool_mode=transaction`.
Use more worker clients than backend connections, with models blocked behind
test gates, to prove idle model calls do not pin those backends.

Inject process death and ambiguous replies around submission, acquisition,
restoration, acceptance, acknowledgment, execution settlement and release.
After every committed boundary, recover without a new submission. Test both
retirement/submission orders, competing owners, expiry, stale journal writes,
stable identity after payload cleanup, steering behind busy starts and opaque
journal-byte preservation. Do not equate execution recovery with exactly-once
external tool side effects.

The transaction model now proves its mapping to `SessionContract.Adapter`.
Implement these transactions against that model and discharge the representation
and native-journal obligations in [PROOF.md](PROOF.md). Atomic commit/rollback,
durability and row exclusion remain trusted storage semantics; the driver and
database engine are not formally verified here.

## Primary references

- [PgBouncer feature matrix](https://www.pgbouncer.org/features.html): transaction
  pooling releases server connections at transaction end; LISTEN and session
  advisory locks are not supported in this mode.
- [PostgreSQL row locks](https://www.postgresql.org/docs/current/explicit-locking.html):
  row locks serialize competing mutations until transaction end.
- [PostgreSQL SELECT](https://www.postgresql.org/docs/current/sql-select.html):
  SKIP LOCKED is appropriate for queue-like consumers, not a generally consistent
  view of all rows.
