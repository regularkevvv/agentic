# Native recovery and failure cleanup

This extension closes two implementation gaps not covered by the original
delivery model. Durable acceptance alone is insufficient: opening the journal
must reconstruct a valid execution frontier, and losing a worker must not be
interpreted as the user cancelling accepted work.

## Contract

```text
durable acceptance -> execution -> committed assistant -> validation/turn hook
                                               -> committed completion -> closure
       |                    |                  |                |
       +---------------- crash or worker failure ---------------+
                            |
                    abandon volatile handle
                            |
                    reopen the SAME logical run
```

- `Close` is normal handle shutdown and may interrupt a running session.
- `RecoveryCloser.Abandon` is required by `actor.Session`. It must preserve
  accepted, unfinished execution without inventing cancellation or settlement.
  The Worker uses it on errors and shutdown, and does not retire unfinished work.
- A cancelled dispatch context cannot revoke an acceptance that already committed.
- Process recovery retains the logical run ID and immutable command receipts.
  `run.recovered` records another attempt, not another logical run.
- A committed final assistant message is a candidate, not new model input.
  `DriveRecover` re-evaluates that candidate without another model request or
  assistant commit. Validators and uncommitted hooks may run again.
- When completion already committed, recovery appends only the missing native
  closure. It does not re-execute validation or turn handling.
- Drained but unapplied steering is reconstructed before completion. Mailbox
  acknowledgment does not transfer responsibility away from the durable journal.

## Checked component models

| Lean model | Main claims | Implementation mapping |
|---|---|---|
| `RecoveryFrontier` | Reachable state safety, stable run identity, no lost accepted steering, legal finite recovery continuation from every reachable prefix | `session/recovery.go`, `DriveRecover`, native queue replay |
| `Cleanup` | Failure cleanup and late caller cancellation preserve durable work; late finalizers cannot cancel it; reopened pending input can be applied | `actor/worker.go`, `LoopView.Abandon`, native session mutex/fault checks |

Both retain checked counterexamples of the old behavior. The old input-only
driver rejected a reachable committed-assistant frontier. The old failure
cleanup could cancel steering without a user interrupt command.

Run `bash verify.sh` here to build the models, audit transitive axioms (including
negative audit fixtures), and replay proof terms through `leanchecker`.

These are component proofs, not a machine-checked refinement of the Go compiler,
goroutine scheduler, PostgreSQL server, or entire harness. The delivery model
assumes atomic fenced storage. The cleanup model assumes every emitting path
checks the fault under the same mutex before committing. The frontier model
abstracts response generation and validation. Their implementation mapping is
checked separately with executable regressions, not asserted as a proved theorem.

## Executable evidence

- `harness/session/recovery_frontier_test.go` cuts real driver journals after
  assistant commit, validation, turn end, completion, driver end, and native
  closure. Reopening must complete once without another model request or run ID.
- `e2e/sessionloop/postgres/frontier_test.go` commits an event, revokes ownership,
  and loses the append reply at seven actual journal boundaries. An independent
  Worker must recover acknowledged steering before inspection, preserve both
  receipts and provider-prefix bytes, and settle the original run exactly once.
- The same PostgreSQL test injects renewal I/O failure while the grant is still
  valid. Failure cleanup must leave acknowledged steering for takeover.
- Existing fencing, acknowledgment, retirement, process-kill, replay, and
  publication tests remain required; these tests do not replace them.

Safety does not imply progress during permanent I/O failure, infinite crashes,
or an unresponsive model/tool. Recovery needs an eventually healthy owner and
storage interval. Uncommitted external effects still require idempotency or
explicit reconciliation; a journal cannot make an arbitrary external API exactly-once.

## Compatibility

Adding `Abandon` to `actor.Session` deliberately makes actor adapters lacking
recovery cleanup fail to compile. Generic non-actor SessionLoop implementations
need not implement it. Wrappers must forward the capability, including cleanup
when opening succeeds but later attachment fails.

`run.recovered` is a new native journal entry. Upgrade all native readers and
workers before they share journals written by this version; mixed old/new
workers are not qualified. Existing journals remain readable. Journal history
is authoritative; new code does not rewrite historical interrupted outcomes.
