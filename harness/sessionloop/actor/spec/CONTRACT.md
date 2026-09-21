# Mailbox and Worker: derived contract

This is the contract implemented by the actor ports and shared Go Worker.
`Model.lean` gives its executable semantics. Lean proves the abstract protocol;
Go conformance and integration tests are separate evidence, not a Go code proof.

## Names and ownership

| Concept | Responsibility |
|---|---|
| Mailbox | Accept immutable commands and retain undelivered work |
| Worker | Independently discover, own and drive sessions; recover unfinished work |
| Ownership | Grant exclusive mutation authority and reject stale grants |
| Lease | The scoped grant, not another service |
| SessionOpener | Obtain the correctly bound executable session under that grant |
| Journal | Persist authoritative harness acceptance and execution/recovery facts |

There is no required Doorbell. Polling or notifications belong inside the
implementation. A transport signal never substitutes for retained work.

## Application-facing surface

The compiled application-facing shape in `actor/mailbox.go` stays small:

```go
type Mailbox interface {
    Submit(context.Context, Command) (Submission, error)
}

type Worker interface {
    Run(context.Context) error
}
```

The request path uses Mailbox. Independent service bootstrap runs Worker.
Submission never invokes Worker, opens a harness, or manages a lease.
The Worker implementation owns discovery, waiting, renewal and recovery.
SessionOpener and Journal are runtime assembly/storage ports; business use
cases do not import lease generations, SQL, clock/heartbeat or fencing types.

These two Go interfaces cannot enforce semantics. Supported durable adapters
must meet the laws below; verified adapters additionally provide a refinement
proof. `Adapter.lean` makes the distinction concrete: operations **and their
proof obligations** form `Adapter`.

## Submission laws

1. Under the durable profile, a success reply means command identity and immutable semantics are durable.
   The command must be discoverable without any future submission or signal.
2. Retrying the same identity and semantics returns the original acceptance.
   Reusing the identity for different semantics is a conflict.
3. These identities survive mailbox cleanup. Deduplication cannot depend only
   on a disposable queue row.
4. A timeout is not evidence of non-commit. Retrying uses the same identity.
5. For a message reference, the referenced content/version must remain stable
   until durable acceptance; never resolve a retry to newly edited content.

The Go `Submission.Guarantee` states the failure profile. The local memory
flavor returns `accepted`, not `durable`: it cannot survive process destruction.
Do not attribute the durable model's crash theorem to this volatile store.

The mathematical `submitted` map is logical history, **not a requirement for
another table or another copy of the text**. An adapter can interpret existing
immutable conversation records, pending envelopes and journal receipts as
that history, provided the mapping survives cleanup and preserves identity.

## Worker and handoff laws

The following are logical primitives, not a proposal to expose twelve public
Go methods or twelve database tables:

| Primitive | Required meaning |
|---|---|
| Discover | Pending mailbox input OR persisted unfinished execution makes a session eligible, subject to explicit suspension policy |
| Acquire | Atomically grant a fresh owner generation and mark execution outstanding before opening or accepting input |
| Open | Reconstruct the same harness session under that ownership; do not silently replace it on restoration failure |
| Accept | Journal the exact command's acceptance before returning a durable receipt; repeat acceptance is idempotent |
| Acknowledge | Remove the exact delivered mailbox item only with durable acceptance evidence |
| Settle | Record the harness's durable resolution, not a Worker guess that processing probably ended |
| Close | End the live handle and outstanding work under it |
| Release | Clear execution-needed state only after durable quiescence and closing; never discard pending input |
| Expire/take over | Revoke old authority while preserving pending and unfinished work; advance the generation on the next grant |

Mailbox deletion and journal acceptance need **not** be one cross-store
transaction. The safe order is:

```text
persist unfinished-execution marker
        |
        v
journal durable acceptance ---- crash here: retry the same command
        |
        v
delete mailbox entry ---------- crash here: discover session, replay journal
        |
        v
journal durable resolution
        |
        v
close handle, retire ownership
```

Retries preserve exact dispatch identity/meaning. The Worker must continue
receiving eligible controls during an active run; it cannot wait for final
settlement before servicing all new input. Full Start/Steer/FollowUp/NextTurn
eligibility is an integration obligation, not proved by this delivery model.

The Go Worker durably records terminal invalid/unsupported/stale-run rejection
through the harness's RejectionRecorder before acknowledgment. Its one journal
append maps to `accept; settle`; AdapterStep.batch preserves safety for such
atomic groups. Busy/suspended work is deferred. Storage, identity conflicts and
unknown errors are not guessed to be terminal; they preserve pending delivery.

## Ownership laws

The same authority guards journal commits, mailbox acknowledgments and
retirement. A standalone lease table whose fence never reaches journal writes
does not satisfy this contract.

Go passes the grant to SessionOpener. The assembled harness binds it to every
journal primitive, for example with `store.WithAuthority` and the local
adapter's `Authority(lease)`. This is an atomic validation-and-mutation scope,
not a preflight check. A database realization must honor the same transaction
context; the wrapper does not create cross-store transactions.

- Acquisition and fence validation have real linearization points.
- Each guarded mutation validates authority atomically with that mutation.
- An old process cannot accept, acknowledge, settle or release the new owner.
- Generation numbers never repeat. Connection identity is not authority.
- Process death does not magically release a durable lease; expiry is explicit.
- Renewal preserves the grant. Its clock implementation is below the contract;
  the model abstracts the point at which authority is revoked as `expire`.

Independent repositories are allowed, but the assembled adapter must prove
that they share a coherent authority and handoff protocol. Implementing each
method signature independently is insufficient.

## The shutdown race

An old observation that the mailbox was empty grants no authority to discard
new work. Retirement may clear the execution marker, but cannot erase pending
input. Discovery includes both sources:

```text
eligible = ownership_available AND
           (pending_mailbox_input OR unfinished_execution)
```

Thus submission before release and submission after release both leave
discoverable work. This relies on the normal Worker receive/discovery contract,
not a best-effort wakeup or an occasional repair cron. Eventual execution still
requires a fair, available worker; readiness alone is not a progress proof.

## Persistent adapter obligations

Each adapter chooses how to represent immutable command identity, pending input,
ownership, unfinished execution and journal history. The contract prescribes
their semantics, not a database, schema layout, table names or application model.

For transaction-backed adapters, keep storage transactions bounded; do not hold
a transaction or database connection during model/tool execution. Establish
operation linearization, conditional fencing, rollback or commit on interruption,
ambiguous replies and duplicate identity behavior for the chosen storage.
Persist the journal/session binding before a mailbox entry can be acknowledged.
Changing the journal backend remains possible, but the replacement must meet
the same durability and authority laws.

The proof does not verify storage isolation, persistence settings, failover,
the Go scheduler, or provider-side effects. Those are explicit adapter/runtime
obligations, not hidden conclusions of the abstract proof.

## Acceptance criteria for each flavor

1. Document its exact interpretation into the model and failure assumptions.
2. Identify each action's linearization point and emitted receipt.
3. Show every crash maps to a permitted durable state; no torn handoffs.
4. Prove safety refinement, including replies and preserved identities.
5. Separately prove the scheduler meets progress obligations; a stuttering
   implementation must not qualify merely because it never corrupts state.
6. Run shared model-based and failure-injection conformance tests against the
   actual code. Keep proof, test and production-validation claims separate.

A volatile local flavor is useful but has a narrower failure model. To claim
the production crash contract, a local `recv` implementation also needs durable
storage; the channel is only how it waits for/receives requests.
