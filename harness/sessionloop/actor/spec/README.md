# Proved session-delivery contract

Status: an executable Lean specification, checked protocol proofs, a receive-based
**Lean reference flavor**, a proved **PostgreSQL transaction model**, and a
**command-publication ordering model**.
The Go ports, shared Worker and local
memory adapter implement the derived contract and have separate conformance,
race and integration tests. They are **not formally proved Go code**. Persistent
storage adapters require their own implementation and verification.

The purpose is to derive a small contract from correctness obligations, then
connect adapters to those laws. Start with [CONTRACT.md](CONTRACT.md).

## File guide

```text
SessionContract/Model.lean          state and atomic operations
               Safety.lean         every permitted finite history stays safe
               Recovery.lean       continuation exists after crashes
               Progress.lean       fair scheduling reaches resolution + cleanup
               Isolation.lean      sessions do not modify one another
               Publication.lean    attribution ordering, replay and crash rebuild
               PublicationExamples.lean old-order counterexamples + fixed trace
               RecoveryStartup.lean early-interrupt responsibility and settlement
               RecoveryFrontier.lean committed candidates, steering and stable runs
               Cleanup.lean        failure cleanup versus explicit interruption
               Adapter.lean        adapter state/operation/reply obligations
               Flavors/Receive.lean  receive-based reference adapter
               Flavors/Postgres.lean PostgreSQL transaction proof root
               Flavors/Postgres/     rows, interleavings, recovery, replies, expiry
               Examples.lean       regression witnesses and invalid designs
SessionContract.lean               imports the entire specification
Audit.lean                         forbids admitted proofs and custom axioms
verify.sh / audit-tests.sh         build, audit, negative tests and kernel replay
```

## Verify

Install Elan, then run from this directory:

```sh
bash verify.sh
```

The project pins Lean 4.34.0 and has no external Lean packages. Verification:

1. Builds every model, proof and regression imported by `SessionContract.lean`.
2. Audits every public theorem in `SessionContract` for transitive axioms.
   Only `propext`, `Classical.choice` and `Quot.sound` are allowed. Admitted
   proofs, custom axioms and trusted native evaluation fail the audit.
3. Checks negative audit fixtures: an admitted proof and a transitive custom
   axiom must both be rejected for the expected reason.
4. Runs the bundled `leanchecker` over stored proof terms in the package.

`Audit.lean` also requires the main theorems by name, so an accidentally empty
proof build cannot pass. GitHub Actions runs the same script.

## What is proved

| Claim | Checked theorem |
|---|---|
| Every permitted interleaving preserves the invariants | `reachable_safe`, `run_safe` |
| Submitted input is still pending or durably accepted | `no_lost_submission` |
| Successful submission/acceptance replies have durable backing | `submission_reply_is_durable`, `accepted_reply_is_durable` |
| Mailbox deletion requires durable acceptance | `deleted_mailbox_has_acceptance` |
| Committed input and acceptance survive arbitrary later actions | `run_submission_persistent`, `run_journal_persistent` |
| Retries cannot change payloads or freshly accept the same ID again | `conflicting_submission`, `retry_acceptance`, `no_second_fresh_acceptance` |
| Ownership grants are unique; stale mutations fail | `one_owner`, `stale_*_rejected`, `takeover_fences_previous_owner` |
| Crashes leave durable state unchanged | `crash_preserves_durable` |
| Unfinished work remains discoverable without mailbox entries | `unfinished_ready_after_expiry` |
| A concurrent release cannot hide a pending input | `pending_ready_after_release` |
| Every safe state has a finite continuation to durable resolution AND mailbox cleanup | `recoverable` |
| Crash after any arbitrary finite history still permits recovery | `crash_at_every_prefix_recoverable` |
| Fair reference-worker scheduling eventually resolves eligible work AND cleans up its mailbox entry | `fair_worker_eventually_settles` |
| Lawful adapters inherit safety, recovery and conditional progress | `every_adapter_safe`, `every_adapter_recoverable`, `every_adapter_progresses` |
| Receive/commit/reply-loss/crash preserve the protocol | `Flavors.Receive.*_refines` |
| A receive-based submission reply cannot precede durable submission | `Flavors.Receive.submitted_response_is_durable` |
| PostgreSQL row state AND replies match the contract | `Flavors.Postgres.calculate_correct` |
| Arbitrary permitted SQL microstep histories preserve safety | `Flavors.Postgres.reachable_safe` |
| A crash at any SQL phase has a concrete transaction recovery path | `Flavors.Postgres.crash_at_every_sql_phase_recoverable` |
| Visible success replies retain durable backing through later actions | `Flavors.Postgres.success_response_has_durable_submission`, `Flavors.Postgres.acceptance_response_has_durable_journal` |
| Steps in one session leave other sessions unchanged | `other_session_unchanged`, `world_step_safe` |
| Publication microsteps preserve the existing protocol | `Publication.history_linearizes`, `Publication.reachable_safe` |
| Observed keyed-command IDs agree with durable replay through subsequent steps | `Publication.live_equals_replay` |
| Committed attribution can be rebuilt and observed after a crash at any phase | `Publication.crash_at_every_phase_replayable` |
| The fixed ordering can actually publish, while the old ordering has a counterexample | `Publication.committed_can_publish`, `Publication.Examples.old_order_breaks_agreement` |
| Append errors before/after commit preserve safety and require reconstruction before observing | `Publication.append_failure_preserves_safety`, `Publication.offline_until_reconstruction`, `Publication.invalidated_cannot_observe` |
| Reconstruction recovers committed attribution or permits an uncommitted retry | `Publication.append_error_reconstruction`, `Publication.uncommitted_error_retry` |
| An early recovery interrupt retains a responsible callback and a settlement continuation | `RecoveryStartup.reachable_owned`, `RecoveryStartup.interrupted_can_settle` |
| Native recovery preserves run identity and acknowledged steering at every modeled prefix | `RecoveryFrontier.reachable_identity`, `RecoveryFrontier.acknowledged_steering_not_lost`, `RecoveryFrontier.every_prefix_recoverable` |
| Failure cleanup cannot manufacture user cancellation and pending input remains recoverable | `Cleanup.reachable_safe`, `Cleanup.abandon_preserves_durable`, `Cleanup.worker_loss_can_recover` |

The delivery/publication proofs are symbolic. The parameter `n` is arbitrary, not a test size.
The command identity domain is `Fin n`: any finite history can be represented
with a sufficiently large domain. Lease generations and history length are
not bounded. `Examples.lean` adds small kernel-checked examples; it does not
replace the general proofs.
`RecoveryStartup` is a separate finite-state component model for one recovered
run's startup/interrupt handoff, including arbitrary repeated permitted steps.
The additional native frontier and cleanup component models, implementation
mapping, executable counterexamples and limits are documented in [RECOVERY.md](RECOVERY.md).

## Counterexamples kept with the proofs

The checked examples include both orders of the submission/shutdown race,
crashing after deleting the last mailbox entry, duplicate submission after
deletion, and an old process retaining a handle after takeover.

They also demonstrate why these designs are invalid:

- Discovering work only through an execution flag can miss a submission that
  races with clearing that flag. Readiness must include durable pending work.
- Deleting an input before durable acceptance breaks conservation.
- Clearing execution state while journal work is unfinished breaks recovery.
- Resetting all storage on crash can still satisfy state-local invariants!
  It violates history preservation. This is why a snapshot assertion alone
  is not a crash-durability proof.

## Exact scope and assumptions

The model covers durable delivery, ownership, acceptance handoff, execution
discoverability and retirement. It abstracts a harness's durable acceptance
and settlement; it does not prove the harness's model/tool behavior.

Safety covers arbitrary action orderings, stale attempts, duplicates, crashes
and expiry. Failed I/O before commit is a stutter. Successful commit followed
by a lost reply is the committed step followed by a retry. Atomicity applies
to individual primitives, **never an entire worker run**. A concrete adapter
must prove that a crash inside its implementation corresponds to a modeled
pre-commit or post-commit state, not a torn state.

The recovery theorem proves existence of a continuation; the progress theorem
separately proves eventual settlement for the reference worker under fair
successful primitive scheduling. Arbitrarily many finite waits are allowed.
For progress, there must eventually be a healthy execution interval: workers
are scheduled, expired authority becomes reclaimable, storage operations can
complete, and the harness eventually produces a durable outcome. The theorem
does not claim progress under perpetual crashes, starvation, or a permanently
unresponsive provider. New submissions are covered by safety; the progress
trace formalizes a healthy worker suffix for the already submitted finite
identity domain, not an unbounded arrival/service-rate theorem.

`Resolved` requires BOTH `settled = true` and `mailbox = false`. The reference
worker acknowledges durably accepted input before advancing it to settlement,
and also cleans up an entry if settlement happened before acknowledgment.
`settled` means the execution obligation has a durable resolution: completed,
failed explicitly, or safely parked pending external input. The detailed
outcome, suspension protocol, general output projection and LLM transcript are
outside this model. The separate publication extension proves only keyed-command
attribution consistency, not equality of complete projected events; see
[PUBLICATION.md](PUBLICATION.md). No proof here permits silently dropping a failed
command.

The trusted storage abstraction does not lose committed durable data. Permanent
loss of every durable copy is outside this failure model. So are Byzantine
workers, forged capabilities and arbitrary memory corruption. Generations do
not wrap: a bounded implementation must prevent reuse and state its capacity
assumption for progress.

Commands here already have stable semantics. Automatic Start/Steer planning,
run-target validation, journal bootstrap/binding, replay bytes/compaction,
application authorization, and external-tool idempotency need their own faithful
integration obligations. SessionOpener is modeled as restoring an intact,
correctly bound journal under the granted authority; its real Go implementation
is not proved here. A non-idempotent external effect with an unknown outcome
may require durable suspension/reconciliation, not automatic repetition.

One authoritative writer does **not** mean two processes can never overlap in
CPU time. The model deliberately preserves an old handle across expiry and
proves that it cannot commit. External effects need their own authorization or
idempotency boundary.

## What the reference flavor demonstrates

`Flavors/Receive.lean` separates volatile request arrival, durable processing, and reply
availability. Losing a reply cannot undo a commit. Losing an unprocessed
request does not contradict durability because no success was returned.

It is a proved **Lean algorithm**, not a proof of Go's runtime or a disk driver.
Its durable core remains an abstract storage requirement. A memory-only Go
channel cannot claim its process-crash guarantees; a local durable adapter
can use a channel for transport and a persisted core for acknowledged state.

## Next implementation boundary

The PostgreSQL model and its exact implementation/trust obligations are described
in [the PostgreSQL proof guide](../../../hosts/postgres/PROOF.md). Its private
writes, commit, rollback and reply phases are explicit; the SQL engine, actual
statements, Go code, clocks and native journal mapping are not proved by Lean.

1. Review the specification and its assumptions as the contract, not only the
   green proof result. Check that the desired failure cases are in the model.
2. The Go local adapter and common Worker implement the derived handoff order.
   Their mapping and narrower failure model are in `../memory/README.md`.
3. Shared conformance and concrete failure-injection tests exercise the code.
   Tests provide evidence; they are not code-level formal verification.
4. Implement persistent flavors behind the same public ports, choosing storage
   and scheduling technology within each adapter's documented failure model.
5. For a formally verified adapter, connect its actual algorithm, linearization
   points, emitted replies and crashes to `Adapter`/`AdapterStep`.
   Proving a separately rewritten sketch is insufficient.

Lean is development tooling only; none of these files add a runtime dependency
to Agentic or its consumers.
