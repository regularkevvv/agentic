# Command publication proof

The attribution fix changes observable behavior, not the mailbox/lease rules.
Previously a live event could have no command ID while later replay of the same
journal entry supplied one. The original delivery model could not express this
bug. `Publication.lean` adds that missing observation boundary.

## Algorithm and proved claims

```text
                    session mutex held
             +-----------------------------------+
acceptance:  acquire -> journal commit -> install  -> unlock -> publish
                                       ID maps
reader:      load journal -> acquire same mutex -> read ID -> release
crash:       discard volatile maps -> rebuild from journal -> expose new view
```

A reader can load committed bytes before `Append` returns. Moving installation
ahead of hub publication alone is therefore insufficient: attribution lookups
must acquire the same mutex too. All paths acquire `inner.mu` before `v.mu`.

The model splits `begin`, `commit`, `install`, `unlock` and `observe`; crashes can
occur between any two. Other delivery-protocol actions can interleave too. The
observation guard checks only availability, an unlocked phase and an existing
durable key. It does **not** assume the command ID is correct.

For arbitrary finite histories and arbitrary command/key domains, Lean proves:

- `live_equals_replay`: every recorded observation carries the same command ID
  as the durable projection, including after subsequent accepts and crashes.
- `reachable_safe`: cached attribution may lag only inside the critical section;
  durable bindings are backed by journal acceptance; previous protocol safety
  still holds.
- `history_linearizes`: erasing publication metadata gives a history of the
  original protocol. No new mailbox, ownership or acceptance semantics appear.
- `crash_at_every_phase_replayable`: at any phase, each committed binding has an
  explicit crash/rebuild/read continuation. Before commit, the original mailbox
  recovery rules still apply; no acceptance is invented.
- `committed_can_publish`: install/unlock/read is an enabled finite continuation.
  Safety is not obtained by prohibiting all observations.

An acceptance can bind several native keys in one commit, as when a resolution
opens a continuation run. The model installs all those bindings before unlock.

`PublicationExamples.lean` starts with a reachable submit/acquire/open/accept
trace and checks both bad designs: premature unlock, and a reader bypassing the
mutex. Both expose `none` where durable replay supplies a command ID. It proves
that the old-order trace is unreachable under the repaired transition rules and
that the old and new examples have identical underlying delivery-protocol state.

## Correspondence to the implementation

| Model obligation | Handwritten Go implementation / evidence |
|---|---|
| Atomic keyed acceptance and durable attribution | `kindCommandAccepted` is appended in the same batch as the run, queue or resolution facts. `durable` in Lean is a projection of those records, **not another table**. |
| Commit, then install, then unlock | `prepareStartWithCommand`, `acceptWithCursorCommand`, `prepareResumeWithCommand`, `prepareResumeIndeterminateWithCommand` call the private `onAccepted` callback after successful append and before releasing `s.mu`. |
| All affected bindings installed together | `dispatchStart`, `dispatchQueue`, `dispatchResolve` callbacks update the appropriate `runCommands`, `queueCommands`, `resolutionCommands` under `v.mu`; a resolution keeps an existing run's original attribution. |
| Readers cannot cross the commit/install gap | `commandForRun`, `commandForQueue`, `commandForResolution` acquire `inner.mu` before `v.mu`, including readers projecting directly from journal loads. Queue consumption also takes the session mutex. |
| Rebuild before exposing a reopened view | `NewLoopView` completes `restoreCommandAcceptances` before returning a view with idempotent-dispatch capability. A decoding error fails construction. |
| Identity does not change after publication | Model `Compatible` requires native keys to be fresh or already bound to the same command. Run/queue identity generation, unique journal sequences and resolve's preserve-existing-run rule are implementation obligations. |
| Real-code regression evidence | `loop_publication_test.go` blocks publication before `Dispatch` returns, exercises live reads, finite replay, concurrent queue consumption and keyed reopen for start and recovery-resolution paths. |

Implementation sources: [view and attribution](../../../session/loop_view.go),
[start](../../../session/execution.go), [queue](../../../session/session.go),
[resolve](../../../session/resume.go),
[regression tests](../../../session/loop_publication_test.go).

## Exact limits

- This proves the **modeled ordering algorithm**, not a formal extraction or
  refinement proof of the actual Go, mutexes, storage driver or journal decoder.
  The table above is a reviewed mapping, supplemented by executable tests.
- Scope is **durably keyed** start, queue and resolution command attribution.
  Unkeyed IDs remain live-handle metadata; persistence across reopen is not
  promised. This does not prove equality of complete events or LLM transcripts.
- `restore` abstracts successful reconstruction of an intact journal index. It
  does not grant a lease or prove SessionOpener; those are separate protocol
  operations. Failed storage reads may delay recovery; corruption is not repaired
  by this theorem. The durable map assumes atomic journal records and no loss of
  committed storage, as in the existing contract.
- Observations are proof-only history, not a new runtime store. The model covers
  attribution readers of one session view; cross-process authority is handled by
  the existing ownership model, not by this process-local mutex.
- Finite recovery continuations do not prove scheduler fairness, eventual I/O
  success, deadlock freedom of the entire Go program or provider progress.

The fix also moves volatile `runDone` registration into the same callback. It
still occurs before launching the drive goroutine; channel completion and
goroutine joining are unchanged. That lifecycle plumbing is not represented by
the attribution theorem and remains covered by the separate Go lifecycle/race
tests. The module-version alignment is build metadata, not a protocol change.

## Verification

```sh
# From harness/sessionloop/actor/spec
bash verify.sh

# From harness
go test -race ./session -run 'TestLoopRace.*AttributionBeforePublication' -count=100
```

The proof root imports both new modules. The audit requires the principal new
theorems and rejects admitted proofs/custom axioms; `leanchecker` rechecks proof
terms. CI now runs that verification for changes to the embedded session,
SessionLoop and PostgreSQL host, not only changes to Lean files. These checks
keep the model exercised; they do not automatically establish Go/model fidelity.
