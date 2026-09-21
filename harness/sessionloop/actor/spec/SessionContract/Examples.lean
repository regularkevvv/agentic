import SessionContract.Recovery
import SessionContract.Flavors.Receive

/-! # Regression examples and counterexamples
Kernel-checked witnesses for shutdown races, takeover, handoff crashes and retries.
Invalid designs deliberately demonstrate what breaks the contract. These examples
supplement, rather than replace, the universally quantified protocol proofs.
-/

namespace SessionContract.Examples

/-- A journal can atomically record a rejected input's acceptance and terminal
resolution. This does not require another mailbox status or an invented run. -/
theorem terminal_rejection_batch_safe (s : State n) (safe : Safe s)
    (g : Lease) (c : CommandId n) (p : Payload) :
    Safe (run s [.accept g c p, .settle g c]) := by
  exact adapter_preserves_safety id (AdapterStep.batch _ rfl) safe

private def first : CommandId 2 := 0
private def second : CommandId 2 := 1

def retiring : State 2 := run initial [
  .submit first 10, .acquire, .openSession 1, .accept 1 first 10,
  .acknowledge 1 first, .settle 1 first, .observeEmpty 1, .closeSession 1]

def arrivedBeforeRelease : State 2 := run retiring [.submit second 20, .release 1]
def arrivedAfterRelease : State 2 := run retiring [.release 1, .submit second 20]

theorem stale_empty_observation : retiring.sawEmpty 1 = true := by decide
theorem arrival_before_release_ready : Ready arrivedBeforeRelease := by decide
theorem arrival_after_release_ready : Ready arrivedAfterRelease := by decide

def acceptedAndDeleted : State 2 := run initial [
  .submit first 10, .acquire, .openSession 1, .accept 1 first 10, .acknowledge 1 first]

def crashedAfterDeletion : State 2 := run acceptedAndDeleted [.crash 1, .expire]

theorem crash_without_mailbox_still_ready :
    crashedAfterDeletion.mailbox first = false ∧ Ready crashedAfterDeletion := by decide

theorem journal_survives_crash : crashedAfterDeletion.journal first = some 10 := by decide

theorem recovery_without_mailbox :
    (run crashedAfterDeletion [.acquire, .openSession 2, .accept 2 first 10,
      .settle 2 first, .closeSession 2, .release 2]).settled first = true := by decide

theorem duplicate_submit_after_deletion :
    execute acceptedAndDeleted (.submit first 10) = (acceptedAndDeleted, .duplicate) := by
  apply retry_submission
  decide

theorem changed_payload_is_rejected :
    (execute acceptedAndDeleted (.submit first 11)).2 = .conflict := by decide

def takenOver : State 2 := run acceptedAndDeleted [.expire, .acquire, .openSession 2]

theorem old_process_can_still_have_a_handle : takenOver.handles 1 = true := by decide
theorem old_process_cannot_settle :
    (execute takenOver (.settle 1 first)).2 = .unavailable := by decide
theorem old_process_cannot_release_new_owner :
    (execute takenOver (.release 1)).2 = .unavailable := by decide

/-- Negative witness: keeping only an execution flag for discovery loses the
submission-vs-release race even though the input is durably in the mailbox. -/
theorem flag_only_discovery_is_insufficient :
    arrivedBeforeRelease.needsExecution = false ∧
    arrivedBeforeRelease.mailbox second = true ∧ Ready arrivedBeforeRelease := by decide

def unsafeEarlyDeletion : State 2 :=
  { step initial (.submit first 10) with mailbox := fun _ => false }

theorem early_deletion_breaks_conservation : ¬ Safe unsafeEarlyDeletion := by
  intro safe
  have conservation := safe.conservation first 10 (by decide)
  have notPending : unsafeEarlyDeletion.mailbox first = false := by decide
  have notAccepted : unsafeEarlyDeletion.journal first = none := by decide
  simp [notPending, notAccepted] at conservation

def unsafeClearExecution : State 2 :=
  { acceptedAndDeleted with needsExecution := false, owner := none }

theorem clearing_unfinished_execution_breaks_recovery : ¬ Safe unsafeClearExecution := by
  intro safe
  have discoverable := safe.unfinished_discoverable first 10 (by decide) (by decide)
  change false = true at discoverable
  contradiction

/-- A reset-to-empty adapter can satisfy state-local invariants but violate
history preservation. This is why refinement must include crash semantics. -/
theorem empty_is_safe_but_not_a_durable_crash :
    Safe (initial : State 2) ∧
    acceptedAndDeleted.submitted first = some 10 ∧
    (initial : State 2).submitted first = none := by
  exact ⟨initial_safe, by decide, rfl⟩

def fiveTicks (s : State 2) (c : CommandId 2) : State 2 :=
  tick (tick (tick (tick (tick s c) c) c) c) c

theorem reference_worker_resolves_and_cleans :
    Resolved (fiveTicks (step initial (.submit first 10)) first) first := by decide

def settledBeforeCleanup : State 2 := run initial [
  .submit first 10, .acquire, .openSession 1, .accept 1 first 10,
  .settle 1 first, .crash 1, .expire]

theorem recovery_cleans_already_settled_input :
    Resolved (fiveTicks settledBeforeCleanup first) first := by decide

def responseWasLost : Flavors.Receive.Local 2 :=
  Flavors.Receive.loseReply (Flavors.Receive.recv (Flavors.Receive.send {} (.submit first 10)))

theorem receive_retry_after_lost_reply :
    (Flavors.Receive.recv (Flavors.Receive.send responseWasLost (.submit first 10))).response =
      some (.submit first 10, .duplicate) := by decide

theorem receive_crash_preserves_submission :
    (Flavors.Receive.crash responseWasLost 1).core.submitted first = some 10 := by decide

theorem unprocessed_channel_request_is_not_acknowledged :
    (Flavors.Receive.send ({} : Flavors.Receive.Local 2) (.submit first 10)).response = none ∧
    (Flavors.Receive.send ({} : Flavors.Receive.Local 2) (.submit first 10)).core.submitted first = none := by decide

end SessionContract.Examples
