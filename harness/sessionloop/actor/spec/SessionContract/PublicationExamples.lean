import SessionContract.Publication

/-! # Publication-ordering regression witnesses
Start from a real submit/acquire/open/accept history. The old order exposes an
empty attribution after journal commit; the repaired order publishes the durable
ID. These kernel-checked witnesses complement, not replace, the universal proofs.
-/

namespace SessionContract.Publication.Examples

def request : Request 1 := ⟨1, 0, 7, [42, 43]⟩

def submitted : View 1 := protocol initialView (.submit 0 7)
def owned : View 1 := protocol submitted .acquire
def opened : View 1 := protocol owned (.openSession 1)
def ready : View 1 := restore opened
def accepting : View 1 := begin ready request
def committed : View 1 := commit accepting request

theorem committed_reachable : Steps initialView committed := by
  have hs : Steps initialView submitted := .next (.refl _) (.protocol _ (.submit 0 7) trivial)
  have ho : Steps initialView owned := .next hs (.protocol _ .acquire trivial)
  have hp : Steps initialView opened := .next ho (.protocol _ (.openSession 1) trivial)
  have hr : Steps initialView ready := .next hp (.restore _ rfl)
  have ha : Steps initialView accepting := .next hr (.begin _ request rfl rfl)
  exact .next ha (.commit _ request rfl (fun _ _ => Or.inl rfl) (by decide))

/-- The original order released the session before installing view metadata.
This deliberately uses `unlock` WITHOUT the new Step.unlock phase guard. -/
def oldObserved : View 1 := observe (unlock committed) 42

theorem old_order_exposes_missing_id :
    oldObserved.observations.head? = some ⟨42, none⟩ ∧
    oldObserved.durable 42 = some 0 := by decide

theorem old_order_breaks_agreement : ¬ Agrees oldObserved := by
  intro agrees
  obtain ⟨c, _, hc⟩ := agrees ⟨42, none⟩ (by decide)
  contradiction

theorem old_order_unreachable : ¬ Steps initialView oldObserved := by
  intro history
  exact old_order_breaks_agreement (reachable_safe history).2.agrees

/-- Installing before publish is insufficient if a journal reader bypasses
the same mutex: it could see committed bytes while Append has not returned. -/
theorem unlocked_reader_also_breaks_agreement :
    ¬ Agrees (observe committed 42) := by
  intro agrees
  obtain ⟨c, _, hc⟩ := agrees ⟨42, none⟩ (by decide)
  contradiction

def fixedObserved : View 1 := observe (unlock (install committed request)) 42

theorem fixed_order_reachable : Steps initialView fixedObserved := by
  have hi := Steps.next committed_reachable (Step.install _ request rfl)
  have hu := Steps.next hi (Step.unlock _ rfl)
  exact .next hu (.observe _ 42 rfl rfl (by decide))

theorem fixed_order_agrees : Agrees fixedObserved :=
  (reachable_safe fixed_order_reachable).2.agrees

theorem same_durable_protocol : oldObserved.core = fixedObserved.core := rfl

theorem continuation_and_resolution_both_installed :
    fixedObserved.cache 42 = some 0 ∧ fixedObserved.cache 43 = some 0 := by decide

/-- Crash immediately after append, before the callback, still permits a
reconstructed read of this command's ID. -/
theorem crash_before_install_replays :
    Steps committed (observe (restore (crash committed 1)) 42) ∧
    (observe (restore (crash committed 1)) 42).observations.head? = some ⟨42, some 0⟩ :=
  crash_at_every_phase_replayable committed 1 rfl

/-- Before append, there is no invented acceptance: the durable mailbox still
holds the submitted input after the cache and in-flight request disappear. -/
theorem crash_before_append_keeps_pending :
    (crash accepting 1).core.mailbox 0 = true ∧
    (crash accepting 1).core.journal 0 = none ∧
    (crash accepting 1).durable 42 = none := by decide

/-- Lost success and actual rollback produce the same unavailable view, but
different durable facts. Never infer rollback from the returned error. -/
theorem unknown_result_after_commit : Steps initialView (invalidate committed) :=
  .next committed_reachable (.appendError _ ⟨request, Or.inr rfl⟩)

theorem unknown_result_keeps_acceptance :
    (invalidate committed).core.journal 0 = some 7 ∧
    (invalidate committed).durable 42 = some 0 ∧
    (invalidate committed).online = false := by decide

theorem unknown_result_replays_after_rebuild :
    Steps (invalidate committed) (observe (restore (invalidate committed)) 42) ∧
    (observe (restore (invalidate committed)) 42).observations.head? = some ⟨42, some 0⟩ :=
  append_error_reconstruction committed rfl

theorem precommit_error_does_not_invent_acceptance :
    Step accepting (invalidate accepting) ∧
    (invalidate accepting).core.mailbox 0 = true ∧
    (invalidate accepting).core.journal 0 = none :=
  ⟨.appendError _ ⟨request, Or.inl rfl⟩, by decide, by decide⟩

theorem precommit_error_can_retry :
    Steps (invalidate accepting)
      (commit (begin (restore (invalidate accepting)) request) request) :=
  uncommitted_error_retry accepting request (fun _ _ => Or.inl rfl) (by decide)

end SessionContract.Publication.Examples
