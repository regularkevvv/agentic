import SessionContract.Flavors.Postgres.Receipts
import SessionContract.Flavors.Postgres.Expiry

/-! # PostgreSQL regression witnesses and rejected algorithms
Small kernel-checked scenarios supplement the universal proofs. Negative
witnesses deliberately bypass one rule and exhibit lost data or false replies;
they prevent mistaking a convenient but incomplete model for the intended design.
-/

namespace SessionContract.Flavors.Postgres.Examples

private def first : CommandId 2 := 0
private def second : CommandId 2 := 1

def retiring : Rows 2 := (rowAdapter 2).run {} [
  .submit first 10, .acquire, .openSession 1, .accept 1 first 10,
  .acknowledge 1 first, .settle 1 first, .observeEmpty 1, .closeSession 1]

theorem arrival_before_retirement_discoverable :
    ready ((rowAdapter 2).run retiring [.submit second 20, .release 1]) := by
  unfold ready work
  decide

theorem arrival_after_retirement_discoverable :
    ready ((rowAdapter 2).run retiring [.release 1, .submit second 20]) := by
  unfold ready work
  decide

theorem flag_only_polling_misses_input :
    let rows := (rowAdapter 2).run retiring [.submit second 20, .release 1]
    rows.session.executionNeeded = false ∧ rows.commands.pending second = true ∧ ready rows := by
  unfold ready work
  decide

def acceptedAndCleaned : Rows 2 := (rowAdapter 2).run {} [
  .submit first 10, .acquire, .openSession 1, .accept 1 first 10, .acknowledge 1 first]

theorem no_mailbox_payload_but_recovery_remains :
    acceptedAndCleaned.commands.pending first = false ∧
    acceptedAndCleaned.commands.identity first = some 10 ∧
    ready (calculate acceptedAndCleaned .expire).1 := by decide

theorem cleanup_does_not_break_retry :
    (calculate acceptedAndCleaned (.submit first 10)).2 = .duplicate ∧
    (calculate acceptedAndCleaned (.submit first 11)).2 = .conflict := by decide

def successor : Rows 2 := (rowAdapter 2).run acceptedAndCleaned [.expire, .acquire, .openSession 2]

theorem paused_old_handle_is_fenced :
    successor.handles 1 = true ∧ successor.session.owner = some 2 ∧
    (calculate successor (.settle 1 first)).2 = .unavailable ∧
    (calculate successor (.release 1)).2 = .unavailable := by decide

theorem expired_renewal_at_equal_deadline_rejected :
    renewDeadline acceptedAndCleaned 1 { now := 50, expiresAt := 50 } 30 = none := by decide

/-- Client A read before B committed. Publishing A's old whole write set without
the required row lock silently destroys B's acknowledged input. -/
def staleTransaction : Transaction 2 :=
  { client := 0, action := .submit first 10, before := {}, phase := .readyToCommit }

def brokenLockless : Database 2 :=
  { committed := (calculate ({} : Rows 2) (.submit second 20)).1,
    active := some staleTransaction }

theorem missing_lock_loses_committed_input :
    brokenLockless.committed.commands.identity second = some 20 ∧
    (commit brokenLockless staleTransaction).committed.commands.identity second = none := by decide

theorem lockless_counterexample_violates_snapshot : ¬ LockedSnapshot brokenLockless := by
  intro valid
  have equality := valid staleTransaction rfl
  have wrong := congrArg (fun r => r.commands.identity second) equality
  change none = some 20 at wrong
  cases wrong

theorem lockless_counterexample_unreachable : ¬ Steps ({} : Database 2) brokenLockless := by
  intro history
  exact lockless_counterexample_violates_snapshot (steps_keep_snapshot history initial_locked_snapshot)

/-- Responding from a private write set before COMMIT is not durable acceptance. -/
theorem premature_success_has_no_backing :
    ¬ Backed ({} : Rows 2) (.submit first 10) .submitted := by
  intro backed
  have wrong := backed.1 first 10 rfl (Or.inl rfl)
  cases wrong

def waitingBehindLock : Database 2 :=
  enqueue (lockRow (enqueue {} 0 (.submit first 10)) 0 (.submit first 10)) 1 (.submit second 20)

theorem contender_cannot_take_held_lock : waitingBehindLock.active ≠ none := by
  simp [waitingBehindLock, enqueue, lockRow]

theorem queued_input_is_not_a_success_reply :
    waitingBehindLock.responses 0 = none ∧ waitingBehindLock.responses 1 = none ∧
    waitingBehindLock.committed.commands.identity first = none ∧
    waitingBehindLock.committed.commands.identity second = none := by decide

theorem crash_between_private_writes_preserves_tables (phase : Phase) :
    let t := { staleTransaction with phase := phase }
    let s : Database 2 := { committed := acceptedAndCleaned, active := some t }
    (crash s).committed = acceptedAndCleaned ∧ (crash s).active = none := by
  exact ⟨rfl, rfl⟩

end SessionContract.Flavors.Postgres.Examples
