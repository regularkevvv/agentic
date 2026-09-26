import Std

/-! # Failed-worker cleanup cannot invent user interruption

This component models the native mutex boundary: abandon first faults the live
handle, then cancels/joins its driver. A finalizer that wins the mutex may commit
its real result; one arriving after faulting cannot mutate durable state.
Cancellation of an already-accepted request is not an interrupt command.
Fencing of each durable write is supplied by the delivery/adapter model.
-/
namespace SessionContract.Cleanup

structure Durable where
  pending : Bool
  applied : Bool
  cancelled : Bool
  stopRequested : Bool
  deriving DecidableEq

structure State where
  durable : Durable
  faulted : Bool
  deriving DecidableEq

def initial : State := ⟨⟨true, false, false, false⟩, false⟩
def abandon (s : State) : State := { s with faulted := true }
def reopen (s : State) : State := { s with faulted := false }
def requestCancelled (s : State) : State := s
def interrupt (s : State) : State :=
  if s.faulted then s else { s with durable.stopRequested := true }
def applyInput (s : State) : State :=
  if !s.faulted && s.durable.pending && !s.durable.stopRequested then
    { s with durable.pending := false, durable.applied := true }
  else s
def finishInterrupt (s : State) : State :=
  if !s.faulted && s.durable.pending && s.durable.stopRequested then
    { s with durable.pending := false, durable.cancelled := true }
  else s

inductive Step : State → State → Prop where
  | abandon (s) : Step s (abandon s)
  | reopen (s) : Step s (reopen s)
  | requestCancelled (s) : Step s (requestCancelled s)
  | interrupt (s) : Step s (interrupt s)
  | applyInput (s) : Step s (applyInput s)
  | finishInterrupt (s) : Step s (finishInterrupt s)

def Safe (s : State) : Prop :=
  (s.durable.pending = true ∨ s.durable.applied = true ∨ s.durable.cancelled = true) ∧
  (s.durable.cancelled = true → s.durable.stopRequested = true)

theorem step_safe (h : Step s t) (safe : Safe s) : Safe t := by
  cases h <;> rcases s with ⟨⟨pending, applied, cancelled, stopRequested⟩, faulted⟩ <;>
    cases pending <;> cases applied <;> cases cancelled <;> cases stopRequested <;> cases faulted <;>
    simp_all [Safe, abandon, reopen, requestCancelled, interrupt, applyInput, finishInterrupt]

inductive Reachable : State → Prop where
  | initial : Reachable initial
  | next : Reachable s → Step s t → Reachable t

theorem reachable_safe (h : Reachable s) : Safe s := by
  induction h with
  | initial => simp [Safe, initial]
  | next _ step ih => exact step_safe step ih

theorem abandon_preserves_durable (s : State) : (abandon s).durable = s.durable := rfl
theorem late_request_cancel_preserves_durable (s : State) :
    (requestCancelled s).durable = s.durable := rfl

theorem late_finalizers_cannot_write (s : State) :
    applyInput (abandon s) = abandon s ∧
    finishInterrupt (abandon s) = abandon s := by
  simp [applyInput, finishInterrupt, abandon]

theorem worker_loss_can_recover (s : State) (pending : s.durable.pending = true)
    (notInterrupted : s.durable.stopRequested = false) :
    (applyInput (reopen (abandon s))).durable.applied = true := by
  simp [applyInput, reopen, abandon, pending, notInterrupted]

/-- The old cleanup cancelled a pending input without an interrupt command. -/
def oldCleanup (s : State) : State :=
  { s with durable.pending := false, durable.cancelled := true, faulted := true }

theorem old_cleanup_breaks_contract : ¬ Safe (oldCleanup initial) := by
  simp [Safe, oldCleanup, initial]

end SessionContract.Cleanup
