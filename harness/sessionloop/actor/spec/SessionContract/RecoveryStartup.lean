/-! # Responsibility for interrupting a recovered run

The recovery callback may be scheduled after Close requests interruption. This
small executable model tracks responsibility separately from session status:
either the pending startup callback or the active driver must finish the run.
Startup consumes its responsibility only by handing it to the driver, settling,
or faulting. An I/O error is an explicit fault, not a claim of durable settlement.

Transitions represent mutex-serialized decisions and completed settlement I/O.
Progress assumes callbacks are scheduled and I/O returns; this is not a proof
of Go scheduling, the settlement journal codec, or arbitrary driver behavior.
-/

namespace SessionContract.RecoveryStartup

inductive Status where
  | running | interrupting | idle | faulted
  deriving DecidableEq

structure State where
  status : Status
  pending : Bool
  driver : Bool
  deriving DecidableEq

def initial : State := ⟨.running, true, false⟩

def Owned (s : State) : Prop :=
  (s.status = .running ∨ s.status = .interrupting) →
    s.pending = true ∨ s.driver = true

def interrupt (s : State) : State :=
  if s.status = .running then { s with status := .interrupting } else s

def finish (success : Bool) : State :=
  ⟨if success then .idle else .faulted, false, false⟩

def start (s : State) (success : Bool) : State :=
  if s.pending then
    match s.status with
    | .running => ⟨.running, false, true⟩
    | .interrupting => finish success
    | _ => { s with pending := false }
  else s

def driverFinish (s : State) (success : Bool) : State :=
  if s.driver then finish success else s

inductive Step : State → State → Prop where
  | interrupt (s) : Step s (interrupt s)
  | start (s) (success) : Step s (start s success)
  | driverFinish (s) (success) : Step s (driverFinish s success)

inductive Reachable : State → Prop where
  | initial : Reachable initial
  | next : Reachable s → Step s t → Reachable t

theorem step_preserves_responsibility (h : Step s t) (owned : Owned s) : Owned t := by
  cases h <;> rcases s with ⟨status, pending, driver⟩ <;>
    cases status <;> cases pending <;> cases driver <;>
    simp_all [Owned, interrupt, start, driverFinish, finish] <;>
    split <;> simp_all

theorem reachable_owned (h : Reachable s) : Owned s := by
  induction h with
  | initial => simp [Owned, initial]
  | next _ step ih => exact step_preserves_responsibility step ih

/-- Even before the driver exists, interruption has a finite completion path.
Failure returns a fault; it never silently strands an interrupting session. -/
theorem interrupt_before_start (success : Bool) :
    start (interrupt initial) success = finish success := by
  rfl

/-- If startup wins the mutex, responsibility goes to the driver's finalizer. -/
theorem interrupt_after_start (success : Bool) :
    driverFinish (interrupt (start initial true)) success = finish success := by
  rfl

/-- For every reachable interrupted state, scheduling the responsible callback
with successful settlement I/O reaches idle, regardless of startup order. -/
theorem interrupted_can_settle (h : Reachable s) (stopped : s.status = .interrupting) :
    start s true = finish true ∨ driverFinish s true = finish true := by
  have owned := reachable_owned h
  rcases s with ⟨status, pending, driver⟩
  cases status <;> cases pending <;> cases driver <;>
    simp_all [Owned, start, driverFinish]

/-- The previous guard consumed startup without settling an early interrupt. -/
def oldStart (s : State) : State :=
  if s.status = .running then ⟨.running, false, true⟩
  else { s with pending := false }

theorem old_guard_strands_reachable_interrupt :
    Reachable (interrupt initial) ∧ ¬ Owned (oldStart (interrupt initial)) := by
  constructor
  · exact .next .initial (.interrupt initial)
  · simp [Owned, oldStart, interrupt, initial]

end SessionContract.RecoveryStartup
