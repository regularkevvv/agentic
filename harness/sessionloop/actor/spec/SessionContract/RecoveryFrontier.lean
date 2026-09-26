import Std

/-! # Recovery of a committed response and steering frontier

Component model below the delivery contract's abstract journal. A run identity
is fixed by acceptance. A response candidate is NOT an input requiring another
model request; a durable completion only needs closure. Queue draining and
injection are separate commits. Crashes preserve both, including the gap.

Each step is a successful fenced journal transition (or loss of the live
handle). The PostgreSQL model supplies authority/atomicity; this model does not
prove the Go decoder, arbitrary providers, or OS scheduling. Progress assumes
the worker is scheduled and model/validation/storage eventually succeed.
-/
namespace SessionContract.RecoveryFrontier

inductive Frontier where
  | input | candidate | completed
  deriving DecidableEq, Repr

inductive InputPhase where
  | absent | queued | drained | applied
  deriving DecidableEq, Repr

structure State where
  run : Nat
  frontier : Frontier
  accepted : Bool
  input : InputPhase
  live : Bool
  closed : Bool
  deriving DecidableEq, Repr

def initial (run : Nat) : State := ⟨run, .input, false, .absent, true, false⟩

def offer (s : State) : State :=
  if s.live && !s.closed && s.input == .absent && s.frontier != .completed then
    { s with accepted := true, input := .queued }
  else s

def crash (s : State) : State := { s with live := false }

def recover (s : State) : State :=
  if !s.live && !s.closed then
    if s.frontier == .completed then { s with closed := true }
    else { s with live := true }
  else s

def respond (s : State) : State :=
  if s.live && !s.closed && s.frontier == .input then
    { s with frontier := .candidate }
  else s

def drain (s : State) : State :=
  if s.live && !s.closed && s.input == .queued then
    { s with input := .drained }
  else s

def inject (s : State) : State :=
  if s.live && !s.closed && s.input == .drained then
    { s with input := .applied, frontier := .input }
  else s

def complete (s : State) : State :=
  if s.live && !s.closed && s.frontier == .candidate &&
      (s.input == .absent || s.input == .applied) then
    { s with frontier := .completed }
  else s

def close (s : State) : State :=
  if s.live && s.frontier == .completed then { s with closed := true, live := false }
  else s

inductive Step : State → State → Prop where
  | offer (s) : Step s (offer s)
  | crash (s) : Step s (crash s)
  | recover (s) : Step s (recover s)
  | respond (s) : Step s (respond s)
  | drain (s) : Step s (drain s)
  | inject (s) : Step s (inject s)
  | complete (s) : Step s (complete s)
  | close (s) : Step s (close s)

inductive Reachable (run : Nat) : State → Prop where
  | initial : Reachable run (initial run)
  | next : Reachable run s → Step s t → Reachable run t

def Safe (s : State) : Prop :=
  (s.accepted = true ↔ s.input ≠ .absent) ∧
  (s.frontier = .completed → s.input = .absent ∨ s.input = .applied) ∧
  (s.closed = true → s.frontier = .completed) ∧
  (s.closed = true → s.live = false)

theorem step_preserves_identity (h : Step s t) : t.run = s.run := by
  cases h <;> simp only [offer, crash, recover, respond, drain, inject, complete, close] <;>
    repeat' first | split | rfl

theorem step_preserves_safety (h : Step s t) (safe : Safe s) : Safe t := by
  cases h <;> rcases s with ⟨run, frontier, accepted, input, live, closed⟩ <;>
    cases frontier <;> cases accepted <;> cases input <;> cases live <;> cases closed <;>
    simp_all [Safe, offer, crash, recover, respond, drain, inject, complete, close]

theorem reachable_safe (h : Reachable run s) : Safe s := by
  induction h with
  | initial => simp [Safe, initial]
  | next _ step ih => exact step_preserves_safety step ih

theorem reachable_identity (h : Reachable run s) : s.run = run := by
  induction h with
  | initial => rfl
  | next _ step ih => exact (step_preserves_identity step).trans ih

theorem acknowledged_steering_not_lost (h : Reachable run s) (accepted : s.accepted = true) :
    s.input = .queued ∨ s.input = .drained ∨ s.input = .applied := by
  have present := (reachable_safe h).1.mp accepted
  cases eq : s.input <;> simp_all

theorem closed_steering_was_applied (h : Reachable run s)
    (accepted : s.accepted = true) (closed : s.closed = true) : s.input = .applied := by
  obtain ⟨known, done, settled, _⟩ := reachable_safe h
  have := done (settled closed)
  simp_all

/-- Recovery never fabricates failure or completion from a response candidate. -/
theorem recovery_keeps_durable_frontier (s : State) :
    (recover (crash s)).frontier = s.frontier ∧
    (recover (crash s)).input = s.input ∧
    (recover (crash s)).run = s.run := by
  rcases s with ⟨run, frontier, accepted, input, live, closed⟩
  cases frontier <;> cases closed <;> simp [recover, crash]

/-- A finite successful continuation exists at EVERY reachable commit prefix,
including after draining but before injecting and after completion before close.
These operations are the recovery dispatcher; irrelevant operations stutter. -/
def finishRecovery (s : State) : State :=
  close (complete (respond (inject (drain (recover (crash s))))))

theorem recovery_continuation_is_legal (h : Reachable run s) :
    Reachable run (finishRecovery s) := by
  exact .next (.next (.next (.next (.next (.next (.next h (.crash _))
    (.recover _)) (.drain _)) (.inject _)) (.respond _)) (.complete _)) (.close _)

theorem every_prefix_recoverable (h : Reachable run s) :
    (finishRecovery s).closed = true ∧
    (finishRecovery s).run = run ∧
    (s.accepted = true → (finishRecovery s).input = .applied) := by
  have safe := reachable_safe h
  have identity := reachable_identity h
  rcases s with ⟨id, frontier, accepted, input, live, closed⟩
  cases frontier <;> cases accepted <;> cases input <;> cases live <;> cases closed <;>
    simp_all [Safe, finishRecovery, crash, recover, drain, inject, respond, complete, close]

/-- The old dispatcher sends a committed candidate to an input-only driver;
that driver rejects the frontier. The candidate itself has not been lost. -/
def oldDriverAccepts (s : State) : Bool := s.frontier == .input

theorem old_dispatch_rejects_reachable_candidate :
    Reachable run (respond (initial run)) ∧
    oldDriverAccepts (respond (initial run)) = false := by
  exact ⟨.next .initial (.respond _), rfl⟩

end SessionContract.RecoveryFrontier
