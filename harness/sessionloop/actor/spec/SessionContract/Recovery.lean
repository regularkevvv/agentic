import SessionContract.Safety

/-! # Recovery
Define the reference worker and a decreasing rank. Construct a finite continuation
to resolution AND mailbox cleanup from every safe submitted state, including a
crash at any history prefix. Existence of recovery is distinct from scheduling it.
-/

namespace SessionContract

def Resolved (s : State n) (c : CommandId n) : Prop :=
  s.settled c = true ∧ s.mailbox c = false

instance (s : State n) (c : CommandId n) : Decidable (Resolved s c) :=
  inferInstanceAs (Decidable (s.settled c = true ∧ s.mailbox c = false))

/-- The small reference Worker's next successful action for one eligible input.
Waiting for I/O is `idle`. `settle` represents a harness-issued durable outcome,
not permission for the Worker to invent a model/tool result. -/
def nextAction (s : State n) (c : CommandId n) : Action n :=
  match s.submitted c with
  | none => .idle
  | some p =>
      if Resolved s c then .idle
      else match s.owner with
      | none => .acquire
      | some g =>
          if s.handles g = false then .openSession g
          else match s.journal c with
          | none => .accept g c p
          | some _ => if s.mailbox c = true then .acknowledge g c else .settle g c

def tick (s : State n) (c : CommandId n) : State n := step s (nextAction s c)

/-- A potential for progress, not a bound on LLM latency or individual I/O calls. -/
def rank (s : State n) (c : CommandId n) : Nat :=
  if (s.submitted c).isNone ∨ Resolved s c then 0
  else match s.owner with
  | none => 5
  | some g =>
      if s.handles g = false then 4
      else if (s.journal c).isNone then 3
      else if s.mailbox c = true then 2 else 1

theorem unfinished_has_work (s : State n) (h : Safe s) (c : CommandId n) (p : Payload)
    (submitted : s.submitted c = some p) (unfinished : s.settled c = false) : Work s := by
  rcases h.conservation c p submitted with pending | accepted
  · exact Or.inr ⟨c, pending⟩
  · exact Or.inl (h.unfinished_discoverable c p accepted unfinished)

theorem outstanding_has_work (s : State n) (h : Safe s) (c : CommandId n) (p : Payload)
    (submitted : s.submitted c = some p) (unresolved : ¬ Resolved s c) : Work s := by
  by_cases pending : s.mailbox c = true
  · exact Or.inr ⟨c, pending⟩
  · have empty : s.mailbox c = false := by cases hm : s.mailbox c <;> simp_all
    have unfinished : s.settled c = false := by
      cases hs : s.settled c <;> simp_all [Resolved]
    exact unfinished_has_work s h c p submitted unfinished

theorem rank_zero_iff_resolved (s : State n) (c : CommandId n) (p : Payload)
    (submitted : s.submitted c = some p) : rank s c = 0 ↔ Resolved s c := by
  simp only [rank, submitted, Option.isNone_some, Bool.false_eq_true, false_or]
  split
  · simp_all
  · cases s.owner <;> simp_all
    all_goals (repeat' split)
    all_goals simp_all

theorem tick_safe (s : State n) (h : Safe s) (c : CommandId n) : Safe (tick s c) :=
  step_safe s h (nextAction s c)

/-- From any invariant state, an eligible input has a strictly decreasing
recovery path. Other requests need not stop submitting new inputs. -/
theorem tick_decreases (s : State n) (h : Safe s) (c : CommandId n) (p : Payload)
    (submitted : s.submitted c = some p) (unresolved : ¬ Resolved s c) :
    rank (tick s c) c < rank s c := by
  have work := outstanding_has_work s h c p submitted unresolved
  cases owner : s.owner with
  | none =>
      have ready : Ready s := ⟨owner, work⟩
      have action : nextAction s c = .acquire := by simp [nextAction, submitted, unresolved, owner]
      rw [tick, action]
      simp only [step, execute, ready, ↓reduceIte, rank, owner]
      all_goals (repeat' split)
      all_goals simp_all [Resolved]
      all_goals omega
  | some g =>
      by_cases closed : s.handles g = false
      · have action : nextAction s c = .openSession g := by
          simp [nextAction, submitted, unresolved, owner, closed]
        rw [tick, action]
        simp only [step, execute, Owns, owner, ↓reduceIte, rank, put_same]
        all_goals (repeat' split)
        all_goals simp_all [Resolved]
        all_goals omega
      · have opened : Opened s g := ⟨owner, by cases hh : s.handles g <;> simp_all⟩
        cases journal : s.journal c with
        | none =>
            have pending : s.mailbox c = true := by
              rcases h.conservation c p submitted with hp | hj
              · exact hp
              · simp [journal] at hj
            have action : nextAction s c = .accept g c p := by
              simp [nextAction, submitted, unresolved, owner, closed, journal]
            rw [tick, action]
            simp [step, execute, opened, journal, submitted, pending, rank, owner, closed, put, Resolved]
        | some q =>
            by_cases pending : s.mailbox c = true
            · have action : nextAction s c = .acknowledge g c := by
                simp [nextAction, submitted, unresolved, owner, closed, journal, pending]
              rw [tick, action]
              simp [step, execute, Owns, owner, journal, rank, submitted, closed, pending, put, Resolved]
              split <;> decide
            · have empty : s.mailbox c = false := by cases hm : s.mailbox c <;> simp_all
              have unfinished : s.settled c = false := by
                cases hs : s.settled c <;> simp_all [Resolved]
              have action : nextAction s c = .settle g c := by
                simp [nextAction, submitted, unresolved, owner, closed, journal, empty]
              rw [tick, action]
              simp [step, execute, opened, journal, rank, submitted, unfinished, empty,
                owner, closed, put, Resolved]

/-- Constructive recoverability: there exists a finite permitted continuation
from EVERY safe state, not only the hand-written regression examples. -/
theorem recoverable (s : State n) (h : Safe s) (c : CommandId n) (p : Payload)
    (submitted : s.submitted c = some p) :
    ∃ actions : List (Action n), Resolved (run s actions) c := by
  generalize hr : rank s c = r
  induction r using Nat.strongRecOn generalizing s with
  | ind r ih =>
      by_cases done : Resolved s c
      · exact ⟨[], done⟩
      · have lower := tick_decreases s h c p submitted done
        rw [hr] at lower
        have persisted := submission_persistent s (nextAction s c) c p submitted
        obtain ⟨rest, finishes⟩ := ih (rank (tick s c) c) lower (tick s c)
          (tick_safe s h c) persisted rfl
        exact ⟨nextAction s c :: rest, finishes⟩

/-- Crashing at any point of any finite history still leaves a recovery path.
Expiry is explicit: a dead worker's durable ownership does not vanish on crash. -/
theorem crash_at_every_prefix_recoverable (s : State n) (h : Safe s)
    (history : List (Action n)) (g : Lease) (c : CommandId n) (p : Payload)
    (submitted : s.submitted c = some p) :
    ∃ recovery : List (Action n),
      Resolved (run (step (step (run s history) (.crash g)) .expire) recovery) c := by
  apply recoverable
  · exact step_safe _ (step_safe _ (run_safe s h history) (.crash g)) .expire
  · exact submission_persistent _ .expire c p
      (submission_persistent _ (.crash g) c p (run_submission_persistent s history c p submitted))

end SessionContract
