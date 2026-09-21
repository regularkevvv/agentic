import SessionContract.Recovery

/-! # Progress
Lift finite recovery to eventual resolution under fair successful scheduling.
Serving another input cannot increase the target's rank; arbitrary finite waits
are allowed. Perpetual failure or starvation is explicitly outside this theorem.
-/

namespace SessionContract

/-- Serving another input cannot move this input backwards in the reference
Worker. This is stronger than proving a hand-picked successful recovery trace. -/
theorem tick_rank_nonincreasing (s : State n) (c d : CommandId n) :
    rank (tick s d) c ≤ rank s c := by
  cases submitted : s.submitted d with
  | none => simp [tick, nextAction, submitted, step, execute]
  | some p =>
    by_cases done : Resolved s d
    · simp [tick, nextAction, submitted, done, step, execute]
    · cases owner : s.owner with
      | none =>
        have action : nextAction s d = .acquire := by simp [nextAction, submitted, done, owner]
        rw [tick, action]
        simp only [step, execute]
        split
        · simp only [rank, owner]
          all_goals (repeat' split)
          all_goals simp_all [Resolved]
          all_goals omega
        · exact Nat.le_refl _
      | some g =>
        by_cases closed : s.handles g = false
        · have action : nextAction s d = .openSession g := by
            simp [nextAction, submitted, done, owner, closed]
          rw [tick, action]
          simp [step, execute, Owns, owner, rank, put, closed, Resolved]
          all_goals (repeat' split)
          all_goals simp_all
          all_goals omega
        · have opened : Opened s g := ⟨owner, by cases hh : s.handles g <;> simp_all⟩
          cases journal : s.journal d with
          | none =>
            have action : nextAction s d = .accept g d p := by
              simp [nextAction, submitted, done, owner, closed, journal]
            rw [tick, action]
            simp only [step, execute, opened, journal, ↓reduceIte]
            split
            · rename_i allowed
              by_cases same : c = d
              · subst c
                simp [rank, submitted, owner, closed, journal, put, Resolved, allowed.1]
              · simp [rank, owner, put, same, Resolved]
            · exact Nat.le_refl _
          | some q =>
            by_cases pending : s.mailbox d = true
            · have action : nextAction s d = .acknowledge g d := by
                simp [nextAction, submitted, done, owner, closed, journal, pending]
              rw [tick, action]
              have owns : Owns s g := owner
              simp only [step, execute, owns, journal, Option.isSome_some, and_self, ↓reduceIte]
              by_cases same : c = d
              · subst c
                simp [rank, submitted, owner, closed, journal, pending, put, Resolved]
                split <;> decide
              · simp [rank, put, same, Resolved]
            · have action : nextAction s d = .settle g d := by
                simp [nextAction, submitted, done, owner, closed, journal, pending]
              rw [tick, action]
              simp only [step, execute, opened, journal, Option.isSome_some, and_self, ↓reduceIte]
              by_cases same : c = d
              · subst c
                have empty : s.mailbox d = false := by cases hm : s.mailbox d <;> simp_all
                simp [rank, put, Resolved, empty]
              · simp [rank, put, same, Resolved]

/-- `none` is an arbitrarily long I/O wait. A scheduled tick is one successful
primitive, not an entire atomic run. Provider completion is needed for settle.
Crashes may occur in the preceding history; this models an eventual healthy suffix. -/
def FollowsWorker (trace : Nat → State n) (schedule : Nat → Option (CommandId n)) : Prop :=
  ∀ t, trace (t + 1) = match schedule t with
    | none => trace t
    | some c => tick (trace t) c

/-- Explicit starvation freedom, independently stated in terms of scheduling,
not as an assumption that commands eventually finish. -/
def Fair (schedule : Nat → Option (CommandId n)) : Prop :=
  ∀ c start, ∃ t, start ≤ t ∧ schedule t = some c

theorem trace_safe (trace : Nat → State n) (schedule : Nat → Option (CommandId n))
    (follows : FollowsWorker trace schedule) (safe : Safe (trace 0)) : ∀ t, Safe (trace t) := by
  intro t
  induction t with
  | zero => exact safe
  | succ t ih =>
    rw [follows t]
    cases schedule t with
    | none => exact ih
    | some c => exact tick_safe _ ih c

theorem trace_submission (trace : Nat → State n) (schedule : Nat → Option (CommandId n))
    (follows : FollowsWorker trace schedule) (c : CommandId n) (p : Payload)
    (submitted : (trace 0).submitted c = some p) : ∀ t, (trace t).submitted c = some p := by
  intro t
  induction t with
  | zero => exact submitted
  | succ t ih =>
    rw [follows t]
    cases schedule t with
    | none => exact ih
    | some d => exact submission_persistent _ (nextAction _ d) c p ih

theorem trace_rank_nonincreasing (trace : Nat → State n)
    (schedule : Nat → Option (CommandId n)) (follows : FollowsWorker trace schedule)
    (c : CommandId n) (t u : Nat) (order : t ≤ u) : rank (trace u) c ≤ rank (trace t) c := by
  have adjacent : ∀ k, rank (trace (k + 1)) c ≤ rank (trace k) c := by
    intro k
    rw [follows k]
    cases schedule k with
    | none => exact Nat.le_refl _
    | some d => exact tick_rank_nonincreasing _ c d
  obtain ⟨delta, rfl⟩ := Nat.exists_eq_add_of_le order
  induction delta with
  | zero => simp
  | succ delta ih => exact Nat.le_trans (adjacent (t + delta)) (ih (by omega))

/-- Liveness: fair primitive scheduling plus responsive dependencies implies
eventual settlement. Neither completion nor a decreasing rank is assumed in Fair. -/
theorem fair_worker_eventually_settles (trace : Nat → State n)
    (schedule : Nat → Option (CommandId n)) (follows : FollowsWorker trace schedule)
    (fair : Fair schedule) (safe : Safe (trace 0))
    (c : CommandId n) (p : Payload) (submitted : (trace 0).submitted c = some p) :
    ∀ start, ∃ t, start ≤ t ∧ Resolved (trace t) c := by
  have allSafe := trace_safe trace schedule follows safe
  have allSubmitted := trace_submission trace schedule follows c p submitted
  intro start
  generalize hr : rank (trace start) c = r
  induction r using Nat.strongRecOn generalizing start with
  | ind r ih =>
      obtain ⟨u, later, scheduled⟩ := fair c start
      by_cases done : Resolved (trace u) c
      · exact ⟨u, later, done⟩
      · have decreases := tick_decreases (trace u) (allSafe u) c p (allSubmitted u) done
        have happens : trace (u + 1) = tick (trace u) c := by simp [follows u, scheduled]
        have bounded := trace_rank_nonincreasing trace schedule follows c start u later
        rw [hr] at bounded
        have lower : rank (trace (u + 1)) c < r := by
          rw [happens]
          exact Nat.lt_of_lt_of_le decreases bounded
        obtain ⟨v, afterU, completes⟩ := ih (rank (trace (u + 1)) c) lower (u + 1) rfl
        exact ⟨v, by omega, completes⟩

end SessionContract
