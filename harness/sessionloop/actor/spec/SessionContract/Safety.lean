import SessionContract.Model

/-! # Safety
Prove invariant preservation for each atomic action, then every finite history.
Derive conservation, immutable acceptance, stale-owner rejection and durable
replies without assuming that an operation's input state already has its result.
-/

namespace SessionContract

theorem initial_safe : Safe (initial : State n) := by
  constructor <;> simp [initial]

theorem safe_same_durable {s t : State n} (same : t.toDurable = s.toDurable)
    (h : Safe s) : Safe t := by
  cases s
  cases t
  cases same
  rcases h with ⟨hp, hj, hc, hs, hu, ho⟩
  exact ⟨hp, hj, hc, hs, hu, ho⟩

theorem submit_safe (s : State n) (h : Safe s) (c : CommandId n) (p : Payload) :
    Safe (step s (.submit c p)) := by
  simp only [step, execute]
  split
  · exact h
  · rename_i missing
    rcases h with ⟨hp, hj, hc, hs, hu, ho⟩
    constructor
    · intro d hd
      by_cases eq : d = c
      · subst d; exact ⟨p, by simp⟩
      · simpa [put, eq] using hp d (by simpa [put, eq] using hd)
    · intro d q hd
      by_cases eq : d = c
      · subst d; have := hj c q hd; simp [missing] at this
      · simpa [put, eq] using hj d q hd
    · intro d q hd
      by_cases eq : d = c
      · subst d; exact Or.inl (by simp)
      · simpa [put, eq] using hc d q (by simpa [put, eq] using hd)
    · exact hs
    · exact hu
    · exact ho

theorem acquire_safe (s : State n) (h : Safe s) : Safe (step s .acquire) := by
  simp only [step, execute]
  split
  · rcases h with ⟨hp, hj, hc, hs, _, _⟩
    constructor
    · exact hp
    · exact hj
    · exact hc
    · exact hs
    · simp
    · simp
  · exact h

theorem open_safe (s : State n) (h : Safe s) (g : Lease) :
    Safe (step s (.openSession g)) := by
  simp only [step, execute]
  split
  · exact safe_same_durable (s := s) rfl h
  · exact h

theorem accept_safe (s : State n) (h : Safe s) (g : Lease) (c : CommandId n) (p : Payload) :
    Safe (step s (.accept g c p)) := by
  simp only [step, execute]
  split
  · split
    · exact h
    · split
      · rename_i allowed
        rcases h with ⟨hp, hj, hc, hs, _, ho⟩
        constructor
        · exact hp
        · intro d q hd
          by_cases eq : d = c
          · subst d; simp only [put_same, Option.some.injEq] at hd
            subst q; exact allowed.2
          · exact hj d q (by simpa [put, eq] using hd)
        · intro d q hd
          by_cases eq : d = c
          · subst d; exact Or.inl allowed.1
          · simpa [put, eq] using hc d q hd
        · intro d hd
          obtain ⟨q, hq⟩ := hs d hd
          by_cases eq : d = c
          · subst d; exact ⟨p, by simp⟩
          · exact ⟨q, by simpa [put, eq] using hq⟩
        · simp
        · exact ho
      · exact h
  · exact h

theorem acknowledge_safe (s : State n) (h : Safe s) (g : Lease) (c : CommandId n) :
    Safe (step s (.acknowledge g c)) := by
  simp only [step, execute]
  split
  · rename_i allowed
    rcases h with ⟨hp, hj, hc, hs, hu, ho⟩
    constructor
    · intro d hd
      by_cases eq : d = c
      · subst d; simp at hd
      · exact hp d (by simpa [put, eq] using hd)
    · exact hj
    · intro d p hd
      by_cases eq : d = c
      · subst d
        cases hjc : s.journal c with
        | none => simp [hjc] at allowed
        | some q =>
          have same := hj c q hjc
          have : q = p := Option.some.inj (same.symm.trans hd)
          subst q; exact Or.inr rfl
      · simpa [put, eq] using hc d p hd
    · exact hs
    · exact hu
    · exact ho
  · exact h

theorem settle_safe (s : State n) (h : Safe s) (g : Lease) (c : CommandId n) :
    Safe (step s (.settle g c)) := by
  simp only [step, execute]
  split
  · rename_i allowed
    rcases h with ⟨hp, hj, hc, hs, hu, ho⟩
    constructor
    · exact hp
    · exact hj
    · exact hc
    · intro d hd
      by_cases eq : d = c
      · subst d
        cases he : s.journal c with
        | none => simp [he] at allowed
        | some p => exact ⟨p, rfl⟩
      · exact hs d (by simpa [put, eq] using hd)
    · intro d p hd unfinished
      by_cases eq : d = c
      · subst d; simp at unfinished
      · exact hu d p hd (by simpa [put, eq] using unfinished)
    · exact ho
  · exact h

theorem release_safe (s : State n) (h : Safe s) (g : Lease) :
    Safe (step s (.release g)) := by
  simp only [step, execute]
  split
  · rename_i allowed
    rcases h with ⟨hp, hj, hc, hs, _, _⟩
    constructor
    · exact hp
    · exact hj
    · exact hc
    · exact hs
    · intro c p accepted unfinished
      change s.journal c = some p at accepted
      change s.settled c = false at unfinished
      rcases allowed.2.2 c with absent | done
      · simp [accepted] at absent
      · simp [unfinished] at done
    · simp
  · exact h

/-- All actions, including failure events and rejected/stale attempts, preserve
the invariant. The invariant is a conclusion, not an action precondition. -/
theorem step_safe (s : State n) (h : Safe s) (a : Action n) : Safe (step s a) := by
  cases a with
  | submit c p => exact submit_safe s h c p
  | acquire => exact acquire_safe s h
  | openSession g => exact open_safe s h g
  | accept g c p => exact accept_safe s h g c p
  | acknowledge g c => exact acknowledge_safe s h g c
  | settle g c => exact settle_safe s h g c
  | release g => exact release_safe s h g
  | observeEmpty g => exact safe_same_durable (s := s) rfl h
  | closeSession g => exact safe_same_durable (s := s) rfl h
  | crash g => exact safe_same_durable (s := s) rfl h
  | idle => exact h
  | expire =>
    rcases h with ⟨hp, hj, hc, hs, hu, _⟩
    exact ⟨hp, hj, hc, hs, hu, by simp [step, execute]⟩

theorem reachable_safe {s : State n} (h : Reachable s) : Safe s := by
  induction h with
  | initial => exact initial_safe
  | next _ a ih => exact step_safe _ ih a

theorem run_safe (s : State n) (h : Safe s) (actions : List (Action n)) :
    Safe (run s actions) := by
  induction actions generalizing s with
  | nil => exact h
  | cons a rest ih => exact ih (step s a) (step_safe s h a)

theorem no_lost_submission {s : State n} (h : Reachable s)
    (c : CommandId n) (p : Payload) (accepted : s.submitted c = some p) :
    s.mailbox c = true ∨ s.journal c = some p :=
  (reachable_safe h).conservation c p accepted

theorem one_owner (s : State n) (g h : Lease) (hg : Owns s g) (hh : Owns s h) :
    g = h := Option.some.inj (hg.symm.trans hh)

theorem stale_accept_rejected (s : State n) (g : Lease) (c : CommandId n) (p : Payload)
    (stale : ¬ Owns s g) : execute s (.accept g c p) = (s, .unavailable) := by
  simp [execute, Opened, stale]

theorem stale_acknowledge_rejected (s : State n) (g : Lease) (c : CommandId n)
    (stale : ¬ Owns s g) : execute s (.acknowledge g c) = (s, .unavailable) := by
  simp [execute, stale]

theorem stale_settle_rejected (s : State n) (g : Lease) (c : CommandId n)
    (stale : ¬ Owns s g) : execute s (.settle g c) = (s, .unavailable) := by
  simp [execute, Opened, stale]

theorem stale_release_rejected (s : State n) (g : Lease)
    (stale : ¬ Owns s g) : execute s (.release g) = (s, .unavailable) := by
  simp [execute, stale]

theorem crash_preserves_durable (s : State n) (g : Lease) :
    (step s (.crash g)).toDurable = s.toDurable := rfl

theorem retry_submission (s : State n) (c : CommandId n) (p : Payload)
    (committed : s.submitted c = some p) :
    execute s (.submit c p) = (s, .duplicate) := by simp [execute, committed]

theorem conflicting_submission (s : State n) (c : CommandId n) (p q : Payload)
    (committed : s.submitted c = some p) (different : p ≠ q) :
    execute s (.submit c q) = (s, .conflict) := by simp [execute, committed, different]

theorem retry_acceptance (s : State n) (g : Lease) (c : CommandId n) (p : Payload)
    (opened : Opened s g) (committed : s.journal c = some p) :
    execute s (.accept g c p) = (s, .accepted) := by simp [execute, opened, committed]

theorem accepted_reply_is_durable (s : State n) (g : Lease) (c : CommandId n) (p : Payload)
    (receipt : (execute s (.accept g c p)).2 = .accepted) :
    (step s (.accept g c p)).journal c = some p := by
  simp only [step, execute] at *
  split at *
  · split at *
    · split at receipt <;> simp_all
    · split at * <;> simp_all
  · simp_all

theorem deleted_mailbox_has_acceptance (s : State n) (g : Lease) (c : CommandId n)
    (wasPending : s.mailbox c = true)
    (deleted : (step s (.acknowledge g c)).mailbox c = false) :
    ∃ p, s.journal c = some p := by
  simp only [step, execute] at deleted
  split at deleted
  · rename_i allowed
    cases hj : s.journal c with
    | none => simp [hj] at allowed
    | some p => exact ⟨p, rfl⟩
  · simp [wasPending] at deleted

/-- Deleting the last mailbox item cannot hide unfinished journal work. -/
theorem unfinished_ready_after_expiry (s : State n) (h : Safe s)
    (c : CommandId n) (p : Payload) (accepted : s.journal c = some p)
    (unfinished : s.settled c = false) : Ready (step s .expire) := by
  exact ⟨rfl, Or.inl (h.unfinished_discoverable c p accepted unfinished)⟩

/-- A pending item is discoverable even if a concurrent release cleared the flag. -/
theorem pending_ready_after_release (s : State n) (g : Lease) (c : CommandId n)
    (pending : s.mailbox c = true) (released : (execute s (.release g)).2 = .ok) :
    Ready (step s (.release g)) := by
  by_cases allowed : Owns s g ∧ s.handles g = false ∧ Quiet s
  · simpa [step, execute, allowed] using
      (show Ready { s with owner := none, needsExecution := false } from
        ⟨rfl, Or.inr ⟨c, pending⟩⟩)
  · simp [execute, allowed] at released

theorem submission_persistent (s : State n) (a : Action n) (c : CommandId n) (p : Payload)
    (committed : s.submitted c = some p) : (step s a).submitted c = some p := by
  cases a <;> simp only [step, execute]
  all_goals (repeat' split)
  all_goals simp_all [put]
  all_goals grind

theorem journal_persistent (s : State n) (a : Action n) (c : CommandId n) (p : Payload)
    (committed : s.journal c = some p) : (step s a).journal c = some p := by
  cases a <;> simp only [step, execute]
  all_goals (repeat' split)
  all_goals simp_all [put]
  all_goals grind

theorem settlement_persistent (s : State n) (a : Action n) (c : CommandId n)
    (committed : s.settled c = true) : (step s a).settled c = true := by
  cases a <;> simp only [step, execute]
  all_goals (repeat' split)
  all_goals simp_all [put]
  all_goals grind

theorem generation_monotone (s : State n) (a : Action n) :
    s.generation ≤ (step s a).generation := by
  cases a <;> simp only [step, execute]
  all_goals (repeat' split)
  all_goals simp_all

theorem run_submission_persistent (s : State n) (actions : List (Action n))
    (c : CommandId n) (p : Payload) (committed : s.submitted c = some p) :
    (run s actions).submitted c = some p := by
  induction actions generalizing s with
  | nil => exact committed
  | cons a rest ih => exact ih (step s a) (submission_persistent s a c p committed)

theorem run_journal_persistent (s : State n) (actions : List (Action n))
    (c : CommandId n) (p : Payload) (committed : s.journal c = some p) :
    (run s actions).journal c = some p := by
  induction actions generalizing s with
  | nil => exact committed
  | cons a rest ih => exact ih (step s a) (journal_persistent s a c p committed)

/-- A fresh acceptance requires an empty receipt slot. Once filled, no later
interleaving can freshly accept that identity again, including after crashes. -/
theorem no_second_fresh_acceptance (s : State n) (actions : List (Action n))
    (c : CommandId n) (p : Payload) (committed : s.journal c = some p) :
    (run s actions).journal c ≠ none := by
  simp [run_journal_persistent s actions c p committed]

theorem submission_reply_is_durable (s : State n) (c : CommandId n) (p : Payload)
    (receipt : (execute s (.submit c p)).2 = .submitted ∨
               (execute s (.submit c p)).2 = .duplicate) :
    (step s (.submit c p)).submitted c = some p := by
  simp only [step, execute] at *
  split at *
  · split at receipt <;> simp_all
  · simp

theorem takeover_fences_previous_owner (s : State n) (safe : Safe s) (old : Lease)
    (owner : Owns s old) (work : Work s) (c : CommandId n) (p : Payload) :
    let successor := step (step s .expire) .acquire
    execute successor (.accept old c p) = (successor, .unavailable) := by
  have ready : Ready (step s .expire) := ⟨rfl, work⟩
  have oldGeneration := safe.owner_generation old owner
  apply stale_accept_rejected
  change ¬ (execute (step s .expire) .acquire).1.owner = some old
  simp only [execute, ready, ↓reduceIte]
  simp [step, execute, oldGeneration]

end SessionContract
