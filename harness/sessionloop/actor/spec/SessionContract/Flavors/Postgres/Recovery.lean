import SessionContract.Flavors.Postgres.Transactions

/-! # Transaction-level conservation and recovery
Connect every microstep history to a sequence of committed contract actions.
Conversely construct real lock/write/commit histories for contract recovery;
existence of an abstract continuation alone is not the concrete recovery claim.
Progress is conditional on fair, successfully completed worker transactions.
-/

namespace SessionContract.Flavors.Postgres

theorem Steps.trans {s t u : Database n} (first : Steps s t) (second : Steps t u) : Steps s u := by
  induction second with
  | refl => exact first
  | next _ last ih => exact .next ih last

theorem step_linearizes {s t : Database n} (h : Step s t) (valid : LockedSnapshot s) :
    ∃ actions, view t = run (view s) actions := by
  have corresponding := step_refines h valid
  cases corresponding with
  | internal same => exact ⟨[], same⟩
  | commit a r corresponding => exact ⟨[a], (congrArg Prod.fst corresponding).symm⟩
  | batch actions corresponding => exact ⟨actions, corresponding.symm⟩

theorem history_linearizes {s t : Database n} (history : Steps s t) (valid : LockedSnapshot s) :
    ∃ actions, view t = run (view s) actions := by
  induction history with
  | refl => exact ⟨[], rfl⟩
  | next earlier last ih =>
      obtain ⟨before, hb⟩ := ih
      obtain ⟨after, ha⟩ := step_linearizes last (steps_keep_snapshot earlier valid)
      exact ⟨before ++ after, by simpa [run, List.foldl_append, hb] using ha⟩

theorem submitted_data_survives {s t : Database n} (history : Steps s t)
    (valid : LockedSnapshot s) (c : CommandId n) (p : Payload)
    (submitted : s.committed.commands.identity c = some p) :
    t.committed.commands.identity c = some p := by
  obtain ⟨actions, same⟩ := history_linearizes history valid
  change (view t).submitted c = some p
  rw [same]
  exact run_submission_persistent (view s) actions c p submitted

theorem journal_data_survives {s t : Database n} (history : Steps s t)
    (valid : LockedSnapshot s) (c : CommandId n) (p : Payload)
    (accepted : s.committed.journal.acceptance c = some p) :
    t.committed.journal.acceptance c = some p := by
  obtain ⟨actions, same⟩ := history_linearizes history valid
  change (view t).journal c = some p
  rw [same]
  exact run_journal_persistent (view s) actions c p accepted

theorem no_lost_committed_input {s : Database n} (history : Steps {} s)
    (c : CommandId n) (p : Payload) (submitted : s.committed.commands.identity c = some p) :
    s.committed.commands.pending c = true ∨ s.committed.journal.acceptance c = some p :=
  (reachable_safe history).conservation c p submitted

/-- A complete healthy transaction, including the four private statement phases
and reply disposal. This constructively connects recovery to the microstep model. -/
theorem transaction_realizable (r : Rows n) (a : Action n) :
    Steps { committed := r } { committed := (calculate r a).1 } := by
  let start : Database n := { committed := r }
  let t0 : Transaction n := { client := 0, action := a, before := r }
  let t1 : Transaction n := { t0 with phase := .checked }
  let t2 : Transaction n := { t0 with phase := .commandsWritten }
  let t3 : Transaction n := { t0 with phase := .journalWritten }
  let t4 : Transaction n := { t0 with phase := .readyToCommit }
  let s1 := enqueue start 0 a
  let s2 := lockRow s1 0 a
  let s3 := advance s2 t0
  let s4 := advance s3 t1
  let s5 := advance s4 t2
  let s6 := advance s5 t3
  let s7 := commit s6 t4
  have h1 : Step start s1 := .enqueue _ _ _ rfl rfl (by intro t h; cases h)
  have h2 : Step s1 s2 := .lock _ _ _ rfl (by simp [s1, enqueue])
  have h3 : Step s2 s3 := .advance _ _ rfl
  have h4 : Step s3 s4 := .advance _ _ rfl
  have h5 : Step s4 s5 := .advance _ _ rfl
  have h6 : Step s5 s6 := .advance _ _ rfl
  have h7 : Step s6 s7 := .commit _ _ rfl rfl
  have history := Steps.next (Steps.next (Steps.next (Steps.next (Steps.next
    (Steps.next (Steps.next (Steps.refl start) h1) h2) h3) h4) h5) h6) h7
  have finished : loseReply s7 0 = ({ committed := (calculate r a).1 } : Database n) := by
    simp [s7, s6, s5, s4, s3, s2, s1, start, t4, t0, commit,
      Transaction.draft, advance, lockRow, enqueue, loseReply]
    constructor <;> funext client <;> simp [put]
  rw [← finished]
  exact .next history (.loseReply _ _)

theorem recovery_actions_realizable (r : Rows n) (actions : List (Action n)) :
    Steps { committed := r } { committed := (rowAdapter n).run r actions } := by
  induction actions generalizing r with
  | nil => exact .refl _
  | cons a rest ih =>
      exact (transaction_realizable r a).trans (ih (calculate r a).1)

/-- Crash at ANY SQL microstep, not just an abstract operation boundary, leaves
a finite concrete transaction continuation to durable resolution and cleanup. -/
theorem crash_at_every_sql_phase_recoverable {s : Database n} (history : Steps {} s)
    (c : CommandId n) (p : Payload) (submitted : s.committed.commands.identity c = some p) :
    ∃ recovered : Database n, Steps (crash s) recovered ∧ Resolved (view recovered) c := by
  let expired := (calculate s.committed .expire).1
  have safe : Safe (project expired) := by
    rw [show project expired = step (view s) .expire from calculate_project _ _]
    exact step_safe _ (reachable_safe history) .expire
  have retained : (project expired).submitted c = some p := by
    rw [show project expired = step (view s) .expire from calculate_project _ _]
    exact submission_persistent _ .expire c p submitted
  obtain ⟨actions, done⟩ := every_adapter_recoverable (rowAdapter n) expired safe c p retained
  exact ⟨{ committed := (rowAdapter n).run expired actions },
    (transaction_realizable s.committed .expire).trans (recovery_actions_realizable expired actions), done⟩

theorem pending_discoverable_after_release (r : Rows n) (g : Lease) (c : CommandId n)
    (pending : r.commands.pending c = true) (released : (calculate r (.release g)).2 = .ok) :
    ready (calculate r (.release g)).1 := by
  have result := pending_ready_after_release (project r) g c pending
    (by simpa [calculate_reply] using released)
  change Ready (project (calculate r (.release g)).1)
  rw [calculate_project]
  exact result

theorem unfinished_discoverable_after_expiry {s : Database n} (history : Steps {} s)
    (c : CommandId n) (p : Payload) (accepted : s.committed.journal.acceptance c = some p)
    (unfinished : s.committed.journal.resolved c = false) :
    ready (calculate s.committed .expire).1 := by
  change Ready (project (calculate s.committed .expire).1)
  rw [calculate_project]
  exact unfinished_ready_after_expiry (view s) (reachable_safe history) c p accepted unfinished

/-- Progress at completed-transaction boundaries. Fairness requires successful
primitive service and eventual harness resolution, not merely polling forever.
transaction_realizable supplies the concrete microsteps for every boundary. -/
theorem fair_committed_worker_progresses (trace : Nat → Rows n)
    (schedule : Nat → Option (CommandId n))
    (follows : ∀ t, trace (t + 1) = match schedule t with
      | none => trace t
      | some c => (calculate (trace t) (nextAction (project (trace t)) c)).1)
    (fair : Fair schedule) (safe : Safe (project (trace 0)))
    (c : CommandId n) (p : Payload) (submitted : (trace 0).commands.identity c = some p) :
    ∀ start, ∃ t, start ≤ t ∧ Resolved (project (trace t)) c :=
  every_adapter_progresses (rowAdapter n) trace schedule follows fair safe c p submitted

end SessionContract.Flavors.Postgres
