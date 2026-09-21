import SessionContract.Progress

/-! # Adapter contract
Connect concrete storage and replies to the protocol. An adapter supplies the
correspondence proofs; safety, recovery and conditional progress follow. Internal
instructions may stutter, but committed operations must match the model exactly.
-/

namespace SessionContract

/-- Adapter implementation contract. This is a proof-carrying specification,
not a claim that Go interfaces enforce these fields. Atomic operations may be
implemented with many internal instructions; a code proof must additionally
establish their linearization, response fidelity and crash correspondence. -/
structure Adapter (n : Nat) where
  Storage : Type
  initialState : Storage
  interpret : Storage → State n
  operate : Storage → Action n → Storage × Reply
  initial_correct : interpret initialState = initial
  operation_correct : ∀ s a,
    execute (interpret s) a = (interpret (operate s a).1, (operate s a).2)

def referenceAdapter : Adapter n where
  Storage := State n
  initialState := initial
  interpret := id
  operate := execute
  initial_correct := rfl
  operation_correct := by intros; rfl

def Adapter.run (impl : Adapter n) (s : impl.Storage)
    (actions : List (Action n)) : impl.Storage :=
  actions.foldl (fun state a => (impl.operate state a).1) s

theorem adapter_step_refines (impl : Adapter n) (s : impl.Storage) (a : Action n) :
    impl.interpret (impl.operate s a).1 = step (impl.interpret s) a := by
  exact (congrArg Prod.fst (impl.operation_correct s a)).symm

theorem adapter_run_refines (impl : Adapter n) (s : impl.Storage)
    (actions : List (Action n)) :
    impl.interpret (impl.run s actions) = run (impl.interpret s) actions := by
  induction actions generalizing s with
  | nil => rfl
  | cons a rest ih =>
      simp only [Adapter.run, run, List.foldl_cons] at *
      rw [ih, adapter_step_refines]

/-- The reusable theorem: an adapter supplies operation laws, not safety as an
assumption. Safety is inherited for every finite interleaving and failure trace. -/
theorem every_adapter_safe (impl : Adapter n) (actions : List (Action n)) :
    Safe (impl.interpret (impl.run impl.initialState actions)) := by
  rw [adapter_run_refines, impl.initial_correct]
  exact run_safe initial initial_safe actions

theorem adapter_reply_fidelity (impl : Adapter n) (s : impl.Storage) (a : Action n) :
    (impl.operate s a).2 = (execute (impl.interpret s) a).2 := by
  exact (congrArg Prod.snd (impl.operation_correct s a)).symm

theorem every_adapter_recoverable (impl : Adapter n) (s : impl.Storage)
    (safe : Safe (impl.interpret s)) (c : CommandId n) (p : Payload)
    (submitted : (impl.interpret s).submitted c = some p) :
    ∃ actions, Resolved (impl.interpret (impl.run s actions)) c := by
  obtain ⟨actions, recovered⟩ := recoverable (impl.interpret s) safe c p submitted
  exact ⟨actions, by simpa [adapter_run_refines] using recovered⟩

def Adapter.tick (impl : Adapter n) (s : impl.Storage) (c : CommandId n) :
    impl.Storage := (impl.operate s (nextAction (impl.interpret s) c)).1

theorem every_adapter_progresses (impl : Adapter n) (trace : Nat → impl.Storage)
    (schedule : Nat → Option (CommandId n))
    (follows : ∀ t, trace (t + 1) = match schedule t with
      | none => trace t | some c => impl.tick (trace t) c)
    (fair : Fair schedule) (safe : Safe (impl.interpret (trace 0)))
    (c : CommandId n) (p : Payload) (submitted : (impl.interpret (trace 0)).submitted c = some p) :
    ∀ start, ∃ t, start ≤ t ∧ Resolved (impl.interpret (trace t)) c := by
  have abstractFollows : FollowsWorker (fun t => impl.interpret (trace t)) schedule := by
    intro t
    dsimp only
    rw [follows t]
    cases hs : schedule t with
    | none => rfl
    | some d => exact adapter_step_refines impl (trace t) (nextAction _ d)
  exact fair_worker_eventually_settles _ schedule abstractFollows fair safe c p submitted

/-- Internal transport steps stutter; durable commit steps preserve replies too.
This admits blocking receive, polling, and split request/commit/response paths. -/
inductive AdapterStep (project : C → State n) : C → C → Prop where
  | internal {s t} (same : project t = project s) : AdapterStep project s t
  | commit {s t} (a : Action n) (r : Reply)
      (corresponds : Transition (project s) a r (project t)) : AdapterStep project s t
  /-- A transaction may commit several model primitives together, e.g. accept
  plus settle for a terminal rejection. Reply fidelity is a separate obligation
  for the compound API, just as it is for the receive flavor's response field. -/
  | batch {s t} (actions : List (Action n))
      (corresponds : run (project s) actions = project t) : AdapterStep project s t

theorem adapter_preserves_safety (project : C → State n) {s t : C}
    (h : AdapterStep project s t) (safe : Safe (project s)) : Safe (project t) := by
  cases h with
  | internal same => simpa [same] using safe
  | commit a r corresponds =>
      have effect : step (project s) a = project t := congrArg Prod.fst corresponds
      rw [← effect]
      exact step_safe _ safe a
  | batch actions corresponds =>
      rw [← corresponds]
      exact run_safe _ safe actions

end SessionContract
