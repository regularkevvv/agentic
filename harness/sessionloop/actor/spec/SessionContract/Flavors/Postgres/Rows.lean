import SessionContract.Adapter

/-! # PostgreSQL row algorithm
Independent column-oriented representations of the three per-session tables.
`identity` denotes exact immutable semantics, not a collision-prone hash. A SQL
digest implementation must discharge that additional representation obligation.
Handles and empty observations are proof witnesses for the native harness, NOT
database columns. This model starts after a stable journal binding exists.
-/

namespace SessionContract.Flavors.Postgres

structure SessionRow where
  fence : Nat := 0
  owner : Option Lease := none
  executionNeeded : Bool := false

structure CommandRows (n : Nat) where
  identity : CommandId n → Option Payload := fun _ => none
  /-- True means the pending envelope exists; false leaves only its identity. -/
  pending : CommandId n → Bool := fun _ => false

structure JournalRows (n : Nat) where
  acceptance : CommandId n → Option Payload := fun _ => none
  resolved : CommandId n → Bool := fun _ => false

structure Rows (n : Nat) where
  session : SessionRow := {}
  commands : CommandRows n := {}
  journal : JournalRows n := {}
  handles : Lease → Bool := fun _ => false
  emptyObservation : Lease → Bool := fun _ => false

def project (r : Rows n) : State n :=
  { submitted := r.commands.identity, mailbox := r.commands.pending,
    journal := r.journal.acceptance, settled := r.journal.resolved,
    needsExecution := r.session.executionNeeded, generation := r.session.fence,
    owner := r.session.owner, handles := r.handles, sawEmpty := r.emptyObservation }

def owns (r : Rows n) (g : Lease) : Prop := r.session.owner = some g
def opened (r : Rows n) (g : Lease) : Prop := owns r g ∧ r.handles g = true
def work (r : Rows n) : Prop := r.session.executionNeeded = true ∨ ∃ c, r.commands.pending c = true
def ready (r : Rows n) : Prop := r.session.owner = none ∧ work r
def quiet (r : Rows n) : Prop := ∀ c, r.journal.acceptance c = none ∨ r.journal.resolved c = true

instance (r : Rows n) (g : Lease) : Decidable (owns r g) := inferInstanceAs (Decidable (_ = _))
instance (r : Rows n) (g : Lease) : Decidable (opened r g) := inferInstanceAs (Decidable (_ ∧ _))
instance (r : Rows n) : Decidable (work r) := inferInstanceAs (Decidable (_ ∨ _))
instance (r : Rows n) : Decidable (ready r) := inferInstanceAs (Decidable (_ ∧ _))
instance (r : Rows n) : Decidable (quiet r) :=
  inferInstanceAs (Decidable (∀ c : CommandId n, r.journal.acceptance c = none ∨ r.journal.resolved c = true))

/-- Calculate the row updates and reply while holding the session row lock.
This does not call the abstract interpreter. Publishing the updates is a
separate COMMIT step in Transactions.lean. Expiry is a database-authorized
operation, never an inference from a worker's local clock. -/
def calculate (r : Rows n) (a : Action n) : Rows n × Reply :=
  match a with
  | .submit c p =>
      match r.commands.identity c with
      | some prior => (r, if prior = p then .duplicate else .conflict)
      | none =>
          ({ r with commands := { identity := put r.commands.identity c (some p),
                                  pending := put r.commands.pending c true } }, .submitted)
  | .acquire =>
      if ready r then
        ({ r with session := { fence := r.session.fence + 1,
                               owner := some (r.session.fence + 1), executionNeeded := true } },
          .acquired (r.session.fence + 1))
      else (r, .unavailable)
  | .openSession g =>
      if owns r g then ({ r with handles := put r.handles g true }, .ok)
      else (r, .unavailable)
  | .accept g c p =>
      if opened r g then
        match r.journal.acceptance c with
        | some prior => (r, if prior = p then .accepted else .conflict)
        | none =>
            if r.commands.pending c = true ∧ r.commands.identity c = some p then
              ({ r with journal := { r.journal with acceptance := put r.journal.acceptance c (some p) },
                        session := { r.session with executionNeeded := true } }, .accepted)
            else (r, .unavailable)
      else (r, .unavailable)
  | .acknowledge g c =>
      if owns r g ∧ (r.journal.acceptance c).isSome = true then
        ({ r with commands := { r.commands with pending := put r.commands.pending c false } }, .ok)
      else (r, .unavailable)
  | .settle g c =>
      if opened r g ∧ (r.journal.acceptance c).isSome = true then
        ({ r with journal := { r.journal with resolved := put r.journal.resolved c true } }, .ok)
      else (r, .unavailable)
  | .observeEmpty g =>
      ({ r with emptyObservation := put r.emptyObservation g (decide (∀ c, r.commands.pending c = false)) }, .ok)
  | .closeSession g => ({ r with handles := put r.handles g false }, .ok)
  | .release g =>
      if owns r g ∧ r.handles g = false ∧ quiet r then
        ({ r with session := { r.session with owner := none, executionNeeded := false } }, .ok)
      else (r, .unavailable)
  | .expire => ({ r with session := { r.session with owner := none } }, .ok)
  | .crash g =>
      ({ r with handles := put r.handles g false, emptyObservation := put r.emptyObservation g false }, .ok)
  | .idle => (r, .ok)

@[simp] theorem project_initial : project ({} : Rows n) = initial := rfl

/-- Row updates preserve the abstract operation AND its reply, for every state
and operation. Safety is derived later, not assumed as a guard here. -/
theorem calculate_correct (r : Rows n) (a : Action n) :
    execute (project r) a = (project (calculate r a).1, (calculate r a).2) := by
  cases a <;> simp only [calculate] <;> (repeat' split) <;>
    simp_all [execute, project, owns, opened, work, ready, quiet,
      Owns, Opened, Work, Ready, Quiet]
  all_goals first | rfl | grind

def rowAdapter (n : Nat) : Adapter n where
  Storage := Rows n
  initialState := {}
  interpret := project
  operate := calculate
  initial_correct := rfl
  operation_correct := calculate_correct

theorem calculate_project (r : Rows n) (a : Action n) :
    project (calculate r a).1 = step (project r) a :=
  (congrArg Prod.fst (calculate_correct r a)).symm

theorem calculate_reply (r : Rows n) (a : Action n) :
    (calculate r a).2 = (execute (project r) a).2 :=
  (congrArg Prod.snd (calculate_correct r a)).symm

theorem row_algorithm_safe (actions : List (Action n)) :
    Safe (project ((rowAdapter n).run {} actions)) := every_adapter_safe (rowAdapter n) actions

theorem stale_fence_rejected (r : Rows n) (g : Lease) (c : CommandId n) (p : Payload)
    (stale : r.session.owner ≠ some g) :
    calculate r (.accept g c p) = (r, .unavailable) := by
  simp [calculate, opened, owns, stale]

theorem stale_release_rejected (r : Rows n) (g : Lease)
    (stale : r.session.owner ≠ some g) :
    calculate r (.release g) = (r, .unavailable) := by
  simp [calculate, owns, stale]

theorem release_preserves_pending (r : Rows n) (g : Lease) :
    (calculate r (.release g)).1.commands = r.commands := by
  simp only [calculate]; split <;> rfl

theorem acknowledgment_preserves_identity (r : Rows n) (g : Lease) (c : CommandId n) :
    (calculate r (.acknowledge g c)).1.commands.identity = r.commands.identity := by
  simp only [calculate]; split <;> rfl

theorem same_identity_retry (r : Rows n) (c : CommandId n) (p : Payload)
    (known : r.commands.identity c = some p) :
    calculate r (.submit c p) = (r, .duplicate) := by simp [calculate, known]

end SessionContract.Flavors.Postgres
