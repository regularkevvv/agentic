import Std

/-!
An executable specification of ONE session's durable delivery protocol.
`n` is arbitrary: proofs are not a bounded state-space search. Commands use a
finite identity universe so the reference interpreter is executable. Any finite
history fits some `n`. Payloads stand for exact immutable command semantics,
including a resolved dispatch kind and run target, not a lossy content hash.

Each action is one linearization point, not one entire request or worker run.
A commit's reply may be lost; retrying the same action is explicitly permitted.
-/
namespace SessionContract

abbrev CommandId (n : Nat) := Fin n
abbrev Payload := Nat
abbrev Lease := Nat

def put {α β : Type} [DecidableEq α] (f : α → β) (key : α) (value : β) : α → β :=
  fun other => if other = key then value else f other

@[simp] theorem put_same {α β : Type} [DecidableEq α]
    (f : α → β) (key : α) (value : β) : put f key value key = value := by
  simp [put]

@[simp] theorem put_other {α β : Type} [DecidableEq α]
    (f : α → β) (key other : α) (value : β) (h : other ≠ key) :
    put f key value other = f other := by simp [put, h]

structure Durable (n : Nat) where
  /-- Immutable accepted submission identities; may be product records/dedup data. -/
  submitted : CommandId n → Option Payload := fun _ => none
  mailbox : CommandId n → Bool := fun _ => false
  /-- Durable acceptance facts, abstracting the harness journal, not UI messages. -/
  journal : CommandId n → Option Payload := fun _ => none
  settled : CommandId n → Bool := fun _ => false
  needsExecution : Bool := false
  generation : Nat := 0
  owner : Option Lease := none

structure State (n : Nat) extends Durable n where
  /-- Old handles can survive expiry; they must still fail every fenced mutation. -/
  handles : Lease → Bool := fun _ => false
  /-- A deliberately stale-able mailbox observation; never a source of authority. -/
  sawEmpty : Lease → Bool := fun _ => false

def initial : State n := { toDurable := {} }

def Owns (s : State n) (g : Lease) : Prop := s.owner = some g
def Opened (s : State n) (g : Lease) : Prop := Owns s g ∧ s.handles g = true
def Work (s : State n) : Prop := s.needsExecution = true ∨ ∃ c, s.mailbox c = true
def Ready (s : State n) : Prop := s.owner = none ∧ Work s
def Quiet (s : State n) : Prop := ∀ c, s.journal c = none ∨ s.settled c = true

instance (s : State n) (g : Lease) : Decidable (Owns s g) := inferInstanceAs (Decidable (s.owner = some g))
instance (s : State n) (g : Lease) : Decidable (Opened s g) := inferInstanceAs (Decidable (Owns s g ∧ s.handles g = true))
instance (s : State n) : Decidable (Work s) := inferInstanceAs (Decidable (s.needsExecution = true ∨ ∃ c, s.mailbox c = true))
instance (s : State n) : Decidable (Ready s) := inferInstanceAs (Decidable (s.owner = none ∧ Work s))
instance (s : State n) : Decidable (Quiet s) := inferInstanceAs (Decidable (∀ c, s.journal c = none ∨ s.settled c = true))

inductive Action (n : Nat) where
  | submit (command : CommandId n) (payload : Payload)
  | acquire
  | openSession (lease : Lease)
  | accept (lease : Lease) (command : CommandId n) (payload : Payload)
  | acknowledge (lease : Lease) (command : CommandId n)
  | settle (lease : Lease) (command : CommandId n)
  | observeEmpty (lease : Lease)
  | closeSession (lease : Lease)
  | release (lease : Lease)
  /-- Authority expiry is serialized with fenced commits, not a worker's clock. -/
  | expire
  | crash (lease : Lease)
  | idle
  deriving Repr, DecidableEq

inductive Reply where
  | submitted
  | duplicate
  | conflict
  | acquired (lease : Lease)
  | accepted
  | ok
  | unavailable
  deriving Repr, DecidableEq

/-- Replies describe the result of a commit. Receiving that reply is NOT atomic
with committing: transports may discard it and retry any action. -/
def execute (s : State n) (a : Action n) : State n × Reply :=
  match a with
  | .submit c p =>
      match s.submitted c with
      | some previous => (s, if previous = p then .duplicate else .conflict)
      | none =>
          ({ s with submitted := put s.submitted c (some p),
                    mailbox := put s.mailbox c true }, .submitted)
  | .acquire =>
      if Ready s then
        ({ s with generation := s.generation + 1, owner := some (s.generation + 1),
                  needsExecution := true }, .acquired (s.generation + 1))
      else (s, .unavailable)
  | .openSession g =>
      if Owns s g then ({ s with handles := put s.handles g true }, .ok)
      else (s, .unavailable)
  | .accept g c p =>
      if Opened s g then
        match s.journal c with
        | some previous => (s, if previous = p then .accepted else .conflict)
        | none =>
            if s.mailbox c = true ∧ s.submitted c = some p then
              ({ s with journal := put s.journal c (some p), needsExecution := true }, .accepted)
            else (s, .unavailable)
      else (s, .unavailable)
  | .acknowledge g c =>
      if Owns s g ∧ (s.journal c).isSome = true then
        ({ s with mailbox := put s.mailbox c false }, .ok)
      else (s, .unavailable)
  | .settle g c =>
      if Opened s g ∧ (s.journal c).isSome = true then
        ({ s with settled := put s.settled c true }, .ok)
      else (s, .unavailable)
  | .observeEmpty g =>
      ({ s with sawEmpty := put s.sawEmpty g (decide (∀ c, s.mailbox c = false)) }, .ok)
  | .closeSession g => ({ s with handles := put s.handles g false }, .ok)
  | .release g =>
      if Owns s g ∧ s.handles g = false ∧ Quiet s then
        ({ s with owner := none, needsExecution := false }, .ok)
      else (s, .unavailable)
  | .expire => ({ s with owner := none }, .ok)
  | .crash g =>
      ({ s with handles := put s.handles g false, sawEmpty := put s.sawEmpty g false }, .ok)
  | .idle => (s, .ok)

def step (s : State n) (a : Action n) : State n := (execute s a).1

def run (s : State n) (actions : List (Action n)) : State n :=
  actions.foldl step s

/-- An implementation must preserve both state effects and reply meaning. -/
def Transition (s : State n) (a : Action n) (r : Reply) (t : State n) : Prop :=
  execute s a = (t, r)

inductive Reachable : State n → Prop where
  | initial : Reachable initial
  | next {s : State n} (h : Reachable s) (a : Action n) : Reachable (step s a)

/-- Preservation obligations; NONE is used as an enablement guard in execute. -/
structure Safe (s : State n) : Prop where
  pending_known : ∀ c, s.mailbox c = true → ∃ p, s.submitted c = some p
  journal_exact : ∀ c p, s.journal c = some p → s.submitted c = some p
  conservation : ∀ c p, s.submitted c = some p → s.mailbox c = true ∨ s.journal c = some p
  settled_accepted : ∀ c, s.settled c = true → ∃ p, s.journal c = some p
  unfinished_discoverable : ∀ c p, s.journal c = some p → s.settled c = false → s.needsExecution = true
  owner_generation : ∀ g, s.owner = some g → g = s.generation

end SessionContract
