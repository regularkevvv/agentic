import SessionContract.Adapter

/-! # Command attribution before observation

This model extends the delivery protocol with the embedded harness's publication
critical section. Journal acceptance and the volatile attribution index are
separate steps: even a reader of the committed journal must wait for that index.
The observer guard checks the mutex phase, NOT the desired attribution equality.

Scope: durably keyed commands, immutable native event keys, one live view per
owned session. A key abstracts a namespaced run ID, queue ID or resolution
sequence; it is not the command ID. The durable map abstracts reconstruction
from journaled acceptances, not an additional physical table. Observations are
ghost history recording what consumers saw. This is an algorithm proof, not a
proof of the Go compiler, mutex implementation or journal decoder.
-/

namespace SessionContract.Publication

abbrev Key := Nat

/-- One acceptance may attribute several native keys (e.g. a resolution and
its new continuation run). They are installed in the same critical section. -/
structure Request (n : Nat) where
  lease : Lease
  command : CommandId n
  payload : Payload
  keys : List Key
  deriving DecidableEq

/-- `accepting`, `committed` and `installed` all hold the session mutex. -/
inductive Phase (n : Nat) where
  | unlocked
  | accepting (request : Request n)
  | committed (request : Request n)
  | installed
  deriving DecidableEq

structure Observation (n : Nat) where
  key : Key
  command : Option (CommandId n)
  deriving DecidableEq

structure View (n : Nat) where
  core : State n := initial
  durable : Key → Option (CommandId n) := fun _ => none
  cache : Key → Option (CommandId n) := fun _ => none
  phase : Phase n := .unlocked
  online : Bool := false
  observations : List (Observation n) := []

def initialView : View n := {}

def bind (index : Key → Option (CommandId n)) (r : Request n) : Key → Option (CommandId n) :=
  fun key => if key ∈ r.keys then some r.command else index key

def begin (s : View n) (r : Request n) : View n :=
  { s with phase := .accepting r }

def commit (s : View n) (r : Request n) : View n :=
  { s with core := step s.core (.accept r.lease r.command r.payload)
           durable := bind s.durable r
           phase := .committed r }

def install (s : View n) (r : Request n) : View n :=
  { s with cache := bind s.cache r, phase := .installed }

def unlock (s : View n) : View n := { s with phase := .unlocked }

def reject (s : View n) (r : Request n) : View n :=
  { s with core := step s.core (.accept r.lease r.command r.payload)
           phase := .unlocked }

def observe (s : View n) (key : Key) : View n :=
  { s with observations := ⟨key, s.cache key⟩ :: s.observations }

def crash (s : View n) (g : Lease) : View n :=
  { s with core := step s.core (.crash g), cache := fun _ => none
           phase := .unlocked, online := false }

/-- Reconstruction finishes before the new view becomes observable. -/
def restore (s : View n) : View n :=
  { s with cache := s.durable, phase := .unlocked, online := true }

/-- Other protocol operations do not modify the attribution projection. -/
def protocol (s : View n) (a : Action n) : View n :=
  { s with core := step s.core a }

def ProtocolOnly : Action n → Prop
  | .accept .. | .crash .. => False
  | _ => True

/-- Native IDs are not rebound to another command. This is an explicit key
identity obligation, independent of the publication ordering being proved. -/
def Compatible (s : View n) (r : Request n) : Prop :=
  ∀ key ∈ r.keys, s.durable key = none ∨ s.durable key = some r.command

inductive Step : View n → View n → Prop where
  | begin (s) (r) (online : s.online = true) (free : s.phase = .unlocked) :
      Step s (begin s r)
  | commit (s) (r) (held : s.phase = .accepting r) (key : Compatible s r)
      (accepted : (execute s.core (.accept r.lease r.command r.payload)).2 = .accepted) :
      Step s (commit s r)
  | install (s) (r) (held : s.phase = .committed r) : Step s (install s r)
  | unlock (s) (held : s.phase = .installed) : Step s (unlock s)
  | reject (s) (r) (held : s.phase = .accepting r)
      (failed : (execute s.core (.accept r.lease r.command r.payload)).2 ≠ .accepted) :
      Step s (reject s r)
  | observe (s) (key) (online : s.online = true) (free : s.phase = .unlocked)
      (present : s.durable key ≠ none) : Step s (observe s key)
  | crash (s) (g) : Step s (crash s g)
  | restore (s) (offline : s.online = false) : Step s (restore s)
  | protocol (s) (a) (other : ProtocolOnly a) : Step s (protocol s a)

def Synchronized (s : View n) : Prop :=
  s.online = true → match s.phase with
    | .committed r => bind s.cache r = s.durable
    | _ => s.cache = s.durable

def Agrees (s : View n) : Prop :=
  ∀ o ∈ s.observations, ∃ c, s.durable o.key = some c ∧ o.command = some c

def Backed (s : View n) : Prop :=
  ∀ k c, s.durable k = some c → ∃ p, s.core.journal c = some p

structure Valid (s : View n) : Prop where
  synchronized : Synchronized s
  agrees : Agrees s
  backed : Backed s

variable {n : Nat} {s t : View n} {k : Key} {c : CommandId n} {o : Observation n}

theorem initial_valid : Valid (initialView : View n) := by
  constructor <;> simp [Synchronized, Agrees, Backed, initialView]

/-- An accepted event key keeps the same meaning through any legal step. -/
theorem binding_persistent (h : Step s t) (bound : s.durable k = some c) :
    t.durable k = some c := by
  cases h <;> try exact bound
  case commit r held key accepted =>
    simp only [commit]
    by_cases hk : k ∈ r.keys
    · rcases key k hk with empty | same
      · simp [bound] at empty
      · simpa [bind, hk, bound] using same.symm
    · simpa [bind, hk] using bound

theorem step_synchronized (h : Step s t) (sync : Synchronized s) : Synchronized t := by
  cases h with
  | begin r online free => simpa [Synchronized, begin, free] using sync
  | commit r held key accepted =>
      intro online
      have eq := sync online
      simp only [held] at eq
      simp [commit, eq]
  | install r held =>
      intro online
      simpa [install, held] using sync online
  | unlock held => simpa [Synchronized, unlock, held] using sync
  | reject r held failed => simpa [Synchronized, reject, held] using sync
  | observe key online free present => exact sync
  | crash g => simp [Synchronized, crash]
  | restore offline => simp [Synchronized, restore]
  | protocol a other => exact sync

theorem step_agrees (h : Step s t) (valid : Valid s) : Agrees t := by
  have keep : ∀ o ∈ s.observations, ∃ c, t.durable o.key = some c ∧ o.command = some c := by
    intro o ho
    obtain ⟨c, bound, seen⟩ := valid.agrees o ho
    exact ⟨c, binding_persistent h bound, seen⟩
  cases h <;> try exact keep
  case observe key online free present =>
    intro o ho
    simp only [observe, List.mem_cons] at ho
    rcases ho with rfl | old
    · have eq := valid.synchronized online
      simp only [free] at eq
      cases hb : s.durable key with
      | none => exact False.elim (present hb)
      | some c => exact ⟨c, hb, by simpa only [eq] using hb⟩
    · exact keep o old

theorem step_backed (h : Step s t) (backed : Backed s) : Backed t := by
  cases h <;> try exact backed
  case commit r held key accepted =>
    intro k c bound
    by_cases hk : k ∈ r.keys
    · have hc : r.command = c := by simpa [commit, bind, hk] using bound
      subst c
      exact ⟨r.payload, accepted_reply_is_durable s.core r.lease r.command r.payload accepted⟩
    · have old : s.durable k = some c := by simpa [commit, bind, hk] using bound
      obtain ⟨p, hp⟩ := backed k c old
      exact ⟨p, journal_persistent _ _ _ _ hp⟩
  case reject r held failed =>
    intro k c bound
    obtain ⟨p, hp⟩ := backed k c bound
    exact ⟨p, journal_persistent _ _ _ _ hp⟩
  case protocol a other =>
    intro k c bound
    obtain ⟨p, hp⟩ := backed k c bound
    exact ⟨p, journal_persistent _ _ _ _ hp⟩

theorem step_valid (h : Step s t) (valid : Valid s) : Valid t :=
  ⟨step_synchronized h valid.synchronized, step_agrees h valid, step_backed h valid.backed⟩

/-- Forgetting projection metadata recovers exactly the existing protocol:
no new mailbox, lease, acceptance or recovery transition is introduced. -/
theorem step_refines (h : Step s t) : AdapterStep (View.core (n := n)) s t := by
  cases h <;> try exact AdapterStep.internal rfl
  case commit r held key accepted =>
    exact AdapterStep.commit (.accept r.lease r.command r.payload) _ rfl
  case reject r held failed =>
    exact AdapterStep.commit (.accept r.lease r.command r.payload) _ rfl
  case crash g => exact AdapterStep.commit (.crash g) _ rfl
  case protocol a other => exact AdapterStep.commit a _ rfl

inductive Steps : View n → View n → Prop where
  | refl (s : View n) : Steps s s
  | next {s t u : View n} (history : Steps s t) (move : Step t u) : Steps s u

theorem steps_valid (history : Steps s t) (valid : Valid s) : Valid t := by
  induction history with
  | refl => exact valid
  | next history move ih => exact step_valid move ih

theorem step_linearizes (h : Step s t) : ∃ actions, t.core = run s.core actions := by
  cases step_refines h with
  | internal same => exact ⟨[], same⟩
  | commit a r corresponding => exact ⟨[a], (congrArg Prod.fst corresponding).symm⟩
  | batch actions corresponding => exact ⟨actions, corresponding.symm⟩

theorem history_linearizes (history : Steps s t) : ∃ actions, t.core = run s.core actions := by
  induction history with
  | refl => exact ⟨[], rfl⟩
  | next earlier last ih =>
      obtain ⟨before, hb⟩ := ih
      obtain ⟨after, ha⟩ := step_linearizes last
      exact ⟨before ++ after, by simpa [run, List.foldl_append, hb] using ha⟩

/-- All finite interleavings, including crashes at any instruction boundary,
preserve both the old protocol's safety and the new observation invariant. -/
theorem reachable_safe (history : Steps (initialView : View n) t) :
    Safe t.core ∧ Valid t := by
  constructor
  · have base : Safe (initialView : View n).core := initial_safe
    induction history with
    | refl => exact base
    | next history move ih => exact adapter_preserves_safety View.core (step_refines move) ih
  · exact steps_valid history initial_valid

theorem live_equals_replay (history : Steps (initialView : View n) t)
    (seen : o ∈ t.observations) : o.command = t.durable o.key := by
  obtain ⟨c, bound, value⟩ := (reachable_safe history).2.agrees o seen
  exact value.trans bound.symm

/-- A crash may erase any partly installed cache. Reconstruction restores all
committed bindings, including one whose acceptance callback never ran. -/
theorem crash_restore_recovers (s : View n) (g : Lease) :
    (restore (crash s g)).durable = s.durable ∧
    (restore (crash s g)).cache = s.durable ∧
    Steps s (restore (crash s g)) := by
  exact ⟨rfl, rfl, .next (.next (.refl s) (.crash s g)) (.restore _ rfl)⟩

/-- At ANY phase, a committed attribution has a concrete crash/rebuild/read
continuation. Unlike abstract core recovery alone, this constructs view steps. -/
theorem crash_at_every_phase_replayable (s : View n) (g : Lease)
    (bound : s.durable k = some c) :
    Steps s (observe (restore (crash s g)) k) ∧
    (observe (restore (crash s g)) k).observations.head? = some ⟨k, some c⟩ := by
  constructor
  · exact .next (crash_restore_recovers s g).2.2
      (.observe _ k rfl rfl (by simp [restore, crash, bound]))
  · simp [observe, restore, crash, bound]

/-- The repair is not vacuous: after a successful append, install and unlock
enable a real observation with the committed command ID. -/
theorem committed_can_publish (s : View n) (r : Request n)
    (held : s.phase = .committed r) (online : s.online = true)
    (sync : Synchronized s) (key : k ∈ r.keys) :
    Steps s (observe (unlock (install s r)) k) ∧
    (observe (unlock (install s r)) k).observations.head? =
      some ⟨k, some r.command⟩ := by
  have bound : s.durable k = some r.command := by
    have eq := sync online
    simp only [held] at eq
    simpa [bind, key] using (congrFun eq k).symm
  constructor
  · exact .next (.next (.next (.refl s) (.install s r held)) (.unlock _ rfl))
      (.observe _ k online rfl (by simp [unlock, install, bound]))
  · simp [observe, unlock, install, bind, key]

end SessionContract.Publication
