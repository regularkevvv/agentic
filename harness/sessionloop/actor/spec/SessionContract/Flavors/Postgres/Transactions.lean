import SessionContract.Flavors.Postgres.Rows

/-! # Interleaved short transactions
Arbitrarily many clients contend for ONE session row. A client locks before
reading/validating, keeps private writes, and publishes them only at COMMIT.
Other clients can enqueue, wait, crash and lose replies between every phase.
PostgreSQL's row exclusion, rollback and durable atomic commit are the trusted
storage semantics modeled here; no entire worker run is assumed atomic.
-/

namespace SessionContract.Flavors.Postgres

abbrev Client := Nat

inductive Phase where
  | read | checked | commandsWritten | journalWritten | readyToCommit
  deriving DecidableEq, Repr

def Phase.next : Phase → Phase
  | .read => .checked
  | .checked => .commandsWritten
  | .commandsWritten => .journalWritten
  | .journalWritten => .readyToCommit
  | .readyToCommit => .readyToCommit

structure Transaction (n : Nat) where
  client : Client
  action : Action n
  before : Rows n
  phase : Phase := .read

/-- The actual private write set at each SQL statement boundary. A half-written
command/journal pair is allowed here, but is never a committed database state. -/
def Transaction.draft (t : Transaction n) : Rows n :=
  let result := (calculate t.before t.action).1
  match t.phase with
  | .read | .checked => t.before
  | .commandsWritten => { t.before with commands := result.commands }
  | .journalWritten => { t.before with commands := result.commands, journal := result.journal }
  | .readyToCommit => result

structure Database (n : Nat) where
  committed : Rows n := {}
  /-- Existence of an active transaction is the session-row lock. -/
  active : Option (Transaction n) := none
  waiting : Client → Option (Action n) := fun _ => none
  responses : Client → Option (Action n × Reply) := fun _ => none

def view (s : Database n) : State n := project s.committed

def enqueue (s : Database n) (client : Client) (a : Action n) : Database n :=
  { s with waiting := put s.waiting client (some a) }

def lockRow (s : Database n) (client : Client) (a : Action n) : Database n :=
  { s with active := some { client, action := a, before := s.committed },
           waiting := put s.waiting client none }

def advance (s : Database n) (t : Transaction n) : Database n :=
  { s with active := some { t with phase := t.phase.next } }

def commit (s : Database n) (t : Transaction n) : Database n :=
  { s with committed := t.draft, active := none,
           responses := put s.responses t.client (some (t.action, (calculate t.before t.action).2)) }

/-- Rollback/cancel/process death can discard only this client's transaction.
An unexpired lease and stale handle witnesses deliberately survive: takeover
must fence them even if the process merely pauses instead of dying. -/
def abort (s : Database n) (client : Client) : Database n :=
  { s with active := s.active.filter (fun t => t.client != client),
           waiting := put s.waiting client none, responses := put s.responses client none }

def loseReply (s : Database n) (client : Client) : Database n :=
  { s with responses := put s.responses client none }

/-- Server/transport failure rolls back uncommitted writes and loses replies.
It does not destroy committed tables or erase ownership/recovery metadata. -/
def crash (s : Database n) : Database n := { committed := s.committed }

/-- Operational SQL/storage rules, not assumed preservation or refinement laws.
Wrong-phase operations and contenders waiting on the row lock simply wait. -/
inductive Step : Database n → Database n → Prop where
  | enqueue (s : Database n) (client : Client) (a : Action n)
      (free : s.waiting client = none) (answered : s.responses client = none)
      (notActive : ∀ t, s.active = some t → t.client ≠ client) :
      Step s (enqueue s client a)
  | lock (s : Database n) (client : Client) (a : Action n)
      (unlocked : s.active = none) (requested : s.waiting client = some a) :
      Step s (lockRow s client a)
  | advance (s : Database n) (t : Transaction n) (held : s.active = some t) :
      Step s (advance s t)
  | commit (s : Database n) (t : Transaction n) (held : s.active = some t)
      (prepared : t.phase = .readyToCommit) : Step s (commit s t)
  | abort (s : Database n) (client : Client) : Step s (abort s client)
  | loseReply (s : Database n) (client : Client) : Step s (loseReply s client)
  | crash (s : Database n) : Step s (crash s)
  | wait (s : Database n) : Step s s

/-- The read snapshot remains current while its transaction holds the row.
This is proved from the operational rules, never checked as a commit guard. -/
def LockedSnapshot (s : Database n) : Prop :=
  ∀ t, s.active = some t → t.before = s.committed

theorem initial_locked_snapshot : LockedSnapshot ({} : Database n) := by
  intro t held; cases held

theorem step_keeps_snapshot {s u : Database n} (h : Step s u)
    (valid : LockedSnapshot s) : LockedSnapshot u := by
  cases h with
  | enqueue => exact valid
  | lock => intro t held; simp only [lockRow, Option.some.injEq] at held; subst t; rfl
  | advance t held =>
      intro next hnext
      simp only [advance, Option.some.injEq] at hnext
      subst next
      exact valid t held
  | commit => intro t held; cases held
  | abort client =>
      intro t held
      have original : s.active = some t := (Option.filter_eq_some_iff.mp held).1
      exact valid t original
  | loseReply => exact valid
  | crash => intro t held; cases held
  | wait => exact valid

/-- All interleaved SQL phases are abstract stutters except COMMIT. Its state
and reply match a contract action because its locked snapshot cannot go stale. -/
theorem step_refines {s u : Database n} (h : Step s u) (valid : LockedSnapshot s) :
    AdapterStep view s u := by
  cases h with
  | commit t held prepared =>
      apply AdapterStep.commit t.action (calculate t.before t.action).2
      have current := valid t held
      simp only [Transition, view, commit, Transaction.draft, prepared]
      rw [← current]
      exact calculate_correct t.before t.action
  | _ => exact AdapterStep.internal rfl

inductive Steps : Database n → Database n → Prop where
  | refl (s : Database n) : Steps s s
  | next {s t u : Database n} (history : Steps s t) (last : Step t u) : Steps s u

theorem steps_keep_snapshot {s u : Database n} (trace : Steps s u)
    (valid : LockedSnapshot s) : LockedSnapshot u := by
  induction trace with
  | refl => exact valid
  | next _ last ih => exact step_keeps_snapshot last ih

theorem interleavings_safe {s u : Database n} (trace : Steps s u)
    (valid : LockedSnapshot s) (safe : Safe (view s)) : Safe (view u) := by
  induction trace with
  | refl => exact safe
  | next history last ih =>
      exact adapter_preserves_safety view (step_refines last (steps_keep_snapshot history valid)) ih

theorem reachable_safe {s : Database n} (trace : Steps {} s) : Safe (view s) :=
  interleavings_safe trace initial_locked_snapshot initial_safe

theorem crash_preserves_committed (s : Database n) : (crash s).committed = s.committed := rfl

theorem rollback_preserves_committed (s : Database n) (client : Client) :
    (abort s client).committed = s.committed := rfl

theorem partial_writes_invisible (s : Database n) (t : Transaction n) :
    (advance s t).committed = s.committed := rfl

theorem one_row_lock (s : Database n) (a b : Transaction n)
    (ha : s.active = some a) (hb : s.active = some b) : a.client = b.client := by
  have : a = b := Option.some.inj (ha.symm.trans hb)
  exact congrArg Transaction.client this

theorem commit_reply_matches (s : Database n) (t : Transaction n)
    (valid : LockedSnapshot s) (held : s.active = some t) (prepared : t.phase = .readyToCommit) :
    execute (view s) t.action = (view (commit s t), (calculate t.before t.action).2) := by
  simp only [view, commit, Transaction.draft, prepared]
  rw [← valid t held]
  exact calculate_correct t.before t.action

theorem commit_releases_row_lock (s : Database n) (t : Transaction n) :
    (commit s t).active = none := rfl

theorem crashed_state_safe {s : Database n} (trace : Steps {} s) : Safe (view (crash s)) :=
  reachable_safe (.next trace (.crash s))

end SessionContract.Flavors.Postgres
