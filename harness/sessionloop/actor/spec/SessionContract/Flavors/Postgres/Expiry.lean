import SessionContract.Flavors.Postgres.Rows

/-! # Database-clock boundary
Clock values stand for readings made AFTER obtaining the session row lock.
They are not worker clocks. The transaction algorithm treats expiration as a
serialized revocation; these lemmas connect its guards to deadline comparison.
Real timestamp representation, clock availability and eventual passage of time
remain implementation/environment obligations, not custom Lean axioms.
-/

namespace SessionContract.Flavors.Postgres

structure DeadlineCheck where
  now : Nat
  expiresAt : Nat

/-- An expired stored owner is revoked before attempting another guarded action.
Expiry plus the action may be committed in one SQL transaction (AdapterStep.batch). -/
def checkedRows (r : Rows n) (clock : DeadlineCheck) : Rows n :=
  if clock.now < clock.expiresAt then r else (calculate r .expire).1

def renewDeadline (r : Rows n) (g : Lease) (clock : DeadlineCheck) (ttl : Nat) : Option Nat :=
  if owns r g ∧ clock.now < clock.expiresAt ∧ 0 < ttl then some (clock.now + ttl) else none

theorem expired_lease_cannot_renew (r : Rows n) (g : Lease) (clock : DeadlineCheck) (ttl : Nat)
    (expired : clock.expiresAt ≤ clock.now) : renewDeadline r g clock ttl = none := by
  simp [renewDeadline, show ¬ clock.now < clock.expiresAt by omega]

theorem stale_lease_cannot_renew (r : Rows n) (g : Lease) (clock : DeadlineCheck) (ttl : Nat)
    (stale : r.session.owner ≠ some g) : renewDeadline r g clock ttl = none := by
  simp [renewDeadline, owns, stale]

theorem renewal_requires_live_authority (r : Rows n) (g : Lease) (clock : DeadlineCheck)
    (ttl deadline : Nat) (renewed : renewDeadline r g clock ttl = some deadline) :
    owns r g ∧ clock.now < clock.expiresAt ∧ clock.now < deadline := by
  simp only [renewDeadline] at renewed
  split at renewed
  · rename_i allowed
    have result : clock.now + ttl = deadline := Option.some.inj renewed
    exact ⟨allowed.1, allowed.2.1, by omega⟩
  · cases renewed

theorem deadline_checked_operation_refines (r : Rows n) (clock : DeadlineCheck) (a : Action n) :
    AdapterStep project r (calculate (checkedRows r clock) a).1 := by
  unfold checkedRows
  split
  · exact AdapterStep.commit a (calculate r a).2 (calculate_correct r a)
  · apply AdapterStep.batch [.expire, a]
    simp only [run, List.foldl_cons, List.foldl_nil]
    rw [calculate_project, calculate_project]

theorem expired_write_rejected (r : Rows n) (clock : DeadlineCheck) (g : Lease)
    (c : CommandId n) (p : Payload) (expired : clock.expiresAt ≤ clock.now) :
    (calculate (checkedRows r clock) (.accept g c p)).2 = .unavailable := by
  simp [checkedRows, show ¬ clock.now < clock.expiresAt by omega, calculate, opened, owns]

end SessionContract.Flavors.Postgres
