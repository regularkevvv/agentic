import SessionContract.Safety

/-! # Session isolation
Lift the single-session protocol to independent sessions. A step changes only
its addressed session and preserves the world's per-session safety invariants.
-/

namespace SessionContract

abbrev World (n : Nat) := Nat → State n

def worldStep (world : World n) (session : Nat) (a : Action n) : World n :=
  put world session (step (world session) a)

theorem other_session_unchanged (world : World n) (session other : Nat) (a : Action n)
    (different : other ≠ session) : worldStep world session a other = world other := by
  simp [worldStep, put, different]

theorem world_step_safe (world : World n) (safe : ∀ session, Safe (world session))
    (session : Nat) (a : Action n) : ∀ other, Safe (worldStep world session a other) := by
  intro other
  by_cases same : other = session
  · subst other; simpa [worldStep] using step_safe (world session) (safe session) a
  · simpa [other_session_unchanged world session other a same] using safe other

end SessionContract
