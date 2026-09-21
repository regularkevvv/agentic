import SessionContract.Adapter

/-! # Receive flavor
A reference adapter with separate send, receive/commit, response loss and crash
steps. The channel is volatile transport; acknowledged storage is not. This proves
the Lean algorithm's correspondence, not Go channels or a concrete disk driver.
-/

namespace SessionContract.Flavors.Receive

/-- A one-request channel reference around the durable protocol. The channel
holds UNACKNOWLEDGED transport requests, never the sole durable work record. -/
structure Local (n : Nat) where
  core : State n := initial
  request : Option (Action n) := none
  response : Option (Action n × Reply) := none

def send (s : Local n) (a : Action n) : Local n :=
  if s.request.isNone ∧ s.response.isNone then { s with request := some a } else s

/-- Successful receive commits the request before making its reply available. -/
def recv (s : Local n) : Local n :=
  match s.request with
  | none => s
  | some a =>
      let result := execute s.core a
      { core := result.1, request := none, response := some (a, result.2) }

def loseReply (s : Local n) : Local n := { s with response := none }

/-- A process failure destroys channel contents, but not acknowledged storage. -/
def crash (s : Local n) (g : Lease) : Local n :=
  { core := step s.core (.crash g) }

theorem send_refines (s : Local n) (a : Action n) : AdapterStep Local.core s (send s a) := by
  apply AdapterStep.internal
  simp only [send]
  split <;> rfl

theorem recv_refines (s : Local n) : AdapterStep Local.core s (recv s) := by
  cases hr : s.request with
  | none => apply AdapterStep.internal; simp [recv, hr]
  | some a =>
      apply AdapterStep.commit a (execute s.core a).2
      simp [Transition, recv, hr]

theorem lost_reply_refines (s : Local n) : AdapterStep Local.core s (loseReply s) :=
  AdapterStep.internal rfl

theorem crash_refines (s : Local n) (g : Lease) : AdapterStep Local.core s (crash s g) :=
  AdapterStep.commit (.crash g) .ok rfl

theorem recv_safe (s : Local n) (safe : Safe s.core) : Safe (recv s).core :=
  adapter_preserves_safety Local.core (recv_refines s) safe

theorem submitted_response_is_durable (s : Local n) (c : CommandId n) (p : Payload)
    (request : s.request = some (.submit c p))
    (response : (recv s).response = some (.submit c p, .submitted) ∨
                (recv s).response = some (.submit c p, .duplicate)) :
    (recv s).core.submitted c = some p := by
  have receipt : (execute s.core (.submit c p)).2 = .submitted ∨
      (execute s.core (.submit c p)).2 = .duplicate := by
    simpa [recv, request] using response
  simpa [recv, request, step] using submission_reply_is_durable s.core c p receipt

end SessionContract.Flavors.Receive
