import SessionContract.Flavors.Postgres.Recovery

/-! # Durable replies across SQL interleavings
A successful response must have committed backing even after other clients
commit, the mailbox is cleaned, or a reply is lost. This is stronger than a
state-local invariant and rejects responding from an uncommitted write set.
-/

namespace SessionContract.Flavors.Postgres

def Backed (r : Rows n) (a : Action n) (reply : Reply) : Prop :=
  (∀ c p, a = .submit c p → (reply = .submitted ∨ reply = .duplicate) →
    r.commands.identity c = some p) ∧
  (∀ g c p, a = .accept g c p → reply = .accepted → r.journal.acceptance c = some p)

theorem calculated_reply_backed (r : Rows n) (a : Action n) :
    Backed (calculate r a).1 a (calculate r a).2 := by
  constructor
  · intro c p same reply
    subst a
    have receipt := submission_reply_is_durable (project r) c p
      (by simpa [calculate_reply] using reply)
    change (project (calculate r (.submit c p)).1).submitted c = some p
    rw [calculate_project]
    exact receipt
  · intro g c p same reply
    subst a
    have receipt := accepted_reply_is_durable (project r) g c p
      (by simpa [calculate_reply] using reply)
    change (project (calculate r (.accept g c p)).1).journal c = some p
    rw [calculate_project]
    exact receipt

theorem calculation_preserves_backing (r : Rows n) (next prior : Action n) (reply : Reply)
    (backed : Backed r prior reply) : Backed (calculate r next).1 prior reply := by
  constructor
  · intro c p same accepted
    change (project (calculate r next).1).submitted c = some p
    rw [calculate_project]
    exact submission_persistent (project r) next c p (backed.1 c p same accepted)
  · intro g c p same accepted
    change (project (calculate r next).1).journal c = some p
    rw [calculate_project]
    exact journal_persistent (project r) next c p (backed.2 g c p same accepted)

def ResponsesBacked (s : Database n) : Prop :=
  ∀ client a reply, s.responses client = some (a, reply) → Backed s.committed a reply

theorem initial_responses_backed : ResponsesBacked ({} : Database n) := by
  intro client a reply present; cases present

theorem step_preserves_responses {s u : Database n} (h : Step s u)
    (valid : LockedSnapshot s) (sound : ResponsesBacked s) : ResponsesBacked u := by
  cases h with
  | enqueue => exact sound
  | lock => exact sound
  | advance => exact sound
  | commit t held prepared =>
      intro client a reply present
      by_cases same : client = t.client
      · subst client
        simp only [commit, put_same, Option.some.injEq, Prod.mk.injEq] at present
        obtain ⟨ha, hr⟩ := present
        subst a
        subst reply
        simpa only [commit, Transaction.draft, prepared] using calculated_reply_backed t.before t.action
      · have old : s.responses client = some (a, reply) := by simpa [commit, put, same] using present
        simpa only [commit, Transaction.draft, prepared, valid t held] using
          calculation_preserves_backing s.committed t.action a reply (sound client a reply old)
  | abort client =>
      intro other a reply present
      by_cases same : other = client
      · simp [abort, put, same] at present
      · exact sound other a reply (by simpa [abort, put, same] using present)
  | loseReply client =>
      intro other a reply present
      by_cases same : other = client
      · simp [loseReply, put, same] at present
      · exact sound other a reply (by simpa [loseReply, put, same] using present)
  | crash => intro client a reply present; cases present
  | wait => exact sound

theorem reachable_responses_backed {s : Database n} (history : Steps {} s) : ResponsesBacked s := by
  induction history with
  | refl => exact initial_responses_backed
  | next earlier last ih =>
      exact step_preserves_responses last (steps_keep_snapshot earlier initial_locked_snapshot) ih

theorem success_response_has_durable_submission {s : Database n} (history : Steps {} s)
    (client : Client) (c : CommandId n) (p : Payload) (reply : Reply)
    (response : s.responses client = some (.submit c p, reply))
    (success : reply = .submitted ∨ reply = .duplicate) :
    s.committed.commands.identity c = some p :=
  (reachable_responses_backed history client _ reply response).1 c p rfl success

theorem acceptance_response_has_durable_journal {s : Database n} (history : Steps {} s)
    (client : Client) (g : Lease) (c : CommandId n) (p : Payload)
    (response : s.responses client = some (.accept g c p, .accepted)) :
    s.committed.journal.acceptance c = some p :=
  (reachable_responses_backed history client _ _ response).2 g c p rfl rfl

theorem lost_success_reply_retry_is_duplicate {s t : Database n} (history : Steps {} s)
    (later : Steps s t) (client : Client) (c : CommandId n) (p : Payload) (reply : Reply)
    (response : s.responses client = some (.submit c p, reply))
    (success : reply = .submitted ∨ reply = .duplicate) :
    calculate t.committed (.submit c p) = (t.committed, .duplicate) := by
  apply same_identity_retry
  exact submitted_data_survives later (steps_keep_snapshot history initial_locked_snapshot) c p
    (success_response_has_durable_submission history client c p reply response success)

end SessionContract.Flavors.Postgres
