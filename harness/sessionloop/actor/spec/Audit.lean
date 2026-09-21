import Lean
import SessionContract

/-! # Proof trust audit
Require the principal results and inspect every contract theorem's transitive
axioms. Reject admitted proofs, custom axioms and trusted native computation;
only Lean's three standard logical axioms are permitted.
-/

open Lean Elab Command in
elab "#audit_session_contract" : command => do
  let required := #[
    ``SessionContract.reachable_safe,
    ``SessionContract.no_lost_submission,
    ``SessionContract.stale_accept_rejected,
    ``SessionContract.accepted_reply_is_durable,
    ``SessionContract.submission_reply_is_durable,
    ``SessionContract.no_second_fresh_acceptance,
    ``SessionContract.crash_at_every_prefix_recoverable,
    ``SessionContract.fair_worker_eventually_settles,
    ``SessionContract.every_adapter_safe,
    ``SessionContract.every_adapter_recoverable,
    ``SessionContract.every_adapter_progresses,
    ``SessionContract.Flavors.Receive.recv_refines,
    ``SessionContract.Flavors.Receive.submitted_response_is_durable,
    ``SessionContract.world_step_safe,
    ``SessionContract.Examples.arrival_before_release_ready,
    ``SessionContract.Examples.arrival_after_release_ready,
    ``SessionContract.Examples.early_deletion_breaks_conservation,
    ``SessionContract.Examples.clearing_unfinished_execution_breaks_recovery,
    ``SessionContract.Examples.reference_worker_resolves_and_cleans,
    ``SessionContract.Examples.recovery_cleans_already_settled_input,
    ``SessionContract.Examples.receive_retry_after_lost_reply]
  unless (← getEnv).contains ``SessionContract.Examples.terminal_rejection_batch_safe do
    throwError "Missing terminal rejection batch proof"
  let env ← getEnv
  for name in required do
    unless env.contains name do throwError "Missing required theorem: {name}"
  let allowed := #[``propext, ``Quot.sound, ``Classical.choice]
  let mut count : Nat := 0
  for (name, info) in env.constants.toList do
    if (`SessionContract).isPrefixOf name then
      match info with
      | .axiomInfo _ => throwError "Custom contract axiom is forbidden: {name}"
      | .thmInfo _ =>
          count := count + 1
          for dependency in (← collectAxioms name) do
            unless allowed.contains dependency do
              throwError "{name} depends on forbidden axiom {dependency}"
      | _ => pure ()
  logInfo m!"Session contract audit: {count} theorems checked; only standard Lean logical axioms."

#audit_session_contract
