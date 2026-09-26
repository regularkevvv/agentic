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
    ``SessionContract.Publication.history_linearizes,
    ``SessionContract.RecoveryStartup.reachable_owned,
    ``SessionContract.RecoveryStartup.interrupt_before_start,
    ``SessionContract.RecoveryStartup.interrupt_after_start,
    ``SessionContract.RecoveryStartup.interrupted_can_settle,
    ``SessionContract.RecoveryStartup.old_guard_strands_reachable_interrupt,
    ``SessionContract.Publication.reachable_safe,
    ``SessionContract.Publication.live_equals_replay,
    ``SessionContract.Publication.crash_at_every_phase_replayable,
    ``SessionContract.Publication.committed_can_publish,
    ``SessionContract.Publication.append_failure_preserves_safety,
    ``SessionContract.Publication.offline_until_reconstruction,
    ``SessionContract.Publication.invalidated_cannot_observe,
    ``SessionContract.Publication.append_error_reconstruction,
    ``SessionContract.Publication.uncommitted_error_retry,
    ``SessionContract.Publication.Examples.unknown_result_after_commit,
    ``SessionContract.Publication.Examples.unknown_result_keeps_acceptance,
    ``SessionContract.Publication.Examples.unknown_result_replays_after_rebuild,
    ``SessionContract.Publication.Examples.precommit_error_does_not_invent_acceptance,
    ``SessionContract.Publication.Examples.precommit_error_can_retry,
    ``SessionContract.Publication.Examples.committed_reachable,
    ``SessionContract.Publication.Examples.old_order_breaks_agreement,
    ``SessionContract.Publication.Examples.old_order_unreachable,
    ``SessionContract.Publication.Examples.unlocked_reader_also_breaks_agreement,
    ``SessionContract.Publication.Examples.fixed_order_reachable,
    ``SessionContract.Publication.Examples.same_durable_protocol,
    ``SessionContract.Publication.Examples.crash_before_install_replays,
    ``SessionContract.Publication.Examples.crash_before_append_keeps_pending,
    ``SessionContract.Flavors.Receive.recv_refines,
    ``SessionContract.Flavors.Receive.submitted_response_is_durable,
    ``SessionContract.Flavors.Postgres.calculate_correct,
    ``SessionContract.Flavors.Postgres.step_keeps_snapshot,
    ``SessionContract.Flavors.Postgres.step_refines,
    ``SessionContract.Flavors.Postgres.history_linearizes,
    ``SessionContract.Flavors.Postgres.reachable_safe,
    ``SessionContract.Flavors.Postgres.transaction_realizable,
    ``SessionContract.Flavors.Postgres.crash_at_every_sql_phase_recoverable,
    ``SessionContract.Flavors.Postgres.fair_committed_worker_progresses,
    ``SessionContract.Flavors.Postgres.success_response_has_durable_submission,
    ``SessionContract.Flavors.Postgres.acceptance_response_has_durable_journal,
    ``SessionContract.Flavors.Postgres.lost_success_reply_retry_is_duplicate,
    ``SessionContract.Flavors.Postgres.pending_discoverable_after_release,
    ``SessionContract.Flavors.Postgres.unfinished_discoverable_after_expiry,
    ``SessionContract.Flavors.Postgres.expired_lease_cannot_renew,
    ``SessionContract.Flavors.Postgres.deadline_checked_operation_refines,
    ``SessionContract.Flavors.Postgres.Examples.missing_lock_loses_committed_input,
    ``SessionContract.Flavors.Postgres.Examples.lockless_counterexample_unreachable,
    ``SessionContract.Flavors.Postgres.Examples.premature_success_has_no_backing,
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
