# Local receive flavor

`memory.New()` implements actor.Mailbox and actor.Adapter. It is usable outside
the actor package, has no dependencies beyond sessionloop and the standard
library, and advertises `accepted`, never process-crash durability.

## Interpretation into the protocol

The relevant adapter is the **assembled** Store + Worker + bound harness, not
the Store in isolation. Journal acceptance and settlement remain harness facts.

| Model | Go realization |
|---|---|
| submitted | immutable `sessionState.history` indexed by actor and command ID |
| mailbox | `pending` IDs referencing that same history; no copied payload |
| generation / owner | `generation` / `lease` under Store.mu |
| needsExecution | set in Acquire, retained across acknowledgment and failure |
| journal / settled | bound harness's acceptance and execution journal |
| handles | Worker-owned session; successful Close ends the live handle |
| expire | expired lease validation / takeover; never deletes retained state |
| internal waits | condition-channel waits and timers; no authoritative work in a channel |

Submission, acquisition, renewal, acknowledgment and retirement linearize
under Store.mu. Authority.Commit holds that SAME mutex through each journal
primitive, including recovery writes. Callbacks must be bounded storage
operations and must not reenter Store. Never hold this scope during model/tool
execution. Retired generations are never reused; overflow fails explicitly.

Receive checks state and captures its wait channel under one mutex. Submission,
acquisition, renewal and release close that channel. A timer at the earliest
lease expiry makes abandoned work discoverable without another message. It
always rechecks state after wakeup. A round-robin cursor prevents one ready
session from permanently hiding another.

## Worker obligations

Only the assembled Worker calls Release after a quiescent harness snapshot and
successful Close. Calling it directly during unfinished journal execution
violates the port's precondition. Similarly, Acknowledge takes a journal receipt
the Worker has checked; it does not independently reimplement journal lookup.
These are composition obligations, not capabilities that Go's type checker
proves. SessionOpener must persist a stable binding and wire journal authority;
the generic Worker cannot detect an opener secretly using an unguarded store.

The Worker never clears execution-needed state on failed open, dispatch, lookup,
acknowledgment, observation or closure. Handoff retries preserve exact input
identity. An acceptance found on replay is acknowledged even when another run
is active. Pending pages are scanned past busy Starts so steering still arrives.

Next-turn input and durable suspensions are parked pending external input.
Terminal stale-run, not-running, unsupported and invalid-command errors are
recorded by RejectionRecorder in the existing journal before acknowledgment.
That record maps to accept + settle in one atomic batch. Other errors retain
the input; a scheduler cannot prove progress during permanent storage failure.

## Failure profile and verification

Worker/handle restarts retain this Store and its paired journal. Destruction of
the process destroys a memory Store: it is NOT an implementation of the Lean
model's durable `crash` action. A durable flavor must persist its corresponding
state and prove crash/commit linearization. The Lean receive flavor explicitly
assumes such a durable core; Go's channel/mutex implementation is tested, not
formally connected to the Lean kernel.

Tests include the shared contract suite, concurrent shutdown submission,
blocked receive, lease-expiry wakeup, stale ownership, journal authority,
lost acceptance replies, crashes around acknowledgment, failed restoration,
steering behind busy pages, and native Agentic journal reopen. Real production
storage, distributed clocks, process-kill durability and external side effects
are outside this flavor's claims.
