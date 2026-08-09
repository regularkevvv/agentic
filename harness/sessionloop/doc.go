// Package sessionloop defines the provider-neutral asynchronous session
// protocol shared by Agentic Harness, the Agentic TUI bridge, and external
// consumers.
//
// The protocol models a session as a long-lived actor: a caller dispatches a
// typed command, the host acknowledges acceptance with a receipt independently
// of completion, ordered authoritative events describe committed transcript
// entries and run lifecycle, a copy-owned snapshot reconciles missed events,
// and a settled run returns the session to idle so a later command can begin
// another run on the same session.
//
// Stream.Next returns Event and nothing else. Caller-to-session content uses
// Input and InputBlock. Incomplete live generation uses a lossy Preview payload
// on an EventPreviewDelta event. Complete, authoritative conversation content
// uses Entry and EntryBlock on an EventEntryCommitted event. Final run status
// uses RunOutcome on an EventRunSettled event.
//
// The module is deliberately standard-library only: it defines the contract,
// portable error sentinels, a conformance suite (package conformance), and an
// in-memory reference host (package testkit). It does not execute a model and
// it never imports Agentic, Harness, the TUI, a provider SDK, or a transport
// library. The full rationale, protocol laws L1-L12, and the migration plan
// live in docs/design/harness-sessionloop-plan.md at the repository root.
package sessionloop
