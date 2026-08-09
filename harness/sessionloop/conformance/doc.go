// Package conformance is the reusable, capability-table-driven test suite
// for sessionloop hosts. It tests the claims a host advertises — never its
// implementation types: every case synchronizes exclusively through
// receipts, events, snapshots, and the optional Gate. No case sleeps.
//
// A host adapter calls Run with a Factory producing a fresh Env per case.
//
// # Factory contract
//
// The factory must return a host whose runs behave deterministically enough
// to observe the protocol laws:
//
//   - a start command carrying one text block must lead to at least one
//     committed assistant entry and one completed settlement;
//   - sessions support repeated runs after settlement;
//   - structural command validation precedes capability and state checks, so
//     invalid commands report sessionloop.ErrInvalidCommand first;
//   - if Env.Gate is nil, timing-dependent cases are skipped with t.Skip.
//
// Optional cases activate only when the matching capability is advertised.
// Several of them need the host to exhibit a behavior a plain echo run never
// shows, so the factory contract includes scripted scenarios keyed on start
// input metadata (see MetaScenario): a host advertising a capability below
// must honor the matching scenario value.
//
//   - events.preview + ScenarioPreview: emit at least one preview before
//     the run settles;
//   - content.tools.detailed + ScenarioTools: commit tool_call and matching
//     tool_result blocks, the tool_call carrying one complete JSON value;
//   - run.suspension.resolve + ScenarioSuspend: suspend the run once and,
//     after an approving resolution of the exact suspension ID, continue the
//     same run to a completed settlement;
//   - output.structured + ScenarioOutput: settle completed with one complete
//     JSON value as the outcome's structured output.
//
// The testkit package's ScenarioRunFunc implements this contract for the
// in-memory reference host.
package conformance
