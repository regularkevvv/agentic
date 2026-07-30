# Harness Framework: Production Design and Delivery Plan

**Status:** Final production design; Phases 1-2 implemented, Phase 3 in review
**Date:** 2026-07-29
**Repository baseline:** `9c33333` (`v0.4.0`)
**Decision owner:** `agentic` maintainers

---

## 1. Executive decision

Add a nested Go module, `github.com/regularkevvv/agentic/harness`, without moving
the existing root module. The harness accepts an already-bound
`agentic.Runner[O]`, resolves its execution-driver capability at build time, and
uses only public `agentic` APIs.

The harness is a cooperative scheduler around Agentic's tool loop. It owns the
session inbox, durable transcript, environment, permissions, capability graph,
and public event subscriptions. Agentic continues to own model requests, typed
output validation, tool execution, retries, limits, and terminal arbitration.

This design requires a real execution seam in Agentic. It is not implementable
by changing a single `return` or by adding static `AgentOption` hooks. The root
module must first gain:

1. explicit start, continue, and suspended-turn resume driver entries;
2. one shared fold for blocking and streaming model transports;
3. terminal arbitration after typed output validation and complete tool-result
   pairing;
4. per-run turn, event, tool-overlay, and tool-gate options that still work
   after dependencies have been bound, plus a tool-result projection seam; and
5. partial, failed, suspended, stopped, interrupted, and completed execution
   states.

The public `Runner[O]` convenience interface remains source-compatible. Built-in
runners additionally implement `Driver[O]`; the harness checks that capability
when it is built.

The retrieval work described by the original spike is no longer proposed work.
Gemini, Ollama, Cohere, and Bedrock embedders plus Voyage and Cohere rerankers
shipped in `v0.3.0`. The Cohere-on-Bedrock limit is 96 inputs per request and the
implementation correctly chunks at that limit.

---

## 2. Review disposition

The review dated 2026-07-21 is correct on all four points. This revision resolves
them as follows.

| Finding | Assessment | Resolution in this design |
|---|---|---|
| Cohere limit contradicts the implementation | Correct | Keep `cohereBatchLimit = 96`; cite the AWS v3/v4 model contracts; mark the work shipped |
| Continue has more than one terminal return | Correct | Replace return-site interception with one terminal-arbitration stage shared by text and output-tool completion |
| No continuation/resume entry exists | Correct | Add explicit `Driver.Drive` start/continue modes plus `Driver.Resume`; neither path appends an implicit user prompt |
| Existing streaming channel already backpressures | Correct | Make `StreamEvent` a compatibility projection of the canonical fold; keep its documented lossless/blocking contract separate from nonblocking harness subscriptions |
| Line anchors drifted | Correct | Refer to stable symbols and behavioral tests, not source line numbers |

The audit also found two blockers not called out in the review:

- `WithTurnHook` and `WithEventSink` were proposed as `AgentOption`s, but the
  harness receives a bound `Runner[O]`; static options cannot be installed for a
  session. They must be `RunOption`s consumed by the new driver.
- A capability cannot add tools to an opaque runner today. The driver therefore
  needs an immutable per-run toolset overlay and a batch tool gate.

---

## 3. Goals, non-goals, and release boundary

### Goals

- Ship a useful default harness and the public primitives required to assemble a
  different one.
- Preserve the root module's lightweight `Runner[O]` API and dependency typing.
- Make steering, follow-up, interruption, deferred tools, resumption, and event
  delivery explicit state-machine operations.
- Preserve a protocol-valid transcript across normal completion, validation
  retries, output tools, interruption, compaction, and crash recovery.
- Keep environment capability and governance policy separate.
- Support typed and text agents through the same harness API.

### Non-goals for `harness/v0.1.0`

- Distributed scheduling, multiple concurrent writers to one session, or
  exactly-once external side effects.
- A TUI or CLI product.
- A vector database.
- Mid-model-request steering or forced termination of arbitrary Go tool
  handlers. Cancellation remains cooperative.
- Codemode, shared-session role switching, or out-of-process subagents. Their
  interfaces are reserved, but implementation comes later.

Local crash-resumable sessions are a goal. That is different from distributed
durable execution: one process owns a session, records a write-ahead JSONL log,
and refuses to guess whether an indeterminate external side effect should be
repeated.

### Release boundary

- Agentic `v0.4.0`: execution driver, shared fold, canonical events, per-run
  overlays/gates/hooks/result processing, tool-call context, and
  terminal-pairing fixes.
- Harness `v0.1.0`: runtime, sessions, repair, environment, permissions,
  capabilities, and default assembly, requiring Agentic `v0.4.0`.

---

# Part I — Design laws

## 4. What the harness owns

A harness answers seven independent questions:

| Concern | Question |
|---|---|
| Action space | What resources can the agent affect? |
| Action mode | Direct tools, code, planning, or delegation? |
| Action interface | Which tools and schemas are visible this turn? |
| Knowledge | Which history, compacted context, memory, and retrieval enter the request? |
| Control | When does the run continue, suspend, stop, or accept steering? |
| Governance | Which proposed effects are allowed, denied, or deferred? |
| Topology | Which child agents exist and what do they capture? |

Two rules organize the implementation:

> Substrates use interfaces. Policies use ordered hooks.

An environment is a substrate: local, memory-backed, containerized, or remote.
Permissions and compaction are policies layered over that substrate. Putting
permissions inside a filesystem does not make shell execution safe, and putting
filesystem methods directly inside capabilities makes alternate backends
impossible.

The dependency direction is strict: runtime and session policy import ports,
never adapters. Durable payloads are opaque bytes behind an injected codec;
session journals are exclusive leases with expected-leaf appends; environments,
event hubs, and result processors are created by session-scoped factories.
Memory, JSONL, local-filesystem, in-process-bus, and spill implementations live
in adapter subpackages. A deployment selects them only at its composition root.

This separation does not turn state-machine policy into configuration. Queue
linearization, append-before-execute ordering, recovery classification, repair,
durable cursors, subscriber lag behavior, usage, and budgets remain invariant
core behavior.

> The durable transcript is truth; each provider request is a derived view.

Repair, compaction, cache-aware injection, and provider normalization transform a
copy. They never rewrite the durable transcript to appease a provider.

## 5. The five runtime nouns

The runtime has five distinct objects:

1. **Inbox** — many writers, one session reader; durable queued input.
2. **Transcript** — append-only authoritative history.
3. **Loop** — Agentic's single fold over model and tool turns.
4. **Provider view** — repaired and compacted messages for one request.
5. **Event stream** — one producer projected to independent subscribers.

The inbox is not an event bus. It changes future execution. The event stream is
not persistence. A subscriber may disappear without changing execution.

Each session is single-flight: at most one fold mutates its transcript. Tool
calls within one admitted batch may run concurrently, but their results are
committed in model call order.

“Batch atomic” in this document means atomic admission with respect to steering
and permissions: either preflight admits execution or no handler starts. It does
not mean external side effects are transactional or rolled back together.

## 6. Cooperative scheduling

A model request and its admitted tool batch form a turn. The inbox is accepted
concurrently but drained only at safe boundaries.

```text
boundary -> build provider view -> model request -> commit assistant
         -> classify calls -> preflight regular-tool batch
         -> execute admitted tools -> validate completion candidate
         -> commit one result per tool call
         -> TurnEnd -> terminal arbitration -> boundary/end/suspend
```

Steering is therefore queue acceptance during a turn, not preemption of that
turn. Pi and Codex both use this shape: input may arrive while work is running,
but the active model/tool critical section reaches a safe boundary before it is
consumed. Pydantic AI's pending-message and deferred-tool mechanisms independently
support the same separation.

Interruption is different. It cancels the current context and requires an unwind
path because a model stream or tool effect may be partial.

## 7. The four input verbs

| Verb | Accepted while | Drained at | Persists across interrupt? |
|---|---|---|---|
| `Steer` | active ordinary run | next turn boundary | no |
| `FollowUp` | active ordinary run | only when it would naturally complete | no |
| `NextTurn` | any session state | before the next explicit prompt | yes |
| `Interrupt` | running or suspended | immediately | n/a |

Rules:

- `Steer` and `FollowUp` return only after their queue entries are durable.
- One entry drains per boundary by default. `DrainAll` is an opt-in session
  policy.
- At terminal arbitration, `Steer` has priority over `FollowUp`; follow-up is
  considered only when no steer is waiting.
- A session atomically changes `Running -> Closing` while checking its queues.
  Input arriving after that transition receives `ErrRunClosing`; it is never
  silently lost.
- `Steer` rejects compact, review, child-only, and other non-steerable turn kinds
  with `ErrTurnNotSteerable`.
- Explicit stop or interrupt marks undrained steer/follow-up entries cancelled.
  `NextTurn` entries remain queued.
- Accepted queue entry IDs and drain/cancel records live in the session log, so
  restart cannot consume an entry twice.

Queue acceptance and boundary drain use the same session mutex and write-ahead
store. A failed acceptance write does not enqueue. At a boundary, the drain
record is synced before its message is injected; if that write fails, the hook
fails and recovery still sees the entry as accepted rather than consumed. This
is the linearization point for the `Steer`/`Closing` race.

If an input call's context is cancelled after its acceptance record syncs, the
method still returns the `QueueReceipt`; it never reports cancellation for an
entry that was durably accepted.

The natural termination predicate is:

```text
validated completion candidate
AND no model-requested continuation
AND no accepted steer or follow-up to drain
```

“The model emitted no tool calls” is not, by itself, a terminal condition.

---

# Part II — Agentic `v0.4.0` execution contract

## 8. Current state and why the old seam is insufficient

At `v0.3.0`:

- `Runner[O]` exposes only `Run(ctx, prompt string, ...RunOption)`.
- `prepareLoopAfterPreflight` always appends a new user text message, even when
  `WithMessages` supplies history.
- `runAfterPreflight` terminates separately for a validated no-tool text response
  and for an output-tool response.
- `typedRuntime.run` parses and validates typed output outside the core loop, so
  a core turn hook cannot know whether a candidate is valid.
- `runStreamTrue` duplicates the fold and writes directly to a bounded channel.
- `processToolUses` under `EndStrategyEarly` can return with assistant tool calls
  that have no matching tool-result messages.
- Bound runners cannot receive session hooks, capability toolsets, or permission
  gates after binding.

Consequently, a hook added at one `return` would be incorrect for output tools,
typed validation, and streaming. A resume implemented with `WithMessages` would
also append a spurious user message after deferred tool results.

## 9. Additive public driver

Keep the existing convenience interface:

```go
type Runner[O any] interface {
    Run(context.Context, string, ...RunOption) (*Result[O], error)
}
```

Add a capability implemented by every built-in agent and bound runner:

```go
type Driver[O any] interface {
    Runner[O]
    Drive(context.Context, DriveInput, ...RunOption) (*Execution[O], error)
    Resume(context.Context, ResumeInput, ...RunOption) (*Execution[O], error)
}

type DriveMode uint8

const (
    DriveStart DriveMode = iota
    DriveContinue
)

type DriveInput struct {
    Mode    DriveMode
    History []Message
    Prompt  *Message
}

type ResumeInput struct {
    History    []Message
    Suspension Suspension
    Decisions  []ToolResumeDecision
    Prompt     *Message // optional user message after tool results
}

type ToolResumeDecision struct {
    CallID string
    Action ToolResumeAction // execute or return supplied result
    Input  map[string]any   // nil means use persisted arguments
    Result *ToolExecutionResult
}

type ToolResumeAction uint8

const (
    ToolResumeInvalid ToolResumeAction = iota
    ToolResumeExecute
    ToolResumeReturn
)

func RequireDriver[O any](r Runner[O]) (Driver[O], error)
```

Validation is strict:

- `DriveStart` requires one `RoleUser` prompt and history with no open tool
  frontier.
- `DriveContinue` forbids a prompt, requires non-empty history, and requires a
  provider-valid, fully paired frontier: ordinarily a user or tool-result
  message.
- `Resume` requires the exact suspension returned by the driver, history whose
  frontier hash matches it, and exactly one decision for every open executable
  call. It completes that interrupted tool turn before starting another model
  request. Its optional prompt must be `RoleUser` and is appended only after all
  call results. Override inputs are validated against the original tool schema
  before any handler in the resumed batch runs.
- System prompts are inserted only if history does not already contain one.
- Input slices are copied; caller mutation cannot change a running execution.

`Run(ctx, "...")` remains a convenience wrapper over `DriveStart`. Existing
`Bind` and `BindProvider` may keep returning `Runner[O]`; the concrete bound
runner also implements `Driver[O]`. The harness accepts `Runner[O]` and calls
`RequireDriver` during `Build`, preserving the settled post-binding boundary
without silently accepting a runner it cannot control.

A third-party type that implements only `Runner[O]` remains valid for Agentic but
cannot be used by the harness; `RequireDriver` returns `ErrDriverRequired` with
no fallback loop. Third parties may implement `Driver[O]` directly. The harness
never reconstructs an agent loop from `Run`.

`BindProvider` resolves and validates dependencies again on `Resume` before any
handler executes. The harness puts session and suspension IDs in the standard
context first, so a provider may rehydrate session-scoped resources. Applications
that require object identity across suspension must use `Bind` or a provider
keyed by that session; the harness never claims to serialize arbitrary Go
dependency values.

### Execution states

```go
type ExecutionStatus uint8

const (
    ExecutionCompleted ExecutionStatus = iota
    ExecutionSuspended
    ExecutionStopped
    ExecutionInterrupted
    ExecutionFailed
)

type Execution[O any] struct {
    Status     ExecutionStatus
    Result     *Result[O]   // transcript, usage, calls, and results so far
    Suspension *Suspension // non-nil only when suspended
}

type Suspension struct {
    ID           string
    Kind         string
    FrontierHash string
    Payload      json.RawMessage // versioned, serializable, kind-specific data
}
```

`Drive` may return both a non-nil partial `Execution` and an error. Provider,
limit, event-sink, turn-hook, and cancellation errors never erase committed
history. `Runner.Run` preserves its existing behavior by returning the completed
`Result` or the typed error from `Drive`.

`FrontierHash` is computed over a versioned canonical encoding of the committed
messages and ordered open call IDs. It is a misuse/stale-state guard, not an
authentication primitive.

The suspension also carries usage, iteration/retry counters, end strategy, and a
fingerprint of the model/output schema/tool definitions that define the
execution. `Resume` restores counters instead of resetting limits and rejects a
semantic configuration mismatch before effects. Event sink, turn hook, stream
transport, and cancellation grace may be reattached; model, output mode,
toolsets, limits, and retry policy may not change mid-suspension. Unknown payload
versions return `ErrSuspensionVersion` rather than attempting a best-effort
resume.

## 10. One fold, two model transports

Refactor `runAfterPreflight` and `runStreamTrue` into one generic outer fold.
Blocking `Model.Request` and streaming `StreamModel.RequestStream` become
transport strategies that both yield one reconstructed assistant message,
usage, finish reason, and preview events.

There is one implementation of:

- iteration and usage limits;
- provider-error handling;
- assistant commits;
- tool preflight, execution, retries, and pairing;
- typed parsing and validation;
- `TurnEnd` and terminal arbitration; and
- partial execution/error semantics.

Implement it as a generic package-level `driveLoop[O]` with a private
`completionEvaluator[O]`. The text facade supplies the existing text validators;
typed facades supply their parser plus typed validators. Remove
`typedRuntime.run`'s outer validation/re-run loop so output retries, hook timing,
usage, and streaming all pass through the same state machine.

`RunStream` becomes a compatibility adapter over this fold. `StreamEvent` stays
the public delta-oriented projection for existing consumers; it does not become
a second authoritative taxonomy.

Its channel contract remains intentionally lossless and backpressured: if a
caller neither ranges `Events` nor calls `Wait`/`Text`, that particular streaming
run may stall. The harness never consumes this adapter and is therefore not
coupled to its channel capacity.

On a hook or sink error during streaming, the fold retains already committed
messages, projects one `StreamEventError`, closes the channel, and makes
`Wait()` return the same error. Add a non-generic
`StreamResult.Snapshot() (ExecutionSnapshot, bool)` with status, messages, calls,
results, usage, and suspension metadata so text and typed agent-stream consumers
can inspect final or partial state after `Wait`; it does not consume the channel
independently. Provider-level streams return `false` because they do not own an
agent execution.

## 11. Per-run control seams

These are `RunOption`s because the harness configures individual sessions after
the runner has been bound:

```go
func WithRunTurnHook(TurnHook) RunOption
func WithRunEventSink(EventSink) RunOption
func WithRunToolsets(...Toolset) RunOption
func WithRunToolGate(ToolGate) RunOption
func WithRunToolResultProcessor(ToolResultProcessor) RunOption
func WithRunModelStreaming(bool) RunOption
func WithRunToolCancellationGrace(time.Duration) RunOption
```

Toolsets form an immutable registry overlay for the execution. A duplicate name
between the agent registry and any overlay fails before the first model request.
No shared agent registry is mutated.

The shared tool scheduler observes cancellation. After
`WithRunToolCancellationGrace`, it stops waiting for non-cooperative handlers,
marks their calls indeterminate/aborted, and ignores late returns. This is an
explicit replacement for the current unconditional `ExecuteBatch` wait and is
required for `Session.Interrupt` to terminate predictably.

The gate preflights the complete executable regular-tool batch before any
handler runs. Output tools are classified separately as framework calls:

```go
type ToolGate interface {
    EvaluateBatch(context.Context, []ToolUse) (ToolBatchDecision, error)
}

type ToolBatchDecision struct {
    Calls    []ToolDisposition
    Deferral *ToolDeferral
}

type ToolDisposition struct {
    Kind     ToolDispositionKind // execute, return, suspend
    Result   *ToolExecutionResult
    Continue bool // expose this result to the model before accepting output
}

type ToolDeferral struct {
    Kind    string
    Payload json.RawMessage
}

type ToolDispositionKind uint8

const (
    ToolDispositionInvalid ToolDispositionKind = iota
    ToolDispositionExecute
    ToolDispositionReturn
    ToolDispositionSuspend
)
```

`EvaluateBatch` is side-effect free and must return one valid disposition per
input call in the same order; a length mismatch or invalid kind fails before any
handler runs. A suspend disposition requires one batch `Deferral`. If any call
suspends, Agentic executes none of the batch and returns `ExecutionSuspended`
with the entire assistant tool frontier still open. The driver wraps the gate
deferral together with every call definition, its original order, and a hash of
the exact history frontier in its serializable `Suspension`. Only
`Driver.Resume` may re-enter that batch. This atomic preflight is what makes
multi-call approval safe.

Before invoking a handler, Agentic adds public call metadata to the standard Go
context:

```go
type ToolCallContext struct {
    ID      string
    Name    string
    Attempt int
}

func CurrentToolCall(context.Context) (ToolCallContext, bool)
```

Dependency-aware handlers already receive that standard context as
`RunContext[D].Ctx`; no handler signature needs to change.

```go
type ToolResultProcessor interface {
    Process(context.Context, ToolUse, ToolExecutionResult) (ToolExecutionResult, error)
}
```

The processor runs after the handler returns and before result commit/formatting.
It may replace model-visible content while retaining the call ID, tool name, and
error truth: an original error cannot become success, and ID/name changes are
rejected. On processor failure, Agentic commits a paired error result stating
that the side effect may have completed but result processing failed, then
emits a failed `TurnEnd`/`RunError` without invoking the turn hook, and returns
`ExecutionFailed`. This seam supports output bounding and artifact spill without
wrapping or re-registering application tools.

## 12. Turn commit and terminal arbitration

The shared fold executes this order for every iteration:

1. Check pre-request limits.
2. Build the repaired/compacted provider view.
3. Perform one blocking or streaming model request.
4. Check post-response limits and reconstruct a complete assistant message.
5. Commit the assistant message and canonical assistant event.
6. Classify regular and output tool calls. A truncated response is never a valid
   completion candidate and none of its partial calls execute.
7. Under `EndStrategyEarly`, select the first output call and skip all other
   calls. Otherwise preflight the complete executable regular-tool batch and
   execute admitted calls. Output tools never enter the permission gate.
8. If the gate suspends, persist the whole open assistant frontier and return
   `ExecutionSuspended`; `Driver.Resume` re-enters this same step.
9. Parse and validate a text or output-tool completion candidate after admitted
   regular tools finish, preserving current exhaustive-tool semantics. A model
   retry or gate result marked `Continue` discards an otherwise valid output
   candidate so the model sees those results first.
10. Build and append exactly one result for every tool call in original model
    order. On validation failure, include the output failure and normal retry
    feedback; regular-tool effects/results are not repeated.
11. Emit `TurnEnd` and call the turn hook, including on validation-retry and
    ordinary tool-only turns.
12. Apply one terminal-arbitration decision.

The hook receives defensive copies of committed state. A non-empty candidate has
already passed parsing and validation:

```go
type Turn struct {
    Index     int
    Messages  []Message
    Assistant Message
    Results   []ToolExecutionResult
    Usage     Usage
    Candidate CompletionCandidate
}

type TurnAction uint8

const (
    TurnDefault TurnAction = iota
    TurnContinue
    TurnStop
)

type TurnDecision struct {
    Action TurnAction
    Inject []Message
}

type TurnHook func(context.Context, Turn) (TurnDecision, error)
```

Decision rules are deliberately non-ambiguous:

- `Inject` requires `TurnContinue`, at least one message, and only `RoleUser`
  messages. Deferred tool results enter through `Driver.Resume` or an explicit
  `DriveContinue` history, never through a turn hook.
- `TurnContinue` without an injection is rejected. This prevents a request whose
  last message is still the assistant completion.
- `TurnStop` with a validated candidate completes with that candidate. Without
  one, it returns `ExecutionStopped` and a partial result.
- `TurnDefault` completes only if the candidate is valid and no model retry or
  tool continuation remains.
- A hook error occurs after the turn commit. It returns the partial execution,
  attempts `RunError` then `RunEnd`, and never rolls back the transcript.

Error-path lifecycle emission is best effort. The original hook or sink error
wins; a failing sink is not recursively called to report its own failure. The
legacy stream adapter still publishes its independent terminal error projection.

This explicitly answers the output-validation question: queued input does not
bypass validation. The candidate is first validated, then treated as an
intermediate valid answer when `TurnContinue` injects new input. Only the final
validated candidate becomes `Result.Output`.

### Output-tool pairing and `EndStrategyEarly`

Every assistant tool call receives a result before terminal arbitration,
including output tools and deliberately skipped calls.

For `EndStrategyEarly`, the first output tool is the candidate and no regular
tool in that same assistant batch executes, regardless of source order. The
output call gets a non-error acknowledgement result; every skipped call gets a
deterministic error result explaining that early output ended the batch. If the
candidate fails validation, its result carries the validation failure and the
normal retry path continues. If queued input continues the run, the accepted
output remains an intermediate, fully paired turn.

For `EndStrategyExhaustive`, regular calls finish before the output candidate is
validated, matching existing side-effect ordering. If a regular-call gate
suspends the batch, the whole assistant frontier remains open; `Driver.Resume`
resolves the regular calls, then parses and validates the output call and appends
all results in source order. If a regular tool requests a model retry or a gate
result has `Continue`, the output call instead receives a deterministic
“discarded because another call requires continuation” result. No typed output
is hidden in an opaque suspension token.

This preserves the existing “do not run side effects after early output” intent
while eliminating histories that end in unmatched calls.

## 13. Canonical Agentic events

The fold emits typed events at deterministic commit points:

```go
type EventNature uint8

const (
    EventPreview EventNature = iota
    EventAuthoritative
    EventLifecycle
)

type Event interface {
    Nature() EventNature
    Type() EventType
    TurnIndex() int
}

type EventSink interface {
    Emit(context.Context, Event) error
}
```

Each event kind is a concrete typed struct; there is no untyped `map[string]any`
payload in the commit path. Harness persistence converts the closed set of root
events into versioned session entries.

Minimum taxonomy:

- preview: text, thinking, and tool-argument deltas;
- authoritative: assistant committed, tool batch planned, tool started, tool
  result committed, output validated, turn messages injected;
- lifecycle: run started, turn started, turn ended, run suspended, run completed,
  run interrupted, run error, run ended.

Agentic calls the sink synchronously. A sink is a low-level commit participant,
not a public subscriber: blocking is allowed and an error stops further effects.
The harness sink persists authoritative and lifecycle state first and only then
fans events out to subscribers.

Legacy `StreamEvent` is a documented projection of the same source events:

| Canonical event | Legacy projection |
|---|---|
| text/thinking/tool-argument preview | corresponding delta event |
| tool started | `StreamEventToolCallStart` |
| tool result committed | `StreamEventToolResult` |
| completed | `StreamEventDone` |
| any terminal error, including hook error | `StreamEventError` |

There is no independent streaming fold and no parallel authoritative channel.

---

# Part III — Harness architecture

## 14. Monorepo layout and module rules

```text
agentic/                       root module: github.com/regularkevvv/agentic
├── go.mod
├── go.work                    committed; uses . and ./harness
├── docs/
├── internal/
├── provider/
├── tool/
└── harness/                   nested module
    ├── go.mod                 github.com/regularkevvv/agentic/harness
    ├── harness.go
    ├── artifact/              ports and opaque handle types
    │   ├── artifacttest/      reusable adapter conformance
    │   ├── file/              file-backed adapter
    │   ├── memory/            in-memory adapter
    │   └── spill/             oversized-result processor adapter
    ├── codec/                 durable payload representation port
    │   └── json/              JSON codec adapter
    ├── env/                   substrate ports and error taxonomy
    │   ├── envtest/           reusable adapter conformance
    │   ├── local/             host-local adapter; not a sandbox
    │   └── memory/            in-memory adapter
    ├── event/                 hub and subscription ports
    │   ├── eventtest/         reusable adapter conformance
    │   └── inproc/            bounded process-local adapter
    ├── repair/
    ├── runtime/               ToolRuntime, clock, and identity ports
    │   └── system/            wall-clock and cryptographic-ID adapters
    ├── session/               state-machine/application core
    └── store/                 journal contracts and opaque entries
        ├── jsonl/             filesystem JSON-lines adapter
        ├── memory/            in-memory adapter
        └── storetest/         reusable adapter conformance
```

Phase 2 creates no empty future-layout packages. Capability, context-policy,
permission, subagent, codemode, memory, skill, and eval packages arrive only
with their implementation phases.

Do not move the root module into `agentic/`; doing so changes its published import
path. Nested-module releases use subdirectory tags such as `harness/v0.1.0`, as
required by Go module versioning.

`go.work` is committed for local cross-module development. No committed
`replace` directive points at `..`. CI proves all three views:

```sh
GOWORK=off go test -race -count=1 ./...
(cd harness && GOWORK=off go test -race -count=1 ./...)
go test -race -count=1 ./... ./harness/...
```

The independent harness job downloads the released root version from its
`go.mod`; the workspace job tests the pending pair. Release Agentic first, update
the harness requirement, then tag `harness/vX.Y.Z`.

The harness imports only root public packages. If it needs an `internal/core`
symbol, that symbol must first be promoted through the root facade.

## 15. Capability graph and builder

The builder accepts the already-bound runner:

```go
func New[O any](runner agentic.Runner[O], opts ...Option) *Builder[O]

type Capability interface {
    ID() string
    Ordering() Ordering
    Register(*Registry) error
}

type Ordering struct {
    Before []string
    After  []string
}
```

`Build`:

1. resolves `agentic.Driver[O]` with `RequireDriver`;
2. rejects duplicate capability IDs, missing ordering references, and cycles;
3. performs a stable topological sort;
4. combines toolsets, gates, context transforms, event middleware, and lifecycle
   hooks into immutable per-session plans; and
5. returns an immutable `Harness[O]` safe for concurrent session creation.

The registry installs one composed root gate. Capabilities contribute ordered
`ToolGateMiddleware` wrappers around an allow-all base instead of competing
independent gates; each wrapper must preserve batch cardinality and may only
narrow/return/suspend calls that are still executable. Context and event
middleware use the same explicit ordered-chain pattern.

Capability registration may contribute only through public registry points.
`Default()` is assembled from ordinary capabilities and has no privileged hook.
That is the acceptance proof that third parties can build a different harness
from the same primitives.

Resolved names are `Capability`, `contextpolicy`, `Steer`, `FollowUp`, and
`NextTurn`.

### Default assembly

The convenient default is explicit about where it may read/write:

```go
type DefaultConfig struct {
    WorkspaceRoot      string // required, absolute
    SessionDir         string // required, absolute and outside model-visible tools
    ContextWindowTokens int   // required, model's advertised input window
}

func Default[O any](runner agentic.Runner[O], cfg DefaultConfig) (*Harness[O], error)
```

`Default` installs `Local` rooted at `WorkspaceRoot`, synchronized `JSONLStore`,
file-backed artifact storage under `SessionDir`, deterministic compaction,
`Repair`, the canonical event bus, and `WorkspaceWrite` permissions. Shell and
network remain `ask`; outside-root paths remain denied. Zero paths are errors,
not process-working-directory defaults. Build canonicalizes both paths and
rejects overlap, so ordinary filesystem/shell tools cannot browse or modify
session logs and spilled results.

The default tool-cancellation grace is one second and is configurable per
harness. The deadline begins when interruption cancels the run context.

Default deterministic compaction triggers when the conservative estimated
request size reaches 70% of `ContextWindowTokens` and targets 50%, leaving room
for tool schemas and output. The estimator conservatively counts the full UTF-8
byte length as tokens unless the caller supplies a model-specific
`TokenCounter`, adds fixed framing overhead per message/content block/tool, and
includes system/context fragments plus serialized tool schemas. An
absent/non-positive context window is a build error, not a guessed model
constant.

The default tool-result processor formats each result once. Results above 64 KiB
are stored in full in the artifact store; the model receives a 24 KiB head, a
24 KiB tail, byte counts, and an opaque artifact handle exposed through a gated
`read_artifact` tool. Handles are unguessable, scoped to the current session,
and cannot express paths. Limits are configurable, but disabling the limit
requires an explicit option. `Default` in `v0.1.0` does not include subagents,
memory, skills, or codemode; those arrive through later ordinary capabilities.

Head/tail cuts preserve UTF-8 boundaries. Non-text results use Agentic's one
canonical JSON formatting pass before size measurement, so persistence,
model-visible previews, and retrieval all refer to the same bytes.

## 16. Runtime and session API

```go
type Harness[O any] struct { /* immutable */ }

func (h *Harness[O]) NewSession(ctx context.Context, opts ...session.Option) (*Session[O], error)
func (h *Harness[O]) ResumeSession(ctx context.Context, id string) (*Session[O], error)

type Session[O any] struct { /* single-flight state machine */ }

func (s *Session[O]) Prompt(context.Context, agentic.Message) (*agentic.Execution[O], error)
func (s *Session[O]) Resume(context.Context, ResumeRequest) (*agentic.Execution[O], error)
func (s *Session[O]) Steer(context.Context, agentic.Message) (QueueReceipt, error)
func (s *Session[O]) FollowUp(context.Context, agentic.Message) (QueueReceipt, error)
func (s *Session[O]) NextTurn(context.Context, agentic.Message) (QueueReceipt, error)
func (s *Session[O]) Interrupt(context.Context) error
func (s *Session[O]) Snapshot(context.Context) (Snapshot, error)
func (s *Session[O]) Subscribe(SubscribeOptions) *Subscription
func (s *Session[O]) WaitForIdle(context.Context) error
func (s *Session[O]) Close(context.Context) error

type Snapshot struct {
    Cursor     uint64
    State      SessionState
    Messages   []agentic.Message
    Pending    []QueueEntry
    Suspension *agentic.Suspension
    Usage      agentic.Usage
}
```

The low-level Phase 2 composition is explicit and adapter-neutral:

```go
type RuntimeConfig struct {
    Sessions         store.Repository
    Codec            codec.Codec
    Events           event.Factory
    Environments     env.Factory
    ResultProcessors artifact.ProcessorFactory
    Clock            runtime.Clock
    IDs              runtime.IDGenerator
    ToolCancellationGrace time.Duration
}
```

`Close` releases the session's journal lease, event hub, and environment without
rewriting durable state. Idle, suspended, and faulted sessions may be closed and
later reopened. Active sessions reject close until they reach a safe state.
Close is idempotent and retryable: a canceled or failed cleanup does not release
the Harness's ownership, and `ResumeSession` completes any outstanding cleanup
before opening fresh session-scoped adapters.

`Prompt` is valid only while idle. A suspended session must use `Resume` or
`Interrupt`; a second prompt cannot accidentally paper over unresolved calls.
`Snapshot` copies state under the session mutex at its returned durable cursor;
it contains no uncommitted preview data.

The runtime calls `Driver.Drive(DriveStart)` for a prompt,
`Driver.Resume` for a gate-suspended tool frontier, and
`Driver.Drive(DriveContinue)` only when complete tool results already exist in
history, such as an externally fulfilled deferred tool. `NextTurn` entries are
appended before the new prompt in FIFO order. All user-supplied messages are
role-checked.

The session persists the prompt/next-turn entries before calling the driver. Its
synchronous event sink persists each Agentic assistant/tool commit as it occurs;
it does not append `Result.Messages` again at the end. `Result.NewMessages()` and
the persisted run-message sequence (prompt/injection entries plus Agentic commit
events) are compared as an invariant check before normal close. A mismatch
faults the session as `ErrCommitProjectionMismatch`.

Session usage is accumulated in durable turn entries and restored on restart. An
optional session budget is converted into remaining Agentic `UsageLimits` for
each start/resume; budget exhaustion returns `ErrBudgetExceeded`, cancels
steer/follow-up entries under the ordinary explicit-stop rule, and leaves
`NextTurn` queued. Provider-reported post-response usage is still committed even
when that response crosses the limit.

### State machine

```text
Idle --Prompt--> Running --natural/error--> Closing --> Idle
Running --suspend--> Suspended --Resume--> Running
Running/Suspended --Interrupt--> Interrupting --> Idle
Running/Closing/Suspended --store failure--> Faulted
Idle/Suspended/Faulted --Close--> Closed
```

The `Closing` transition and queue check occur under the session mutex. Public
callbacks are never invoked while that mutex is held.

A write-ahead store failure after in-memory state changes is not ordinary idle:
the session enters `Faulted`, cancels work, rejects new input, and makes
`WaitForIdle` return the storage error. After storage is repaired, the caller
reopens it through `ResumeSession`, which runs normal log recovery. A queue
acceptance write that fails before in-memory enqueue does not fault the session.

## 17. Durable transcript and recovery

The session persistence port is an append-only journal with opaque payloads:

```go
type Entry struct {
    Schema   uint16
    Seq      uint64
    ID       string // UUIDv7
    ParentID string
    Kind     string
    Payload  []byte
}

type Repository interface {
    Create(context.Context, string, ...PendingEntry) (Journal, Commit, error)
    Open(context.Context, string) (Journal, error)
}

type Journal interface {
    SessionID() string
    Load(context.Context) (Snapshot, error)
    Append(context.Context, Cursor, ...PendingEntry) (Commit, error)
    Close(context.Context) error
}

type Codec interface {
    Encode(any) ([]byte, error)
    Decode([]byte, any) error
}
```

Opening a journal acquires its exclusive writer lease. Every append supplies the
expected exact leaf cursor and returns `ErrConflict` on stale state. This makes
single ownership a storage contract rather than a mutex hidden in one Harness
or one concrete store value. The session core chooses which facts require
synchronous durability; adapters implement that acknowledgement requirement.
Payload encoding and journal envelope encoding are independent choices.

Entry kinds include messages, queue accepted/drained/cancelled, turn boundaries,
tool batch planned, tool started, tool result, suspension, resolution,
compaction, branch movement, model/toolset changes, and run termination.

The schema retains `ParentID` and branch-movement entries, but `v0.1.0` exposes
one active leaf and no public fork/move API. Readers must preserve unknown branch
entries for forward compatibility. Public branching is deferred until a
single-writer store coordinator can prevent two branch sessions from writing the
same file independently.

The `store/jsonl` adapter has one writer per session, opens with append
semantics, and serializes writes. It calls `fsync` before
acknowledging queue acceptance, tool start/result, suspension/resolution, and
turn/run termination, and syncs the parent directory when creating a session
file. `v0.1.0` has no buffered mode that weakens the word
“durable.” A trailing partial JSON line after a crash is copied to a diagnostic
sidecar and the incomplete tail is truncated before append resumes; earlier
valid entries remain. Moving the leaf or creating a branch is another append,
not an in-place rewrite or an “atomic rename.” It refuses a second
process-local/file-lock owner; this is exclusion, not a distributed lease.
`store/memory` implements the same port and passes the same reusable conformance
suite.

Recovery folds valid entries from root to leaf, reapplies the last compaction,
restores unconsumed inbox entries, and examines tool states:

- planned but not started: safe to resolve or execute after permission checks;
- started with a committed result: complete;
- started without a result: **indeterminate**. Never auto-retry unless the tool
  declares an idempotency contract keyed by call ID. Otherwise suspend for an
  operator-provided result or denial.

This is honest local durability without claiming exactly-once external effects.

## 18. Repair and provider projection

Repair remains an `agentic.HistoryProcessor` and is always the terminal transform
after compaction:

```go
type FrontierMode uint8

const (
    CloseInterruptedFrontier FrontierMode = iota
    PreserveDeferredFrontier
)

func Repair(mode FrontierMode, pending PendingCalls) agentic.HistoryProcessor
```

It is deterministic, idempotent, and non-mutating. It:

- creates stable synthetic error results for open calls after interrupt,
  truncation, or abandoned recovery;
- removes orphan results only from the provider projection;
- excludes incomplete assistant preview content that was never authoritatively
  committed;
- preserves completed thinking blocks, IDs, provider names, and signatures; and
- never splits a call/result pair during compaction.

Matching is scoped to one assistant tool frontier, not global ID uniqueness.
Duplicate call IDs within a frontier or multiple results for one call are hard
`ErrTranscriptInvalid` failures; repair does not silently choose one.

`PreserveDeferredFrontier` is valid only when the active suspension lists exactly
the open tool call IDs. It does not send that open frontier to a provider. Once
resume results have been appended, normal projection continues.

Required property tests:

```text
Repair(Repair(history)) == Repair(history)
Repair(history) is byte-stable across invocations
every outbound call has exactly one later matching result
compaction followed by repair preserves the same open-call set
```

## 19. Deferred tools, permissions, and resume

Permissions are a policy capability over the environment, not part of the
environment itself. The initial ruleset is hierarchical
`pattern -> allow|ask|deny`, with most-specific match winning and default deny.
Ship `ReadOnly` and `WorkspaceWrite` presets.

Capability tools translate calls into structured
`PermissionRequest{Capability, Action, CanonicalResource}` values before rule
matching. An application tool without effect metadata is matched as
`tool:<name>` and remains denied unless the application adds a rule. `ReadOnly`
allows filesystem reads/listing and denies writes, shell, and network;
`WorkspaceWrite` allows canonicalized reads/writes inside the configured root,
asks for shell/network, and denies paths outside it. Command-string matching is
advisory policy, never a sandbox boundary.

The permission capability implements `agentic.ToolGate`. It evaluates the whole
batch before effects:

- all allowed: execute the batch;
- any denied: execute none and return deterministic error results for every call,
  identifying denied calls and calls skipped by atomic policy; every disposition
  sets `Continue`, so a same-turn output candidate cannot hide the denial;
- any asked and none denied: execute none and return one suspension containing
  the whole batch plus the IDs that require a user resolution.

The v0.1 policy is always atomic for non-allow outcomes. Mixed-batch execution is
not configurable in this release; this avoids executing an allowed side effect
beside another call the user has not approved.

```go
type ResumeRequest struct {
    SuspensionID string
    Resolutions  []ToolResolution
    Prompt       *agentic.Message // optional user text after all results
}

type ToolResolution struct {
    CallID       string
    Action       ResolutionAction // approve, deny, external-result
    OverrideArgs map[string]any
    Result       any
    Reason       string
}
```

Resume requires exactly one resolution for every `RequiredResolutionID` in the
suspension and rejects unknown, duplicate, or missing IDs before effects. Calls
that were already allowed require no redundant user approval. The harness then
builds one root `ToolResumeDecision` for every open executable call: approved or
pre-allowed calls use `execute`, while denial and external results use `return`.
`Driver.Resume` validates the suspension ID, history-frontier hash, and complete
decision set; executes admitted handlers; appends all results in original call
order; appends the optional user prompt; completes `TurnEnd`; and re-enters the
fold. No implicit user message is added.

The low-level stateless primitive has the same validation:

```go
runtime.Resume(ctx, driver, transcript, suspension, request, opts...)
```

This primitive delegates the actual tool re-entry to `Driver.Resume`; it does not
try to invoke opaque runner tools from the harness. `Session.Resume` supplies the
persisted transcript and suspension so normal callers cannot resume the wrong
frontier.

## 20. Environment threading

```go
package env

type CanonicalResource struct {
    Scheme  string
    ID      string // opaque outside the backend
    Display string
}

type Environment interface {
    Files() FileSystem
    Shell() (Shell, bool)
}

type Lease interface {
    Environment
    Close(context.Context) error
}

type Factory interface {
    Open(context.Context, string /* session ID */) (Lease, error)
}
```

Initial implementations are `Local` and `Memory`; container and remote backends
come later. Errors map to a closed backend-independent code set. Relative paths
resolve against environment `Cwd`. Canonicalization returns a backend-qualified
opaque resource rather than a backend-specific raw string. Authorization checks
use that canonical resource and the operation must guard against path
replacement between check and use.

`Local` rejects path traversal and follows an explicit symlink policy, but its
permission rules are not an OS security boundary against a hostile shell command
or another local process racing the filesystem. A caller needing containment
must supply a container/VM environment; governance never upgrades an
uncontained substrate into a sandbox.

The harness attaches its runtime services to the standard execution context:

```go
type ToolRuntime struct {
    Environment env.Environment
    SessionID   string
    Emit        func(ToolUpdate)
}

func FromContext(context.Context) (ToolRuntime, bool)
```

Agentic separately attaches `ToolCallContext`. Capability tools retrieve both
from `RunContext[D].Ctx`. Application dependency values remain untouched, and a
bound runner preserves its exact dependency type.

Environment cleanup is bounded and best effort. Cleanup failures emit lifecycle
errors but do not replace an earlier run failure.

`Harness` asks the factory for a distinct lease for every session; it never
shares one environment instance implicitly across sessions.

## 21. Harness events and backpressure

The harness assigns a monotonically increasing durable cursor to each
authoritative/lifecycle entry and persists it inline before publication.
Preview events carry the latest durable cursor plus a transient per-turn ordinal;
they do not advance the durable cursor. Persistence is not a subscriber.

Each `Subscription` has a bounded queue and a nonblocking producer path:

- a full queue may drop/coalesce `Preview` events; the next delivered event
  includes `EventsDropped{Preview: n}`;
- a full queue for an `Authoritative` or `Lifecycle` event disconnects that
  subscriber with `ErrSubscriberLagged{LastCursor}`;
- the run never waits for a public subscriber;
- the consumer calls `Snapshot` and re-subscribes from a durable cursor to
  recover authoritative state.

```go
type SubscribeOptions struct {
    AfterCursor uint64
    Buffer      int
    Preview     bool
}

type Subscription struct {
    Events <-chan Event
    Err    <-chan error // exactly one terminal delivery, then close
}

func (s *Subscription) Close()
```

Historical replay contains authoritative/lifecycle events only. Preview starts
live after replay, so a cursor never promises recovery of token deltas.

Session code depends on an `event.Hub` and `event.Factory`. The bounded
process-local implementation lives in `event/inproc`; alternate hubs must pass
the same preview-gap, lag-disconnect, replay, close, and race conformance tests.

This is the only coherent way to combine bounded memory, nonblocking execution,
and no silent loss of authoritative events. “Never drop and never block” without
disconnect/spooling is impossible.

Callbacks, if offered as convenience wrappers, are invoked by the subscriber's
consumer goroutine and cannot fail the run. Low-level Agentic `EventSink` and
`TurnHook` are different: they are synchronous execution participants and their
errors return a partial execution as specified in §§10–13.

## 22. Interruption contract

`Interrupt`:

1. atomically marks the run interrupting and cancels its context;
2. prevents new tool admissions;
3. waits up to a configurable grace period for cancellation-aware model/tool
   work;
4. commits deterministic aborted/indeterminate results for every open call;
5. records a contextual marker explaining that interruption was deliberate,
   processes may still be running, and side effects may have partially occurred;
6. marks queued steer/follow-up entries cancelled while retaining `NextTurn`;
7. emits `RunInterrupted` and `RunEnd`; and
8. returns the session to idle.

The context passed to `Interrupt` bounds the caller's wait, not the cancellation
request. Once accepted, interruption continues even if that context expires;
the caller may observe completion through `WaitForIdle` or events.

Go cannot safely kill an arbitrary handler goroutine. Tool documentation must
require prompt context handling. A handler still running after grace is detached
from the session; its late return is ignored and an event records that fact.

The contextual marker is durable but filtered from user-facing chat and normal
memory extraction. The provider projection receives it as a harness-authored
context fragment using the best provider-supported role, with tagged user text
as the portable fallback.

## 23. Context policy and cache geometry

Context transformations distinguish durable from ephemeral injection:

```go
type TransformContext struct {
    Durable   *Transcript
    Ephemeral *[]agentic.Message
}
```

- Durable content survives future turns.
- Ephemeral content is re-derived for one request and never written back.
- Mutable reminders, budgets, and plans occupy one designated tail after the
  stable cached prefix.
- `Repair` is always last, after compaction and injection.

Ship two compactors: deterministic structured extraction and LLM summary with a
recent-message reserve. Both select cuts only at protocol-valid boundaries;
`Repair` is still the final guard.

## 24. Subagents and codemode

Subagents are Phase 4. They are bound runners created with an explicit capture
list:

```go
type Capture struct {
    History     Mode // isolate default
    Dependencies Mode // share default
    Environment Mode // share or narrow
    Tools       Mode // isolate default
    Permissions Mode // narrow default
    Budget      Mode // share default
}
```

Children get separate inboxes and transcripts. Parent steering never enters a
child inbox unless addressed to that child. Delegation tools are removed from
inherited toolsets unless recursion is explicitly enabled with a depth limit.
Child events are tagged and projected onto the parent bus; the bounded child
summary is the tool result.

Codemode is Phase 5 behind an `Executor` interface. It must use the same tool
gate, suspension, event, and recovery contracts; it does not introduce another
loop. Monty/gomonty maturity is contained behind that interface, with a
subprocess backend preferred for crash isolation.

---

# Part IV — Shipped retrieval track

## 25. `v0.3.0` status

The retrieval work landed in `3cd5c46` and was published by tag `v0.3.0`:

- `core.Reranker` and root facade helpers;
- Voyage AI and Cohere rerankers;
- Gemini, Ollama, Cohere, and Bedrock embedders;
- shared internal batch fan-out/chunking utilities; and
- retrieval transport/e2e coverage.

The Bedrock embedder supports Titan fan-out and Cohere batching. Both AWS Bedrock
model contracts document a maximum of 96 inputs for Cohere Embed v3/v4, so the
current `cohereBatchLimit = 96` and `embedbatch.Chunked` call are intentional.
The original statement that this cap could not be sourced was wrong.

The following original polish item did **not** ship and is not a harness
dependency:

- optional public `BatchLimit()` hints and an `agentic.EmbedChunked` helper.

Track that separately as retrieval API work; do not put it on the harness
critical path.

---

# Part V — Delivery and proof

## 26. Implementation sequence

| Phase | Deliverable | Exit condition |
|---|---|---|
| 0 | This corrected production design | Complete |
| 1 | Root `v0.4.0` shared fold and `Driver[O]` | Blocking/streaming/typed parity and all terminal-pair invariants pass |
| 2 | Monorepo scaffolding plus port-driven harness runtime/session/events/repair/env/artifacts and concrete adapters | Both modules pass independently with `GOWORK=off`; adapter conformance, architecture direction, restart, spill, and lag tests pass |
| 3 | Capability graph, context policy, permissions, deferred resume, `Default()` | `harness/v0.1.0` experimental; default uses only public capability APIs |
| 4 | Capture-restricted subagents and remaining toolset wrappers | Parent/child routing, budget, recursion, and cancellation tests pass |
| 5 | Codemode, memory, evals, optional product surface | Separate proposal and release |

Do not combine Phase 1 with harness scaffolding in one review. The execution
contract needs focused proof before a second module depends on it.

## 27. Required tests

### Agentic `v0.4.0`

- parity matrix: text/typed × blocking/streaming × no-tool/output-tool/regular
  tools × exhaustive/early;
- valid typed output plus queued continuation produces a later final output;
- invalid typed output retries before the turn hook sees a terminal candidate;
- every output and skipped tool call has exactly one result;
- `DriveContinue` sends no implicit prompt and accepts history ending in tool
  results;
- `Driver.Resume` rejects a wrong frontier hash or incomplete decision set before
  executing handlers, then completes the suspended turn in original call order;
- `TurnContinue` without valid injected input is rejected;
- hook/sink errors return committed partial executions identically in blocking
  and streaming modes;
- batch gate suspension runs zero handlers;
- per-run tool overlays do not mutate the shared runner; and
- result processors preserve call identity/error truth and pair failures; and
- race tests for concurrent runner reuse.

### Harness `v0.1.0`

- linearized steer/follow-up versus the `Running -> Closing` race;
- durable queue acceptance, exactly-once drain, and restart recovery;
- durable cumulative usage and session-budget enforcement across restart;
- store failure before and after in-memory mutation, including `Faulted`
  rejection and reopen recovery;
- subscriber lag: preview gaps, authoritative disconnect, snapshot/resubscribe;
- crash points before tool start, after start, and after result;
- indeterminate tools never auto-repeat without idempotency declaration;
- deferred resume rejects missing/unknown/duplicate IDs before effects;
- oversized tool output spills once, remains session-scoped, and is retrievable
  only through its opaque handle;
- interrupt repairs every frontier and preserves `NextTurn`;
- repair idempotence and byte stability, including after compaction;
- reusable conformance suites for journal, event-hub, environment, and artifact
  adapters;
- a dependency-direction test proving runtime/session core imports no concrete
  adapter and no root `internal/...` package;
- independent session environment leases and deterministic cleanup/reopen;
- `Default()` can be reconstructed from exported capability APIs; and
- `go test -race` for session, event, and store packages.

Both modules inherit the repository's **97% aggregate coverage gate** from the
first testable PR. Critical state-machine packages have no coverage carve-out.
Keep the existing compile-fail fixtures for generic misuse and add fixtures for
incorrect driver/output pairing where compile-time enforcement is possible.

## 28. CI and release gates

Before `v0.4.0`:

```sh
make fmt
make vet
make lint
make test
make coverage-check
```

Before `harness/v0.1.0`, run equivalent targets inside `harness/`, the root
`GOWORK=off` job, the harness `GOWORK=off` job, and the workspace integration
job. Add a fresh temporary consumer that imports the released root and harness
versions with clean `GOMODCACHE` and `GOCACHE`; local workspace success is not a
release proof.

## 29. Explicitly deferred risks

- Exactly-once external side effects require tool-specific idempotency or an
  external transaction system.
- Distributed multi-process session ownership requires a lease-capable
  repository; the JSONL adapter only refuses concurrent local/file-lock owners.
- Remote/container environments need backend-specific TOCTTOU and cancellation
  audits.
- Codemode depends on experimental runtimes and remains isolated behind
  `Executor`.
- Subagent topology presets are recipes over capture semantics, not part of the
  v0.1 runtime.
- Public transcript fork/move APIs are deferred until branch ownership has a
  single-writer implementation.

These are deferred scope, not unresolved decisions in the v0.1 architecture.

---

## 30. Reference index

Reference behavior was checked against pinned local source snapshots:

- Pi `3da591ab74ab9ab407e72ed882600b2c851fae21`: separate start/continue entry
  points, boundary-drained steering, and harness commit ordering.
- Pydantic AI `9688a9cca2fcf81451cdac35e701250dbdfec75e`:
  pending-message priorities, undrained-message errors, deferred results, and
  history continuation without a prompt.
- Pydantic AI Harness `e4839c654b855ce52d09348930e6858c165a1e74`:
  capability ordering, persistence, environments, and codemode boundaries.
- OpenAI Codex `3b948d9dd8d2e13a36e95a05efb6bb2288b801c4`:
  typed steering rejection, safe-boundary draining, and explicit lag signaling.
- OpenCode `32ce0f4b0d1a5015c965676c9feae341b13b87a5` and claw-code
  `2d5f83698893e45e5340f0f1fe854ae7ad87b22b`: transcript repair,
  permission, and interruption comparisons.

Authoritative external contracts:

- Go multi-module repositories and workspaces:
  <https://go.dev/doc/modules/managing-source> and
  <https://go.dev/doc/tutorial/workspaces>
- Pydantic AI deferred tools and message history:
  <https://ai.pydantic.dev/deferred-tools/> and
  <https://ai.pydantic.dev/message-history/>
- Cohere Embed v3 on Bedrock:
  <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html>
- Cohere Embed v4 on Bedrock:
  <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html>
- Pi source:
  <https://github.com/earendil-works/pi/tree/3da591ab74ab9ab407e72ed882600b2c851fae21>
- OpenAI Codex source:
  <https://github.com/openai/codex/tree/3b948d9dd8d2e13a36e95a05efb6bb2288b801c4>
