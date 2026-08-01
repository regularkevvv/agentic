# Harness Phase 5: Code Execution, Memory, Skills, and Evals

**Status:** Accepted implementation proposal
**Date:** 2026-08-01
**Depends on:** Agentic `v0.5.1` and harness Phases 2-4
**Release target:** first harness module release (`harness/v0.1.0`)

This proposal is the Phase 5 boundary required by
[`spike-harness-framework.md`](spike-harness-framework.md). It completes the
remaining named library surfaces without turning one interpreter, memory
database, skill format, or evaluation reporter into a runtime dependency.

## 1. API audit and decisions

The released root driver already owns the only model/tool fold, immutable
per-run tool overlays, atomic tool gates, serializable suspension, exact resume,
canonical events, and result processing. Phase 5 does not add a second agent
loop.

One narrow root facade was missing for code execution: a tool handler could not
yield a nested, serializable deferral after it had started. Returning an error
would commit a result and ask the model to retry, losing the existing durable
`ExecutionSuspended` contract. Agentic `v0.5.0` supplies the optional,
capability-neutral suspendable-handler protocol required here:

- a handler explicitly declares that it may suspend;
- a suspendable handler must be the only regular call in its outer
  batch, which is checked before any handler starts;
- it may return a typed `ToolHandlerSuspension` containing an ordinary
  `ToolDeferral`;
- the driver leaves the outer call frontier open and returns the same
  `ExecutionSuspended` used by gate suspension; and
- on `Driver.Resume`, the handler receives the original deferral and an opaque,
  caller-validated decision payload through `CurrentToolResume`.

The protocol is public and capability-neutral. The root still knows nothing
about code execution, permissions, or harness storage. Existing handlers and
the `Runner` API are unchanged.

The pinned reference harness uses a snapshot/resume interpreter and routes
nested calls through its host tool manager. It converts an unresolved nested
approval into a model retry. Agentic deliberately does not copy that last
behavior: nested approval must suspend durably or fail before an effect, never
be smuggled through model-visible retry text.

## 2. Phase 5 scope

Phase 5 ships four independent surfaces:

1. `codemode`: a capability-neutral executor protocol, selected-tool wrapper,
   durable nested-call projection, exact nested deferral/resume, and a
   crash-isolated subprocess protocol adapter.
2. `memory`: a namespaced, bounded, optimistic-concurrency store port; memory
   and durable JSONL adapters; on-demand tools; and optional ephemeral prompt
   injection.
3. `skill`: a bounded catalog/source port; memory and filesystem `SKILL.md`
   adapters; and on-demand list/read tools.
4. `eval`: deterministic case execution, generic evaluators, harness-session
   subjects, aggregate metrics, and JSON reporting.

`harness.Default` remains unchanged. Code execution, cross-session memory,
skills, and evals are opt-in because each requires application choices about
trust, tenancy, storage, or cost.

The optional product surface remains optional: Phase 5 does not add a CLI,
TUI, hosted service, vector database, provider-specific evaluator, or bundled
language runtime. Out-of-process subagent workers, public transcript branches,
and distributed session scheduling remain outside this phase.

## 3. Code execution contracts

### 3.1 Executor port

An executor is a session-independent state machine:

```go
type Executor interface {
    Start(context.Context, Request) (Step, error)
    Resume(context.Context, Checkpoint, []CallResult) (Step, error)
}

type Step struct {
    Checkpoint Checkpoint
    Calls      []Call
    Output     any
    Stdout     string
    Done       bool
}
```

Exactly one of `Done` or a non-empty call batch is valid. A nonterminal step
must include an opaque checkpoint. Calls have executor-stable IDs, names, and
JSON-shaped keyword arguments. The host never interprets checkpoint bytes.
Executor call IDs remain unchanged at the protocol boundary; nested handlers,
effect idempotency, and result processors receive a collision-safe host ID
derived from the outer call, executor step, and executor call ID.
Inputs and outputs are defensively copied and bounded.

The executor does not receive tool handlers. It can only yield proposed calls.
The codemode host validates those calls against the selected frozen toolset,
passes the complete batch through the harness's already-composed gate, records
planned/started/result/suspended facts durably, applies the session result
processor, and resumes the executor with ordered results.

### 3.2 Suspension and recovery

If the nested gate suspends, no nested handler starts. Codemode combines the
executor checkpoint, proposed calls, and gate deferral into a versioned outer
deferral. The root suspendable-handler seam returns the ordinary session
suspension. A codemode resume planner validates missing, unknown, and duplicate
nested resolutions before producing one root resume decision for the outer
`run_code` call.

Approved and pre-allowed nested calls execute in original order semantics;
denials and external results are supplied to the executor without invoking a
handler. A resumed program may yield another suspension. Every transition is
bounded by a configured maximum step count.

The crash rules are conservative:

- once an outer suspension is durable, its checkpoint may be resumed without
  starting a nested call twice;
- a crash after the outer call starts but before its suspension or result is
  durable is conservatively indeterminate, even if no nested start was seen;
- crash after a nested start but before its result: the outer `run_code` call is
  indeterminate and is never automatically repeated; and
- crash after a nested result but before the outer result: the outer call is
  still indeterminate, because arbitrary executor progress is not assumed
  replay-safe.

### 3.3 Tool projection and safety

CodeMode wraps explicitly selected capability toolsets. Selected definitions
are hidden from the model and rendered into the `run_code` catalog; unselected
tools remain native. Name collisions, invalid identifiers, duplicate selected
names, and selecting another suspendable/code-execution tool are build errors.

The subprocess adapter speaks one versioned JSON request/response per process
invocation and treats the process as untrusted. It enforces context
cancellation, output limits, exit-status reporting, and schema validation. A
subprocess provides crash isolation only. It is **not** an OS sandbox; the
configured executable determines language semantics and containment. No shell
command string is accepted—the adapter executes one fixed binary plus fixed
arguments.

## 4. Memory contracts

Memory is a generic store, not session persistence and not a vector database.
The model never supplies a tenant scope.

```go
type Store interface {
    Read(context.Context, Scope, string, ReadOptions) (File, error)
    List(context.Context, Scope, ListOptions) ([]string, error)
    Mutate(context.Context, Scope, Mutation) (MutationResult, error)
}

type Searcher interface {
    Search(context.Context, Scope, SearchOptions) (SearchResult, error)
}
```

All reads, listings, and searches require positive finite bounds. Paths are
relative slash-separated identifiers; absolute paths, traversal, empty
segments, and backend control filenames are rejected. Mutations combine:

- compare-and-swap against an expected opaque version;
- one atomic append, replace, or delete operation; and
- an idempotency key plus request fingerprint.

Replaying the same key and fingerprint returns the original result. Reusing a
key for different arguments is an operation conflict. A stale version is a
normal conflict and never overwrites newer data.

The in-memory and JSONL adapters pass the same conformance suite. JSONL uses a
per-scope file lock, sync-before-ack mutation records, parent-directory sync on
creation, and the established diagnostic-sidecar/truncate rule for only a
trailing partial record.

The memory capability contributes bounded `read_memory`, `write_memory`,
`delete_memory`, and `search_memory` tools. Operation IDs derive from the
session and tool-call IDs. Optional prompt injection reads a bounded main note
and file listing into an ephemeral, delimited user-role context fragment on
each provider request. Stored content is data, not trusted system instruction.
Injection errors are explicitly fail-open or fail-closed; scope-resolution
errors always fail closed.

## 5. Skill contracts

Skills use a source port so discovery is independent of files:

```go
type Source interface {
    List(context.Context, Scope, int) ([]Descriptor, error)
    Read(context.Context, Scope, string, int) (Skill, error)
}
```

Descriptors contain a stable name and bounded description. A loaded skill
contains bounded instructions plus optional opaque resource names. The model
cannot choose the source scope. Duplicate names, invalid identifiers, and
oversized content fail deterministically.

The capability exposes `list_skills` and `read_skill`. Skill instructions enter
the durable transcript as an ordinary tool result, so restart does not depend
on process-local activation state. They are explicitly framed as
application-provided guidance but remain below the agent's root system prompt.

The filesystem adapter discovers one directory per skill and parses the
`name`/`description` frontmatter from `SKILL.md`, rejects symlink escape, sorts
results bytewise, and never returns an underlying path. The memory adapter is a
real concurrent implementation used by tests and embedders, not a placeholder.

## 6. Evaluation contracts

Evals are ordinary Go values and do not mutate production configuration:

```go
type Subject[I, O any] interface {
    Run(context.Context, Case[I]) Outcome[O]
}

type Evaluator[I, O any] interface {
    Evaluate(context.Context, Case[I], Outcome[O]) (Score, error)
}
```

The runner validates unique case/evaluator IDs, executes each requested sample
in an isolated subject invocation, bounds concurrency and per-sample time, and
returns results in stable case/sample/evaluator order regardless of completion
order. Cancellation stops admission and is represented honestly; evaluator
errors never become passing scores.

The harness subject opens a fresh session per sample, optionally applies a
budget, captures durable events, closes the session, and returns output,
transcript, usage, cursor, duration, and execution error. Built-in deterministic
evaluators cover exact text, substring, error expectation, and custom
functions. Aggregate metrics report pass counts and means without hiding
missing/error scores. The JSON reporter writes one versioned document to an
`io.Writer` and owns no files or network clients.

## 7. Required proof

In addition to all Phase 2-4 gates, Phase 5 must prove:

- suspendable handlers are isolated before effects and preserve every existing
  gate-suspension invariant;
- nested allow/deny/ask batches use the composed gate, and nested ask resumes
  exactly once through the original executor checkpoint;
- nested started-without-result recovery is indeterminate and never repeats;
- subprocess cancellation, malformed protocol, oversized output, and non-zero
  exit behavior;
- memory adapter conformance, stale CAS, operation replay/conflict, bounded
  reads/search, scope isolation, partial-tail repair, and cross-instance races;
- memory injection is ephemeral, bounded, current on every request, and uses no
  model-controlled scope;
- skill ordering, bounds, duplicate rejection, symlink escape rejection, and
  scope isolation;
- eval isolation, deterministic ordering under concurrency, cancellation,
  timeout, scorer error honesty, aggregation, and JSON stability;
- root, harness release-view, and combined workspace race suites;
- root and harness aggregate coverage at or above 97%; and
- architecture tests proving core packages import ports rather than concrete
  adapters and the harness imports no root `internal/...` package.

## 8. Release

No harness module tag exists at this proposal's baseline. The prerequisite root
facade shipped in Agentic `v0.5.0`; Agentic `v0.5.1` adds the public nested
tool-invocation context helper that prevents selected handlers from inheriting
an outer code-execution call or resume context. The harness module must require
`v0.5.1` and pass with `GOWORK=off`. After the Phase 5 PR is merged
and all hosted checks are green, release `harness/v0.1.0` from that merge
commit. Before tagging, verify a fresh consumer with clean module and build
caches against released `github.com/regularkevvv/agentic v0.5.1` and the
candidate harness module. The implementation PR itself must not move the root
module path or add a committed `replace` directive.
