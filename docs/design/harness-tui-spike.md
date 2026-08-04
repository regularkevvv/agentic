# Harness CLI/TUI Spike

**Status:** Proposed after a code-grounded spike
**Date:** 2026-08-02
**Depends on:** the experimental `github.com/regularkevvv/agentic/harness`
module and a small presentation-observation addition described below

## Decision

Create a sixth, pure-Go nested module at `tui/`, published as
`github.com/regularkevvv/agentic/tui`. Its normal import, build, and runtime
path must not require GoMonty or preparation of a native Monty shared library.
It is a terminal **client** of a small, presentation-neutral interactive-session
port. It does not contain another agent loop, tool gate, persistence model,
permission policy, provider transport, or model catalogue.

The module supplies two routes:

1. An application implements the port for its own compatible harness.
2. `tui/adapter/harness` adapts Agentic Harness sessions once Harness exposes
   a typed observation projection. `tui/standard` is an optional convenience
   assembly that builds an Agentic model and `harness.Default` from explicit
   application configuration.

“Any harness” therefore means *any harness that implements this interactive
session contract*, not every arbitrary Go agent library without an adapter.
That distinction keeps the interface honest: a terminal client needs durable
snapshots, a live event subscription, queue semantics, interruption, and
operator resolution to render a correct interactive task.

## What the current repository already provides

The Harness is close to a strong terminal-client substrate:

- `harness.Harness[O]` is a concrete immutable session factory. It takes a
  ready `agentic.Runner[O]` and requires the stronger `agentic.Driver[O]`
  capability; it does not expose one pre-existing public `Harness` interface.
- A `Session` has a single-flight state machine, durable snapshots, durable
  cursor-ordered events, `Prompt`, `Steer`, `FollowUp`, `NextTurn`, `Interrupt`,
  `Resume`, and `Close`.
- Preview events are intentionally nonblocking and may be dropped. Durable
  records have cursors, and a lagged authoritative subscription reports an
  error. This is exactly the recovery model a TUI needs.
- `harness.Default` assembles the local filesystem, JSONL session repository,
  artifact spilling, context policy, and the `WorkspaceWrite` permission
  policy. It is explicitly governance, not an OS sandbox. Shell and network
  effects are `Ask`; local workspace filesystem operations are allowed.

The core boundary is good: an application can construct its own `Runner`,
runtime adapters, and capability graph, then pass the bound runner to
`harness.NewRuntime` or `harness.New(...).Build()`. The TUI must preserve that
direction. In particular, the TUI must never rebuild an agent from a provider
string behind an application's back.

## The two gaps to close before building the UI

The gaps are deliberately small Harness additions, not reasons for the TUI to
reach into private state or deserialize private wire formats.

### 1. Presentation-neutral typed observation

The public subscription currently emits `harness.Event` records whose `Payload`
is encoded by the session's configured `codec.Codec`. A session built with a
custom codec cannot be decoded by an out-of-module TUI, and `harness.Default`
does not expose its codec. Re-parsing JSON in a terminal adapter would work
only by accident and make the event wire format a hidden public API.

Add a small public observation projection owned by Harness, for example:

```go
// Package observe has no terminal imports. It is useful to TUI, CLI, web,
// audit, and test consumers.
package observe

type Event struct {
    Cursor    uint64       // zero only for a preview
    Nature    Nature       // Preview, Authoritative, Lifecycle
    SessionID string
    ParentID  string
    Agent     string
    Depth     int
    Turn      int
    Kind      Kind
    TextDelta string       // populated only for text preview
    Thinking  *Thinking    // redaction metadata is retained
    Tool      *ToolEvent   // call ID, name, state, safe presentation data
    Usage     *Usage
    Suspension *Suspension
    Dropped   uint64
}

type Subscription interface {
    Events() <-chan Event
    Errors() <-chan error
    Close()
}
```

`Session.Observe` performs the codec-specific decoding inside Harness and
returns copies. Its stream preserves the current invariant: durable records
are cursor ordered; previews are best effort; `Dropped > 0` means the client
must redraw from `Snapshot`; a lag error means it must resubscribe from its
last cursor and redraw. It is not a terminal schema and does not embed Bubble
Tea, ANSI, markdown, or configuration.

The projection must not make tool arguments automatically visible. The default
tool event contains call identity and phase. An application may set
`RuntimeConfig.ToolSummarizer` at the capability boundary to attach an
explicitly safe, bounded summary. A later `tui.ToolPresenter` sees only that
projection and selects its category, title, and detail. This keeps secrets in
a tool input or result out of a default terminal transcript while allowing
tool-specific presentation without hardcoding it into the renderer.

### 2. Public operator view of a permission suspension

`runtime.InspectDeferred` correctly validates permission suspension IDs and
call sets, but the `PermissionRequest` list is encoded in an unexported payload
type in `permission`. A UI currently cannot display the canonical action and
resource without knowing that private JSON schema.

Add `permission.InspectSuspension(agentic.Suspension) ([]Approval, error)`,
where an `Approval` contains only:

```go
type Approval struct {
    CallID   string
    ToolName string
    Request  PermissionRequest // capability, action, canonical resource
}
```

It must validate through the existing runtime inspector and return approvals in
the original executable-call order. The UI presents this data, collects one
explicit approve/deny decision for every required ID, then invokes the existing
`Session.Resume`. It never treats an unknown suspension kind as a permission
prompt and never turns “Ask” into “Allow”. A capability-specific suspension can
later provide its own `SuspensionPresenter`; until then the UI shows the durable
kind and offers a safe exit/handoff rather than inventing a resume payload.

## The `tui` public port

The port is intentionally textual and non-generic. A terminal needs rendered
conversation data, not an application output type `O`; the typed result stays
with the application that owns its harness.

```go
package tui

type Host interface {
    NewSession(context.Context, SessionOptions) (Session, error)
    ResumeSession(context.Context, string) (Session, error)
}

// Session operations may block until the underlying run reaches an idle,
// suspended, interrupted, or failed durable boundary. The UI invokes them in
// workers, never from its reducer.
type Session interface {
    ID() string
    Snapshot(context.Context) (Snapshot, error)
    Subscribe(SubscribeOptions) Subscription

    Submit(context.Context, Input) error
    Steer(context.Context, Input) error
    FollowUp(context.Context, Input) error
    NextTurn(context.Context, Input) error
    Resolve(context.Context, Resolution) error
    Interrupt(context.Context) error
    Close(context.Context) error
}

type Snapshot struct {
    Cursor      uint64
    State       State
    Transcript  []Entry
    Pending     []QueuedInput
    Suspension  *Suspension
    Usage       Usage
}

type Resolution struct {
    SuspensionID string
    Decisions    []Decision // exactly one for each required approval
    Prompt       *Input
}
```

`Entry`, `Event`, `Suspension`, `Tool`, `Usage`, and `Failure` are small TUI
data-transfer values. They have no `agentic.Message`, `agentic.ToolUse`, raw
JSON, `any`, model object, or provider key. This is what makes a custom host
possible without giving it access to Agentic internals. The adapter converts
the full Agentic transcript to conservative text/tool/thinking entries.

The contract requires an adapter to implement these semantics:

- `Submit` starts only from idle; `Steer`/`FollowUp`/`NextTurn` are distinct,
  durable requests, rather than one overloaded “send” operation.
- `Snapshot` is a full reconciliation source of truth.
- events may be transient, but a durable sequence never moves backward;
  dropped previews and lag are observable;
- `Resolve` is valid only for the exact current suspension and validates the
  complete decision set before resuming; and
- `Close` releases local resources but does not erase the durable session.

Optional interfaces add session listing, transcript export, custom tool
renderers, and provider/model selection. They must not enlarge the minimum
runtime port.

### Agentic adapter

`tui/adapter/harness` is the only package that imports Harness. Its primary
constructor is deliberately narrow:

```go
host, err := harnessui.New(runtime) // runtime is *harness.Harness[O]
```

It creates/resumes sessions, maps `Input` onto `Prompt`/queue methods, maps
the new observation stream into `tui.Event`, maps `Snapshot`, and uses the
new permission inspector to construct approval views. It does not call model
or provider constructors. The adapter reports an unsupported non-permission
suspension honestly; it does not decode or fabricate a resume request.

## Package and dependency layout

```text
tui/                                  nested module; terminal client API
  go.mod                              requires released Agentic and Harness core; no GoMonty or replace
  README.md                           embedding, configuration, accessibility
  port.go snapshot.go event.go         public compatible-harness contract
  app/                                Bubble Tea reducer and background bridge
  render/                             transcript, tool, approval, status views
  adapter/harness/                    Agentic Harness adapter
  standard/                           optional standard application assembly
  config/                             config loading, precedence, validation
  cmd/agentic-harness/                thin executable; no alternate agent loop
  internal/testhost/                  deterministic scripted compatible host
```

Use Bubble Tea v2, Bubbles v2, and Lip Gloss v2. Bubble Tea is a good fit for
Go here because its Elm-style `Init`/`Update`/`View` reducer makes the only UI
state mutation happen in one place, while blocking Harness calls and event
subscriptions run outside it. Version 2 adds explicit paste messages,
declarative alternate-screen and cursor settings, keyboard/mouse capability
handling, and a cell renderer that down-samples colors. It is sufficient for a
serious chat terminal without carrying a JavaScript or Rust runtime.

Do not make Bubble Tea a Harness dependency. A headless CLI reporter, a web
client, or an application with a different terminal framework can consume the
same `observe` projection.

The module is pure Go and belongs in `go.work`. It must depend on released
`github.com/regularkevvv/agentic` and
`github.com/regularkevvv/agentic/harness`, with no `replace` directive. That
gives it the same release-view discipline as Harness: `GOWORK=off` proves a
fresh consumer path. In the workspace, `go.work` supplies the local root and
Harness modules while implementing both sides of the new observation seam.

### Isolating the optional GoMonty backend

The existing `harness/codemode` package is a portable capability: it owns the
code-mode protocol and depends only on its `Executor` port. It should remain in
the Harness module. Its sibling `harness/codemode/gomonty`, however, imports
GoMonty. GoMonty's Go bindings are cgo-free, but they load an explicitly
prepared native Monty shared library at runtime. Today that makes GoMonty a
direct requirement of `harness/go.mod`, even for consumers that never select
Code Mode.

Before adding `tui/`, extract **only** that backend into a seventh nested
module, retaining its import path:

```text
harness/codemode/gomonty/             optional native-backed executor module
  go.mod                              requires released Harness and GoMonty
```

The extraction removes GoMonty (and its indirect dependencies) from
`harness/go.mod`. An application that wants Monty Code Mode explicitly adds
the `harness/codemode/gomonty` module, constructs the executor, and supplies
it to the portable `codemode` capability; the TUI merely attaches to the
resulting Harness. A consumer that imports only `tui` or its Harness adapter
neither compiles the GoMonty adapter nor needs a prepared Monty runtime.

This is intentionally not a separate module for all of `codemode`: the core
capability does not carry a native dependency and splitting it would add a
module boundary without buying isolation. The GoMonty adapter's tests, release
checks, architecture map entry, and `go.work` membership must move with the
new module. It is still cgo-free, so it belongs in `go.work` under the
repository's current rule, but it needs its own prepared-native-runtime CI
proof. `tui/standard` must not construct this executor implicitly.

## Application configuration and provider ownership

There are two configuration layers, which must not be conflated.

### TUI configuration: safe defaults

These are UI/application mechanics and may have defaults:

```toml
[ui]
alternate_screen = "auto" # auto | always | never
color = "auto"            # auto | always | never; honors NO_COLOR
thinking = "collapsed"    # visible | collapsed | hidden
tool_details = "collapsed"
preview_hz = 60

[session]
# empty means derive an OS data-directory location keyed by canonical workspace,
# never a child of workspace because harness.Default rejects overlap
directory = ""
resume_last = true
```

`auto` alternate-screen mode follows the Codex lesson: use a fullscreen
alternate buffer where it works well, but preserve scrollback in known
multiplexer cases and always permit `--no-alt-screen`. The transcript itself
has an in-app viewport/pager, so normal scrollback is not the only recovery
route. Color auto-detection must honor `NO_COLOR` and degrade cleanly to ASCII;
screen-reader and log-friendly modes are first-class, not an afterthought.

### Standard application configuration: explicit model profile

Provider, model, context-window size, system prompt, resolved workspace root,
session directory, and permission policy belong to the application assembly.
They are not defaults in the reusable `tui` package because choosing a model
silently chooses cost, capability, credential source, and context geometry.
The standard executable resolves its workspace from explicit `--workspace`,
then an optional profile root, then its current working directory; the reusable
Harness assembly still receives an explicit absolute boundary.

`tui/standard` has a registry of explicit `ProviderFactory` values and a
configurable named profile:

```toml
[profile.work]
provider = "anthropic"
model = "claude-sonnet-4-6"
context_window_tokens = 200000
system_prompt_file = "~/.config/agentic/work.md"
permission = "workspace-write" # read-only | workspace-write | custom
```

The example is a template, not a claim that this model is universally
available or appropriate. The launcher resolves `--profile`, then command
flags, workspace config, user config, and only then neutral UI defaults. It
fails before constructing a harness if provider/model/context geometry is
missing or the selected provider cannot obtain its documented credential. A
custom application may bypass `standard` entirely and provide its own `tui.Host`.

`ProviderFactory` has a small explicit seam:

```go
type ProviderFactory interface {
    ID() string
    New(context.Context, ModelConfig) (agentic.Model, error)
}
```

The standard command registers the Agentic providers it chooses to ship. It
does not import every provider into the core TUI library, and it never writes
keys to configuration or session journals.

## Interaction model

```text
terminal input ──> Bubble Tea reducer ──> operation worker ──> tui.Session
       ^                                         │                  │
       │                                         │                  ▼
       └── rendered model <── typed UI messages ─┴───── subscription/snapshot
                                                               │
                                                   Harness adapter / custom host
```

The Bubble Tea model owns only presentation state: current snapshot, transient
preview buffers, draft, focus/overlay, viewport, key bindings, error banner,
and timing. It never calls a blocking session method from `Update`.

An operation worker owns one call to `Submit`, queue, `Resolve`, or
`Interrupt`; it returns an outcome message to the reducer. A subscription
bridge forwards typed events to the reducer. A 60 Hz coalescer batches text
and thinking deltas in arrival order. If a preview is dropped, an
authoritative event is missed, or the subscriber lags, the bridge obtains a
fresh snapshot and resubscribes from the last durable cursor. No “best effort”
stream is allowed to silently become the transcript.

### First usable release

The first interactive release should make a real task pleasant and auditable:

- scrollable durable transcript with streamed assistant text;
- a multiline composer with bracketed-paste support, command history, and
  external-editor escape hatch;
- distinct `Enter` (submit), queue/follow-up action, steer action, and
  `Ctrl+C` interrupt semantics visible in the footer;
- collapsible thinking and tool timeline with start/result/error state;
- approval overlay that shows the canonical capability/action/resource, with
  explicit approve and deny for every required call;
- session ID, state, model-profile label supplied by the host, usage, pending
  message count, and current workspace in a compact status line;
- `new`, `resume`, `help`, `clear`, `export`, and `quit` commands; and
- normal and alternate-screen modes, width/height resilience, `NO_COLOR`, and
  a plain transcript export suitable for CI and support tickets.

Do not place broad plugin ecosystems, images, mouse-first interaction,
automatic compaction controls, branch trees, model pickers, or arbitrary tool
argument viewers in this release. The port permits later extensions, but an
approval-safe durable interactive task is the first bar.

## Lessons carried from the references

The local PI and Codex source trees were inspected, along with current Bubble
Tea upstream material.

| Reference | Evidence | Adopt | Do not copy yet |
|---|---|---|---|
| PI | `packages/agent/src/harness` keeps the reusable harness separate from `packages/coding-agent/src/modes/interactive`; its interactive mode subscribes to events and routes prompt/steer/abort asynchronously. | Separate reusable port, renderer, and standard application; refresh from durable state. | PI's complete extension/model/session-tree product surface. |
| Codex | `codex-rs/tui` uses an explicit app event bus, a stateful composer, approval overlays, snapshots, a transcript pager, and configurable alternate screen. | One UI reducer/message bus, operator overlay, viewport, careful composer, alt-screen modes. | Its large app-server protocol and product-specific configuration. |
| OpenCode | Its current OpenTUI/Solid client consumes a separate SDK event stream and batches UI events at approximately 16 ms. Its permission dialog can show diffs/input and offers once, always-for-process, and reject choices. | Keep the runtime/client split, bounded event coalescing, responsive overlays, and readable approval layouts. | Process-wide “allow always”, raw generic tool arguments/results, and the full server/SDK/product stack. |
| Codex streaming design | Stream chunks remain ordered, use queue age/depth to catch up, and record transitions. | Coalesce previews, preserve order, resync on loss, instrument backlog. | 120 Hz animation tuning before measuring Agentic event rates. |
| Bubble Tea v2 | A Go-native reducer with text areas, viewports, declarative terminal state, paste messages, cursor/mouse/key support, and modern renderer support. | Bubble Tea/Bubbles/Lip Gloss as the implementation stack. | A custom terminal renderer or a JavaScript/Rust bridge. |

## Security and durability invariants

1. The harness remains the only authority for tool execution, permission
   evaluation, durable acceptance, and resume planning.
2. The UI can approve only an existing exact suspension; it cannot broaden a
   permission policy, bulk-approve future calls, or execute a command itself.
   In particular, it has no OpenCode-like “allow always for this process”
   control; changing policy requires constructing a new Harness explicitly.
3. The default screen never renders raw tool input, raw tool result, API key,
   or opaque deferral data. A diff, command, or other detail view must come
   from a capability-owned, bounded, redacted presenter rather than generic
   decoding of a tool input. Safe tool presentation is opt-in and bounded.
4. `harness.Default` is displayed as local-host execution, not a sandbox.
5. Closing the UI closes the local session lease only after a running operation
   reaches a known boundary or the user explicitly chooses an interrupt path.
6. Recovery starts from a durable snapshot/cursor. Previews are not durable
   evidence and are always replaceable by a snapshot redraw.
7. Config files retain profile names and paths, never credentials. Existing
   provider environment/credential mechanisms remain authoritative.

## Implementation sequence and proof

### Phase T0 — presentation seam in Harness

- First extract `harness/codemode/gomonty` into its optional nested module and
  prove separately that (a) `harness` and `tui` build with `CGO_ENABLED=0` and
  no prepared Monty runtime, and (b) an explicit GoMonty Code Mode application
  still performs its native preparation and checkpoint/resume proof.
- Add typed observation and permission-inspection APIs with no TUI imports.
- Test every event mapping, copy/boundary behavior, dropped-preview resync,
  authoritative-lag resubscription, known/unknown suspension handling, and
  no raw sensitive input in the default projection.
- Add a deterministic scripted driver that exercises text preview, thinking,
  tool start/result, a permission suspension, denial, approval, interruption,
  recovery, and child event metadata.

### Phase T1 — reusable TUI and Harness adapter

- Add the `tui` module, adapter, deterministic test host, and pure view-model
  reducers before the executable.
- Use golden/snapshot tests at narrow, normal, and wide terminal dimensions;
  include no-color, inline, alternate-screen, unreadable cursor, large paste,
  dropped-preview, lag/replay, queued message, approval, and resize cases.
- Race-test the event bridge and interrupt/close paths. Test that no call made
  from `Update` blocks the UI loop.

### Phase T2 — standard command

- Add profile/config precedence and provider registry tests with fake
  factories. Validate missing geometry and credentials before runtime startup.
- Add an offline, credential-free runnable example using the scripted driver.
- Add an opt-in live smoke path with an explicitly named provider profile. It
  proves real provider -> Agentic runner -> Harness -> TUI event stream ->
  operator action, but never becomes the default unit-test gate.

### Release and repository integration

- Update `ARCHITECTURE.md`, the module map and directory map in
  `architecture_test.go`, `go.work`, `docs/README.md`, Make targets, and CI in
  the same change. The TUI and optional GoMonty backend are both cgo-free and
  therefore belong in the workspace; the backend additionally needs its own
  native-runtime preparation and compatibility CI, and TUI must not depend on
  it.
- Run root, Harness, TUI, and combined-workspace race suites; run the TUI
  module under `GOWORK=off` against released Harness. Add a fresh temporary
  consumer that imports only `github.com/regularkevvv/agentic/tui` and starts
  the deterministic compatible host.
- Retain the repository's coverage/lint/vet gates. Do not commit an executable
  created by `go build`.

The acceptance demonstration is not a mocked screen alone: a scripted
Harness task streams progress, asks for an actual durable permission decision,
is denied and resumed, is closed, then reopened by ID with the same transcript
and cursor. The optional live smoke repeats the flow with an explicitly
configured real provider.

## Open decisions for implementation

- Whether session listing belongs in the first Harness observation port or is
  a separate optional repository/listing port. It is not currently available
  from `harness.Harness`.
- Whether transcript export is TUI-owned (derived text) or a first-class
  Harness audit exporter. Start TUI-owned; add a Harness export only if other
  consumers need the same redaction semantics.
- Which provider factories ship in `tui/standard`. The core library should
  remain provider-neutral even if the executable chooses a curated registry.
- Whether an opaque custom-suspension presenter belongs in Harness capability
  registration or as a TUI adapter extension. Do not solve it by exposing raw
  deferral JSON.
