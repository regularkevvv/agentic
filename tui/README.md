# Agentic Harness TUI

`github.com/regularkevvv/agentic/tui` is a provider-neutral terminal client for
durable interactive sessions. Applications can implement `tui.Host` directly
or attach an already assembled Agentic Harness through `tui/adapter/harness`.
The core TUI never chooses a provider, executes a tool, or broadens permission
policy.

The composer keeps a padded three-row surface while its text area grows with
wrapped or multiline input. Transcript gutters, quiet user cards, and
assistant bullets replace explicit `USER`/`ASSISTANT` headings. The transcript
renders a conservative Markdown subset, folds one tool call's
planned/running/result records into one activity row, groups adjacent tool-only
records by semantic category, and keeps provider thinking collapsed by default.
Thinking is model reasoning, not user-facing
progress commentary; `Ctrl+T` cycles collapsed, hidden, and explicitly visible
modes.

By default, the standard command requires a coherent local environment: file
tools are confined to the workspace with `os.Root`, while command processes use
Seatbelt on macOS and Landlock plus seccomp on Linux. Both facets share the same
workspace root. Permission prompts still govern requested effects independently
of kernel isolation. Network creation is denied, writes are limited to the
workspace and a private per-command temporary directory, and unrelated
host/user paths are not readable. Windows currently fails closed because no
native backend is implemented. This default adds no third-party sandbox package;
it uses the existing `x/sys` dependency and platform facilities.

Applications may instead set `standard.Config.Environments` to any `env.Factory`
whose session leases present filesystem and shell operations over one logical
workspace, such as a container, VM, or remote executor. Because Standard also
installs `delegate_task`, these leases must implement `env.Narrower`; incomplete
leases are closed and rejected when the session opens. Standard never falls
back from a supplied factory to host execution. `ExecutionLabel` is display-
only metadata and is not treated as evidence of confinement.

The standard assembly also registers `delegate_task`. A model may issue up to
four delegation calls in one tool batch; they run as fork/join child sessions
with isolated history, narrowed workspace tools and permissions, and shared
usage accounting. The parent waits for all results. This is deliberately not a
background spawn/wait protocol.

Run the credential-free demonstration:

```sh
go run ./examples/offline
```

Type a task to see streamed text, tool state, usage, and cache-hit metrics. A
task containing `permission` opens the exact-decision approval overlay. The
standard launcher is available with `go run ./cmd/agentic-harness --offline`.

For a live profile, create `~/.config/agentic/config.toml` (the OS-specific user
config directory is used) and set the selected provider's existing environment
credential:

```toml
[ui]
alternate_screen = "auto"
color = "auto"
thinking = "collapsed"
tool_details = "collapsed"
preview_hz = 60

[profile.work]
provider = "anthropic"
model = "claude-sonnet-4-6"
context_window_tokens = 200000
system_prompt_file = "~/.config/agentic/work.md"
permission = "workspace-write"
```

Launch the command from the repository it should operate on. The standard
executable uses its current working directory as the workspace root. An
explicit `--workspace PATH` takes precedence, followed by an optional
`[profile.work.workspace] root = "..."` profile setting, then the current
directory. The Harness still canonicalizes the resolved root and confines its
file environment to it.

Install the command once from this module, then launch it from any repository:

```sh
go install ./cmd/agentic-harness
cd /path/to/repository
"$(go env GOPATH)/bin/agentic-harness" --profile work
```

No workspace flag is needed in the normal case; the second command operates on
`/path/to/repository` because that is its current working directory.

The standard launcher currently registers `anthropic`, `openai`, and
`openrouter`. Configuration files never contain credentials. `NO_COLOR` is
honored in automatic color mode; `--no-alt-screen` preserves scrollback.
`session.resume_last` defaults to true and uses a mode-0600 sidecar pointer in
the configured session directory; it never inspects or rewrites Harness
journals. Set it to false or pass `--resume ID` for explicit control.

The footer shows only keys relevant to the current state; `/help` contains the
complete inventory. `Enter` submits, `Alt+S` steers,
`Alt+F` queues a follow-up, `Alt+N` queues the next turn, `Ctrl+C` interrupts,
`Shift+Enter` or `Ctrl+J` inserts a newline, Up/Down navigates input history,
and `Ctrl+E` opens `$VISUAL`/`$EDITOR` after Bubble Tea releases the terminal.
`Ctrl+T` cycles thinking visibility and `Ctrl+G` toggles safe tool summaries.
The mouse wheel or trackpad and Page Up/Page Down scroll the transcript. New
output follows the bottom until the operator scrolls up, then stays put until
they return to the bottom. Submitted tasks appear immediately and reconcile
with the durable Harness event without rendering twice.
Commands include `/new`, `/resume ID`, `/help`, `/clear`, `/export`,
`/thinking`, `/tools`, and `/quit`. Export is a redacted plain-text value derived
from the conservative snapshot unless the host supplies its own exporter; raw
tool arguments and results are not part of the default view.

Harness applications may set `RuntimeConfig.ToolSummarizer` at the capability
boundary to turn raw calls into bounded, non-sensitive display text. Compatible
hosts may then populate `tui.Tool.Presentation`, and Agentic Harness applications
may pass `adapter/harness.WithToolPresenter`, to provide a category, title, and
detail for tool activity. Presenters receive only that already-redacted TUI tool
projection; they cannot inspect raw arguments or results. The standard assembly
shows file paths and command argument vectors while omitting stdin and environment
variables and redacting common credential flags, assignments, and headers. After
the operator submits a complete approval
decision set, the approval card is replaced immediately by a muted resolving
status; it is restored only if resolution fails.

The tag-triggered release workflow additionally copies `testdata/consumer` to
a fresh directory, turns its `go.mod.template` into a standalone module with no
`replace`, and runs a compatible host through the published TUI API. Keeping
the manifest as a template avoids adding another repository module while
preserving a reproducible clean-consumer proof.
