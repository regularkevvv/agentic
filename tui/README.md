# Agentic Harness TUI

`github.com/regularkevvv/agentic/tui` is a provider-neutral terminal client for
durable interactive sessions. Applications can implement `tui.Host` directly
or attach an already assembled Agentic Harness through `tui/adapter/harness`.
The core TUI never chooses a provider, executes a tool, or broadens permission
policy.

The standard command's status line explicitly labels execution as
`local-host governance (not an OS sandbox)`. Permission prompts govern effects;
they do not add kernel isolation.

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

[profile.work.workspace]
root = "/absolute/path/to/repository"
```

The standard launcher currently registers `anthropic`, `openai`, and
`openrouter`. Configuration files never contain credentials. `NO_COLOR` is
honored in automatic color mode; `--no-alt-screen` preserves scrollback.
`session.resume_last` defaults to true and uses a mode-0600 sidecar pointer in
the configured session directory; it never inspects or rewrites Harness
journals. Set it to false or pass `--resume ID` for explicit control.

Interaction keys are shown in the footer. `Enter` submits, `Alt+S` steers,
`Alt+F` queues a follow-up, `Alt+N` queues the next turn, `Ctrl+C` interrupts,
`Shift+Enter` or `Ctrl+J` inserts a newline, Up/Down navigates input history,
and `Ctrl+E` opens `$VISUAL`/`$EDITOR` after Bubble Tea releases the terminal.
`Ctrl+T` cycles thinking visibility and `Ctrl+G` toggles safe tool summaries.
Commands include `/new`, `/resume ID`, `/help`, `/clear`, `/export`,
`/thinking`, `/tools`, and `/quit`. Export is a redacted plain-text value derived
from the conservative snapshot unless the host supplies its own exporter; raw
tool arguments and results are not part of the default view.

The tag-triggered release workflow additionally copies `testdata/consumer` to
a fresh directory, turns its `go.mod.template` into a standalone module with no
`replace`, and runs a compatible host through the published TUI API. Keeping
the manifest as a template avoids adding another repository module while
preserving a reproducible clean-consumer proof.
