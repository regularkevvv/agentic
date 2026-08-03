# Examples

Each example demonstrates a core capability of Agentic.

These programs are part of the nested `e2e` module, not the library, so
anything they import to be readable stays out of every consumer's module graph.
The commands below are run from the repository root, where the committed
`go.work` resolves the module.

| Example | What it shows |
|---------|---------------|
| `basic/` | Simple agent with no tools |
| `tools/` | Auto-registered tool with struct-tag schema |
| `structured/` | Typed agent with validated structured output |
| `retrieval/` | Hybrid retrieval: dense and learned sparse from one call |
| `sparse/` | What a learned sparse vector is, and what it costs |
| `codemode/` | Full Harness session through Codemode, GoMonty, a Monty worker, and a nested Go tool |
| `tui/` | Real Harness-to-TUI adapter with streaming, permission deny/approve, cache proof, and durable recovery |

## Setup

```bash
cp .env.example .env   # fill in at least OPENAI_API_KEY
```

## Run

```bash
go run ./e2e/examples/basic
go run ./e2e/examples/tools
go run ./e2e/examples/structured
go run ./e2e/examples/retrieval  # needs DEEPINFRA_TOKEN
go run ./e2e/examples/sparse     # needs DEEPINFRA_TOKEN
go run ./e2e/examples/codemode   # no credential; explicitly downloads and verifies GoMonty
go run ./e2e/examples/tui        # no credential; deterministic local acceptance flow
```

An opt-in live TUI smoke asks a configured real provider to request a shell
tool, approves the exact durable suspension, and verifies the workspace effect:

```bash
cd e2e
AGENTIC_TUI_LIVE=1 \
AGENTIC_TUI_LIVE_PROVIDER=anthropic \
AGENTIC_TUI_LIVE_MODEL=claude-sonnet-4-6 \
ANTHROPIC_API_KEY=... \
go test -tags=e2e -run TestLiveProviderHarnessTUIFlow ./tui
```

The Codemode example is deterministic and does not call an LLM provider. Its
scripted model requests `run_code`; the downloaded GoMonty runtime starts a real
Monty worker, the Monty program calls a selected Go tool, and the restored
program returns `42` through the durable Harness session. No native binary is
stored in this repository. GoMonty verifies the release archive, shared library,
and worker against hashes in its published source module before execution.

Download is the default. To build the same manifest-pinned runtime from the
reviewed Rust source instead, use:

```bash
go run ./e2e/examples/codemode -prepare=build -timeout=15m
```

The opt-in test runs the same shared scenario as the example:

```bash
cd e2e
GOMONTY_CACHE_DIR="$(mktemp -d)" GOMONTY_E2E=1 GOWORK=off \
  go test -race -count=1 -timeout=3m ./codemode
```

`GOMONTY_CACHE_DIR` is optional. Giving it an empty temporary directory makes
the command prove a fresh download; GoMonty still rechecks cached files before
every load when its normal user cache is used. The worker subprocess provides
crash isolation and timeout enforcement, not an OS security sandbox.

Pass a custom prompt as an argument:

```bash
go run ./e2e/examples/tools "What's the weather in Tokyo in fahrenheit?"
```

## Credentials

Each example reads its key from the environment. Put them in a `.env` file at
the repository root — it is gitignored, and the examples walk up from wherever
you run them to find it:

```
OPENAI_API_KEY=...
DEEPINFRA_TOKEN=...
```

Or export them in your shell instead. If you keep a key in the macOS Keychain,
note that this does not make it an environment variable; pass it in explicitly:

```bash
export DEEPINFRA_TOKEN=$(security find-generic-password -a "$USER" -s deepinfra-token -w)
```
