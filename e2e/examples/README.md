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
```

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
