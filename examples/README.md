# Examples

Each example demonstrates a core capability of Agentic.

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
go run ./examples/basic
go run ./examples/tools
go run ./examples/structured
go run ./examples/retrieval  # needs DEEPINFRA_TOKEN
go run ./examples/sparse     # needs DEEPINFRA_TOKEN
```

Pass a custom prompt as an argument:

```bash
go run ./examples/tools "What's the weather in Tokyo in fahrenheit?"
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
