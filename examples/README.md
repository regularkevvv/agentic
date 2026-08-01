# Examples

Each example demonstrates a core capability of Agentic.

| Example | What it shows |
|---------|---------------|
| `basic/` | Simple agent with no tools |
| `tools/` | Auto-registered tool with struct-tag schema |
| `structured/` | Typed agent with validated structured output |
| `retrieval/` | Hybrid retrieval: dense and learned sparse from one call |

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
```

Pass a custom prompt as an argument:

```bash
go run ./examples/tools "What's the weather in Tokyo in fahrenheit?"
```
