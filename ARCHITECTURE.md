# Architecture

Where every kind of code lives, and why. If you are adding something and this
document does not say where it goes, that is a bug in this document — say so in
the pull request rather than guessing.

Most of what follows is enforced by `architecture_test.go` in the root package.
Those tests exist so that this file cannot quietly become fiction: change the
layout without changing this document and the build fails.

## The shape

```
agentic/                    the library — one import path for callers
  doc.go                    package documentation
  agent.go driver.go …      chat: the agent loop
  embedding.go              retrieval: the three vector entry points
  representation.go
  reranking.go
  aliases.go                re-exports internal/core and internal/retrieval
  architecture_test.go      enforces this document

  tool/                     tool builders, registry, toolsets
  mcp/                      Model Context Protocol client and toolset

  provider/                 every provider, flat — see the README table
    test/                   the one exception: doubles and the contract suite
    local/onnx/             nested module (CGO)

  internal/                 not importable by callers
    core/                   chat primitives
    retrieval/              vector primitives
      wire/ batch/ embedbatch/
    providerhttp/           shared HTTP retry and bounded reads
    testutil/               doubles for the root package's own tests

  harness/                  nested module: durable sessions (experimental)
    codemode/gomonty/       nested module: optional GoMonty executor
  tui/                      nested module: reusable terminal client
  e2e/                      nested module: live tests and runnable examples
    localinference/         nested module (CGO): the example that uses onnx

  docs/                     what exists; docs/design/ records why
  testdata/                 compile-failure fixtures for the type-safety tests
```

## The two halves

Agentic solves two problems that share providers and nothing else.

**Chat** is the agent loop: send a conversation to a model, run the tools it
asks for, fold the results back, repeat. `internal/core`.

**Retrieval** is turning text into vectors and ordering results: `Embedder`,
`RepresentationEncoder`, `Reranker`. `internal/retrieval`.

Neither half references the other. No type in `internal/core` mentions a vector
and no type in `internal/retrieval` mentions a message, and a test asserts it,
because this is the boundary that erodes first — one convenience field is all it
takes, and after that the halves cannot be reasoned about separately.

The root package is deliberately *not* split, because Go gives a directory one
package and callers should have one import path. The line still holds there: the
retrieval surface is `embedding.go`, `representation.go`, and `reranking.go`, and
everything else is chat.

## Where does it go?

| What you are adding | Where | Why |
|---|---|---|
| A new provider | `provider/<vendor>/` | Flat. Declare capability with `var _ retrieval.Embedder = …`; a test fails if you don't |
| A provider that needs CGO or a native library | its own nested module | Must not enter the dependency graph of `go get agentic` |
| Something two or more providers share | `internal/providerhttp/` if transport, otherwise `internal/retrieval` or `internal/core` | The half it belongs to decides |
| A chat concept (message, tool call, stream event) | `internal/core/`, re-exported in `aliases.go` | |
| A retrieval concept (vector, space, encoder) | `internal/retrieval/`, re-exported in `aliases.go` | |
| A tool builder or toolset | `tool/` | Public; the root package aliases the types |
| A test double for *callers* | `provider/test/` | Public, because callers need it |
| A test double for *this repo* | `internal/testutil/` | Not part of the API |
| A contract suite a provider must pass | `provider/test/conformance/` | Beside the doubles, importable by an out-of-tree provider |
| A runnable example | `e2e/examples/<name>/` | The `e2e` module, so a table writer never reaches a caller's binary |
| A compatible-harness terminal client | `tui/` | Pure-Go client port, renderer, app, Harness adapter, and explicit standard assembly |
| A Code Mode backend that loads a native runtime | `harness/codemode/<backend>/` as a nested module | The portable Code Mode capability remains in Harness; the optional runtime stays out of its graph |
| A live test against a real API | `e2e/providers/`, behind `//go:build e2e` | Needs a key; never runs in the default `make test` |
| A record of why a decision was made | `docs/design/` | Not maintained afterwards; see `docs/README.md` |
| A description of what exists | `docs/` | Maintained; wrong if the code moves and it doesn't |

## The seven modules

A nested module exists to keep something out of a dependency graph, and for no
other reason. Depth means nothing to Go — `provider/local/onnx` is no more
"inside" the root module than `harness` is — so the only question that matters
is what `go get github.com/regularkevvv/agentic` should pull.

| Module | In `go.work` | Resolves the root as | Exists because |
|---|:-:|---|---|
| `.` | ✓ | — | The library |
| `harness/` | ✓ | **Agentic `v0.6.0`** | Experimental; its dependencies and its API churn should not be the library's |
| `harness/codemode/gomonty/` | ✓ | **Harness `v0.2.0`, GoMonty `v0.0.15`** | Optional native-backed Code Mode execution must not enter the core Harness graph |
| `tui/` | ✓ | **Agentic `v0.6.0`, Harness `v0.2.0`** | Bubble Tea and terminal application dependencies must not enter either library graph |
| `e2e/` | ✓ | this checkout | Live tests and examples need keys, table writers, and fixtures that no caller should inherit |
| `provider/local/onnx/` | | this checkout | CGO, ONNX Runtime, and a statically linked tokenizer |
| `e2e/localinference/` | | this checkout | The example that imports it, kept apart for the same reason |

The last two are absent from `go.work` on purpose: including them would make
`go build ./...` at the repository root fail for anyone who has not installed a
native ONNX Runtime. They are built, tested, and linted by the `onnx` CI job, so
they are gated — just not by the commands you run every day. The GoMonty
bindings are cgo-free and prepare their native runtime explicitly at execution,
so that optional module remains in `go.work`. `make lint-all` covers all seven
when you do have the ONNX libraries.

**`harness/`, `harness/codemode/gomonty/`, and `tui/` have no `replace`
directives, and that is deliberate.** They are released in dependency order:
root Agentic, Harness, the optional GoMonty adapter, then TUI. Their
`GOWORK=off` release checks rehearse what `go get` gives a user. Locally
`go.work` supplies the modules being changed together, so the distinction is
invisible day to day. The tag-triggered `release-view.yml` workflow enforces
this order against published dependencies; it can also be dispatched manually
for one module. A test fails if a replace directive erases the boundary.

The e2e modules replace the libraries they exercise because they are repository
acceptance code rather than release consumers. The two CGO modules also replace
the root for the same reason.

## What is enforced

Not by convention — by tests that fail.

| Rule | Where |
|---|---|
| Every provider declares its capabilities in code | `architecture_test.go` |
| The chat and retrieval halves do not reference each other | `architecture_test.go` |
| `go.work` lists every non-CGO module and no CGO module | `architecture_test.go` |
| Release modules have no replace; repository acceptance/CGO modules resolve this checkout | `architecture_test.go` |
| TUI and the optional GoMonty adapter contain no replace or cross-import | their module `architecture_test.go` files |
| `internal/` holds only the four documented packages | `architecture_test.go` |
| Every top-level directory is one this document names | `architecture_test.go` |
| No compiled binary is tracked in git | `architecture_test.go` |
| Every relative link in every Markdown file resolves | `architecture_test.go` |
| `e2e` imports only public packages, as a caller would | `e2e/architecture_test.go` |
| Harness core packages do not import harness adapters | `harness/architecture_test.go` |
| `provider/local/onnx` does not import the root's internals | `provider/local/onnx/architecture_test.go` |
| A package comment names the package it documents | `ST1000`, in `.golangci.yml` |
| Coverage stays above 97% | `make coverage-check` |

## Things that look wrong and are not

**`provider/test` is not a provider.** It holds the doubles and the contract
suite for providers, which is why it sits with them. The capability test skips
it by name.

**`provider/local/` holds one thing.** It is the grouping directory for
providers that run in your process rather than over a network. One member today;
the alternative is `provider/onnx` sitting next to fifteen network clients while
being nothing like them.

**The root package is a long flat list of files.** Go gives a directory one
package, and splitting the public API across import paths to make the source
tree prettier would be a worse trade than the flat list. The internal split is
where the structure lives.

**`aliases.go` is hundreds of lines of type aliases.** That is the price of one
import path for callers, paid once, in a file with no logic in it.
