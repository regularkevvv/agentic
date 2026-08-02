# Documentation

Two kinds of document live here, and the directory is the difference.

**`docs/` describes what exists.** These are maintained: if the code changes and
the document does not, the document is wrong and should be fixed.

- [multi-representation-inference.md](multi-representation-inference.md) — dense,
  learned sparse, and token multi-vector encoding: the interface, vector-space
  identity, and every provider that implements it

**`docs/design/` records why.** These are written before or during the work and
then left alone. They are not maintained and are not a description of the
current code — several describe things that were built differently, or not at
all. Read them for the reasoning, not for the shape.

- [design/spike-harness-framework.md](design/spike-harness-framework.md) — the
  durable-harness design, phases 1–4
- [design/harness-phase5.md](design/harness-phase5.md) — code execution, memory,
  skills, and evals
- [design/multi-representation-inference-plan.md](design/multi-representation-inference-plan.md)
  — the encoding plan, with an amendment noting the two things it describes that
  no longer exist
- [design/module-boundaries-plan.md](design/module-boundaries-plan.md) — the
  nested-module split, with twelve recorded deviations from what was planned

For where code goes rather than why, see [ARCHITECTURE.md](../ARCHITECTURE.md).
For the public API, see [pkg.go.dev](https://pkg.go.dev/github.com/regularkevvv/agentic).
