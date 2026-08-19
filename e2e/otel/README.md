# OpenTelemetry Collector E2E proof

This is both a runnable example and an automated smoke test. It does not call
an external model or require credentials: deterministic models exercise the
real Agentic lifecycle, the real `agentic/otel` adapter sends OTLP/gRPC, and a
pinned OpenTelemetry Collector writes each signal as OTLP JSON. The test then
parses those files using the official OTLP protobufs and asserts the result.

Run the complete proof:

```bash
make -C e2e/otel smoke
```

The command creates an isolated Compose volume, cross-compiles one static test
binary for the Docker engine's architecture, starts Collector `v0.158.0`, runs
the scenarios, verifies the exports, and removes the binary, containers, and
volume even if an assertion fails. The verifier is mounted into the same
digest-pinned image, avoiding a second container toolchain and dependency
download.

The scenarios cover:

- nested agent inheritance and agent/model/tool span parentage;
- a suspended invocation followed by a separately traced resume;
- streaming time-to-first-chunk and inter-chunk telemetry;
- a provider failure and its private, trace-correlated exception log;
- an instrumented embedding request with its actual vector dimension;
- opt-in inference details and an explicit evaluation record;
- content filtering and default privacy using a deliberate tripwire secret.

The verifier requires exactly 16 adapter spans across five scenario traces,
all eight metric instruments, and five correlated GenAI log records. It also
checks semantic attributes, durable correlation IDs, span kinds, parent IDs,
error status, redacted opt-in content, and absence of the raw tripwire secret.

To run only the example against the Compose Collector:

```bash
docker compose -f e2e/otel/docker-compose.yml up -d collector
go run ./e2e/examples/otel
docker compose -f e2e/otel/docker-compose.yml down --volumes
```

The example produces telemetry and reports the application scenarios. The
`make` target is the authoritative proof because it additionally inspects and
asserts Collector-exported traces, metrics, and logs.
