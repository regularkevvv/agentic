# Session-loop implementations

The interaction contract is `harness/sessionloop`. The delivery/ownership
contract and shared worker are in `harness/sessionloop/actor`.

| Implementation | Status | Delivery and execution |
|---|---|---|
| [localchannel](localchannel/README.md) | Implemented | Local-channel adapter + independent worker + configured native Harness journal |
| [postgres](../../e2e/sessionloop/postgres/README.md) | Executable e2e reference; transaction model proved | Persisted mailbox, fenced ownership and native journal through PostgreSQL or a transaction pool |

The CLI uses `localchannel`, through `tui/adapter/sessionloop`. The UI does not
receive journal handles, leases, fences or database connections.

`harness.NewSessionLoopHost` remains the direct native-session adapter used
inside a worker. It is not a mailbox or a scheduling implementation.

The local delivery mechanism is explicitly named `actor/localchannel`.
`actor/memory` is a deprecated source-compatible alias, not another flavor.

Lean proves the abstract contract and modeled adapters. Tests establish the
connection to the Go implementation; they are not a formal Go or SQL proof.

The PostgreSQL flavor intentionally lives only in `e2e/sessionloop/postgres`. Its driver,
schema, pool and test containers add no dependencies to the production modules.
