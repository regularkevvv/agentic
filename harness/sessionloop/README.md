# sessionloop

`github.com/regularkevvv/agentic/harness/sessionloop` is the provider-neutral
asynchronous session protocol shared by Agentic Harness, the Agentic TUI
bridge, and external consumers. It is a standard-library-only module: its
`go.mod` has zero `require` directives, so importing it never places Agentic,
Harness, the TUI, a provider SDK, or a terminal library in your module graph.

The module defines the contract — types, validation, cloning, error
sentinels — plus a reusable conformance suite (`conformance`) and an
in-memory reference host (`testkit`). It does not execute a model. The full
design rationale lives in
[`docs/design/harness-sessionloop-plan.md`](../../docs/design/harness-sessionloop-plan.md).

## The command/event mental model

A session is a long-lived actor. Callers never block on an answer; they
dispatch commands, receive acceptance receipts, and observe truth as ordered
events reconciled by snapshots.

```text
caller                         host session
  │  Dispatch(command) ──────────▶ validate + accept
  │  ◀────────── Receipt           (acceptance ≠ completion)
  │                                run executes asynchronously
  │  Stream.Next ◀───────────────  authoritative events, in order:
  │                                command.accepted, entry.committed,
  │                                run.started, …, run.settled
  │  Stream.Next ◀╌╌╌╌╌╌╌╌╌╌╌╌╌╌  preview deltas (lossy, droppable)
  │  Snapshot ◀──────────────────  copy-owned authoritative view
```

- **Dispatch → receipt.** `Dispatch` returns once the command is accepted
  under the receipt's declared guarantee (`accepted` or `durable`). The
  outcome arrives later through events and snapshots.
- **Two-tier stream.** Authoritative events are ordered facts with replay
  positions. Previews are lossy progress that never becomes transcript truth;
  losing them is observable (`Dropped`), losing authoritative events is
  terminal (`ErrLagged`).
- **Snapshot reconciliation.** On lag, gaps, or an unknown position, discard
  speculative state, load a `Snapshot`, and `Subscribe` after its position.
- **Settlement and reuse.** A run settles exactly once — completed,
  interrupted, or failed — and the session returns to idle for the next run.
  Suspension is a durable pause of the same run, not settlement.

### Protocol laws

| Law | Summary |
|---|---|
| L1 | One active run per session; starting while busy fails explicitly |
| L2 | Acceptance is not completion; receipts are not results |
| L3 | Every acknowledgement declares its guarantee: accepted or durable |
| L4 | The dispatch context governs acceptance only; runs belong to the session |
| L5 | Authoritative data is never inferred from previews |
| L6 | Replay positions are opaque; a zero position is not replayable |
| L7 | Snapshots are copy-owned authoritative truth and reconcile stream gaps |
| L8 | Targeted commands carry a RunID and cannot cross runs (`ErrStaleRun`) |
| L9 | Capabilities are honest; unsupported operations fail with `ErrUnsupported` |
| L10 | Settlement is singular and follows every entry of its run |
| L11 | Closing releases the handle, never the durable session |
| L12 | Content authority and privacy remain application-owned |

## Capabilities

Baseline behavior — new/open, start, snapshots, authoritative entries,
settlement, close — is required of every host and is not a capability.
Optional behavior is advertised explicitly and never emulated:

| Capability | Meaning |
|---|---|
| `acceptance.durable` | Receipts are crash-durable before they return |
| `events.replay` | Authoritative events replay strictly after a position |
| `events.preview` | Lossy preview deltas are published |
| `input.steer` | Steer the active run at its next steerable boundary |
| `input.follow_up` | Continue the active run after its current candidate |
| `input.next_turn` | Queue input for a future start; survives interruption |
| `run.interrupt` | Interrupt the active run |
| `run.suspension.resolve` | Durable suspensions resolved by exact ID |
| `dispatch.idempotent` | Idempotency keys deduplicate dispatches durably |
| `content.tools.detailed` | Tool call/result blocks carry structured detail |
| `output.structured` | Completed outcomes carry projected JSON output |

Unknown capability strings survive round trips, so vendors can extend the set
without breaking consumers.

## A runnable fake-host example

The `testkit` host is a complete in-memory implementation advertising every
capability except `dispatch.idempotent`:

```go
package main

import (
	"context"
	"fmt"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/testkit"
)

func main() {
	ctx := context.Background()
	host := testkit.New() // default run: echo one assistant entry

	session, _ := host.NewSession(ctx, sessionloop.SessionOptions{})
	defer session.Close(ctx)

	stream, _ := session.Subscribe(ctx, sessionloop.SubscribeOptions{})
	defer stream.Close()

	receipt, _ := session.Dispatch(ctx, sessionloop.Command{
		Kind: sessionloop.CommandStart,
		Input: &sessionloop.Input{Content: []sessionloop.Block{
			{Kind: sessionloop.BlockText, Text: "hello"},
		}},
	})
	fmt.Println("accepted run", receipt.RunID, "at", receipt.Position.Sequence)

	for {
		event, err := stream.Next(ctx)
		if err != nil {
			break
		}
		if event.Kind == sessionloop.EventEntryCommitted {
			fmt.Println(event.Entry.Role, event.Entry.Content[0].Text)
		}
		if event.Kind == sessionloop.EventRunSettled && event.RunID == receipt.RunID {
			fmt.Println("settled:", event.Outcome.Kind)
			break
		}
	}
}
```

## Conformance

Host adapters prove their claims with the reusable suite:

```go
func TestMyHostConformance(t *testing.T) {
	conformance.Run(t, func(t *testing.T) conformance.Env {
		host := newMyHost(t)
		return conformance.Env{Host: host, Gate: host} // Gate is optional
	})
}
```

Baseline cases always run; optional cases activate only for advertised
capabilities; timing-dependent cases skip without a `Gate`. See the
`conformance` package documentation for the full factory contract, including
the scripted scenario metadata.

## Development

`make check` runs formatting, vet, lint, race tests, and the independent 97%
coverage gate. The `conformance` and `testkit` packages are excluded from the
coverage measurement but are exercised by the main package's tests, which run
the conformance suite against the testkit host. An architecture test keeps
the module's `go.mod` free of `require`/`replace` directives and every import
inside the standard library.
