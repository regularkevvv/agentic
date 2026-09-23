# Session-loop end-to-end tests

| Flavor | Location | Execution and storage |
|---|---|---|
| Local channel | Tests in this directory | Independent worker, process-local mailbox, JSONL journal |
| PostgreSQL | [postgres/](postgres/README.md) | Independent worker, durable mailbox and fenced journal; direct PostgreSQL and transaction-pooled PgBouncer |

Both exercise the same actor contracts and shared worker. The PostgreSQL adapter
is a reference implementation owned by e2e, not a production-module dependency.

## Local mailbox/worker acceptance

Run from the repository root:

```sh
just sessionloop-e2e
```

This credential-free e2e imports only public APIs and is run by CI with the
race detector. Only the model is scripted. The assembly is:

```text
Mailbox.Submit -> actor/localchannel -> independent Worker.Run
                                     |
                      SessionOpener + fenced journal authority
                                     |
                   AssembleDefault -> Agent -> real read_file tool
                                     |
                              JSONL journal on disk
```

The actor ID is the pre-created journal ID: an explicit identity binding.
Each opener constructs fresh Harness and repository objects and opens that ID;
there is no fallback to creating a replacement session.

The tests exercise competing workers, submit-without-execution, steering past a
busy Start with one-item pages, mailbox cleanup after durable acceptance,
duplicate/conflicting IDs, worker replacement, journal receipt replay, and
failure immediately before/after mailbox deletion. Recovery requires neither
another submission nor a repair cron.

They also compare serialized message prefixes and tool schemas across turns
and disk reopen, and check stable prompt-cache identity. This tests cache-friendly
request continuity, **not** an actual provider KV-cache hit.

The memory adapter retains state across worker replacement, not process death.
These are Go integration tests, not a formal refinement proof, a live-provider
test, or a process-kill durability claim. The Lean durable receive flavor and
the Go process-local flavor have different failure boundaries.
