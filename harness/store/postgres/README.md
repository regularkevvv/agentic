# Harness PostgreSQL journal

This optional module implements `harness/store.Repository` on PostgreSQL. It
stores the append-only journal transactionally and holds one PostgreSQL
advisory lock on a dedicated connection for the lifetime of each open session.
That makes session ownership exclusive across processes and Kubernetes pods,
not merely inside one Go process.

```go
repository, err := postgres.Open(ctx, databaseURL,
    postgres.WithSchema("agentic_harness"),
)
if err != nil {
    return err
}
defer repository.Close()
if err := repository.Migrate(ctx); err != nil {
    return err
}

journal, _, err := repository.Create(ctx, "conversation-123")
```

`Create` and `Open` return `store.ErrSessionOpen` while another journal owns
the same schema/session pair. Closing the journal releases that lock without
deleting its history. The module intentionally has no `replace` directive;
applications use a `go.work` file while developing coordinated unpublished
changes.

The conformance suite needs a disposable PostgreSQL database:

```sh
AGENTIC_POSTGRES_TEST_URL='postgres://agentic:agentic@127.0.0.1:5432/agentic?sslmode=disable' \
  go test -race ./...
```
