// Command main is the fresh no-replace consumer proof for the PostgreSQL
// journal release view. It crosses migration, create, append, load, and close
// against a real database service.
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/regularkevvv/agentic/harness/store"
	postgresstore "github.com/regularkevvv/agentic/harness/store/postgres"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	repository, err := postgresstore.Open(ctx, os.Getenv("AGENTIC_POSTGRES_TEST_URL"), postgresstore.WithSchema("release_consumer"))
	must(err)
	defer repository.Close()
	must(repository.Migrate(ctx))

	journal, created, err := repository.Create(ctx, "consumer", store.PendingEntry{Kind: "created", Payload: []byte("one")})
	must(err)
	defer func() { must(journal.Close(context.Background())) }()
	appended, err := journal.Append(ctx, created.Cursor, store.PendingEntry{Kind: "message", Payload: []byte("two")})
	must(err)
	snapshot, err := journal.Load(ctx)
	must(err)
	if len(snapshot.Entries) != 2 || !snapshot.Cursor.Equal(appended.Cursor) || string(snapshot.Entries[1].Payload) != "two" {
		panic(fmt.Sprintf("incompatible PostgreSQL journal: %#v", snapshot))
	}
	fmt.Printf("compatible_postgres=%s entries=%d cursor=%d\n", journal.SessionID(), len(snapshot.Entries), snapshot.Cursor.Seq)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
