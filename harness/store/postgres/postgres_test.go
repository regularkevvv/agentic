package postgres

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/regularkevvv/agentic/harness/store"
	"github.com/regularkevvv/agentic/harness/store/storetest"
)

func TestPureValidationAndSequencing(t *testing.T) {
	if _, err := Open(context.Background(), ""); err == nil {
		t.Fatal("empty DSN succeeded")
	}
	for _, schema := range []string{"", "Bad", "has-dash", "9starts_wrong"} {
		repository := &Repository{schema: defaultSchema}
		if err := WithSchema(schema)(repository); err == nil {
			t.Fatalf("schema %q succeeded", schema)
		}
	}
	entries, err := sequence(store.Cursor{Seq: 2, EntryID: "parent"}, []store.PendingEntry{
		{Kind: "one", Payload: []byte("payload"), Durability: store.DurabilitySync},
		{Kind: "two"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 || entries[0].Seq != 3 || entries[0].ParentID != "parent" ||
		entries[1].Seq != 4 || entries[1].ParentID != entries[0].ID || tail(entries) != entries[1].Cursor() {
		t.Fatalf("entries = %#v", entries)
	}
	entries[0].Payload[0] = 'X'
	if _, err := sequence(store.Cursor{}, []store.PendingEntry{{}}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("empty entry = %v", err)
	}
	if _, err := sequence(store.Cursor{}, []store.PendingEntry{{Kind: "bad", Durability: store.DurabilitySync + 1}}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("bad durability = %v", err)
	}
	if !tail(nil).IsZero() {
		t.Fatal("empty tail is non-zero")
	}
	if !isUniqueViolation(&pgconn.PgError{Code: "23505"}) || isUniqueViolation(errors.New("other")) {
		t.Fatal("unique violation classification changed")
	}
}

func TestRepositoryConformance(t *testing.T) {
	dsn := os.Getenv("AGENTIC_POSTGRES_TEST_URL")
	if dsn == "" {
		t.Skip("AGENTIC_POSTGRES_TEST_URL is not set")
	}
	schema := fmt.Sprintf("agentic_test_%d", time.Now().UnixNano())
	var repositories []*Repository
	t.Cleanup(func() {
		if len(repositories) == 0 {
			return
		}
		_, _ = repositories[0].pool.Exec(context.Background(), "DROP SCHEMA "+schema+" CASCADE")
		for _, repository := range repositories {
			repository.Close()
		}
	})
	factory := func(t *testing.T) store.Repository {
		t.Helper()
		repository, err := Open(context.Background(), dsn, WithSchema(schema))
		if err != nil {
			t.Fatal(err)
		}
		if err := repository.Migrate(context.Background()); err != nil {
			t.Fatal(err)
		}
		repositories = append(repositories, repository)
		return repository
	}
	storetest.Run(t, factory)

	first := factory(t).(*Repository)
	second := factory(t).(*Repository)
	journal, _, err := first.Create(context.Background(), "cross_process", store.PendingEntry{Kind: "created"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := second.Open(context.Background(), "cross_process"); !errors.Is(err, store.ErrSessionOpen) {
		t.Fatalf("second repository Open = %v", err)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	reopened, err := second.Open(context.Background(), "cross_process")
	if err != nil {
		t.Fatal(err)
	}
	if err := reopened.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestRepositoryValidationFailureAndCorruptionPaths(t *testing.T) {
	dsn := os.Getenv("AGENTIC_POSTGRES_TEST_URL")
	if dsn == "" {
		t.Skip("AGENTIC_POSTGRES_TEST_URL is not set")
	}
	ctx := context.Background()
	schema := fmt.Sprintf("agentic_failures_%d", time.Now().UnixNano())
	repository, err := Open(ctx, dsn, WithSchema(schema))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = repository.pool.Exec(context.Background(), "DROP SCHEMA "+schema+" CASCADE")
		repository.Close()
	})
	if err := repository.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	if err := repository.Migrate(ctx); err != nil {
		t.Fatalf("idempotent migrate = %v", err)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, _, err := repository.Create(canceled, "canceled"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled create = %v", err)
	}
	if _, err := repository.Open(canceled, "canceled"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled open = %v", err)
	}
	if _, _, err := repository.Create(ctx, ""); err == nil {
		t.Fatal("invalid create session succeeded")
	}
	if _, _, err := repository.Create(ctx, "invalid-pending", store.PendingEntry{}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("invalid create pending = %v", err)
	}
	if _, err := repository.Open(ctx, ""); err == nil {
		t.Fatal("invalid open session succeeded")
	}
	if _, err := repository.Open(ctx, "missing"); !errors.Is(err, store.ErrSessionNotFound) {
		t.Fatalf("missing open = %v", err)
	}

	journalValue, commit, err := repository.Create(ctx, "lifecycle")
	if err != nil || !commit.Cursor.IsZero() {
		t.Fatalf("empty create = %+v, %v", commit, err)
	}
	journal := journalValue.(*Journal)
	if journal.SessionID() != "lifecycle" {
		t.Fatalf("SessionID = %q", journal.SessionID())
	}
	if snapshot, err := journal.Load(ctx); err != nil || len(snapshot.Entries) != 0 {
		t.Fatalf("empty load = %+v, %v", snapshot, err)
	}
	if _, err := journal.Load(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled load = %v", err)
	}
	if _, err := journal.Append(canceled, store.Cursor{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled append = %v", err)
	}
	if _, err := journal.Append(ctx, store.Cursor{}, store.PendingEntry{}); !errors.Is(err, store.ErrCorruptLog) {
		t.Fatalf("invalid append = %v", err)
	}
	if _, _, err := repository.Create(ctx, "lifecycle"); !errors.Is(err, store.ErrSessionOpen) {
		t.Fatalf("create while open = %v", err)
	}
	if err := journal.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if err := journal.Close(ctx); err != nil {
		t.Fatalf("idempotent close = %v", err)
	}
	if _, err := journal.Load(ctx); !errors.Is(err, store.ErrJournalClosed) {
		t.Fatalf("closed load = %v", err)
	}
	if _, err := journal.Append(ctx, store.Cursor{}); !errors.Is(err, store.ErrJournalClosed) {
		t.Fatalf("closed append = %v", err)
	}
	if _, _, err := repository.Create(ctx, "lifecycle"); !errors.Is(err, store.ErrSessionExists) {
		t.Fatalf("duplicate create = %v", err)
	}

	unlockedValue, _, err := repository.Create(ctx, "unlock-missing")
	if err != nil {
		t.Fatal(err)
	}
	unlocked := unlockedValue.(*Journal)
	var released bool
	if err := unlocked.connection.QueryRow(ctx, `SELECT pg_advisory_unlock(hashtextextended($1, 0))`, schema+":unlock-missing").Scan(&released); err != nil || !released {
		t.Fatalf("manual unlock = %v, %v", released, err)
	}
	if err := unlocked.Close(ctx); !errors.Is(err, store.ErrJournalClosed) {
		t.Fatalf("close without held lock = %v", err)
	}

	terminatedValue, _, err := repository.Create(ctx, "terminated-lock")
	if err != nil {
		t.Fatal(err)
	}
	terminated := terminatedValue.(*Journal)
	var backendPID int
	if err := terminated.connection.QueryRow(ctx, `SELECT pg_backend_pid()`).Scan(&backendPID); err != nil {
		t.Fatal(err)
	}
	if _, err := repository.pool.Exec(ctx, `SELECT pg_terminate_backend($1)`, backendPID); err != nil {
		t.Fatal(err)
	}
	if err := terminated.Close(ctx); err == nil {
		t.Fatal("close of terminated lock connection succeeded")
	}

	if _, err := repository.pool.Exec(ctx, fmt.Sprintf(`ALTER TABLE %s.entries DROP CONSTRAINT entries_kind_check, DROP CONSTRAINT entries_durability_check`, schema)); err != nil {
		t.Fatal(err)
	}
	corrupt := []struct {
		name       string
		sequence   int
		id, parent string
		kind       string
		durability int
		schema     int
	}{
		{name: "schema", sequence: 1, id: "schema", kind: "kind", schema: 99},
		{name: "sequence", sequence: 2, id: "sequence", kind: "kind", schema: int(store.CurrentSchema)},
		{name: "empty-id", sequence: 1, kind: "kind", schema: int(store.CurrentSchema)},
		{name: "empty-kind", sequence: 1, id: "empty-kind", schema: int(store.CurrentSchema)},
		{name: "parent", sequence: 1, id: "parent", parent: "unexpected", kind: "kind", schema: int(store.CurrentSchema)},
		{name: "durability", sequence: 1, id: "durability", kind: "kind", durability: 9, schema: int(store.CurrentSchema)},
	}
	for _, value := range corrupt {
		if _, err := repository.pool.Exec(ctx, fmt.Sprintf(`INSERT INTO %s.sessions(session_id) VALUES($1)`, schema), value.name); err != nil {
			t.Fatal(err)
		}
		if _, err := repository.pool.Exec(ctx, fmt.Sprintf(`INSERT INTO %s.entries(session_id,seq,id,parent_id,kind,payload,durability,schema_version) VALUES($1,$2,$3,$4,$5,$6,$7,$8)`, schema),
			value.name, value.sequence, value.id, value.parent, value.kind, []byte{}, value.durability, value.schema); err != nil {
			t.Fatal(err)
		}
		if _, err := repository.Open(ctx, value.name); !errors.Is(err, store.ErrCorruptLog) {
			t.Fatalf("open corrupt %s = %v", value.name, err)
		}
	}

	missingSchema := fmt.Sprintf("agentic_missing_%d", time.Now().UnixNano())
	missing, err := Open(ctx, dsn, WithSchema(missingSchema))
	if err != nil {
		t.Fatal(err)
	}
	defer missing.Close()
	if _, _, err := missing.Create(ctx, "session"); err == nil {
		t.Fatal("create without migration succeeded")
	}
	if _, err := missing.Open(ctx, "session"); err == nil {
		t.Fatal("open without migration succeeded")
	}

	closed, err := Open(ctx, dsn, WithSchema(schema))
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	closed.Close()
	if err := closed.Migrate(ctx); err == nil {
		t.Fatal("migrate on closed pool succeeded")
	}
	if _, _, err := closed.Create(ctx, "closed-pool"); err == nil {
		t.Fatal("create on closed pool succeeded")
	}
}

func TestOpenOptionsAndPureHelperFailures(t *testing.T) {
	if _, err := Open(context.Background(), "postgres://bad host"); err == nil {
		t.Fatal("invalid DSN succeeded")
	}
	if _, err := Open(context.Background(), "postgres://localhost/db", nil); err == nil {
		t.Fatal("nil option succeeded")
	}
	wantOption := errors.New("option")
	if _, err := Open(context.Background(), "postgres://localhost/db", func(*Repository) error { return wantOption }); !errors.Is(err, wantOption) {
		t.Fatalf("option error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Open(ctx, "postgres://127.0.0.1:1/unreachable?connect_timeout=1"); err == nil {
		t.Fatal("canceled ping succeeded")
	}

	if id := newID(); len(id) != 32 {
		t.Fatalf("entry ID = %q", id)
	}

	queryFailure := errors.New("query")
	if _, err := loadEntries(context.Background(), fakeQueryer{err: queryFailure}, "schema", "session"); !errors.Is(err, queryFailure) {
		t.Fatalf("load query error = %v", err)
	}
	if _, err := loadEntries(context.Background(), fakeQueryer{rows: &fakeRows{next: true, scanErr: errors.New("scan")}}, "schema", "session"); err == nil {
		t.Fatal("load scan error succeeded")
	}
	if _, err := loadEntries(context.Background(), fakeQueryer{rows: &fakeRows{iterErr: errors.New("iterate")}}, "schema", "session"); err == nil {
		t.Fatal("load iteration error succeeded")
	}
	if cursor, err := loadTail(context.Background(), fakeRowQueryer{err: pgx.ErrNoRows}, "schema", "session"); err != nil || !cursor.IsZero() {
		t.Fatalf("empty tail = %+v, %v", cursor, err)
	}
	if _, err := loadTail(context.Background(), fakeRowQueryer{err: queryFailure}, "schema", "session"); !errors.Is(err, queryFailure) {
		t.Fatalf("tail error = %v", err)
	}
	executor := &fakeExecer{err: errors.New("insert")}
	if err := insertEntries(context.Background(), executor, "schema", "session", []store.Entry{{Seq: 1}}); err == nil {
		t.Fatal("insert error succeeded")
	}
	executor.err = nil
	if err := insertEntries(context.Background(), executor, "schema", "session", []store.Entry{{Seq: 1}}); err != nil {
		t.Fatal(err)
	}
	if payload, ok := executor.last[5].([]byte); !ok || payload == nil {
		t.Fatalf("nil payload was not normalized: %#v", executor.last)
	}
	(&Repository{}).Close()
	var nilRepository *Repository
	nilRepository.Close()
	if err := (&Journal{closed: true}).Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := (&Journal{}).Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestDatabaseTransactionFailurePaths(t *testing.T) {
	dsn := os.Getenv("AGENTIC_POSTGRES_TEST_URL")
	if dsn == "" {
		t.Skip("AGENTIC_POSTGRES_TEST_URL is not set")
	}
	ctx := context.Background()
	openRepository := func(t *testing.T, label string) *Repository {
		t.Helper()
		schema := fmt.Sprintf("agentic_%s_%d", label, time.Now().UnixNano())
		repository, err := Open(ctx, dsn, WithSchema(schema))
		if err != nil {
			t.Fatal(err)
		}
		if err := repository.Migrate(ctx); err != nil {
			repository.Close()
			t.Fatal(err)
		}
		t.Cleanup(func() {
			_, _ = repository.pool.Exec(context.Background(), "DROP SCHEMA "+schema+" CASCADE")
			repository.Close()
		})
		return repository
	}

	t.Run("create entry insert", func(t *testing.T) {
		repository := openRepository(t, "create_insert")
		if _, err := repository.pool.Exec(ctx, "DROP TABLE "+repository.schema+".entries"); err != nil {
			t.Fatal(err)
		}
		if _, _, err := repository.Create(ctx, "session", store.PendingEntry{Kind: "entry"}); err == nil {
			t.Fatal("create with missing entries table succeeded")
		}
	})

	t.Run("create deferred commit", func(t *testing.T) {
		repository := openRepository(t, "create_commit")
		installDeferredFailure(t, ctx, repository, "sessions")
		if _, _, err := repository.Create(ctx, "session"); err == nil {
			t.Fatal("create deferred commit failure succeeded")
		}
	})

	t.Run("append tail read", func(t *testing.T) {
		repository := openRepository(t, "append_tail")
		value, _, err := repository.Create(ctx, "session")
		if err != nil {
			t.Fatal(err)
		}
		journal := value.(*Journal)
		defer journal.Close(ctx)
		if _, err := repository.pool.Exec(ctx, "DROP TABLE "+repository.schema+".entries"); err != nil {
			t.Fatal(err)
		}
		if _, err := journal.Append(ctx, store.Cursor{}, store.PendingEntry{Kind: "entry"}); err == nil {
			t.Fatal("append with missing entries table succeeded")
		}
	})

	t.Run("append entry insert", func(t *testing.T) {
		repository := openRepository(t, "append_insert")
		value, _, err := repository.Create(ctx, "session")
		if err != nil {
			t.Fatal(err)
		}
		journal := value.(*Journal)
		defer journal.Close(ctx)
		installImmediateFailure(t, ctx, repository, "entries")
		if _, err := journal.Append(ctx, store.Cursor{}, store.PendingEntry{Kind: "entry"}); err == nil {
			t.Fatal("append insert trigger failure succeeded")
		}
	})

	t.Run("append deferred commit", func(t *testing.T) {
		repository := openRepository(t, "append_commit")
		value, _, err := repository.Create(ctx, "session")
		if err != nil {
			t.Fatal(err)
		}
		journal := value.(*Journal)
		defer journal.Close(ctx)
		installDeferredFailure(t, ctx, repository, "entries")
		if _, err := journal.Append(ctx, store.Cursor{}, store.PendingEntry{Kind: "entry"}); err == nil {
			t.Fatal("append deferred commit failure succeeded")
		}
	})
}

func installDeferredFailure(t *testing.T, ctx context.Context, repository *Repository, table string) {
	t.Helper()
	name := "fail_" + table
	ddl := fmt.Sprintf(`
CREATE FUNCTION %s.%s() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'injected deferred failure'; END $$;
CREATE CONSTRAINT TRIGGER %s AFTER INSERT ON %s.%s
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION %s.%s()`,
		repository.schema, name, name, repository.schema, table, repository.schema, name)
	if _, err := repository.pool.Exec(ctx, ddl); err != nil {
		t.Fatal(err)
	}
}

func installImmediateFailure(t *testing.T, ctx context.Context, repository *Repository, table string) {
	t.Helper()
	name := "fail_" + table
	ddl := fmt.Sprintf(`
CREATE FUNCTION %s.%s() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'injected insert failure'; END $$;
CREATE TRIGGER %s BEFORE INSERT ON %s.%s
FOR EACH ROW EXECUTE FUNCTION %s.%s()`,
		repository.schema, name, name, repository.schema, table, repository.schema, name)
	if _, err := repository.pool.Exec(ctx, ddl); err != nil {
		t.Fatal(err)
	}
}

type fakeQueryer struct {
	rows pgx.Rows
	err  error
}

func (q fakeQueryer) Query(context.Context, string, ...any) (pgx.Rows, error) { return q.rows, q.err }

type fakeRows struct {
	next, returned   bool
	scanErr, iterErr error
}

func (r *fakeRows) Close()                                       {}
func (r *fakeRows) Err() error                                   { return r.iterErr }
func (r *fakeRows) CommandTag() pgconn.CommandTag                { return pgconn.CommandTag{} }
func (r *fakeRows) FieldDescriptions() []pgconn.FieldDescription { return nil }
func (r *fakeRows) Next() bool {
	if r.next && !r.returned {
		r.returned = true
		return true
	}
	return false
}
func (r *fakeRows) Scan(...any) error      { return r.scanErr }
func (r *fakeRows) Values() ([]any, error) { return nil, nil }
func (r *fakeRows) RawValues() [][]byte    { return nil }
func (r *fakeRows) Conn() *pgx.Conn        { return nil }

type fakeRowQueryer struct{ err error }

func (q fakeRowQueryer) QueryRow(context.Context, string, ...any) pgx.Row { return fakeRow{err: q.err} }

type fakeRow struct{ err error }

func (r fakeRow) Scan(...any) error { return r.err }

type fakeExecer struct {
	err  error
	last []any
}

func (e *fakeExecer) Exec(_ context.Context, _ string, values ...any) (pgconn.CommandTag, error) {
	e.last = values
	return pgconn.NewCommandTag("INSERT 0 1"), e.err
}
