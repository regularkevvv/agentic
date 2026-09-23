// Native end-to-end scenarios exercise durable handoff, steering, replay and
// independently connected workers without provider credentials or fake storage.
package postgres

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
)

func TestNativeWorkerSteeringAndReplay(t *testing.T) {
	f := newFixture(t)
	f.model.readFile = true
	if err := os.WriteFile(filepath.Join(f.config.WorkspaceRoot, "fixture.txt"), []byte("postgres-e2e-content"), 0o600); err != nil {
		t.Fatal(err)
	}
	gate := make(chan struct{})
	f.model.gate = gate
	start := command(f.id, "start", "first")
	submit(t, f.store, start)
	if len(f.model.requests) != 0 {
		t.Fatal("Submit executed the model")
	}
	a, b := observe(reconnect(t, f.store)), observe(reconnect(t, f.store))
	stopA, stopB := f.worker(t, a.Adapter.(*Store), a), f.worker(t, b.Adapter.(*Store), b)
	// Either worker can win. Collect acknowledgments without preferring one.
	ack := func() sessionloop.Receipt {
		select {
		case r := <-a.acks:
			return r
		case r := <-b.acks:
			return r
		case <-time.After(15 * time.Second):
			t.Fatal("missing acceptance")
		}
		return sessionloop.Receipt{}
	}
	retired := func() {
		select {
		case <-a.releases:
		case <-b.releases:
		case <-time.After(15 * time.Second):
			t.Fatal("missing retirement")
		}
	}
	first := ack()
	if first.CommandID != "start" || first.Rejection != "" {
		t.Fatalf("start=%+v", first)
	}
	await(t, f.model.entered)
	submit(t, f.store, command(f.id, "queued", "second"))
	steer := command(f.id, "steer", "steering")
	steer.Command.Kind, steer.Command.RunID = sessionloop.CommandSteer, first.RunID
	submit(t, f.store, steer)
	steered := ack()
	if steered.CommandID != "steer" || steered.QueueID == "" {
		t.Fatalf("steering behind busy start=%+v", steered)
	}
	close(gate)
	if r := ack(); r.CommandID != "queued" || r.Rejection != "" {
		t.Fatalf("queued=%+v", r)
	}
	retired()
	stopA()
	stopB()
	// New connection pool, worker and Harness objects: only SQL state survives.
	fresh := reconnect(t, f.store)
	c := observe(fresh)
	submit(t, fresh, command(f.id, "continue", "third"))
	stopC := f.worker(t, fresh, c)
	if r := await(t, c.acks); r.CommandID != "continue" || r.Rejection != "" {
		t.Fatalf("continue=%+v", r)
	}
	await(t, c.releases)
	stopC()
	s := f.inspect(t)
	snap, err := s.Snapshot(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var users []string
	for _, e := range snap.Entries {
		if e.Role == sessionloop.RoleUser {
			users = append(users, e.Blocks[0].Text)
		}
	}
	if !reflect.DeepEqual(users, []string{"first", "steering", "second", "third"}) || snap.State != sessionloop.StateIdle {
		t.Fatalf("replay users=%v state=%s", users, snap.State)
	}
	normalized, _ := start.Normalize()
	r, found, err := s.Acceptance(t.Context(), normalized.Command)
	if err != nil || !found || r != first {
		t.Fatalf("receipt changed: %+v found=%v %v", r, found, err)
	}
	if duplicate := submit(t, fresh, start); !duplicate.Duplicate {
		t.Fatal("dedup lost after payload removal")
	}
	if _, err := fresh.Submit(t.Context(), command(f.id, "start", "changed")); !errors.Is(err, actor.ErrCommandConflict) {
		t.Fatalf("identity conflict=%v", err)
	}
	err = f.store.transaction(t.Context(), func(tx pgx.Tx) error {
		var pending, tombstones int
		if err := tx.QueryRow(t.Context(), "SELECT count(*) FILTER(WHERE payload IS NOT NULL),count(*) FILTER(WHERE payload IS NULL) FROM commands WHERE session_id=$1", string(f.id)).Scan(&pending, &tombstones); err != nil {
			return err
		}
		if pending != 0 || tombstones != 4 {
			t.Errorf("payloads=%d tombstones=%d", pending, tombstones)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	f.model.mu.Lock()
	defer f.model.mu.Unlock()
	if !f.model.toolSeen || len(f.model.requests) != 4 {
		t.Fatalf("tool/replay requests=%d tool seen=%v", len(f.model.requests), f.model.toolSeen)
	}
	for i, req := range f.model.requests {
		if req.cache != string(f.id) {
			t.Fatalf("cache identity changed: %q", req.cache)
		}
		if i == 0 {
			continue
		}
		previous := f.model.requests[i-1]
		if len(req.messages) < len(previous.messages) {
			t.Fatal("history shortened")
		}
		for k, prefix := range previous.messages {
			if !bytes.Equal(prefix, req.messages[k]) {
				t.Fatalf("request %d rewrote prefix %d", i, k)
			}
		}
	}
}

func TestBlockedModelsDoNotPinBackends(t *testing.T) {
	f := newFixture(t)
	gate := make(chan struct{})
	f.model.gate = gate
	h, err := host(f.config, f.model, f.store.Bootstrap())
	if err != nil {
		t.Fatal(err)
	}
	ids := []actor.ActorID{f.id}
	for range 5 {
		s, err := h.NewSession(t.Context(), sessionloop.SessionOptions{})
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, actor.ActorID(s.ID()))
		if err := s.Close(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	for _, id := range ids {
		submit(t, f.store, command(id, "start", "block in model"))
	}
	a := observe(reconnect(t, f.store))
	stop := f.worker(t, a.Adapter.(*Store), a)
	for range ids {
		await(t, f.model.entered)
	}
	// All six models are now blocked, yet unrelated SQL must still complete.
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	if err := f.store.pool.Ping(ctx); err != nil {
		t.Fatalf("models pinned database: %v", err)
	}
	if dsn := os.Getenv("AGENTIC_PGBOUNCER_ADMIN_DSN"); dsn != "" {
		cfg, err := pgx.ParseConfig(dsn)
		if err != nil {
			t.Fatal(err)
		}
		cfg.DefaultQueryExecMode = pgx.QueryExecModeSimpleProtocol
		admin, err := pgx.ConnectConfig(ctx, cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer admin.Close(context.Background())
		rows, err := admin.Query(ctx, "SHOW DATABASES")
		if err != nil {
			t.Fatal(err)
		}
		fields := rows.FieldDescriptions()
		checked := false
		for rows.Next() {
			values, err := rows.Values()
			if err != nil {
				t.Fatal(err)
			}
			row := map[string]any{}
			for i, field := range fields {
				row[field.Name] = values[i]
			}
			if row["name"] == "e2e" {
				if row["pool_mode"] != "transaction" || row["pool_size"] != int32(2) {
					t.Fatalf("wrong pool configuration: %v", row)
				}
				checked = true
			}
		}
		if err := rows.Err(); err != nil {
			t.Fatal(err)
		}
		rows.Close()
		if !checked {
			t.Fatal("did not verify transaction pool configuration")
		}
	}
	close(gate)
	for range ids {
		await(t, a.releases)
	}
	stop()
}
