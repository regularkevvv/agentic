// Real subprocess death exercises committed handoff boundaries. The parent
// starts a fresh pool and worker without resubmitting or setting a recovery flag.
package postgres

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
)

// TestWorkerProcess is invoked only as a child of the crash matrix. Boundary
// hooks block outside the transaction: the parent kills rather than cancels it.
func TestWorkerProcess(t *testing.T) {
	stage := os.Getenv("AGENTIC_PG_CHILD_STAGE")
	if stage == "" {
		t.Skip("subprocess helper")
	}
	s, err := Connect(t.Context(), os.Getenv("AGENTIC_POSTGRES_DSN"), os.Getenv("AGENTIC_PG_CHILD_SCHEMA"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	id := actor.ActorID(os.Getenv("AGENTIC_PG_CHILD_ACTOR"))
	workspace := os.Getenv("AGENTIC_PG_CHILD_WORKSPACE")
	config := harness.DefaultConfig{WorkspaceRoot: workspace, SessionDir: os.Getenv("AGENTIC_PG_CHILD_SESSION_DIR"), ContextWindowTokens: 16384, PromptCacheRetention: agentic.PromptCacheShort}
	boundary := func() { fmt.Println("BOUNDARY"); <-t.Context().Done() }
	if stage == "submission" {
		submit(t, s, command(id, "crash", "accepted once"))
		boundary()
		return
	}
	if stage == "acquisition" || stage == "restoration" || stage == "private-journal" {
		lease, err := s.Acquire(t.Context(), id, "killed-owner", time.Second)
		if err != nil {
			t.Fatal(err)
		}
		if stage == "restoration" {
			h, err := host(config, &model{}, s.Repository(lease))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := h.OpenSession(t.Context(), sessionloop.SessionID(id)); err != nil {
				t.Fatal(err)
			}
		}
		if stage == "private-journal" {
			handle, err := s.Repository(lease).Open(t.Context(), string(id))
			if err != nil {
				t.Fatal(err)
			}
			err = s.transaction(t.Context(), func(tx pgx.Tx) error {
				row, err := handle.(*journal).owned(t.Context(), tx)
				if err != nil {
					return err
				}
				if _, err := appendEntries(t.Context(), tx, string(id), row.cursor, []store.PendingEntry{{Kind: "must-rollback", Payload: []byte("private"), Durability: store.DurabilitySync}}); err != nil {
					return err
				}
				boundary() // killed after INSERT + leaf update, before COMMIT
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			return
		}
		boundary()
		return
	}
	f := &fixture{store: s, id: id, model: &model{}, config: config}
	a := observe(s)
	switch stage {
	case "acceptance":
		a.beforeAck = boundary
		f.model.gate = make(chan struct{})
	case "cleanup":
		a.afterAck = boundary
		f.model.gate = make(chan struct{})
	case "settlement":
		a.beforeRelease = boundary
	case "retirement":
		a.afterRelease = boundary
	default:
		t.Fatalf("unknown crash stage %q", stage)
	}
	f.worker(t, s, a)
	<-t.Context().Done()
}

func killAt(t *testing.T, f *fixture, stage string) {
	t.Helper()
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(binary, "-test.run=^TestWorkerProcess$", "-test.timeout=25s", "-test.v")
	cmd.Env = append(os.Environ(), "AGENTIC_PG_CHILD_STAGE="+stage, "AGENTIC_PG_CHILD_SCHEMA="+f.store.schema, "AGENTIC_PG_CHILD_ACTOR="+string(f.id), "AGENTIC_PG_CHILD_WORKSPACE="+f.config.WorkspaceRoot, "AGENTIC_PG_CHILD_SESSION_DIR="+f.config.SessionDir)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	t.Cleanup(func() {
		if !waited {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	})
	lines := make(chan string, 128)
	go func() {
		defer close(lines)
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
	}()
	var output strings.Builder
	timer := time.NewTimer(10 * time.Second)
	defer timer.Stop()
	for {
		select {
		case line, ok := <-lines:
			if !ok {
				_ = cmd.Wait()
				waited = true
				t.Fatalf("child failed before %s: %s\n%s", stage, output.String(), stderr.String())
			}
			if line == "BOUNDARY" {
				goto reached
			}
			output.WriteString(line + "\n")
		case <-timer.C:
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
			waited = true
			t.Fatalf("child did not reach %s: %s\n%s", stage, output.String(), stderr.String())
		}
	}
reached:
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err == nil {
		t.Fatal("child exited normally instead of being killed")
	}
	waited = true
}

func TestProcessCrashRecovery(t *testing.T) {
	for _, stage := range []string{"submission", "acquisition", "restoration", "private-journal", "acceptance", "cleanup", "settlement", "retirement"} {
		t.Run(stage, func(t *testing.T) {
			f := newFixture(t)
			c := command(f.id, "crash", "accepted once")
			if stage != "submission" {
				submit(t, f.store, c)
			}
			killAt(t, f, stage)
			// A receipt, when already committed, must retain exactly the same
			// native entry identity after cleanup/recovery. No second transcript.
			var accepted *int64
			err := f.store.transaction(t.Context(), func(tx pgx.Tx) error {
				return tx.QueryRow(t.Context(), "SELECT accepted_at_seq FROM commands WHERE session_id=$1 AND id='crash'", string(f.id)).Scan(&accepted)
			})
			if err != nil {
				t.Fatal(err)
			}
			fresh := reconnect(t, f.store)
			if stage == "retirement" {
				ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
				_, err := fresh.Receive(ctx)
				cancel()
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("retired work rediscovered: %v", err)
				}
			} else {
				a := observe(fresh)
				stop := f.worker(t, fresh, a)
				await(t, a.releases)
				stop()
			}
			s := f.inspect(t)
			normalized, _ := c.Normalize()
			r, found, err := s.Acceptance(t.Context(), normalized.Command)
			if err != nil || !found || r.Rejection != "" {
				t.Fatalf("lost acceptance: %+v %v", r, err)
			}
			if accepted != nil && r.Position.Sequence != uint64(*accepted) {
				t.Fatalf("reaccepted instead of replayed: %+v original=%d", r, *accepted)
			}
			snap, err := s.Snapshot(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			users := 0
			for _, entry := range snap.Entries {
				if entry.Role == sessionloop.RoleUser {
					users++
					if entry.Blocks[0].Text != "accepted once" {
						t.Fatalf("wrong recovered input: %+v", entry)
					}
				}
			}
			if users != 1 || snap.State != sessionloop.StateIdle {
				t.Fatalf("users=%d state=%s", users, snap.State)
			}
			if retry := submit(t, fresh, c); !retry.Duplicate || retry.Sequence != 1 {
				t.Fatalf("lost committed identity: %+v", retry)
			}
		})
	}
}
