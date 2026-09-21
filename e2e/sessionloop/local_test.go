// Package sessionloop_test exercises the public local actor adapter against the
// default Harness, real filesystem tools and a freshly reopened JSONL journal.
package sessionloop_test

import (
	"errors"
	"reflect"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
)

func TestLocalMailboxWorkerEndToEnd(t *testing.T) {
	f := newLocalFixture(t)
	f.model.readFile = true
	f.model.resume = make(chan struct{})
	start := f.command("start", "read the fixture")
	f.submit(t, start)
	if f.model.count() != 0 {
		t.Fatal("submission executed the harness before a worker was started")
	}

	// Two independent workers compete for the same actor; only its owner opens
	// the journal. Page size one deliberately puts a busy Start before steering.
	stopA := f.startWorker(t, "worker-a")
	stopB := f.startWorker(t, "worker-b")
	started := f.ack(t, "start")
	await(t, f.ctx, f.model.entered)
	queued := f.command("queued", "the next turn")
	f.submit(t, queued)
	steer := f.command("steer", "include the fixture contents")
	steer.Command.Kind, steer.Command.RunID = sessionloop.CommandSteer, started.RunID
	f.submit(t, steer)
	steered := f.ack(t, "steer")
	if steered.QueueID == "" {
		t.Fatal("steering was not accepted into the active run")
	}
	close(f.model.resume)
	f.ack(t, "queued")
	f.retired(t)
	stopA()
	stopB()
	f.noErrors(t)

	// Keep the process-local mailbox, but replace the worker and reconstruct
	// every Harness/repository object. Continuation must come from disk.
	continued := f.command("continued", "continue after worker restart")
	f.submit(t, continued)
	stopC := f.startWorker(t, "worker-c")
	f.ack(t, "continued")
	f.retired(t)
	stopC()
	f.noErrors(t)

	s := f.reopen(t)
	snapshot, err := s.Snapshot(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	var users, completions []string
	for _, entry := range snapshot.Entries {
		if entry.Role == sessionloop.RoleUser {
			users = append(users, entry.Blocks[0].Text)
			if entry.Origin == sessionloop.OriginSteer && entry.RunID != started.RunID {
				t.Fatal("steering moved to a different run")
			}
		}
		if entry.Role == sessionloop.RoleAssistant {
			for _, block := range entry.Blocks {
				if block.Text != "" {
					completions = append(completions, block.Text)
				}
			}
		}
	}
	want := []string{"read the fixture", "include the fixture contents", "the next turn", "continue after worker restart"}
	if snapshot.State != sessionloop.StateIdle || !reflect.DeepEqual(users, want) {
		t.Fatalf("restored state=%s users=%v, want %v", snapshot.State, users, want)
	}
	if want := []string{"completed request 2", "completed request 3", "completed request 4"}; !reflect.DeepEqual(completions, want) {
		t.Fatalf("runs did not complete successfully: %v, want %v", completions, want)
	}
	for _, command := range []actor.Command{start, steer, queued, continued} {
		receipt := acceptance(t, f, s, command)
		if receipt.SessionID != sessionloop.SessionID(f.id) || receipt.Rejection != "" {
			t.Fatalf("wrong acceptance after reopening: %+v", receipt)
		}
	}
	if acceptance(t, f, s, start) != started || acceptance(t, f, s, steer) != steered {
		t.Fatal("disk replay changed the original acceptance receipts")
	}
	duplicate, err := f.mailbox.Submit(f.ctx, start)
	if err != nil || !duplicate.Duplicate {
		t.Fatalf("identity was lost after mailbox cleanup: %+v, %v", duplicate, err)
	}
	changed := f.command("start", "different payload")
	if _, err := f.mailbox.Submit(f.ctx, changed); !errors.Is(err, actor.ErrCommandConflict) {
		t.Fatalf("changed semantics reused an accepted command ID: %v", err)
	}
	f.model.assertContinuity(t, string(f.id), 4)
}

func TestLocalJournalHandoffRecovery(t *testing.T) {
	for _, boundary := range []string{"before deletion", "after deletion"} {
		t.Run(boundary, func(t *testing.T) {
			f := newLocalFixture(t)
			f.adapter.failure = boundary
			f.adapter.fail.Store(true)
			command := f.command("recover", "accepted once")
			f.submit(t, command)
			stop := f.startWorker(t, "first-owner")
			if err := await(t, f.ctx, f.errors); !errors.Is(err, errInjected) {
				t.Fatalf("unexpected worker failure: %v", err)
			}
			stop()
			original := await(t, f.ctx, f.adapter.attempts)

			// No second Submit and no repair cron. Receive rediscovers either
			// pending input or unfinished execution when the old lease expires.
			stop = f.startWorker(t, "recovery-owner")
			f.retired(t)
			stop()
			f.noErrors(t)
			s := f.reopen(t)
			if got := acceptance(t, f, s, command); got != original {
				t.Fatalf("recovery changed acceptance: got %+v, want %+v", got, original)
			}
			snapshot, err := s.Snapshot(f.ctx)
			if err != nil {
				t.Fatal(err)
			}
			users := 0
			for _, entry := range snapshot.Entries {
				if entry.Role == sessionloop.RoleUser {
					users++
					if entry.RunID != original.RunID || entry.Blocks[0].Text != "accepted once" {
						t.Fatalf("retry created a different input/run: %+v", entry)
					}
				}
			}
			if users != 1 || snapshot.State != sessionloop.StateIdle {
				t.Fatalf("recovery did not settle the original input: users=%d state=%s", users, snapshot.State)
			}
		})
	}
}

func acceptance(t *testing.T, f *localFixture, s actor.Session, command actor.Command) sessionloop.Receipt {
	t.Helper()
	normalized, err := command.Normalize()
	if err != nil {
		t.Fatal(err)
	}
	r, found, err := s.Acceptance(f.ctx, normalized.Command)
	if err != nil || !found || r.Guarantee != sessionloop.AcceptanceDurable || r.Position.IsZero() {
		t.Fatalf("missing durable acceptance: %+v found=%v err=%v", r, found, err)
	}
	return r
}
