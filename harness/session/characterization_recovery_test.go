package session

// Characterization scenarios 7 and 10 from the sessionloop plan (section
// 8.2): process-loss recovery from committed JSONL crash fixtures and
// close/reopen ownership over the JSONL store.
//
// The crash fixtures under testdata/characterization/*.jsonl are committed
// files. They are regenerated only under -update-characterization by driving
// a complete session against a JSONL repository and truncating the resulting
// journal at line granularity, which simulates torn process loss exactly as
// the store recovers it (see testdata/characterization/README.md). Tests
// always copy a fixture into t.TempDir() before Recover and never mutate the
// committed file.

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	storejsonl "github.com/regularkevvv/agentic/harness/store/jsonl"
)

const (
	crashRepairFixture        = "char_crash_repair"
	crashIndeterminateFixture = "char_crash_indeterminate"
)

// truncateThroughKind keeps the journal lines up to and including the first
// line of the given kind. Line-level truncation matches the JSONL store's
// crash-atomicity unit.
func truncateThroughKind(t *testing.T, raw []byte, kind string) []byte {
	t.Helper()
	lines := strings.Split(strings.TrimSuffix(string(raw), "\n"), "\n")
	for index, line := range lines {
		var envelope struct {
			Kind string `json:"kind"`
		}
		if err := json.Unmarshal([]byte(line), &envelope); err != nil {
			t.Fatalf("parse journal line %d: %v", index+1, err)
		}
		if envelope.Kind == kind {
			return []byte(strings.Join(lines[:index+1], "\n") + "\n")
		}
	}
	t.Fatalf("kind %s not found in generated journal", kind)
	return nil
}

func writeCrashFixture(t *testing.T, fixture string, content []byte) {
	t.Helper()
	path := characterizationPath(fixture + ".jsonl")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatal(err)
	}
}

// regenerateCrashFixtures rebuilds both committed crash journals: a complete
// session is driven against a JSONL repository, closed cleanly, and its
// journal file is copied and truncated after the crash point.
func regenerateCrashFixtures(t *testing.T) {
	t.Helper()

	// Fixture 1: process loss immediately after the durable acceptance batch
	// (run.opened + prompt), before any driver progress.
	repairRoot := t.TempDir()
	repairRepository, err := storejsonl.New(repairRoot)
	if err != nil {
		t.Fatal(err)
	}
	repairConfig := characterizationConfig(t, &countingDriver{}, repairRepository)
	repairConfig.ID = crashRepairFixture
	repairSession, err := New(context.Background(), repairConfig)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := repairSession.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "crash prompt")); err != nil {
		t.Fatal(err)
	}
	if err := repairSession.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	repairRaw, err := os.ReadFile(filepath.Join(repairRoot, crashRepairFixture+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	writeCrashFixture(t, crashRepairFixture, truncateThroughKind(t, repairRaw, kindMessage))

	// Fixture 2: process loss after agentic.tool_started with no
	// agentic.tool_result — the indeterminate frontier.
	indeterminateRoot := t.TempDir()
	indeterminateRepository, err := storejsonl.New(indeterminateRoot)
	if err != nil {
		t.Fatal(err)
	}
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "crash-1", Name: "effect", Input: map[string]any{"value": "x"}})},
		textStep("done"),
	}}
	agent := agentic.NewAgent("", model)
	agentic.AddTool(agent,
		func(context.Context, characterizationToolInput) (string, error) { return "ok", nil },
		agentic.AutoToolName("effect"),
		agentic.AutoToolDescription("Apply a deterministic effect"),
	)
	indeterminateConfig := characterizationConfig(t, agent, indeterminateRepository)
	indeterminateConfig.ID = crashIndeterminateFixture
	indeterminateSession, err := New(context.Background(), indeterminateConfig)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := indeterminateSession.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "crash prompt")); err != nil {
		t.Fatal(err)
	}
	if err := indeterminateSession.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	indeterminateRaw, err := os.ReadFile(filepath.Join(indeterminateRoot, crashIndeterminateFixture+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	writeCrashFixture(t, crashIndeterminateFixture, truncateThroughKind(t, indeterminateRaw, kindToolStarted))
}

// recoverCrashFixture copies a committed crash journal into a fresh temp
// root, opens a JSONL repository over the copy, and recovers it. Recovery
// mints IDs with an "_r" suffix so fixture IDs and recovery IDs never mix.
func recoverCrashFixture(t *testing.T, fixture string, driver agentic.Driver[string]) (*Session[string], Config[string]) {
	t.Helper()
	data, err := os.ReadFile(characterizationPath(fixture + ".jsonl"))
	if err != nil {
		t.Fatalf("crash fixture %s missing (regenerate with -update-characterization): %v", fixture, err)
	}
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, fixture+".jsonl"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	repository, err := storejsonl.New(root)
	if err != nil {
		t.Fatal(err)
	}
	config := characterizationConfig(t, driver, repository)
	config.ID = fixture
	config.IDs = characterizationRecoveryIDs()
	session, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatalf("Recover(%s) error = %v", fixture, err)
	}
	return session, config
}

// Scenario 7: process loss after run.opened. The repairable variant closes
// the abandoned run and continues on a new recovery run with exactly one
// DriveContinue; the indeterminate variant suspends durably with zero drives.
func TestCharacterizationProcessLossAfterRunOpened(t *testing.T) {
	if *updateCharacterization {
		regenerateCrashFixtures(t)
	}

	t.Run("RepairableFrontierContinuesExactlyOnce", func(t *testing.T) {
		driver := &countingDriver{}
		session, config := recoverCrashFixture(t, crashRepairFixture, driver)
		if err := session.WaitForIdle(context.Background()); err != nil {
			t.Fatal(err)
		}
		if driver.Count() != 1 || driver.Last().Mode != agentic.DriveContinue {
			t.Fatalf("recovery drives = %#v", driver.drives)
		}
		last := driver.Last()
		if len(last.History) != 1 || last.History[0].GetTextContent() != "crash prompt" || last.Prompt != nil {
			t.Fatalf("recovery drive input = %#v", last)
		}

		entries := loadJournalEntries(t, session)
		wantKinds := []string{
			kindSessionCreated, kindRunOpened, kindMessage,
			kindRecovered, kindRunClosed, kindRunOpened, kindRunClosed,
		}
		if fmtKinds(journalKinds(entries)) != fmtKinds(wantKinds) {
			t.Fatalf("recovered journal kinds = %v, want %v", journalKinds(entries), wantKinds)
		}
		recovered, err := decodePayload[struct{ State string }](config.Codec, entries[3])
		if err != nil {
			t.Fatal(err)
		}
		if recovered.State != "continue" {
			t.Fatalf("session.recovered payload = %#v", recovered)
		}
		abandoned, err := decodePayload[runClosedPayload](config.Codec, entries[4])
		if err != nil {
			t.Fatal(err)
		}
		if abandoned.ID != "run_c1" || abandoned.Status != agentic.ExecutionInterrupted ||
			abandoned.Error != "process stopped before run termination" {
			t.Fatalf("abandoned run.closed payload = %#v", abandoned)
		}
		reopened, err := decodePayload[runOpenedPayload](config.Codec, entries[5])
		if err != nil {
			t.Fatal(err)
		}
		if reopened.ID != "run_r1" || reopened.Mode != "continue" || !reopened.Recovery {
			t.Fatalf("recovery run.opened payload = %#v", reopened)
		}
		completed, err := decodePayload[runClosedPayload](config.Codec, entries[6])
		if err != nil {
			t.Fatal(err)
		}
		if completed.ID != "run_r1" || completed.Status != agentic.ExecutionCompleted {
			t.Fatalf("recovery run.closed payload = %#v", completed)
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("IndeterminateFrontierSuspendsWithoutDriving", func(t *testing.T) {
		driver := &countingDriver{}
		session, _ := recoverCrashFixture(t, crashIndeterminateFixture, driver)
		snapshot, err := session.Snapshot(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if snapshot.State != Suspended || snapshot.Suspension == nil ||
			snapshot.Suspension.Kind != "harness.recovery.indeterminate" {
			t.Fatalf("indeterminate recovery snapshot = %#v", snapshot)
		}
		if driver.Count() != 0 {
			t.Fatalf("indeterminate recovery drove the driver %d times", driver.Count())
		}
		entries := loadJournalEntries(t, session)
		if countEntries(entries, kindRecoverySuspension) != 1 {
			t.Fatalf("recovery.suspension entries = %d in %v",
				countEntries(entries, kindRecoverySuspension), journalKinds(entries))
		}
		if entries[len(entries)-1].Kind != kindRecoverySuspension {
			t.Fatalf("recovery.suspension is not the journal tail: %v", journalKinds(entries))
		}
		if err := session.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	})
}

func fmtKinds(kinds []string) string {
	return strings.Join(kinds, ",")
}

// Scenario 10: Close is idempotent, reopening through Recover works, a
// second Recover while a journal lease is held fails with ErrSessionOpen at
// both the in-process and file-lock layers, and closing releases the lease.
func TestCharacterizationIdempotentCloseAndReopenOwnership(t *testing.T) {
	root := t.TempDir()
	repository, err := storejsonl.New(root)
	if err != nil {
		t.Fatal(err)
	}
	driver := &countingDriver{}
	config := characterizationConfig(t, driver, repository)
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "persist once")); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("first Close error = %v", err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatalf("second Close must be idempotent, got %v", err)
	}

	reopened, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatalf("reopen after Close error = %v", err)
	}
	if reopened.State() != Idle {
		t.Fatalf("reopened state = %s", reopened.State())
	}
	snapshot, err := reopened.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Messages) != 1 || snapshot.Messages[0].GetTextContent() != "persist once" {
		t.Fatalf("reopened messages = %#v", snapshot.Messages)
	}

	// While the reopened session holds the lease, a second Recover fails on
	// the same repository (in-process registry) and on a fresh repository
	// over the same root (file lock).
	if _, err := Recover(context.Background(), config); !errors.Is(err, ErrSessionOpen) {
		t.Fatalf("same-repository concurrent Recover error = %v", err)
	}
	other, err := storejsonl.New(root)
	if err != nil {
		t.Fatal(err)
	}
	otherConfig := config
	otherConfig.Repository = other
	if _, err := Recover(context.Background(), otherConfig); !errors.Is(err, ErrSessionOpen) {
		t.Fatalf("cross-repository concurrent Recover error = %v", err)
	}

	// Closing releases the lease: the previously rejected repository can now
	// open the session.
	if err := reopened.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	final, err := Recover(context.Background(), otherConfig)
	if err != nil {
		t.Fatalf("reopen after lease release error = %v", err)
	}
	if final.State() != Idle {
		t.Fatalf("final state = %s", final.State())
	}
	if err := final.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}
