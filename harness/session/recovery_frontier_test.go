package session

// Journal-cut regressions use the real driver and committed event bytes. They
// model process loss, not Close (which intentionally requests interruption).
import (
	"context"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestRecoveryCommittedAssistantFrontier(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{textStep("durable reply")}}
	cfg := sessionConfig(t, agentic.NewAgent("", model), storememory.New(), artifactmemory.New(), spill.Config{})
	seed, err := New(t.Context(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := seed.Prompt(t.Context(), agentic.NewTextMessage(agentic.RoleUser, "one input")); err != nil {
		t.Fatal(err)
	}
	entries := loadJournalEntries(t, seed)
	if err := seed.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{kindAssistantCommitted, kindOutputValidated, kindTurnEnded, kindRunCompleted, kindRunEnded, kindRunClosed} {
		t.Run(kind, func(t *testing.T) {
			repository := storememory.New()
			var prefix []store.PendingEntry
			for _, entry := range entries {
				prefix = append(prefix, store.PendingEntry{Kind: entry.Kind, Payload: entry.Payload, Durability: entry.Durability})
				if entry.Kind == kind {
					break
				}
			}
			journal, _, err := repository.Create(t.Context(), cfg.ID, prefix...)
			if err != nil {
				t.Fatal(err)
			}
			if err := journal.Close(t.Context()); err != nil {
				t.Fatal(err)
			}
			recoveryModel := &scriptedModel{steps: []modelStep{textStep("must not regenerate a committed reply")}}
			config := cfg
			config.Repository = repository
			config.Driver = agentic.NewAgent("", recoveryModel)
			// Reopening must not allocate a substitute run identity.
			config.IDs = idsFunc(func(string) (string, error) { t.Error("recovery allocated a new run"); return "unexpected", nil })
			recovered, err := Recover(t.Context(), config)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			defer cancel()
			if err := recovered.WaitForIdle(ctx); err != nil {
				t.Fatal(err)
			}
			replayed := loadJournalEntries(t, recovered)
			var closures []runClosedPayload
			for _, entry := range replayed {
				if entry.Kind == kindRunClosed {
					closed, err := decodePayload[runClosedPayload](config.Codec, entry)
					if err != nil {
						t.Fatal(err)
					}
					closures = append(closures, closed)
				}
			}
			if len(closures) != 1 || closures[0].Status != agentic.ExecutionCompleted {
				t.Errorf("committed reply must settle once, not become a failed attempt: %+v", closures)
			}
			if calls := len(recoveryModel.Calls()); calls != 0 {
				t.Errorf("regenerated committed reply: model calls=%d", calls)
			}
			snap, err := recovered.Snapshot(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if len(snap.Messages) != 2 || snap.Messages[1].GetTextContent() != "durable reply" {
				t.Errorf("recovery changed transcript: %+v", snap.Messages)
			}
			if err := recovered.Close(ctx); err != nil {
				t.Fatal(err)
			}
		})
	}
}
