package session

// These tests force interruption before the recovery goroutine is scheduled.
// No timing sleeps are needed: startup itself must settle or fault the run.

import (
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestRecoveryStartupInterruptedBeforeDriver(t *testing.T) {
	for _, fails := range []bool{false, true} {
		name := "settles"
		if fails {
			name = "storage_error_faults"
		}
		t.Run(name, func(t *testing.T) {
			s := bareRecoverySession(t)
			journal, _, err := storememory.New().Create(t.Context(), s.id)
			if err != nil {
				t.Fatal(err)
			}
			s.journal = journal
			boom := errors.New("settlement storage failure")
			if fails {
				s.journal = &failingJournal{Journal: journal, owner: &failingRepository{kind: kindRunClosed, err: boom}}
			}
			if _, err := s.requestInterrupt(t.Context(), ""); err != nil {
				t.Fatal(err)
			}
			if s.State() != Interrupting || s.runCancel != nil {
				t.Fatal("fixture must interrupt before driver startup")
			}
			s.continueRecovered()
			if s.driver.(*countingDriver).Count() != 0 {
				t.Fatal("interrupted recovery started a model request")
			}
			if fails {
				if s.State() != Faulted || !errors.Is(s.fault, boom) {
					t.Fatalf("state=%s fault=%v", s.State(), s.fault)
				}
				return
			}
			if s.State() != Idle || s.run != nil {
				t.Fatalf("recovery stranded: state=%s run=%v", s.State(), s.run)
			}
			snapshot, err := journal.Load(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			var closed []store.Entry
			for _, entry := range snapshot.Entries {
				if entry.Kind == kindRunClosed {
					closed = append(closed, entry)
				}
			}
			if len(closed) != 1 {
				t.Fatalf("durable run closures=%d", len(closed))
			}
		})
	}
}
