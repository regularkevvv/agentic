package session

// Failure cleanup must join an uncooperative driver without inventing a user
// interrupt. Both ordinary dispatch and automatic reconstruction own drivers.
import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestAbandonDeadlineRetainsJournalAndAcceptedRun(t *testing.T) {
	for _, recovering := range []bool{false, true} {
		name := "dispatch"
		if recovering {
			name = "recovery"
		}
		t.Run(name, func(t *testing.T) {
			entered, release := make(chan struct{}), make(chan struct{})
			var releaseOnce sync.Once
			unblock := func() { releaseOnce.Do(func() { close(release) }) }
			driver := &countingDriver{before: func(agentic.DriveInput) error {
				close(entered)
				<-release // deliberately ignores context cancellation
				return nil
			}}
			repo := storememory.New()
			cfg := sessionConfig(t, driver, repo, artifactmemory.New(), spill.Config{})
			s, err := New(t.Context(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			if recovering {
				if _, err := s.prepareStart(t.Context(), agentic.NewTextMessage(agentic.RoleUser, "retained"), t.Context()); err != nil {
					t.Fatal(err)
				}
				// No dispatch goroutine was started; only the accepted prefix exists.
				if err := s.Abandon(t.Context()); err != nil {
					t.Fatal(err)
				}
				s, err = Recover(t.Context(), cfg)
				if err != nil {
					t.Fatal(err)
				}
			}
			view, err := NewLoopView(s, LoopConfig[string]{CloseRoot: s.Close})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = view.Abandon(context.Background()) }()
			defer unblock() // registered last, unblocks cleanup on test failure
			if !recovering {
				if _, err := view.Dispatch(t.Context(), sessionloopStartCommand("retained")); err != nil {
					t.Fatal(err)
				}
			}
			select {
			case <-entered:
			case <-time.After(3 * time.Second):
				t.Fatal("driver did not start")
			}
			before := loadJournalEntries(t, s)
			ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
			defer cancel()
			if err := view.Abandon(ctx); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("cleanup must wait for live driver: %v", err)
			}
			if s.State() != Faulted || s.journalClosed {
				t.Fatal("cleanup released a live driver's journal")
			}
			unblock()
			if err := view.Abandon(t.Context()); err != nil {
				t.Fatal(err)
			}
			j, err := repo.Open(t.Context(), cfg.ID)
			if err != nil {
				t.Fatal(err)
			}
			loaded, err := j.Load(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(before, loaded.Entries) {
				t.Fatal("late finalizer changed the durable journal")
			}
			if err := j.Close(t.Context()); err != nil {
				t.Fatal(err)
			}
			cfg.Driver = &countingDriver{}
			next, err := Recover(t.Context(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = next.Abandon(context.Background()) }()
			ctx, cancel = context.WithTimeout(t.Context(), 3*time.Second)
			defer cancel()
			if err := next.WaitForIdle(ctx); err != nil {
				t.Fatal(err)
			}
			closures := 0
			for _, entry := range loadJournalEntries(t, next) {
				if entry.Kind != kindRunClosed {
					continue
				}
				outcome, err := decodePayload[runClosedPayload](cfg.Codec, entry)
				if err != nil {
					t.Fatal(err)
				}
				if outcome.Status != agentic.ExecutionCompleted {
					t.Fatalf("invented terminal outcome: %+v", outcome)
				}
				closures++
			}
			if closures != 1 {
				t.Fatalf("closures=%d", closures)
			}
		})
	}
}

func TestAbandonRetriesRootCleanupFailure(t *testing.T) {
	sentinel := errors.New("journal close transport error")
	calls := 0
	view, native := newLoopViewForTest(t, &countingDriver{}, storememory.New(), func(_ *Config[string], cfg *LoopConfig[string]) {
		cfg.CloseRoot = func(context.Context) error {
			calls++
			if calls == 1 {
				return sentinel
			}
			return nil
		}
	})
	defer func() { _ = native.Close(context.Background()) }()
	if err := view.Abandon(t.Context()); !errors.Is(err, sentinel) {
		t.Fatalf("first close=%v", err)
	}
	if err := view.Abandon(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := view.Abandon(t.Context()); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("root cleanup calls=%d, expected one retry then memoization", calls)
	}
}

func TestNativeAbandonDeadlineAndClosedRetry(t *testing.T) {
	cfg := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	s, err := New(t.Context(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	s.recoveryDone = done
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := s.Abandon(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("join=%v", err)
	}
	close(done)
	if err := s.Abandon(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := s.Abandon(t.Context()); err != nil {
		t.Fatal(err)
	}
	if s.State() != Closed {
		t.Fatal("closed session reopened by cleanup retry")
	}
}
