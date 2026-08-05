package session

// Characterization infrastructure for the sessionloop plan (S0, plan sections
// 8.2 and 12.2). These helpers freeze the CURRENT public behavior of
// harness/session before any acceptance/execution split. Everything is driven
// through the public surface; the only package-private access is read-only:
// the kind* constants, decodePayload, and journal loads for golden capture.
//
// Determinism contract: fixedClock + a per-prefix counting idsFunc +
// jsoncodec.New() are pinned as part of the fixture contract. Goldens are
// regenerated with:
//
//	go test ./harness/session/ -run TestCharacterization -update-characterization

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/codec"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/store"
)

var updateCharacterization = flag.Bool(
	"update-characterization",
	false,
	"regenerate characterization goldens and crash fixtures under testdata/characterization",
)

// characterizationTime is the fixed UTC instant every characterization
// session observes through Config.Clock.
var characterizationTime = time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

// characterizationWait bounds every blocking receive as a failure detector
// only; passing tests never wait on it.
const characterizationWait = 10 * time.Second

// characterizationIDs returns a deterministic per-prefix ID generator that
// yields "<prefix>_c<counter>".
func characterizationIDs() harnessruntime.IDGenerator {
	var mu sync.Mutex
	counters := map[string]int{}
	return idsFunc(func(prefix string) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		counters[prefix]++
		return fmt.Sprintf("%s_c%d", prefix, counters[prefix]), nil
	})
}

// characterizationRecoveryIDs is the generator injected when reopening a
// committed crash fixture, so recovery-minted IDs ("run_r1") never collide
// with the fixture's own IDs ("run_c1").
func characterizationRecoveryIDs() harnessruntime.IDGenerator {
	var mu sync.Mutex
	counters := map[string]int{}
	return idsFunc(func(prefix string) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		counters[prefix]++
		return fmt.Sprintf("%s_r%d", prefix, counters[prefix]), nil
	})
}

// characterizationConfig builds a session config through the canonical
// sessionConfig helper and pins the deterministic clock and ID ports. The
// codec stays jsoncodec.New() as installed by sessionConfig.
func characterizationConfig(t *testing.T, driver agentic.Driver[string], repository store.Repository) Config[string] {
	t.Helper()
	config := sessionConfig(t, driver, repository, artifactmemory.New(), spill.Config{})
	config.Clock = fixedClock{value: characterizationTime}
	config.IDs = characterizationIDs()
	return config
}

// normalizedJournalEntry is the golden shape for one durable journal entry.
// Store-generated entry IDs are replaced with deterministic "e<seq>"
// placeholders; every other value is already deterministic under the pinned
// clock/ID/codec contract, except the session ID which is normalized to
// "SESSION_ID".
type normalizedJournalEntry struct {
	Seq      uint64 `json:"seq"`
	Kind     string `json:"kind"`
	EntryID  string `json:"entry_id"`
	ParentID string `json:"parent_id"`
	Payload  any    `json:"payload"`
}

// normalizedEvent is the golden shape for one published event record.
type normalizedEvent struct {
	Name   string `json:"name"`
	Nature string `json:"nature"`
	Type   int    `json:"type"`
	Cursor uint64 `json:"cursor"`
}

func normalizeJournal(t *testing.T, payloadCodec codec.Codec, sessionID string, entries []store.Entry) []normalizedJournalEntry {
	t.Helper()
	normalized := make([]normalizedJournalEntry, len(entries))
	for index, entry := range entries {
		var payload any
		if err := payloadCodec.Decode(entry.Payload, &payload); err != nil {
			t.Fatalf("decode %s entry at sequence %d: %v", entry.Kind, entry.Seq, err)
		}
		if strings.HasPrefix(entry.Kind, "agentic.") {
			record, ok := payload.(map[string]any)
			if !ok {
				t.Fatalf("agentic entry %s at sequence %d did not decode to an object: %T", entry.Kind, entry.Seq, payload)
			}
			if encoded, ok := record["Payload"].(string); ok && encoded != "" {
				raw, err := base64.StdEncoding.DecodeString(encoded)
				if err != nil {
					t.Fatalf("decode inner payload of %s at sequence %d: %v", entry.Kind, entry.Seq, err)
				}
				var inner any
				if err := payloadCodec.Decode(raw, &inner); err != nil {
					t.Fatalf("decode inner payload of %s at sequence %d: %v", entry.Kind, entry.Seq, err)
				}
				record["Payload"] = inner
			}
		}
		parent := ""
		if entry.Seq > 1 {
			parent = fmt.Sprintf("e%d", entry.Seq-1)
		}
		normalized[index] = normalizedJournalEntry{
			Seq:      entry.Seq,
			Kind:     entry.Kind,
			EntryID:  fmt.Sprintf("e%d", entry.Seq),
			ParentID: parent,
			Payload:  normalizeSessionID(payload, sessionID),
		}
	}
	return normalized
}

// normalizeSessionID replaces every string exactly equal to the session ID
// with a stable placeholder so goldens survive the per-run session counter.
func normalizeSessionID(value any, sessionID string) any {
	switch current := value.(type) {
	case map[string]any:
		for key, item := range current {
			current[key] = normalizeSessionID(item, sessionID)
		}
		return current
	case []any:
		for index, item := range current {
			current[index] = normalizeSessionID(item, sessionID)
		}
		return current
	case string:
		if current == sessionID {
			return "SESSION_ID"
		}
		return current
	default:
		return value
	}
}

func natureString(nature agentic.EventNature) string {
	switch nature {
	case agentic.EventPreview:
		return "preview"
	case agentic.EventAuthoritative:
		return "authoritative"
	case agentic.EventLifecycle:
		return "lifecycle"
	default:
		return fmt.Sprintf("nature(%d)", nature)
	}
}

func normalizeEvents(records []event.Record) []normalizedEvent {
	normalized := make([]normalizedEvent, len(records))
	for index, record := range records {
		normalized[index] = normalizedEvent{
			Name:   record.Name,
			Nature: natureString(record.Nature),
			Type:   int(record.Type),
			Cursor: record.Cursor,
		}
	}
	return normalized
}

func marshalGolden(t *testing.T, value any) []byte {
	t.Helper()
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(encoded, '\n')
}

func characterizationPath(name string) string {
	return filepath.Join("testdata", "characterization", name)
}

// goldenBytesEqual is the single comparison primitive every golden check
// uses; TestCharacterizationGoldensDetectMutations proves it bites.
func goldenBytesEqual(want, got []byte) bool {
	return bytes.Equal(want, got)
}

func compareGolden(t *testing.T, name string, got []byte) {
	t.Helper()
	path := characterizationPath(name)
	if *updateCharacterization {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("missing golden %s (regenerate with -update-characterization): %v", name, err)
	}
	if !goldenBytesEqual(want, got) {
		t.Fatalf("golden %s mismatch\n--- want ---\n%s\n--- got ---\n%s", name, want, got)
	}
}

func loadJournalEntries(t *testing.T, session *Session[string]) []store.Entry {
	t.Helper()
	loaded, err := session.journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return loaded.Entries
}

func journalKinds(entries []store.Entry) []string {
	kinds := make([]string, len(entries))
	for index, entry := range entries {
		kinds[index] = entry.Kind
	}
	return kinds
}

func firstEntryIndex(entries []store.Entry, kind string) int {
	for index, entry := range entries {
		if entry.Kind == kind {
			return index
		}
	}
	return -1
}

func countEntries(entries []store.Entry, kind string) int {
	count := 0
	for _, entry := range entries {
		if entry.Kind == kind {
			count++
		}
	}
	return count
}

// replayDurableRecords subscribes after cursor zero and reads the full
// replayed durable history, asserting strict cursor monotonicity.
func replayDurableRecords(t *testing.T, session *Session[string], count int) []event.Record {
	t.Helper()
	subscription := session.Subscribe(SubscribeOptions{Buffer: count + 16})
	defer subscription.Close()
	records := make([]event.Record, 0, count)
	guard := time.After(characterizationWait)
	for len(records) < count {
		select {
		case record, ok := <-subscription.Events:
			if !ok {
				t.Fatalf("event stream closed after %d of %d records", len(records), count)
			}
			records = append(records, record)
		case err := <-subscription.Err:
			t.Fatalf("event stream error after %d records: %v", len(records), err)
		case <-guard:
			t.Fatalf("timed out waiting for %d replayed records, got %d", count, len(records))
		}
	}
	for index := 1; index < len(records); index++ {
		if records[index].Cursor <= records[index-1].Cursor {
			t.Fatalf("replayed cursors are not strictly monotonic: %d then %d at index %d",
				records[index-1].Cursor, records[index].Cursor, index)
		}
	}
	return records
}

func awaitSignal(t *testing.T, signal <-chan struct{}, label string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(characterizationWait):
		t.Fatalf("timed out waiting for %s", label)
	}
}

func awaitPrompt(t *testing.T, done <-chan promptOutcome, label string) promptOutcome {
	t.Helper()
	select {
	case outcome := <-done:
		return outcome
	case <-time.After(characterizationWait):
		t.Fatalf("timed out waiting for %s", label)
		return promptOutcome{}
	}
}

func awaitErr(t *testing.T, done <-chan error, label string) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(characterizationWait):
		t.Fatalf("timed out waiting for %s", label)
		return nil
	}
}

// awaitState spins (no sleeps) until the session reaches the wanted state,
// with a bounded deadline as a failure detector.
func awaitState(t *testing.T, session *Session[string], want State) {
	t.Helper()
	deadline := time.Now().Add(characterizationWait)
	for session.State() != want {
		if time.Now().After(deadline) {
			t.Fatalf("session never reached %s (state %s)", want, session.State())
		}
		runtime.Gosched()
	}
}

type promptOutcome struct {
	execution *agentic.Execution[string]
	err       error
}

func promptAsync(session *Session[string], text string) <-chan promptOutcome {
	done := make(chan promptOutcome, 1)
	go func() {
		execution, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, text))
		done <- promptOutcome{execution: execution, err: err}
	}()
	return done
}

// TestCharacterizationGoldensDetectMutations proves the golden comparison is
// sensitive: reordering two adjacent journal entries or changing one payload
// field or one event nature must produce a byte-level difference.
func TestCharacterizationGoldensDetectMutations(t *testing.T) {
	if *updateCharacterization {
		t.Skip("goldens are being regenerated in this run")
	}
	want, err := os.ReadFile(characterizationPath("tool_free_prompt.golden.json"))
	if err != nil {
		t.Fatalf("missing committed golden (run -update-characterization first): %v", err)
	}
	var entries []normalizedJournalEntry
	if err := json.Unmarshal(want, &entries); err != nil {
		t.Fatal(err)
	}
	if len(entries) < 3 {
		t.Fatalf("golden too small to mutate: %d entries", len(entries))
	}
	if !goldenBytesEqual(want, marshalGolden(t, entries)) {
		t.Fatal("golden does not round-trip byte-stably; the comparison cannot be trusted")
	}

	swapped := make([]normalizedJournalEntry, len(entries))
	copy(swapped, entries)
	swapped[1], swapped[2] = swapped[2], swapped[1]
	if goldenBytesEqual(want, marshalGolden(t, swapped)) {
		t.Fatal("swapping two adjacent journal entries was not detected")
	}

	var mutated []normalizedJournalEntry
	if err := json.Unmarshal(want, &mutated); err != nil {
		t.Fatal(err)
	}
	payload, ok := mutated[1].Payload.(map[string]any)
	if !ok {
		t.Fatalf("golden entry 1 payload is not an object: %T", mutated[1].Payload)
	}
	payload["Mode"] = "mutated"
	if goldenBytesEqual(want, marshalGolden(t, mutated)) {
		t.Fatal("changing a payload field was not detected")
	}

	wantEvents, err := os.ReadFile(characterizationPath("tool_free_prompt.events.golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	var events []normalizedEvent
	if err := json.Unmarshal(wantEvents, &events); err != nil {
		t.Fatal(err)
	}
	if !goldenBytesEqual(wantEvents, marshalGolden(t, events)) {
		t.Fatal("events golden does not round-trip byte-stably")
	}
	events[0].Nature = "mutated"
	if goldenBytesEqual(wantEvents, marshalGolden(t, events)) {
		t.Fatal("changing an event nature was not detected")
	}
}
