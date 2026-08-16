package session

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/artifact"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	"github.com/regularkevvv/agentic/harness/store"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type modelStep struct {
	message agentic.Message
	usage   agentic.Usage
	entered chan struct{}
	release <-chan struct{}
	err     error
}

type scriptedModel struct {
	mu    sync.Mutex
	steps []modelStep
	calls []agentic.ChatRequest
}

func (m *scriptedModel) Name() string { return "test:session" }

func (m *scriptedModel) Request(ctx context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	m.mu.Lock()
	index := len(m.calls)
	copy := *request
	copy.Messages = cloneMessages(request.Messages)
	m.calls = append(m.calls, copy)
	step := m.steps[len(m.steps)-1]
	if index < len(m.steps) {
		step = m.steps[index]
	}
	m.mu.Unlock()
	if step.entered != nil {
		select {
		case <-step.entered:
		default:
			close(step.entered)
		}
	}
	if step.release != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-step.release:
		}
	}
	if step.err != nil {
		return nil, step.err
	}
	finish := agentic.FinishReasonStop
	if len(step.message.GetToolUses()) > 0 {
		finish = agentic.FinishReasonToolCalls
	}
	return &agentic.ChatResponse{Model: m.Name(), Message: step.message, Usage: step.usage, FinishReason: finish, RawFinishReason: string(finish)}, nil
}

func (m *scriptedModel) Calls() []agentic.ChatRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]agentic.ChatRequest, len(m.calls))
	copy(result, m.calls)
	return result
}

type countingDriver struct {
	mu        sync.Mutex
	drives    []agentic.DriveInput
	badCommit bool
	before    func(agentic.DriveInput) error
}

type correlationInstrumentation struct {
	agents chan agentic.AgentOperation
}

func (i *correlationInstrumentation) StartAgent(ctx context.Context, operation agentic.AgentOperation) (context.Context, agentic.AgentOperationSpan) {
	i.agents <- operation
	return ctx, correlationAgentSpan{}
}

func (*correlationInstrumentation) StartModelRequest(ctx context.Context, _ agentic.ModelOperation) (context.Context, agentic.ModelOperationSpan) {
	return ctx, correlationModelSpan{}
}

func (*correlationInstrumentation) StartTool(ctx context.Context, _ agentic.ToolOperation) (context.Context, agentic.ToolOperationSpan) {
	return ctx, correlationToolSpan{}
}

type correlationAgentSpan struct{}

func (correlationAgentSpan) End(agentic.AgentOperationResult) {}

type correlationModelSpan struct{}

func (correlationModelSpan) ObserveStreamEvent(agentic.StreamEvent) {}
func (correlationModelSpan) End(agentic.ModelOperationResult)       {}

type correlationToolSpan struct{}

func (correlationToolSpan) End(agentic.ToolOperationResult) {}

type rejectingAssistantCodec struct {
	codec.Codec
}

func (c rejectingAssistantCodec) Encode(value any) ([]byte, error) {
	if _, ok := value.(event.AssistantPayload); ok {
		return nil, errors.New("assistant payload rejected")
	}
	return c.Codec.Encode(value)
}

type previewEvent struct {
	turn int
}

func (e previewEvent) Nature() agentic.EventNature { return agentic.EventPreview }
func (e previewEvent) Type() agentic.EventType     { return agentic.EventTypeTextPreview }
func (e previewEvent) TurnIndex() int              { return e.turn }

func (d *countingDriver) Run(ctx context.Context, prompt string, opts ...agentic.RunOption) (*agentic.Result[string], error) {
	message := agentic.NewTextMessage(agentic.RoleUser, prompt)
	execution, err := d.Drive(ctx, agentic.DriveInput{Mode: agentic.DriveStart, Prompt: &message}, opts...)
	if execution == nil {
		return nil, err
	}
	return execution.Result, err
}

func (d *countingDriver) Drive(_ context.Context, input agentic.DriveInput, _ ...agentic.RunOption) (*agentic.Execution[string], error) {
	if d.before != nil {
		if err := d.before(input); err != nil {
			return nil, err
		}
	}
	d.mu.Lock()
	d.drives = append(d.drives, input)
	d.mu.Unlock()
	messages := cloneMessages(input.History)
	if input.Prompt != nil {
		messages = append(messages, cloneMessages([]agentic.Message{*input.Prompt})[0])
	}
	if d.badCommit {
		messages = append(messages, agentic.NewTextMessage(agentic.RoleAssistant, "uncommitted"))
	}
	return &agentic.Execution[string]{Status: agentic.ExecutionCompleted, Result: &agentic.Result[string]{Messages: messages}}, nil
}

func (d *countingDriver) Resume(context.Context, agentic.ResumeInput, ...agentic.RunOption) (*agentic.Execution[string], error) {
	return nil, errors.New("unexpected Resume")
}

func (d *countingDriver) Count() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.drives)
}

func (d *countingDriver) Last() agentic.DriveInput {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.drives[len(d.drives)-1]
}

var testSessionCounter atomic.Uint64

func sessionConfig[O any](t *testing.T, driver agentic.Driver[O], repository store.Repository, artifacts artifact.Store, spillConfig spill.Config) Config[O] {
	t.Helper()
	environments, err := envmemory.NewFactory(envmemory.Config{Cwd: "/workspace"})
	if err != nil {
		t.Fatal(err)
	}
	processors, err := spill.NewFactory(artifacts, spillConfig)
	if err != nil {
		t.Fatal(err)
	}
	return Config[O]{
		ID:               fmt.Sprintf("session_%d", testSessionCounter.Add(1)),
		Driver:           driver,
		Repository:       repository,
		Codec:            jsoncodec.New(),
		Events:           inproc.NewFactory(),
		Environments:     environments,
		ResultProcessors: processors,
		Clock:            system.NewClock(),
		IDs:              system.NewIDs(),
	}
}

func textStep(text string) modelStep {
	return modelStep{message: agentic.NewTextMessage(agentic.RoleAssistant, text), usage: agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}}
}

func TestSteerAndFollowUpLinearizeWithClosing(t *testing.T) {
	for _, kind := range []QueueKind{QueueSteer, QueueFollowUp} {
		kind := kind
		t.Run(string(kind), func(t *testing.T) {
			for iteration := 0; iteration < 50; iteration++ {
				driver := &countingDriver{}
				repository := storememory.New()
				config := sessionConfig(t, driver, repository, artifactmemory.New(), spill.Config{})
				session, err := New(context.Background(), config)
				if err != nil {
					t.Fatal(err)
				}
				session.mu.Lock()
				session.run = &activeRun{id: "race"}
				session.transitionLocked(Running)
				session.mu.Unlock()

				start := make(chan struct{})
				decisionCh := make(chan agentic.TurnDecision, 1)
				errorCh := make(chan error, 1)
				go func() {
					<-start
					decision, hookErr := session.turnHook(context.Background(), agentic.Turn{Candidate: agentic.CompletionCandidate{Source: agentic.CompletionText}})
					if hookErr != nil {
						errorCh <- hookErr
						return
					}
					decisionCh <- decision
				}()
				receiptCh := make(chan QueueReceipt, 1)
				acceptErrCh := make(chan error, 1)
				go func() {
					<-start
					var receipt QueueReceipt
					var acceptErr error
					if kind == QueueSteer {
						receipt, acceptErr = session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "queued"))
					} else {
						receipt, acceptErr = session.FollowUp(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "queued"))
					}
					if acceptErr != nil {
						acceptErrCh <- acceptErr
						return
					}
					receiptCh <- receipt
				}()
				close(start)

				var decision agentic.TurnDecision
				select {
				case err := <-errorCh:
					t.Fatal(err)
				case decision = <-decisionCh:
				}
				select {
				case receipt := <-receiptCh:
					if receipt.ID == "" || decision.Action != agentic.TurnContinue || len(decision.Inject) != 1 {
						t.Fatalf("accepted receipt=%#v decision=%#v", receipt, decision)
					}
					snapshot, _ := session.Snapshot(context.Background())
					if len(snapshot.Pending) != 0 {
						t.Fatalf("accepted input remained queued: %#v", snapshot.Pending)
					}
				case acceptErr := <-acceptErrCh:
					if !errors.Is(acceptErr, ErrRunClosing) || decision.Action != agentic.TurnDefault || session.State() != Closing {
						t.Fatalf("rejected err=%v decision=%#v state=%s", acceptErr, decision, session.State())
					}
				}
			}
		})
	}
}

func TestDurableQueueDrainExactlyOnceAcrossRestart(t *testing.T) {
	repository := storememory.New()
	artifacts := artifactmemory.New()
	model := &scriptedModel{steps: []modelStep{textStep("done")}}
	driver := agentic.NewAgent("", model)
	config := sessionConfig(t, driver, repository, artifacts, spill.Config{})
	first, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := first.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "durable-next"))
	if err != nil || receipt.Cursor == 0 {
		t.Fatalf("NextTurn = %#v, %v", receipt, err)
	}
	if err := first.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	restarted, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := restarted.Snapshot(context.Background())
	if len(snapshot.Pending) != 1 || snapshot.Pending[0].ID != receipt.ID {
		t.Fatalf("recovered pending = %#v", snapshot.Pending)
	}
	if _, err := restarted.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); err != nil {
		t.Fatal(err)
	}
	calls := model.Calls()
	if len(calls) != 1 || len(calls[0].Messages) != 2 || calls[0].Messages[0].GetTextContent() != "durable-next" || calls[0].Messages[1].GetTextContent() != "prompt" {
		t.Fatalf("driver messages = %#v", calls)
	}
	if err := restarted.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	again, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ = again.Snapshot(context.Background())
	if len(snapshot.Pending) != 0 {
		t.Fatalf("queue drained more than once: %#v", snapshot.Pending)
	}
	loaded, loadErr := again.journal.Load(context.Background())
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	accepted, drained := 0, 0
	for _, entry := range loaded.Entries {
		if entry.Kind == kindQueueAccepted {
			accepted++
		}
		if entry.Kind == kindQueueDrained {
			drained++
		}
	}
	if accepted != 1 || drained != 1 {
		t.Fatalf("accepted=%d drained=%d", accepted, drained)
	}
}

func TestPromptPassesDurableSessionAndRunIDsToInstrumentation(t *testing.T) {
	repository := storememory.New()
	observer := &correlationInstrumentation{agents: make(chan agentic.AgentOperation, 1)}
	model := &scriptedModel{steps: []modelStep{textStep("done")}}
	config := sessionConfig(t,
		agentic.NewAgent("", model, agentic.WithInstrumentation(observer)),
		repository,
		artifactmemory.New(),
		spill.Config{},
	)
	current, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := current.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); err != nil {
		t.Fatal(err)
	}
	operation := <-observer.agents
	if operation.Run.ConversationID != config.ID || operation.Run.RunID == "" {
		t.Fatalf("instrumentation correlation = %#v", operation.Run)
	}

	loaded, err := current.journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range loaded.Entries {
		if entry.Kind != kindRunOpened {
			continue
		}
		opened, err := decodePayload[runOpenedPayload](config.Codec, entry)
		if err != nil {
			t.Fatal(err)
		}
		if operation.Run.RunID != opened.ID {
			t.Fatalf("instrumentation run ID %q != durable run ID %q", operation.Run.RunID, opened.ID)
		}
		return
	}
	t.Fatal("journal contains no run.opened entry")
}

func TestPromptAndNextTurnAreDurableBeforeDriverExecution(t *testing.T) {
	repository := storememory.New()
	driver := &countingDriver{}
	config := sessionConfig(t, driver, repository, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "queued first")); err != nil {
		t.Fatal(err)
	}
	inspected := false
	driver.before = func(input agentic.DriveInput) error {
		loaded, err := session.journal.Load(context.Background())
		if err != nil {
			return err
		}
		snapshot, err := session.Snapshot(context.Background())
		if err != nil {
			return err
		}
		if loaded.Cursor.Seq != snapshot.Cursor {
			return fmt.Errorf("driver observed cursor %v while snapshot was at %d", loaded.Cursor, snapshot.Cursor)
		}
		var persisted []messagePayload
		for _, entry := range loaded.Entries {
			if entry.Kind != kindMessage {
				continue
			}
			payload, err := decodePayload[messagePayload](config.Codec, entry)
			if err != nil {
				return err
			}
			persisted = append(persisted, payload)
		}
		if len(persisted) != 2 ||
			persisted[0].Source != string(QueueNextTurn) || persisted[0].Message.GetTextContent() != "queued first" ||
			persisted[1].Source != "prompt" || persisted[1].Message.GetTextContent() != "prompt second" {
			return fmt.Errorf("persisted driver inputs = %#v", persisted)
		}
		if len(input.History) != 1 || input.History[0].GetTextContent() != "queued first" ||
			input.Prompt == nil || input.Prompt.GetTextContent() != "prompt second" {
			return fmt.Errorf("drive input = %#v", input)
		}
		inspected = true
		return nil
	}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt second")); err != nil {
		t.Fatal(err)
	}
	if !inspected {
		t.Fatal("driver did not inspect durable inputs")
	}
}

func TestUsageAndBudgetRestoreAcrossRestart(t *testing.T) {
	repository := storememory.New()
	artifacts := artifactmemory.New()
	model := &scriptedModel{steps: []modelStep{textStep("first"), textStep("crosses")}}
	driver := agentic.NewAgent("", model)
	config := sessionConfig(t, driver, repository, artifacts, spill.Config{})
	maxTotal := 20
	first, err := New(context.Background(), config, WithBudget(agentic.UsageLimits{MaxTotalTokens: &maxTotal}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "one")); err != nil {
		t.Fatal(err)
	}
	if err := first.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	restarted, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := restarted.Snapshot(context.Background())
	if snapshot.Usage.TotalTokens != 15 || snapshot.Usage.Requests != 1 {
		t.Fatalf("restored usage = %#v", snapshot.Usage)
	}
	if _, err := restarted.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "two")); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("crossing error = %v", err)
	}
	snapshot, _ = restarted.Snapshot(context.Background())
	if snapshot.Usage.TotalTokens != 30 || snapshot.Usage.Requests != 2 {
		t.Fatalf("post-crossing usage = %#v", snapshot.Usage)
	}
	if _, err := restarted.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "three")); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("exhausted error = %v", err)
	}
}

func TestResultMessagesAreProjectedButNeverAppendedAgainAtClose(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{textStep("one assistant")}}
	config := sessionConfig(t, agentic.NewAgent("", model), storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "one prompt")); err != nil {
		t.Fatal(err)
	}
	snapshot, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Messages) != 2 ||
		snapshot.Messages[0].GetTextContent() != "one prompt" ||
		snapshot.Messages[1].GetTextContent() != "one assistant" {
		t.Fatalf("projected messages = %#v", snapshot.Messages)
	}
	loaded, err := session.journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	assistantCommits := 0
	for _, entry := range loaded.Entries {
		if entry.Kind == kindAssistantCommitted {
			assistantCommits++
		}
	}
	if assistantCommits != 1 {
		t.Fatalf("assistant commit entries = %d", assistantCommits)
	}
}

type failingRepository struct {
	base store.Repository
	mu   sync.Mutex
	kind string
	err  error
}

func (f *failingRepository) Create(ctx context.Context, id string, entries ...store.PendingEntry) (store.Journal, store.Commit, error) {
	journal, commit, err := f.base.Create(ctx, id, entries...)
	if err != nil {
		return nil, store.Commit{}, err
	}
	return &failingJournal{Journal: journal, owner: f}, commit, nil
}

func (f *failingRepository) Open(ctx context.Context, id string) (store.Journal, error) {
	journal, err := f.base.Open(ctx, id)
	if err != nil {
		return nil, err
	}
	return &failingJournal{Journal: journal, owner: f}, nil
}

type failingJournal struct {
	store.Journal
	owner *failingRepository
}

func (j *failingJournal) Append(ctx context.Context, cursor store.Cursor, entries ...store.PendingEntry) (store.Commit, error) {
	j.owner.mu.Lock()
	for _, entry := range entries {
		if j.owner.kind != "" && entry.Kind == j.owner.kind {
			err := j.owner.err
			j.owner.kind = ""
			j.owner.mu.Unlock()
			return store.Commit{}, err
		}
	}
	j.owner.mu.Unlock()
	return j.Journal.Append(ctx, cursor, entries...)
}

func (f *failingRepository) fail(kind string, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.kind = kind
	f.err = err
}

func TestStorageFailureBeforeAndAfterMutationFaultAndReopen(t *testing.T) {
	base := storememory.New()
	repository := &failingRepository{base: base}
	artifacts := artifactmemory.New()
	driver := &countingDriver{}
	config := sessionConfig(t, driver, repository, artifacts, spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	writeErr := errors.New("disk unavailable")
	repository.fail(kindQueueAccepted, writeErr)
	if _, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "not accepted")); !errors.Is(err, writeErr) {
		t.Fatalf("queue error = %v", err)
	}
	snapshot, _ := session.Snapshot(context.Background())
	if snapshot.State != Idle || len(snapshot.Pending) != 0 {
		t.Fatalf("pre-mutation failure changed state: %#v", snapshot)
	}

	model := &scriptedModel{steps: []modelStep{textStep("will fault"), textStep("recovered")}}
	realDriver := agentic.NewAgent("", model)
	config.Driver = realDriver
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	reopenedBase, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	repository.fail(kindAssistantCommitted, writeErr)
	if _, err := reopenedBase.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "run")); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulting Prompt error = %v", err)
	}
	if state := reopenedBase.State(); state != Faulted {
		t.Fatalf("state = %s", state)
	}
	if _, err := reopenedBase.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "reject")); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("faulted input error = %v", err)
	}
	if err := reopenedBase.WaitForIdle(context.Background()); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("WaitForIdle error = %v", err)
	}

	if err := reopenedBase.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if err := recovered.WaitForIdle(context.Background()); err != nil {
		t.Fatal(err)
	}
	if recovered.State() != Idle {
		t.Fatalf("recovered state = %s", recovered.State())
	}
}

func TestCommitProjectionMismatchFaultsSession(t *testing.T) {
	driver := &countingDriver{badCommit: true}
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); !errors.Is(err, ErrSessionFaulted) || !errors.Is(err, ErrCommitProjectionMismatch) {
		t.Fatalf("Prompt error = %v", err)
	}
	if session.State() != Faulted {
		t.Fatalf("state = %s", session.State())
	}
}

func TestAuthoritativeCodecFailureFaultsSession(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{textStep("cannot persist")}}
	config := sessionConfig(t, agentic.NewAgent("", model), storememory.New(), artifactmemory.New(), spill.Config{})
	config.Codec = rejectingAssistantCodec{Codec: jsoncodec.New()}
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "prompt")); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("Prompt error = %v", err)
	}
	if session.State() != Faulted {
		t.Fatalf("state = %s", session.State())
	}
}

func TestSessionSubscriberLagSnapshotAndResubscribe(t *testing.T) {
	model := &scriptedModel{steps: []modelStep{textStep("first"), textStep("second")}}
	driver := agentic.NewAgent("", model)
	config := sessionConfig(t, driver, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	lagged := session.Subscribe(SubscribeOptions{Buffer: 1})
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "one")); err != nil {
		t.Fatal(err)
	}
	for range lagged.Events {
	}
	var lag *event.ErrSubscriberLagged
	if err := <-lagged.Err; !errors.As(err, &lag) {
		t.Fatalf("lag error = %v", err)
	}
	snapshot, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	resumed := session.Subscribe(SubscribeOptions{AfterCursor: snapshot.Cursor, Buffer: 64})
	defer resumed.Close()
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "two")); err != nil {
		t.Fatal(err)
	}
	record := <-resumed.Events
	if record.Cursor <= snapshot.Cursor {
		t.Fatalf("resubscribed cursor = %d after %d", record.Cursor, snapshot.Cursor)
	}
}

func TestPreviewOrdinalsResetPerTurnAndToolUpdatesUseCurrentTurn(t *testing.T) {
	config := sessionConfig(t, &countingDriver{}, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := session.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	subscription := session.Subscribe(SubscribeOptions{AfterCursor: snapshot.Cursor, Buffer: 64, Preview: true})
	defer subscription.Close()
	session.mu.Lock()
	session.run = &activeRun{id: "preview"}
	session.transitionLocked(Running)
	session.mu.Unlock()

	if err := session.Emit(context.Background(), previewEvent{turn: 0}); err != nil {
		t.Fatal(err)
	}
	if err := session.Emit(context.Background(), previewEvent{turn: 0}); err != nil {
		t.Fatal(err)
	}
	if err := session.Emit(context.Background(), previewEvent{turn: 1}); err != nil {
		t.Fatal(err)
	}
	session.emitToolUpdate(harnessruntime.ToolUpdate{Kind: "progress"})

	wantTurns := []int{0, 0, 1, 1}
	wantOrdinals := []uint64{1, 2, 1, 2}
	for i := range wantTurns {
		record := <-subscription.Events
		if record.Turn != wantTurns[i] || record.Ordinal != wantOrdinals[i] {
			t.Fatalf("preview %d = turn %d ordinal %d", i, record.Turn, record.Ordinal)
		}
	}

	const concurrent = 16
	var updates sync.WaitGroup
	updates.Add(concurrent)
	for i := 0; i < concurrent; i++ {
		go func() {
			defer updates.Done()
			session.emitToolUpdate(harnessruntime.ToolUpdate{Kind: "concurrent"})
		}()
	}
	updates.Wait()
	for i := 0; i < concurrent; i++ {
		record := <-subscription.Events
		wantOrdinal := uint64(i + 3)
		if record.Turn != 1 || record.Ordinal != wantOrdinal {
			t.Fatalf("concurrent preview %d = turn %d ordinal %d, want %d", i, record.Turn, record.Ordinal, wantOrdinal)
		}
	}

	session.mu.Lock()
	session.run = nil
	session.transitionLocked(Idle)
	session.mu.Unlock()
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

type largeInput struct {
	Value string `json:"value"`
}

func TestOversizedToolOutputSpillsOnceAndToolRuntimeIsPlumbed(t *testing.T) {
	call := agentic.ToolUse{ID: "large-call", Name: "large", Input: map[string]any{"value": "x"}}
	model := &scriptedModel{steps: []modelStep{
		{message: agentic.NewToolUseMessage(call), usage: agentic.Usage{TotalTokens: 1}},
		textStep("done"),
	}}
	agent := agentic.NewAgent("", model)
	full := strings.Repeat("0123456789", 20)
	var sawRuntime atomic.Bool
	agentic.AddTool(agent, func(ctx context.Context, _ largeInput) (string, error) {
		runtime, ok := harnessruntime.FromContext(ctx)
		callContext, callOK := agentic.CurrentToolCall(ctx)
		if ok && callOK && runtime.SessionID != "" && runtime.Environment != nil && callContext.ID == call.ID {
			sawRuntime.Store(true)
			runtime.Emit(harnessruntime.ToolUpdate{Kind: "large.progress"})
		}
		return full, nil
	}, agentic.AutoToolName("large"), agentic.AutoToolDescription("Return a large test result"))
	artifacts := artifactmemory.New()
	config := sessionConfig(t, agent, storememory.New(), artifacts, spill.Config{Threshold: 32, Head: 8, Tail: 8})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "run")); err != nil {
		t.Fatal(err)
	}
	if !sawRuntime.Load() {
		t.Fatal("tool did not observe both ToolRuntime and ToolCallContext")
	}
	snapshot, _ := session.Snapshot(context.Background())
	var visible string
	for _, message := range snapshot.Messages {
		for _, result := range message.GetToolResults() {
			if result.ToolUseID == call.ID {
				visible = result.Content
			}
		}
	}
	if !strings.Contains(visible, "harness artifact art_") || artifacts.Count(config.ID) != 1 {
		t.Fatalf("visible=%q count=%d", visible, artifacts.Count(config.ID))
	}
	first := strings.SplitN(visible, ";", 2)[0]
	handle := artifact.Handle(strings.TrimPrefix(first, "[harness artifact "))
	stored, err := artifacts.Get(context.Background(), config.ID, handle)
	if err != nil || string(stored) != full {
		t.Fatalf("stored=%q err=%v", stored, err)
	}
	if _, err := artifacts.Get(context.Background(), "other", handle); !errors.Is(err, artifact.ErrArtifactNotFound) {
		t.Fatalf("cross-session artifact error = %v", err)
	}
}

func TestDriverInsertedSystemMessageProjectsAndRestores(t *testing.T) {
	log := storememory.New()
	model := &scriptedModel{steps: []modelStep{textStep("first"), textStep("second")}}
	driver := agentic.NewAgent("durable system", model)
	config := sessionConfig(t, driver, log, artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "one")); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := recovered.Snapshot(context.Background())
	if len(snapshot.Messages) == 0 || snapshot.Messages[0].Role != agentic.RoleSystem || snapshot.Messages[0].GetTextContent() != "durable system" {
		t.Fatalf("restored messages = %#v", snapshot.Messages)
	}
	if _, err := recovered.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "two")); err != nil {
		t.Fatal(err)
	}
}

func TestInterruptCancelsTransientQueuesAndPreservesNextTurn(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	model := &scriptedModel{steps: []modelStep{{message: agentic.NewTextMessage(agentic.RoleAssistant, "late"), entered: entered, release: release}}}
	agent := agentic.NewAgent("durable interrupt system", model)
	config := sessionConfig(t, agent, storememory.New(), artifactmemory.New(), spill.Config{})
	session, err := New(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	promptDone := make(chan error, 1)
	go func() {
		_, promptErr := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "run"))
		promptDone <- promptErr
	}()
	<-entered
	if _, err := session.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "steer")); err != nil {
		t.Fatal(err)
	}
	if _, err := session.FollowUp(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "follow")); err != nil {
		t.Fatal(err)
	}
	next, err := session.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "next"))
	if err != nil {
		t.Fatal(err)
	}
	interruptCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := session.Interrupt(interruptCtx); err != nil {
		t.Fatal(err)
	}
	if err := <-promptDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("Prompt error = %v", err)
	}
	snapshot, _ := session.Snapshot(context.Background())
	if snapshot.State != Idle || len(snapshot.Pending) != 1 || snapshot.Pending[0].ID != next.ID {
		t.Fatalf("interrupt snapshot = %#v", snapshot)
	}
	for _, message := range snapshot.Messages {
		if strings.Contains(message.GetTextContent(), "harness_context") {
			t.Fatalf("interrupt marker leaked into user-facing snapshot: %#v", snapshot.Messages)
		}
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	session, err = Recover(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ = session.Snapshot(context.Background())
	if len(snapshot.Pending) != 1 || snapshot.Pending[0].ID != next.ID {
		t.Fatalf("recovered interrupt queue = %#v", snapshot.Pending)
	}

	close(release)
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "continue")); err != nil {
		t.Fatal(err)
	}
	calls := model.Calls()
	if len(calls) != 2 {
		t.Fatalf("model calls = %d, want 2", len(calls))
	}
	markerIndex := -1
	nextIndex := -1
	for i, message := range calls[1].Messages {
		text := message.GetTextContent()
		if strings.Contains(text, `<harness_context type="interruption">`) && strings.Contains(text, "deliberately interrupted") {
			markerIndex = i
		}
		if text == "next" {
			nextIndex = i
		}
	}
	if markerIndex < 0 || nextIndex < 0 || markerIndex >= nextIndex {
		t.Fatalf("provider history did not preserve marker ordering: %#v", calls[1].Messages)
	}
	snapshot, _ = session.Snapshot(context.Background())
	for _, message := range snapshot.Messages {
		if strings.Contains(message.GetTextContent(), "harness_context") {
			t.Fatalf("interrupt marker leaked after subsequent prompt: %#v", snapshot.Messages)
		}
	}
}
