package harness

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	providertest "github.com/regularkevvv/agentic/provider/test"

	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/env"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type runnerOnly struct{}

func (runnerOnly) Run(context.Context, string, ...agentic.RunOption) (*agentic.Result[string], error) {
	return &agentic.Result[string]{}, nil
}

type facadeDriver struct {
	mu       sync.Mutex
	badFirst bool
}

type streamingModel struct {
	streamCalls atomic.Int32
	plainCalls  atomic.Int32
}

func (*streamingModel) Name() string { return "test:streaming" }

func (m *streamingModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	m.plainCalls.Add(1)
	return nil, errors.New("plain request used")
}

func (m *streamingModel) RequestStream(context.Context, *agentic.ChatRequest) (*agentic.StreamResult, error) {
	m.streamCalls.Add(1)
	events := make(chan agentic.StreamEvent, 2)
	events <- agentic.StreamEvent{Type: agentic.StreamEventTextDelta, Delta: "streamed"}
	events <- agentic.StreamEvent{Type: agentic.StreamEventDone, FinishReason: agentic.FinishReasonStop}
	close(events)
	return agentic.NewStreamResult(events), nil
}

func (d *facadeDriver) Run(ctx context.Context, prompt string, options ...agentic.RunOption) (*agentic.Result[string], error) {
	message := agentic.NewTextMessage(agentic.RoleUser, prompt)
	execution, err := d.Drive(ctx, agentic.DriveInput{Mode: agentic.DriveStart, Prompt: &message}, options...)
	return execution.Result, err
}

func (d *facadeDriver) Drive(_ context.Context, input agentic.DriveInput, _ ...agentic.RunOption) (*agentic.Execution[string], error) {
	messages := append([]agentic.Message(nil), input.History...)
	if input.Prompt != nil {
		messages = append(messages, *input.Prompt)
	}
	d.mu.Lock()
	bad := d.badFirst
	d.badFirst = false
	d.mu.Unlock()
	if bad {
		messages = append(messages, agentic.NewTextMessage(agentic.RoleAssistant, "not committed"))
	}
	return &agentic.Execution[string]{Status: agentic.ExecutionCompleted, Result: &agentic.Result[string]{Messages: messages}}, nil
}

func (d *facadeDriver) Resume(context.Context, agentic.ResumeInput, ...agentic.RunOption) (*agentic.Execution[string], error) {
	return nil, errors.New("unexpected Resume")
}

func runtimeConfig(t *testing.T) RuntimeConfig {
	t.Helper()
	environments, err := envmemory.NewFactory(envmemory.Config{Cwd: "/workspace"})
	if err != nil {
		t.Fatal(err)
	}
	processors, err := spill.NewFactory(artifactmemory.New(), spill.Config{})
	if err != nil {
		t.Fatal(err)
	}
	return RuntimeConfig{
		Sessions:         storememory.New(),
		Codec:            jsoncodec.New(),
		Events:           inproc.NewFactory(),
		Environments:     environments,
		ResultProcessors: processors,
		Clock:            system.NewClock(),
		IDs:              system.NewIDs(),
	}
}

func TestNewRuntimeRequiresDriverAndExplicitSubstrates(t *testing.T) {
	t.Parallel()
	if _, err := NewRuntime[string](runnerOnly{}, RuntimeConfig{}); !errors.Is(err, agentic.ErrDriverRequired) {
		t.Fatalf("runner-only error = %v", err)
	}
	driver := &facadeDriver{}
	if _, err := NewRuntime[string](driver, RuntimeConfig{}); err == nil {
		t.Fatal("missing store succeeded")
	}
}

func TestHarnessUsesStableSessionPromptCacheAcrossRecovery(t *testing.T) {
	model := providertest.NewTestModel(
		providertest.ModelResponse{Text: "first"},
		providertest.ModelResponse{Text: "second"},
	)
	runtime, err := NewRuntime[string](agentic.NewAgent("system", model), runtimeConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	session, err := runtime.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	id := session.ID()
	if _, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "one")); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	resumed, err := runtime.ResumeSession(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resumed.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "two")); err != nil {
		t.Fatal(err)
	}
	for index, request := range model.Calls() {
		if request.PromptCache == nil || request.PromptCache.Key != id || request.PromptCache.Retention != agentic.PromptCacheShort {
			t.Fatalf("request %d cache = %#v", index, request.PromptCache)
		}
	}
	_ = resumed.Close(context.Background())

	disabledConfig := runtimeConfig(t)
	disabledConfig.PromptCacheRetention = agentic.PromptCacheNone
	disabledModel := providertest.NewTestModel(providertest.ModelResponse{Text: "done"})
	disabled, err := NewRuntime[string](agentic.NewAgent("system", disabledModel), disabledConfig)
	if err != nil {
		t.Fatal(err)
	}
	disabledSession, _ := disabled.NewSession(context.Background())
	if _, err := disabledSession.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "one")); err != nil {
		t.Fatal(err)
	}
	if disabledModel.Calls()[0].PromptCache != nil {
		t.Fatalf("disabled request cache = %#v", disabledModel.Calls()[0].PromptCache)
	}
	_ = disabledSession.Close(context.Background())
}

func TestHarnessRejectsInvalidPromptCacheRetention(t *testing.T) {
	config := runtimeConfig(t)
	config.PromptCacheRetention = "forever"
	if _, err := NewRuntime[string](&facadeDriver{}, config); err == nil {
		t.Fatal("invalid prompt-cache retention succeeded")
	}
}

func TestHarnessCanSelectStreamingModelExecution(t *testing.T) {
	model := &streamingModel{}
	config := runtimeConfig(t)
	config.ModelStreaming = true
	runtime, err := NewRuntime[string](agentic.NewAgent("system", model), config)
	if err != nil {
		t.Fatal(err)
	}
	session, err := runtime.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(context.Background()) }()
	execution, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "stream"))
	if err != nil {
		t.Fatal(err)
	}
	if execution.Result == nil || execution.Result.Output != "streamed" || model.streamCalls.Load() != 1 || model.plainCalls.Load() != 0 {
		t.Fatalf("execution=%#v stream=%d plain=%d", execution, model.streamCalls.Load(), model.plainCalls.Load())
	}
}

func TestHarnessConcurrentSessionsAndFaultedReopen(t *testing.T) {
	driver := &facadeDriver{}
	harness, err := NewRuntime[string](driver, runtimeConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	const workers = 24
	ids := make(chan string, workers)
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			session, createErr := harness.NewSession(context.Background())
			if createErr != nil {
				t.Errorf("NewSession: %v", createErr)
				return
			}
			ids <- session.ID()
		}()
	}
	wg.Wait()
	close(ids)
	seen := make(map[string]bool)
	for id := range ids {
		if seen[id] {
			t.Fatalf("duplicate session ID %q", id)
		}
		seen[id] = true
	}

	driver.mu.Lock()
	driver.badFirst = true
	driver.mu.Unlock()
	faulted, err := harness.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := faulted.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "fault")); !errors.Is(err, ErrSessionFaulted) {
		t.Fatalf("fault error = %v", err)
	}
	reopened, err := harness.ResumeSession(context.Background(), faulted.ID())
	if err != nil {
		t.Fatal(err)
	}
	if err := reopened.WaitForIdle(context.Background()); err != nil {
		t.Fatal(err)
	}
	if reopened.State() != SessionIdle {
		t.Fatalf("reopened state = %s", reopened.State())
	}
	if _, err := harness.ResumeSession(context.Background(), reopened.ID()); !errors.Is(err, ErrSessionOpen) {
		t.Fatalf("healthy duplicate owner error = %v", err)
	}
}

func TestRepositoryLeaseConflictUsesPublicSessionError(t *testing.T) {
	config := runtimeConfig(t)
	first, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	owned, err := first.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = owned.Close(context.Background()) }()

	second, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := second.ResumeSession(context.Background(), owned.ID()); !errors.Is(err, ErrSessionOpen) {
		t.Fatalf("repository lease error = %v", err)
	}
}

type prefixedCodec struct {
	base codec.Codec
}

func (c prefixedCodec) Encode(value any) ([]byte, error) {
	payload, err := c.base.Encode(value)
	if err != nil {
		return nil, err
	}
	return append([]byte{0xff}, payload...), nil
}

func (c prefixedCodec) Decode(payload []byte, target any) error {
	if len(payload) == 0 || payload[0] != 0xff {
		return errors.New("missing test codec prefix")
	}
	return c.base.Decode(payload[1:], target)
}

type trackingEnvironmentFactory struct {
	mu     sync.Mutex
	opens  int
	closes int
}

func (f *trackingEnvironmentFactory) Open(ctx context.Context, _ string) (env.Lease, error) {
	lease, err := envmemory.New("/", nil)
	if err != nil {
		return nil, err
	}
	f.mu.Lock()
	f.opens++
	f.mu.Unlock()
	return &trackingLease{Lease: lease, owner: f}, nil
}

func (f *trackingEnvironmentFactory) counts() (int, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.opens, f.closes
}

type trackingLease struct {
	env.Lease
	owner *trackingEnvironmentFactory
	once  sync.Once
}

func (l *trackingLease) Close(ctx context.Context) error {
	err := l.Lease.Close(ctx)
	if err == nil {
		l.once.Do(func() {
			l.owner.mu.Lock()
			l.owner.closes++
			l.owner.mu.Unlock()
		})
	}
	return err
}

func TestRuntimeUsesOpaqueCodecAndSessionScopedEnvironmentLeases(t *testing.T) {
	driver := &facadeDriver{}
	config := runtimeConfig(t)
	config.Codec = prefixedCodec{base: jsoncodec.New()}
	environments := &trackingEnvironmentFactory{}
	config.Environments = environments
	runtime, err := NewRuntime[string](driver, config)
	if err != nil {
		t.Fatal(err)
	}
	first, err := runtime.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := first.NextTurn(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "durable"))
	if err != nil || receipt.Cursor == 0 {
		t.Fatalf("NextTurn = %#v, %v", receipt, err)
	}
	id := first.ID()
	if err := first.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if first.State() != SessionClosed {
		t.Fatalf("closed state = %s", first.State())
	}

	journal, err := config.Sessions.Open(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Entries) < 2 || json.Valid(loaded.Entries[1].Payload) {
		t.Fatalf("journal payload unexpectedly exposes JSON: %q", loaded.Entries[1].Payload)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}

	reopened, err := runtime.ResumeSession(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := reopened.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Pending) != 1 || snapshot.Pending[0].ID != receipt.ID {
		t.Fatalf("recovered pending = %#v", snapshot.Pending)
	}
	if err := reopened.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	opens, closes := environments.counts()
	if opens != 2 || closes != 2 {
		t.Fatalf("environment leases opened=%d closed=%d", opens, closes)
	}
}

func TestCloseRetriesIncompleteCleanupBeforeResume(t *testing.T) {
	config := runtimeConfig(t)
	environments := &trackingEnvironmentFactory{}
	config.Environments = environments
	runtime, err := NewRuntime[string](&facadeDriver{}, config)
	if err != nil {
		t.Fatal(err)
	}
	first, err := runtime.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	id := first.ID()
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := first.Close(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("Close error = %v", err)
	}
	if opens, closes := environments.counts(); opens != 1 || closes != 0 {
		t.Fatalf("cleanup after canceled Close opened=%d closed=%d", opens, closes)
	}

	reopened, err := runtime.ResumeSession(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if opens, closes := environments.counts(); opens != 2 || closes != 1 {
		t.Fatalf("cleanup before resume opened=%d closed=%d", opens, closes)
	}
	if err := reopened.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if opens, closes := environments.counts(); opens != 2 || closes != 2 {
		t.Fatalf("final cleanup opened=%d closed=%d", opens, closes)
	}
}
