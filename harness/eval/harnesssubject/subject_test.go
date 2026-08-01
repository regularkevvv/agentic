package harnesssubject

import (
	"context"
	"errors"
	"sync"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	evalcore "github.com/regularkevvv/agentic/harness/eval"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type model struct {
	mu       sync.Mutex
	requests int
}

func (m *model) Name() string { return "test:eval-subject" }

func (m *model) Request(_ context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	m.mu.Lock()
	m.requests++
	m.mu.Unlock()
	last := request.Messages[len(request.Messages)-1].GetTextContent()
	return &agentic.ChatResponse{
		Model: m.Name(), Message: agentic.NewTextMessage(agentic.RoleAssistant, "answer:"+last),
		Usage:        agentic.Usage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3, Requests: 1},
		FinishReason: agentic.FinishReasonStop, RawFinishReason: string(agentic.FinishReasonStop),
	}, nil
}

func testHarness(t *testing.T) *harness.Harness[string] {
	return harnessForModel(t, &model{})
}

func harnessForModel(t *testing.T, model agentic.Model) *harness.Harness[string] {
	t.Helper()
	environments, err := envmemory.NewFactory(envmemory.Config{Cwd: "/workspace"})
	if err != nil {
		t.Fatal(err)
	}
	processors, _ := spill.NewFactory(artifactmemory.New(), spill.Config{})
	runtime, err := harness.NewRuntime(agentic.NewAgent("", model), harness.RuntimeConfig{
		Sessions: storememory.New(), Codec: jsoncodec.New(), Events: inproc.NewFactory(), Environments: environments,
		ResultProcessors: processors, Clock: system.NewClock(), IDs: system.NewIDs(),
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime
}

type failingModel struct{}

func (failingModel) Name() string { return "test:failing" }
func (failingModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	return nil, errors.New("model failed")
}

func TestSubjectUsesFreshSessionAndCapturesDurableOutcome(t *testing.T) {
	maximum := 10
	subject, err := New(Config[string, string]{
		Harness: testHarness(t),
		Prompt: func(value string) (agentic.Message, error) {
			return agentic.NewTextMessage(agentic.RoleUser, value), nil
		},
		Budget: &agentic.UsageLimits{MaxTotalTokens: &maximum},
	})
	if err != nil {
		t.Fatal(err)
	}
	first := subject.Run(context.Background(), evalcore.Case[string]{ID: "one", Input: "hello", Samples: 1})
	second := subject.Run(context.Background(), evalcore.Case[string]{ID: "two", Input: "world", Samples: 1})
	for index, outcome := range []evalcore.Outcome[Output[string]]{first, second} {
		if outcome.Error != nil || outcome.Output.Value == "" || outcome.Output.Cursor == 0 ||
			outcome.Output.Usage.TotalTokens != 3 || len(outcome.Output.Messages) != 2 ||
			len(outcome.Output.Events) == 0 || outcome.Output.Status != agentic.ExecutionCompleted {
			t.Fatalf("outcome %d = %#v", index, outcome)
		}
	}
	if first.Output.Messages[0].GetTextContent() == second.Output.Messages[0].GetTextContent() {
		t.Fatal("samples did not receive isolated inputs")
	}
}

func TestSubjectValidationAndErrors(t *testing.T) {
	if _, err := New[string, string](Config[string, string]{}); err == nil {
		t.Fatal("empty config succeeded")
	}
	if _, err := New(Config[string, string]{Harness: testHarness(t), Prompt: func(string) (agentic.Message, error) { return agentic.Message{}, nil }, EventBuffer: -1}); err == nil {
		t.Fatal("negative event buffer succeeded")
	}
	promptErr := errors.New("prompt failed")
	subject, _ := New(Config[string, string]{Harness: testHarness(t), Prompt: func(string) (agentic.Message, error) {
		return agentic.Message{}, promptErr
	}})
	outcome := subject.Run(context.Background(), evalcore.Case[string]{ID: "error", Samples: 1})
	if !errors.Is(outcome.Error, promptErr) || outcome.Duration <= 0 {
		t.Fatalf("prompt outcome = %#v", outcome)
	}
	if firstError(nil, promptErr, errors.New("later")) != promptErr {
		t.Fatal("firstError did not preserve order")
	}
}

func TestSubjectSessionOptionsCreationAndRunErrors(t *testing.T) {
	optionsCalled := false
	subject, err := New(Config[string, string]{
		Harness: testHarness(t),
		Prompt: func(value string) (agentic.Message, error) {
			return agentic.NewTextMessage(agentic.RoleUser, value), nil
		},
		SessionOptions: func(current evalcore.Case[string]) []harness.SessionOption {
			optionsCalled = current.ID == "options"
			return []harness.SessionOption{harness.WithDrainAll(true)}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	outcome := subject.Run(context.Background(), evalcore.Case[string]{ID: "options", Input: "value", Samples: 1})
	if outcome.Error != nil || !optionsCalled {
		t.Fatalf("session options outcome = %#v called=%v", outcome, optionsCalled)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	outcome = subject.Run(canceled, evalcore.Case[string]{ID: "canceled", Input: "value", Samples: 1})
	if !errors.Is(outcome.Error, context.Canceled) || outcome.ErrorMessage == "" {
		t.Fatalf("creation failure = %#v", outcome)
	}

	failing, err := New(Config[string, string]{
		Harness: harnessForModel(t, failingModel{}),
		Prompt: func(value string) (agentic.Message, error) {
			return agentic.NewTextMessage(agentic.RoleUser, value), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	outcome = failing.Run(context.Background(), evalcore.Case[string]{ID: "failure", Input: "value", Samples: 1})
	if outcome.Error == nil || outcome.ErrorMessage == "" {
		t.Fatalf("run failure = %#v", outcome)
	}
}
