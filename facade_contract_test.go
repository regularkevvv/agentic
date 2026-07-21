package agentic_test

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	testprovider "github.com/regularkevvv/agentic/provider/test"
)

type facadeAnswer struct {
	Value string `json:"value"`
}

var _ agentic.Runner[string] = (*agentic.Agent)(nil)
var _ agentic.StreamingRunner[string] = (*agentic.Agent)(nil)
var _ agentic.Runner[facadeAnswer] = (*agentic.TypedAgent[facadeAnswer])(nil)

func TestFourFacadeContracts(t *testing.T) {
	plain := agentic.NewAgent("system", testprovider.NewTestModel(testprovider.ModelResponse{Text: "plain"}))
	plainResult, err := plain.Run(context.Background(), "prompt")
	if err != nil || plainResult.Output != "plain" {
		t.Fatalf("plain: result=%#v err=%v", plainResult, err)
	}

	type deps struct{ Prefix string }
	withDeps := agentic.NewAgentWithDeps[*deps]("system", testprovider.NewTestModel(testprovider.ModelResponse{Text: "deps"}))
	depsResult, err := withDeps.Run(context.Background(), "prompt", &deps{Prefix: "ok"})
	if err != nil || depsResult.Output != "deps" {
		t.Fatalf("deps: result=%#v err=%v", depsResult, err)
	}

	typed := agentic.NewTypedAgent[facadeAnswer]("system", testprovider.NewTestModel(testprovider.ModelResponse{
		ToolCalls: []agentic.ToolUse{{ID: "out_1", Name: "__output__", Input: map[string]interface{}{"value": "typed"}}},
	}), "answer")
	typedResult, err := typed.Run(context.Background(), "prompt")
	if err != nil || typedResult.Output.Value != "typed" {
		t.Fatalf("typed: result=%#v err=%v", typedResult, err)
	}

	typedDeps := agentic.NewTypedAgentWithDeps[facadeAnswer, *deps]("system", testprovider.NewTestModel(testprovider.ModelResponse{
		ToolCalls: []agentic.ToolUse{{ID: "out_2", Name: "__output__", Input: map[string]interface{}{"value": "typed deps"}}},
	}), "answer")
	typedDepsResult, err := typedDeps.Run(context.Background(), "prompt", &deps{})
	if err != nil || typedDepsResult.Output.Value != "typed deps" {
		t.Fatalf("typed deps: result=%#v err=%v", typedDepsResult, err)
	}
}

func TestDependencyPreflightHasZeroExternalEffects(t *testing.T) {
	type deps struct{ Tenant string }

	t.Run("typed nil", func(t *testing.T) {
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "unused"})
		var promptCalls atomic.Int32
		var validatorCalls atomic.Int32
		agent := agentic.NewAgentWithDeps[*deps]("", model).
			SetDynamicPrompt(func(agentic.RunContext[*deps]) (string, error) {
				promptCalls.Add(1)
				return "prompt", nil
			}).
			SetDepsValidator(func(context.Context, *deps) error {
				validatorCalls.Add(1)
				return nil
			})
		var nilDeps *deps
		_, err := agent.Run(context.Background(), "go", nilDeps)
		if !errors.Is(err, agentic.ErrNilDeps) {
			t.Fatalf("expected ErrNilDeps, got %v", err)
		}
		if model.CallCount() != 0 || promptCalls.Load() != 0 || validatorCalls.Load() != 0 {
			t.Fatalf("preflight leaked effects: model=%d prompt=%d validator=%d", model.CallCount(), promptCalls.Load(), validatorCalls.Load())
		}
		stream, err := agent.RunStream(context.Background(), "go", nilDeps)
		if stream != nil || !errors.Is(err, agentic.ErrNilDeps) {
			t.Fatalf("stream preflight: stream=%#v err=%v", stream, err)
		}
		if model.CallCount() != 0 || promptCalls.Load() != 0 || validatorCalls.Load() != 0 {
			t.Fatalf("stream preflight leaked effects: model=%d prompt=%d validator=%d", model.CallCount(), promptCalls.Load(), validatorCalls.Load())
		}
	})

	t.Run("semantic validation", func(t *testing.T) {
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "unused"})
		var promptCalls atomic.Int32
		agent := agentic.NewAgentWithDeps[*deps]("", model).
			SetDynamicPrompt(func(agentic.RunContext[*deps]) (string, error) {
				promptCalls.Add(1)
				return "prompt", nil
			}).
			SetDepsValidator(func(_ context.Context, deps *deps) error {
				if deps.Tenant == "" {
					return errors.New("tenant is required")
				}
				return nil
			})
		_, err := agent.Run(context.Background(), "go", &deps{})
		if err == nil || !strings.Contains(err.Error(), "tenant is required") {
			t.Fatalf("expected dependency validation error, got %v", err)
		}
		if model.CallCount() != 0 || promptCalls.Load() != 0 {
			t.Fatalf("validation leaked effects: model=%d prompt=%d", model.CallCount(), promptCalls.Load())
		}
	})

	t.Run("provider error and typed nil", func(t *testing.T) {
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "unused"})
		agent := agentic.NewAgentWithDeps[*deps]("system", model)
		var calls atomic.Int32
		runner := agent.BindProvider(func(context.Context) (*deps, error) {
			calls.Add(1)
			return nil, errors.New("provider failed")
		})
		if calls.Load() != 0 {
			t.Fatal("provider ran during binding")
		}
		_, err := runner.Run(context.Background(), "go")
		if err == nil || !strings.Contains(err.Error(), "dependency provider: provider failed") || calls.Load() != 1 || model.CallCount() != 0 {
			t.Fatalf("provider error: err=%v calls=%d model=%d", err, calls.Load(), model.CallCount())
		}

		nilRunner := agent.BindProvider(func(context.Context) (*deps, error) { return nil, nil })
		_, err = nilRunner.Run(context.Background(), "go")
		if !errors.Is(err, agentic.ErrNilDeps) || model.CallCount() != 0 {
			t.Fatalf("typed nil provider: err=%v model=%d", err, model.CallCount())
		}
	})

	t.Run("explicit zero struct is valid", func(t *testing.T) {
		type valueDeps struct{ Count int }
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"})
		agent := agentic.NewAgentWithDeps[valueDeps]("system", model)
		if _, err := agent.Run(context.Background(), "go", valueDeps{}); err != nil {
			t.Fatalf("zero struct must be accepted: %v", err)
		}
	})
}

type concurrentDependencyModel struct {
	mu      sync.Mutex
	prompts map[string]int
}

func (m *concurrentDependencyModel) Name() string { return "concurrent-deps" }

func (m *concurrentDependencyModel) Request(_ context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	prompt := request.Messages[0].GetTextContent()
	m.mu.Lock()
	m.prompts[prompt]++
	m.mu.Unlock()
	return &agentic.ChatResponse{
		Message:      agentic.NewTextMessage(agentic.RoleAssistant, prompt),
		FinishReason: agentic.FinishReasonStop,
	}, nil
}

func TestBoundRunnersShareCoreWithoutSharingDeps(t *testing.T) {
	type deps struct{ User string }
	model := &concurrentDependencyModel{prompts: make(map[string]int)}
	agent := agentic.NewAgentWithDeps[*deps]("", model).SetDynamicPrompt(func(ctx agentic.RunContext[*deps]) (string, error) {
		return ctx.Deps.User, nil
	})
	aliceDeps := &deps{User: "alice"}
	bobDeps := &deps{User: "bob"}
	alice := agent.Bind(aliceDeps)
	bob := agent.Bind(bobDeps)

	const runs = 32
	type invocation struct {
		want string
		run  func() (*agentic.Result[string], error)
	}
	invocations := []invocation{
		{want: "alice", run: func() (*agentic.Result[string], error) { return alice.Run(context.Background(), "go") }},
		{want: "bob", run: func() (*agentic.Result[string], error) { return bob.Run(context.Background(), "go") }},
		{want: "alice", run: func() (*agentic.Result[string], error) { return agent.Run(context.Background(), "go", aliceDeps) }},
		{want: "bob", run: func() (*agentic.Result[string], error) { return agent.Run(context.Background(), "go", bobDeps) }},
	}
	var wg sync.WaitGroup
	errs := make(chan error, runs*len(invocations))
	for _, call := range invocations {
		for range runs {
			wg.Add(1)
			go func(invocation invocation) {
				defer wg.Done()
				result, err := invocation.run()
				if err == nil && result.Output != invocation.want {
					err = fmt.Errorf("got output %q, want %q", result.Output, invocation.want)
				}
				errs <- err
			}(call)
		}
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	if model.prompts["alice"] != runs*2 || model.prompts["bob"] != runs*2 {
		t.Fatalf("dependency values crossed runs: %#v", model.prompts)
	}
}

func TestTypedDependencyValidationRetryPreflightsOnce(t *testing.T) {
	type deps struct{ Marker string }
	type answer struct {
		Value string `json:"value"`
	}
	model := testprovider.NewTestModel(
		testprovider.ModelResponse{ToolCalls: []agentic.ToolUse{{ID: "out_1", Name: "__output__", Input: map[string]interface{}{"value": "first"}}}},
		testprovider.ModelResponse{ToolCalls: []agentic.ToolUse{{ID: "out_2", Name: "__output__", Input: map[string]interface{}{"value": "second"}}}},
	)
	var preflightCalls atomic.Int32
	var validatorCalls atomic.Int32
	agent := agentic.NewTypedAgentWithDeps[answer, *deps](
		"system", model, "answer", agentic.WithMaxValidationRetries(1),
	).SetDepsValidator(func(context.Context, *deps) error {
		preflightCalls.Add(1)
		return nil
	}).AddOutputValidatorWithDeps(
		agentic.TypedOutputValidatorWithDepsFunc[*deps, answer](func(ctx agentic.RunContext[*deps], output answer) error {
			if ctx.Deps.Marker != "expected" {
				return fmt.Errorf("wrong deps %q", ctx.Deps.Marker)
			}
			if validatorCalls.Add(1) == 1 {
				return agentic.NewValidationError("retry")
			}
			return nil
		}),
	)
	result, err := agent.Run(context.Background(), "go", &deps{Marker: "expected"})
	if err != nil || result.Output.Value != "second" {
		t.Fatalf("result=%#v err=%v", result, err)
	}
	if preflightCalls.Load() != 1 || validatorCalls.Load() != 2 {
		t.Fatalf("preflight=%d validator=%d", preflightCalls.Load(), validatorCalls.Load())
	}
}

func TestDependencyToolRetryContext(t *testing.T) {
	type deps struct{}
	type input struct{}
	type output struct{}
	model := testprovider.NewTestModel(
		testprovider.ModelResponse{ToolCalls: []agentic.ToolUse{{ID: "retry_1", Name: "input", Input: map[string]interface{}{}}}},
		testprovider.ModelResponse{ToolCalls: []agentic.ToolUse{{ID: "retry_2", Name: "input", Input: map[string]interface{}{}}}},
		testprovider.ModelResponse{Text: "done"},
	)
	var retries []int
	agent := agentic.NewAgentWithDeps[*deps]("system", model, agentic.WithRetries(agentic.RetryConfig{MaxRetries: 2}))
	agentic.AddToolWithDeps(agent, func(ctx agentic.RunContext[*deps], _ input) (output, error) {
		retries = append(retries, ctx.Retry)
		if ctx.Retry == 0 {
			return output{}, agentic.Retry("again")
		}
		return output{}, nil
	}, agentic.AutoToolDescription("retry tool"))
	if _, err := agent.Run(context.Background(), "go", &deps{}); err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(retries) != "[0 1]" {
		t.Fatalf("unexpected retry context %v", retries)
	}
}

func TestResultHistoryParity(t *testing.T) {
	history := agentic.NewTextMessage(agentic.RoleUser, "old")
	text := agentic.NewAgent("system", testprovider.NewTestModel(testprovider.ModelResponse{Text: "text"}))
	textResult, err := text.Run(context.Background(), "new", agentic.WithMessages(history))
	if err != nil {
		t.Fatal(err)
	}
	typed := agentic.NewTypedAgentWithMode[facadeAnswer]("system", testprovider.NewTestModel(testprovider.ModelResponse{Text: `{"value":"typed"}`}), agentic.NewNativeOutput[facadeAnswer]("answer", "answer"))
	typedResult, err := typed.Run(context.Background(), "new", agentic.WithMessages(history))
	if err != nil {
		t.Fatal(err)
	}
	if len(textResult.AllMessages()) != len(typedResult.AllMessages()) || len(textResult.NewMessages()) != len(typedResult.NewMessages()) {
		t.Fatalf("history behavior differs: text=%d/%d typed=%d/%d", len(textResult.AllMessages()), len(textResult.NewMessages()), len(typedResult.AllMessages()), len(typedResult.NewMessages()))
	}
	if typedResult.History() == nil || textResult.History() == nil {
		t.Fatal("expected history helpers on both result types")
	}
}

func TestNegativeCompileContracts(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping compiler fixtures in short mode")
	}
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Dir(currentFile)
	fixtures, err := filepath.Glob(filepath.Join(root, "testdata", "compilefail", "*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(fixtures) < 5 {
		t.Fatalf("expected compile-failure fixtures, found %d", len(fixtures))
	}
	for _, fixture := range fixtures {
		fixture := fixture
		t.Run(filepath.Base(fixture), func(t *testing.T) {
			command := exec.Command("go", "test", ".")
			command.Dir = fixture
			output, err := command.CombinedOutput()
			if err == nil {
				t.Fatalf("fixture unexpectedly compiled:\n%s", output)
			}
			if !strings.Contains(string(output), "compile") && !strings.Contains(string(output), "cannot") && !strings.Contains(string(output), "not enough arguments") {
				t.Fatalf("unexpected compiler output:\n%s", output)
			}
		})
	}
}
