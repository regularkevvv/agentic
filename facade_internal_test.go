package agentic

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/testutil"
	testprovider "github.com/regularkevvv/agentic/provider/test"
)

type facadeDeps struct{ Value string }
type facadeInput struct {
	_     struct{} `tool:"facade tool"`
	Value string   `json:"value"`
}
type facadeOutput struct{ Value string }
type facadeAnswer struct {
	Value string `json:"value"`
}

func facadeToolDefinition(t *testing.T, name string) (Tool, ToolHandler) {
	t.Helper()
	return MustToolPlain(name, "facade tool", func(input facadeInput) (facadeOutput, error) {
		return facadeOutput{Value: input.Value}, nil
	})
}

func TestFacadeConfigurationMethods(t *testing.T) {
	toolA, handlerA := facadeToolDefinition(t, "tool_a")
	toolB, handlerB := facadeToolDefinition(t, "tool_b")
	toolC, handlerC := facadeToolDefinition(t, "tool_c")
	set := NewToolset().Add(toolB, handlerB)

	plain := NewAgent("system", testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"}))
	if plain.AddAutoTool(toolA, handlerA) != plain {
		t.Fatal("plain AddAutoTool must be fluent")
	}

	depsAgent := NewAgentWithDeps[*facadeDeps]("system", testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"}))
	if depsAgent.AddAutoTool(toolA, handlerA) != depsAgent || depsAgent.AddToolset(set) != depsAgent {
		t.Fatal("dependency facade configuration must be fluent")
	}
	registry := NewRegistry()
	if depsAgent.SetRegistry(registry) != depsAgent || depsAgent.SetOutputToolNames(map[string]bool{"done": true}) != depsAgent {
		t.Fatal("dependency registry/output configuration must be fluent")
	}
	depsAgent.AddTool(toolC, handlerC)
	depsAgent.dependencyType((*facadeDeps)(nil))

	stream, err := depsAgent.RunStream(context.Background(), "go", &facadeDeps{})
	if err != nil || stream.Wait() != nil {
		t.Fatalf("dependency stream fallback: stream=%#v err=%v", stream, err)
	}
}

func TestTypedFacadeConfigurationAndBinding(t *testing.T) {
	model := testprovider.NewTestModel(testprovider.ModelResponse{Text: `{"value":"ok"}`})
	output := NewNativeOutput[facadeAnswer]("answer", "answer")
	plain := NewTypedAgentDynamic[facadeAnswer](func(context.Context) (string, error) { return "dynamic", nil }, model, "answer")
	plain.runtime.output = output
	configureAgentOutput(plain.runtime.core, output)
	plain.AddOutputValidator(TypedOutputValidatorFunc[facadeAnswer](func(context.Context, facadeAnswer) error { return nil }))
	stream, err := plain.RunStream(context.Background(), "go")
	if err != nil || stream.Wait() != nil {
		t.Fatalf("typed stream: stream=%#v err=%v", stream, err)
	}

	depsModel := testprovider.NewTestModel(testprovider.ModelResponse{Text: `{"value":"ok"}`})
	typed := NewTypedAgentWithDepsMode[facadeAnswer, *facadeDeps]("system", depsModel, output)
	if identity := NewIdentityHandoff("identity", "identity", typed); identity == nil {
		t.Fatal("identity handoff is nil")
	}
	var validatorCalls int
	typed.SetDepsValidator(func(context.Context, *facadeDeps) error { return nil }).
		AddTextOutputValidator(OutputValidatorWithDepsFunc[*facadeDeps](func(RunContext[*facadeDeps], string) error { return nil })).
		SetToolPrepare(func(_ RunContext[*facadeDeps], tools []Tool) ([]Tool, error) { return tools, nil }).
		AddOutputValidator(TypedOutputValidatorFunc[facadeAnswer](func(context.Context, facadeAnswer) error {
			validatorCalls++
			return nil
		})).
		AddOutputValidatorWithDeps(TypedOutputValidatorWithDepsFunc[*facadeDeps, facadeAnswer](func(RunContext[*facadeDeps], facadeAnswer) error {
			validatorCalls++
			return nil
		}))

	toolA, handlerA := facadeToolDefinition(t, "typed_a")
	toolB, handlerB := facadeToolDefinition(t, "typed_b")
	toolC, handlerC := facadeToolDefinition(t, "typed_c")
	if typed.AddTool(toolA, handlerA) != typed || typed.AddAutoTool(toolB, handlerB) != typed || typed.AddToolset(NewToolset().Add(toolC, handlerC)) != typed {
		t.Fatal("typed dependency tool methods must be fluent")
	}
	newRegistry := NewRegistry()
	if typed.SetRegistry(newRegistry) != typed {
		t.Fatal("typed dependency SetRegistry must be fluent")
	}
	typed.dependencyType((*facadeDeps)(nil))

	result, err := typed.Run(context.Background(), "go", &facadeDeps{})
	if err != nil || result.Output.Value != "ok" || validatorCalls != 2 {
		t.Fatalf("typed deps run: result=%#v err=%v validators=%d", result, err, validatorCalls)
	}

	bound := typed.Bind(&facadeDeps{})
	if _, err := bound.Run(context.Background(), "go"); err != nil {
		t.Fatalf("typed Bind: %v", err)
	}
	provided := typed.BindProvider(func(context.Context) (*facadeDeps, error) { return &facadeDeps{}, nil })
	if _, err := provided.Run(context.Background(), "go"); err != nil {
		t.Fatalf("typed BindProvider: %v", err)
	}
	stream, err = typed.RunStream(context.Background(), "go", &facadeDeps{})
	if err != nil || stream.Wait() != nil {
		t.Fatalf("typed deps stream: stream=%#v err=%v", stream, err)
	}
	if streamRunner, ok := bound.(StreamRunner); !ok {
		t.Fatal("bound typed runner should expose optional streaming")
	} else if stream, err := streamRunner.RunStream(context.Background(), "go"); err != nil || stream.Wait() != nil {
		t.Fatalf("bound typed stream: stream=%#v err=%v", stream, err)
	}
}

func TestTypedFacadeRegistrationMethods(t *testing.T) {
	toolA, handlerA := facadeToolDefinition(t, "plain_typed_a")
	toolB, handlerB := facadeToolDefinition(t, "plain_typed_b")
	typed := NewTypedAgentWithMode[facadeAnswer]("system", testprovider.NewTestModel(testprovider.ModelResponse{Text: `{"value":"ok"}`}), NewNativeOutput[facadeAnswer]("answer", "answer"))
	if typed.AddTool(toolA, handlerA) != typed || typed.AddAutoTool(toolB, handlerB) != typed {
		t.Fatal("typed tool methods must be fluent")
	}
	AddTool(typed, func(context.Context, facadeInput) (facadeOutput, error) { return facadeOutput{}, nil }, AutoToolName("context_typed"), AutoToolDescription("context tool"))
}

func TestAdapterAndPreflightDefensiveErrors(t *testing.T) {
	wrong := core.NewDependencyEnvelope("wrong")
	if _, err := adaptDynamicPrompt(func(RunContext[*facadeDeps]) (string, error) { return "", nil })(context.Background(), wrong); err == nil {
		t.Fatal("expected dynamic prompt extraction error")
	}
	if err := adaptDepsValidator(func(context.Context, *facadeDeps) error { return nil })(context.Background(), wrong); err == nil {
		t.Fatal("expected dependency validator extraction error")
	}
	if err := adaptTextValidator(OutputValidatorWithDepsFunc[*facadeDeps](func(RunContext[*facadeDeps], string) error { return nil }))(context.Background(), wrong, "output"); err == nil {
		t.Fatal("expected text validator extraction error")
	}
	if _, err := adaptToolPrepare(func(RunContext[*facadeDeps], []Tool) ([]Tool, error) { return nil, nil })(context.Background(), wrong, nil); err == nil {
		t.Fatal("expected tool prepare extraction error")
	}
	if err := adaptTypedValidator(TypedOutputValidatorWithDepsFunc[*facadeDeps, facadeAnswer](func(RunContext[*facadeDeps], facadeAnswer) error { return nil }))(context.Background(), wrong, facadeAnswer{}); err == nil {
		t.Fatal("expected typed validator extraction error")
	}

	noDeps := NewAgent("system", testprovider.NewTestModel()).core
	if err := noDeps.preflight(context.Background(), wrong); err == nil {
		t.Fatal("no-deps core must reject a dependency envelope")
	}
	withDeps := NewAgentWithDeps[*facadeDeps]("system", testprovider.NewTestModel()).core
	if err := withDeps.preflight(context.Background(), wrong); err == nil {
		t.Fatal("dependency core must reject the wrong static type")
	}
}

func TestRootToolConvenienceWrappers(t *testing.T) {
	contextTool, contextHandler, err := ToolWithContext("context_tool", "context tool", func(context.Context, facadeInput) (facadeOutput, error) {
		return facadeOutput{}, nil
	})
	if err != nil || contextHandler == nil || contextTool.Function.Name != "context_tool" {
		t.Fatalf("ToolWithContext: tool=%#v handler=%#v err=%v", contextTool, contextHandler, err)
	}
	if _, _, err := AutoToolWithContext(func(context.Context, facadeInput) (facadeOutput, error) { return facadeOutput{}, nil }); err != nil {
		t.Fatalf("AutoToolWithContext: %v", err)
	}
	if _, handler := MustAutoToolWithContext(func(context.Context, facadeInput) (facadeOutput, error) { return facadeOutput{}, nil }); handler == nil {
		t.Fatal("MustAutoToolWithContext returned nil handler")
	}

	agent := NewAgent("system", testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"}))
	AddTool(agent, func(context.Context, facadeInput) (facadeOutput, error) { return facadeOutput{}, nil }, AutoToolName("add_context"), AutoToolDescription("context"))
	AddToolWithContext(agent, func(context.Context, facadeInput) (facadeOutput, error) { return facadeOutput{}, nil }, AutoToolName("add_context_alias"), AutoToolDescription("context"))

	channelHandler := func(context.Context, facadeInput) (<-chan facadeOutput, error) {
		ch := make(chan facadeOutput, 1)
		ch <- facadeOutput{}
		close(ch)
		return ch, nil
	}
	if _, _, err := ChannelTool("channel", "channel", channelHandler, WithChannelTimeout(time.Second)); err != nil {
		t.Fatalf("ChannelTool: %v", err)
	}
	if _, handler := MustChannelTool("must_channel", "channel", channelHandler); handler == nil {
		t.Fatal("MustChannelTool returned nil")
	}
	approval := func(context.Context, ToolUse) (bool, error) { return true, nil }
	if _, _, err := ApprovalTool("approval", "approval", func(context.Context, facadeInput) (facadeOutput, error) { return facadeOutput{}, nil }, approval); err != nil {
		t.Fatalf("ApprovalTool: %v", err)
	}
}

func TestHandoffFacadeMethodsAndDefensivePaths(t *testing.T) {
	ready := NewHandoff("ready", "ready", NewAgent("child", testprovider.NewTestModel()))
	if NewAgentWithDeps[*facadeDeps]("parent", testprovider.NewTestModel()).AddHandoff(ready) == nil {
		t.Fatal("dependency AddHandoff returned nil")
	}
	if NewTypedAgent[facadeAnswer]("parent", testprovider.NewTestModel(), "answer").AddHandoff(ready) == nil {
		t.Fatal("typed AddHandoff returned nil")
	}
	typedDeps := NewTypedAgentWithDeps[facadeAnswer, *facadeDeps]("parent", testprovider.NewTestModel(), "answer")
	if typedDeps.AddHandoff(ready) == nil {
		t.Fatal("typed deps AddHandoff returned nil")
	}

	child := NewAgentWithDeps[*facadeDeps]("child", testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"}))
	mapped := NewIdentityTextHandoff("mapped", "mapped", child)
	if typedDeps.AddHandoffWithDeps(mapped) == nil {
		t.Fatal("typed deps AddHandoffWithDeps returned nil")
	}
	handler := &handoffWithDepsHandler[*facadeDeps]{handoff: mapped}
	if _, err := handler.Execute(context.Background(), map[string]interface{}{"task": "go"}, core.NewDependencyEnvelope(&facadeDeps{})); err != nil {
		t.Fatalf("identity mapper execution: %v", err)
	}

	failedChild := NewAgentWithDeps[*facadeDeps]("child", &testutil.StubModel{Err: errors.New("boom")})
	failed := NewIdentityTextHandoff("failed", "failed", failedChild)
	if _, err := (&handoffWithDepsHandler[*facadeDeps]{handoff: failed}).Execute(context.Background(), map[string]interface{}{"task": "go"}, core.NewDependencyEnvelope(&facadeDeps{})); err == nil || !strings.Contains(err.Error(), "handoff to") {
		t.Fatalf("expected child error, got %v", err)
	}

	toolMessage := NewToolUseMessage(ToolUse{ID: "one", Name: "lookup"})
	resultMessage := NewToolResultMessage("one", "result", false)
	summary, err := summarizeHandoffHistory(context.Background(), []Message{toolMessage, resultMessage}, &handoffConfig{})
	if err != nil || !strings.Contains(summary, "assistant called lookup") || !strings.Contains(summary, "tool result") {
		t.Fatalf("unexpected default summary %q err=%v", summary, err)
	}

	defer func() {
		if recover() == nil {
			t.Fatal("expected invalid handoff tool panic")
		}
	}()
	_ = handoffTool("", "invalid")
}
