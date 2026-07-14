package agentic

import (
	"context"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/testutil"
	testprovider "github.com/regularkevvv/agentic/provider/test"
)

type aliasInput struct {
	_    struct{} `tool:"Alias greeting tool"`
	Name string   `json:"name" description:"Name to greet"`
}

type aliasOutput struct {
	Greeting string `json:"greeting"`
}

type aliasDeps struct {
	Prefix string
}

type aliasTypedOutput struct {
	Result string `json:"result"`
}

func TestAddToolset(t *testing.T) {
	type In struct {
		X int `json:"x"`
	}
	type Out struct {
		Y int `json:"y"`
	}

	t1, h1 := MustToolPlain("tool_a", "tool a", func(in In) (Out, error) { return Out{Y: 1}, nil })
	t2, h2 := MustToolPlain("tool_b", "tool b", func(in In) (Out, error) { return Out{Y: 2}, nil })

	ts := NewToolset().Add(t1, h1).Add(t2, h2)
	model := &testutil.StubModel{
		NameValue: "test",
		Response: &ChatResponse{
			Choices: []Choice{{Message: NewTextMessage(RoleAssistant, "done"), FinishReason: FinishReasonStop}},
		},
	}

	agent := NewAgent("test", model).AddToolset(ts)

	if agent.core.registry == nil {
		t.Fatal("expected registry to be created")
	}
	if !agent.core.registry.Has("tool_a") {
		t.Error("expected registry to have tool_a")
	}
	if !agent.core.registry.Has("tool_b") {
		t.Error("expected registry to have tool_b")
	}
}

func TestAliasToolWrappers(t *testing.T) {
	if tool := MustNewToolFromStruct("alias_struct", "alias struct", aliasInput{}); tool.Function.Name != "alias_struct" {
		t.Fatalf("unexpected tool name %q", tool.Function.Name)
	}

	plainTool, plainHandler, err := ToolPlain("alias_plain", "alias plain", func(input aliasInput) (aliasOutput, error) {
		return aliasOutput{Greeting: "hello " + input.Name}, nil
	})
	if err != nil {
		t.Fatalf("ToolPlain: %v", err)
	}
	if plainTool.Function.Name != "alias_plain" {
		t.Fatalf("unexpected plain tool name %q", plainTool.Function.Name)
	}
	if out, err := plainHandler.Execute(context.Background(), map[string]interface{}{"name": "world"}, nil); err != nil || out.(aliasOutput).Greeting != "hello world" {
		t.Fatalf("unexpected plain handler result: out=%#v err=%v", out, err)
	}

	depsTool, depsHandler, err := ToolWithDeps[aliasInput, aliasOutput, *aliasDeps](
		"alias_with_deps",
		"alias deps",
		func(ctx RunContext[*aliasDeps], input aliasInput) (aliasOutput, error) {
			return aliasOutput{Greeting: ctx.Deps.Prefix + input.Name}, nil
		},
	)
	if err != nil {
		t.Fatalf("ToolWithDeps: %v", err)
	}
	if depsTool.Function.Name != "alias_with_deps" {
		t.Fatalf("unexpected deps tool name %q", depsTool.Function.Name)
	}
	if out, err := depsHandler.Execute(context.Background(), map[string]interface{}{"name": "world"}, core.NewDependencyEnvelope(&aliasDeps{Prefix: "hi "})); err != nil || out.(aliasOutput).Greeting != "hi world" {
		t.Fatalf("unexpected deps handler result: out=%#v err=%v", out, err)
	}
}

func TestAliasToolsetAndAutoWrappers(t *testing.T) {
	t1, h1, err := AutoTool(func(input aliasInput) (aliasOutput, error) {
		return aliasOutput{Greeting: "hello " + input.Name}, nil
	}, AutoToolName("alias_auto"), AutoToolDescription("alias auto tool"))
	if err != nil {
		t.Fatalf("AutoTool: %v", err)
	}
	if t1.Function.Name != "alias_auto" || t1.Function.Description != "alias auto tool" {
		t.Fatalf("unexpected auto tool metadata: %#v", t1.Function)
	}

	t2, h2, err := AutoToolWithDeps[aliasInput, aliasOutput, aliasDeps](
		func(ctx RunContext[aliasDeps], input aliasInput) (aliasOutput, error) {
			return aliasOutput{Greeting: ctx.Deps.Prefix + input.Name}, nil
		},
		AutoToolName("alias_auto_deps"),
	)
	if err != nil {
		t.Fatalf("AutoToolWithDeps: %v", err)
	}

	if _, handler := MustAutoToolWithDeps[aliasInput, aliasOutput, aliasDeps](
		func(ctx RunContext[aliasDeps], input aliasInput) (aliasOutput, error) {
			return aliasOutput{Greeting: ctx.Deps.Prefix + input.Name}, nil
		},
		AutoToolName("alias_auto_must"),
	); handler == nil {
		t.Fatal("expected MustAutoToolWithDeps to return a handler")
	}

	set1 := NewToolset().Add(t1, h1)
	set2 := NewToolset().Add(t2, h2)
	combined := CombineToolsets(set1, set2)
	filtered := FilterToolset(combined, func(name string) bool { return name == "alias_auto" })
	prefixed := PrefixToolset(filtered, "pref")

	registry := NewRegistry()
	if err := RegisterToolset(registry, prefixed); err != nil {
		t.Fatalf("RegisterToolset: %v", err)
	}
	if !registry.Has("pref__alias_auto") {
		t.Fatalf("expected prefixed tool to be registered")
	}
}

func TestAliasAgentAndDeferredWrappers(t *testing.T) {
	plainModel := testprovider.NewTestModel(
		testprovider.ModelResponse{
			ToolCalls: []ToolUse{{
				ID:    "call_1",
				Name:  "alias_input",
				Input: map[string]interface{}{"name": "world"},
			}},
		},
		testprovider.ModelResponse{Text: "done"},
	)
	plainAgent := AddToolPlain(NewAgent("system", plainModel), func(input aliasInput) (aliasOutput, error) {
		return aliasOutput{Greeting: "hello " + input.Name}, nil
	})
	if _, err := plainAgent.Run(context.Background(), "run tool"); err != nil {
		t.Fatalf("AddTool run: %v", err)
	}

	depsModel := testprovider.NewTestModel(
		testprovider.ModelResponse{
			ToolCalls: []ToolUse{{
				ID:    "call_1",
				Name:  "alias_input",
				Input: map[string]interface{}{"name": "world"},
			}},
		},
		testprovider.ModelResponse{Text: "done"},
	)
	depsAgent := AddToolWithDeps(NewAgentWithDeps[*aliasDeps]("system", depsModel), func(ctx RunContext[*aliasDeps], input aliasInput) (aliasOutput, error) {
		return aliasOutput{Greeting: ctx.Deps.Prefix + input.Name}, nil
	})
	if _, err := depsAgent.Run(context.Background(), "run tool", &aliasDeps{Prefix: "hi "}); err != nil {
		t.Fatalf("AddToolWithDeps run: %v", err)
	}

	typed := NewTypedAgent[aliasTypedOutput](
		"system",
		testprovider.NewTestModel(testprovider.ModelResponse{Text: `{"result":"ok"}`}),
		"Return typed output",
	)
	before := typed.runtime.core.registry.Count()
	AddToolPlain(typed, func(input aliasInput) (aliasOutput, error) {
		return aliasOutput{Greeting: "hello " + input.Name}, nil
	})
	if typed.runtime.core.registry.Count() <= before {
		t.Fatalf("expected typed agent registry count to increase")
	}

	approval := WithApproval(func(ctx context.Context, toolCall ToolUse) (bool, error) {
		return true, nil
	})
	timeout := WithChannelTimeout(time.Second)

	asyncTool, asyncHandler, err := ChannelTool("alias_channel", "alias channel", func(ctx context.Context, input aliasInput) (<-chan aliasOutput, error) {
		ch := make(chan aliasOutput, 1)
		ch <- aliasOutput{Greeting: "hello " + input.Name}
		close(ch)
		return ch, nil
	}, approval, timeout)
	if err != nil {
		t.Fatalf("ChannelTool: %v", err)
	}
	if asyncTool.Function.Name != "alias_channel" {
		t.Fatalf("unexpected channel tool name %q", asyncTool.Function.Name)
	}
	if out, err := asyncHandler.Execute(context.Background(), map[string]interface{}{"name": "world"}, nil); err != nil || out.(aliasOutput).Greeting != "hello world" {
		t.Fatalf("unexpected channel output: out=%#v err=%v", out, err)
	}

	if _, handler := MustChannelTool("alias_channel_must", "alias channel", func(ctx context.Context, input aliasInput) (<-chan aliasOutput, error) {
		ch := make(chan aliasOutput, 1)
		ch <- aliasOutput{Greeting: "must " + input.Name}
		close(ch)
		return ch, nil
	}); handler == nil {
		t.Fatal("expected MustChannelTool to return a handler")
	}

	if _, handler, err := ApprovalTool("alias_channel_approval", "alias channel approval", func(ctx context.Context, input aliasInput) (aliasOutput, error) {
		return aliasOutput{Greeting: "approved " + input.Name}, nil
	}, func(ctx context.Context, toolCall ToolUse) (bool, error) {
		return true, nil
	}); err != nil || handler == nil {
		t.Fatalf("ApprovalTool: handler=%#v err=%v", handler, err)
	}

	if _, handler := MustApprovalTool("alias_channel_approval_must", "alias channel approval", func(ctx context.Context, input aliasInput) (aliasOutput, error) {
		return aliasOutput{Greeting: "approved " + input.Name}, nil
	}, func(ctx context.Context, toolCall ToolUse) (bool, error) {
		return true, nil
	}); handler == nil {
		t.Fatal("expected MustApprovalTool to return a handler")
	}
}
