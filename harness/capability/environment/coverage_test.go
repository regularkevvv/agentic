package environment

import (
	"context"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
)

func TestEveryEnvironmentHandlerRequiresRuntime(t *testing.T) {
	value, err := New(Config{Files: true, Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	handlers := handlersByName(t, plan)
	inputs := map[string]map[string]any{
		ToolReadFile:      {"path": "file"},
		ToolWriteFile:     {"path": "file", "content": "content"},
		ToolListFiles:     {"path": "."},
		ToolStatFile:      {"path": "."},
		ToolMakeDirectory: {"path": "dir"},
		ToolRemovePath:    {"path": "file"},
		ToolRunCommand:    {"name": "command"},
	}
	for name, input := range inputs {
		if _, err := handlers[name].Execute(context.Background(), input, nil); err == nil {
			t.Fatalf("%s without runtime succeeded", name)
		}
	}
}

func TestEnvironmentRegistrationAndCanonicalizationFailures(t *testing.T) {
	files, err := New(Config{Files: true})
	if err != nil {
		t.Fatal(err)
	}
	duplicate := func(id, name string) capability.Capability {
		return capability.Func{
			Name: id,
			Apply: func(registry *capability.Registry) error {
				return registry.AddEffectResolver(name, capability.EffectResolverFunc(func(
					context.Context,
					agentic.ToolUse,
					env.Environment,
				) (capability.Effect, error) {
					return capability.Effect{}, nil
				}))
			},
		}
	}
	if _, err := capability.Compile(duplicate("duplicate-file", ToolReadFile), files); !errors.Is(err, capability.ErrDuplicateEffect) {
		t.Fatalf("duplicate file effect = %v", err)
	}
	shell, err := New(Config{Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := capability.Compile(duplicate("duplicate-shell", ToolRunCommand), shell); !errors.Is(err, capability.ErrDuplicateEffect) {
		t.Fatalf("duplicate shell effect = %v", err)
	}

	environment, err := envmemory.New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := environment.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := fileEffect("read").ResolveEffect(context.Background(), agentic.ToolUse{
		Input: map[string]any{"path": "file"},
	}, environment); !env.HasCode(err, env.CodeClosed) {
		t.Fatalf("closed canonicalization = %v", err)
	}
}

func TestEnvironmentRegisterPropagatesFrozenRegistry(t *testing.T) {
	value, err := New(Config{Files: true, Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	var captured *capability.Registry
	_, err = capability.Compile(capability.Func{
		Name: "capture",
		Apply: func(registry *capability.Registry) error {
			captured = registry
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := value.Register(captured); !errors.Is(err, capability.ErrRegistryFrozen) {
		t.Fatalf("frozen registration = %v", err)
	}
}
