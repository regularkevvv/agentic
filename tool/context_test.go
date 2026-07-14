package tool

import (
	"context"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

type contextInput struct {
	_     struct{} `tool:"context-aware tool"`
	Value string   `json:"value"`
}

type contextOutput struct{ Value string }
type contextValueKey struct{}

func TestContextToolHandler(t *testing.T) {
	key := contextValueKey{}
	tool, handler, err := ToolWithContext("context_tool", "context tool", func(ctx context.Context, input contextInput) (contextOutput, error) {
		return contextOutput{Value: ctx.Value(key).(string) + input.Value}, nil
	}, WithToolMaxRetries(4))
	if err != nil || tool.Function.Name != "context_tool" {
		t.Fatalf("ToolWithContext: tool=%#v err=%v", tool, err)
	}
	ctx := context.WithValue(context.Background(), key, "prefix-")
	result, err := handler.Execute(ctx, map[string]interface{}{"value": "ok"}, core.NewDependencyEnvelope(struct{}{}))
	if err != nil || result.(contextOutput).Value != "prefix-ok" {
		t.Fatalf("Execute: result=%#v err=%v", result, err)
	}
	configurable := handler.(interface{ ToolConfig() *core.ToolConfig })
	if configurable.ToolConfig() == nil || *configurable.ToolConfig().MaxRetries != 4 {
		t.Fatalf("tool config missing: %#v", configurable.ToolConfig())
	}
	if handler.Name() != "context_tool" {
		t.Fatalf("unexpected name %q", handler.Name())
	}

	if _, err := handler.Execute(ctx, map[string]interface{}{"value": 12}, nil); err == nil || !strings.Contains(err.Error(), "unmarshal") {
		t.Fatalf("expected unmarshal error, got %v", err)
	}
	if _, err := handler.Execute(ctx, map[string]interface{}{"value": func() {}}, nil); err == nil || !strings.Contains(err.Error(), "marshal") {
		t.Fatalf("expected marshal error, got %v", err)
	}
}

func TestContextToolConstructorsAndAutoRegistration(t *testing.T) {
	if _, _, err := ToolWithContext("", "description", func(context.Context, contextInput) (contextOutput, error) { return contextOutput{}, nil }); err == nil {
		t.Fatal("expected invalid tool definition")
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected MustToolWithContext panic")
			}
		}()
		MustToolWithContext("", "description", func(context.Context, contextInput) (contextOutput, error) { return contextOutput{}, nil })
	}()

	tool, handler, err := AutoWithContext(func(context.Context, contextInput) (contextOutput, error) { return contextOutput{}, nil })
	if err != nil || tool.Function.Name != "context" || handler == nil {
		t.Fatalf("AutoWithContext: tool=%#v handler=%#v err=%v", tool, handler, err)
	}
	if _, handler := MustAutoWithContext(func(context.Context, contextInput) (contextOutput, error) { return contextOutput{}, nil }, WithName("custom")); handler == nil {
		t.Fatal("MustAutoWithContext returned nil")
	}
}
