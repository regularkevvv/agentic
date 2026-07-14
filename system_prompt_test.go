package agentic_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	testprovider "github.com/regularkevvv/agentic/provider/test"
)

func TestStaticSystemPromptSegments(t *testing.T) {
	model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"})
	agent := agentic.NewAgent("ignored", model,
		agentic.WithSystemPrompts("You are helpful.", "", "Always return JSON."),
	)
	if _, err := agent.Run(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	got := model.Calls()[0].Messages[0].GetTextContent()
	if got != "You are helpful.\n\nAlways return JSON." {
		t.Fatalf("unexpected joined prompt %q", got)
	}
}

func TestDynamicSystemPrompts(t *testing.T) {
	t.Run("without dependencies", func(t *testing.T) {
		type contextKey struct{}
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"})
		key := contextKey{}
		ctx := context.WithValue(context.Background(), key, "value")
		agent := agentic.NewAgentDynamic(func(runCtx context.Context) (string, error) {
			return "dynamic=" + runCtx.Value(key).(string), nil
		}, model)
		if _, err := agent.Run(ctx, "hi"); err != nil {
			t.Fatal(err)
		}
		if got := model.Calls()[0].Messages[0].GetTextContent(); got != "dynamic=value" {
			t.Fatalf("unexpected prompt %q", got)
		}
	})

	t.Run("with exact dependencies", func(t *testing.T) {
		type deps struct{ User string }
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"})
		agent := agentic.NewAgentWithDepsDynamic[*deps](func(ctx agentic.RunContext[*deps]) (string, error) {
			return "user=" + ctx.Deps.User, nil
		}, model)
		if _, err := agent.Run(context.Background(), "hi", &deps{User: "Kevin"}); err != nil {
			t.Fatal(err)
		}
		if got := model.Calls()[0].Messages[0].GetTextContent(); got != "user=Kevin" {
			t.Fatalf("unexpected prompt %q", got)
		}
	})
}

func TestSystemPromptPrecedenceAndErrors(t *testing.T) {
	t.Run("static segments override dynamic constructor", func(t *testing.T) {
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"})
		agent := agentic.NewAgentDynamic(func(context.Context) (string, error) {
			return "dynamic", nil
		}, model, agentic.WithSystemPrompts("configured"))
		if _, err := agent.Run(context.Background(), "hi"); err != nil {
			t.Fatal(err)
		}
		if got := model.Calls()[0].Messages[0].GetTextContent(); got != "configured" {
			t.Fatalf("unexpected prompt %q", got)
		}
	})

	t.Run("dynamic error is wrapped before model call", func(t *testing.T) {
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "unused"})
		agent := agentic.NewAgentDynamic(func(context.Context) (string, error) {
			return "", errors.New("boom")
		}, model)
		_, err := agent.Run(context.Background(), "hi")
		if err == nil || !strings.Contains(err.Error(), "system prompt: boom") || model.CallCount() != 0 {
			t.Fatalf("err=%v calls=%d", err, model.CallCount())
		}
	})

	t.Run("empty prompt omits system message", func(t *testing.T) {
		model := testprovider.NewTestModel(testprovider.ModelResponse{Text: "ok"})
		agent := agentic.NewAgent("", model)
		if _, err := agent.Run(context.Background(), "hi"); err != nil {
			t.Fatal(err)
		}
		if messages := model.Calls()[0].Messages; len(messages) != 1 || messages[0].Role != agentic.RoleUser {
			t.Fatalf("unexpected messages %#v", messages)
		}
	})
}
