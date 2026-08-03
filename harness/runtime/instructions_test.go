package runtime

import (
	"context"
	"testing"
)

func TestExchangeInstructionProviderFunc(t *testing.T) {
	want := ExchangeContext{SessionID: "session", RunID: "run", Scope: Scope{SessionID: "session"}}
	provider := ExchangeInstructionProviderFunc(func(ctx context.Context, got ExchangeContext) (string, error) {
		if ctx == nil || got != want {
			t.Fatalf("context=%v exchange=%+v", ctx, got)
		}
		return "trusted", nil
	})
	got, err := provider.ResolveExchangeInstructions(context.Background(), want)
	if err != nil || got != "trusted" {
		t.Fatalf("instructions=%q err=%v", got, err)
	}
}
