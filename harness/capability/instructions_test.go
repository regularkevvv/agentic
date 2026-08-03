package capability

import (
	"context"
	"errors"
	"strings"
	"testing"

	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

func TestCompileExchangeInstructionProvider(t *testing.T) {
	provider := harnessruntime.ExchangeInstructionProviderFunc(func(context.Context, harnessruntime.ExchangeContext) (string, error) {
		return "trusted", nil
	})
	value := Func{Name: "instructions", Apply: func(registry *Registry) error {
		return registry.AddExchangeInstructionProvider(provider)
	}}
	plan, err := Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ExchangeInstructionProvider() == nil {
		t.Fatal("compiled plan lost instruction provider")
	}
	duplicate := Func{Name: "duplicate", Apply: func(registry *Registry) error {
		return registry.AddExchangeInstructionProvider(provider)
	}}
	if _, err := Compile(value, duplicate); !errors.Is(err, ErrInstructionsConfigured) {
		t.Fatalf("duplicate provider error = %v", err)
	}
}

func TestCompileRejectsNilExchangeInstructionProvider(t *testing.T) {
	value := Func{Name: "nil-instructions", Apply: func(registry *Registry) error {
		return registry.AddExchangeInstructionProvider(nil)
	}}
	if _, err := Compile(value); err == nil || !strings.Contains(err.Error(), "must not be nil") {
		t.Fatalf("error = %v", err)
	}
}
