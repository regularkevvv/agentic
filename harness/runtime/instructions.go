package runtime

import "context"

// ExchangeContext identifies one application-level exchange before its first
// durable run entry is written. A suspended/resumed exchange retains the same
// resolved instruction value.
type ExchangeContext struct {
	SessionID string
	RunID     string
	Scope     Scope
}

// ExchangeInstructionProvider resolves trusted, non-user instructions once at
// the start of each application-level exchange.
type ExchangeInstructionProvider interface {
	ResolveExchangeInstructions(context.Context, ExchangeContext) (string, error)
}

// ExchangeInstructionProviderFunc adapts a function to the provider port.
type ExchangeInstructionProviderFunc func(context.Context, ExchangeContext) (string, error)

func (f ExchangeInstructionProviderFunc) ResolveExchangeInstructions(ctx context.Context, exchange ExchangeContext) (string, error) {
	return f(ctx, exchange)
}
