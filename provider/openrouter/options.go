package openrouter

import "github.com/regularkevvv/agentic/internal/core"

// ProviderKey is the key this package reads from
// [core.ChatRequest.ProviderOptions].
const ProviderKey = "openrouter"

// ProviderName identifies this provider on the thinking blocks it produces, so
// a reasoning turn is not replayed to a provider that would reject it.
const ProviderName = ProviderKey

// Options carries OpenRouter-specific request settings.
//
// Attach it to a request under [ProviderKey]:
//
//	req.ProviderOptions = map[string]any{
//		openrouter.ProviderKey: openrouter.Options{
//			Provider: &openrouter.ProviderRouting{
//				Order: []string{"anthropic", "openai"},
//			},
//		},
//	}
//
// A value stored under the key that is not an Options (or *Options) is
// ignored, so a request built for one provider can be retried against another
// without failing.
type Options struct {
	// Provider sets routing preferences for this request. Nil omits them,
	// leaving the gateway's own routing.
	Provider *ProviderRouting
}

// ProviderRouting mirrors OpenRouter's top-level "provider" object, which
// selects which upstream providers may serve a request and in what order.
// See https://openrouter.ai/docs/features/provider-routing, and pydantic-ai
// models/openrouter.py:190 (OpenRouterProviderConfig) for the same field set.
//
// Zero-valued fields are omitted, so the gateway's own defaults apply.
type ProviderRouting struct {
	// Order lists provider slugs to try in order, e.g. ["anthropic", "openai"].
	Order []string `json:"order,omitempty"`
	// AllowFallbacks permits backup providers when the primary is unavailable.
	AllowFallbacks *bool `json:"allow_fallbacks,omitempty"`
	// RequireParameters restricts routing to providers that support every
	// parameter in the request.
	RequireParameters *bool `json:"require_parameters,omitempty"`
	// DataCollection is "allow" or "deny" and controls whether providers that
	// may store data are eligible.
	DataCollection string `json:"data_collection,omitempty"`
	// Only restricts routing to these provider slugs.
	Only []string `json:"only,omitempty"`
	// Ignore skips these provider slugs.
	Ignore []string `json:"ignore,omitempty"`
	// Quantizations filters by quantization level, e.g. ["fp8", "bf16"].
	Quantizations []string `json:"quantizations,omitempty"`
	// Sort orders eligible providers by "price", "throughput", or "latency".
	Sort string `json:"sort,omitempty"`
}

// optionsFrom extracts this provider's own Options from a request. It returns
// the zero Options when the request is nil, the key is absent, or the key holds
// a value of an unexpected type, so a foreign or malformed entry never fails
// the request.
func optionsFrom(req *core.ChatRequest) Options {
	if req == nil || req.ProviderOptions == nil {
		return Options{}
	}
	switch v := req.ProviderOptions[ProviderKey].(type) {
	case Options:
		return v
	case *Options:
		if v == nil {
			return Options{}
		}
		return *v
	default:
		return Options{}
	}
}
