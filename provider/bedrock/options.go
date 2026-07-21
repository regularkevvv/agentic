package bedrock

import "github.com/regularkevvv/agentic/internal/core"

// ProviderKey is the key this package reads from
// [core.ChatRequest.ProviderOptions]. Every other key is ignored, so a single
// request may carry options for several providers at once.
const ProviderKey = "bedrock"

// providerName identifies reasoning blocks produced by this provider. A
// thinking block tagged with any other name is not replayed to Bedrock, which
// rejects reasoning signatures it did not issue.
const providerName = "bedrock"

// redactedThinkingID marks a thinking block whose text the model encrypted.
// It matches the identifier [core.ThinkingBlock.IsRedacted] tests for.
const redactedThinkingID = "redacted_thinking"

// Options carries Bedrock-specific request settings.
//
// Attach it to a request under [ProviderKey]:
//
//	req.ProviderOptions = map[string]any{
//		bedrock.ProviderKey: bedrock.Options{GuardrailIdentifier: "gr-123"},
//	}
//
// Both Options and *Options are accepted. A value of any other type under the
// "bedrock" key is ignored rather than failing the request.
type Options struct {
	// GuardrailIdentifier selects an Amazon Bedrock guardrail to evaluate the
	// request and response against. Empty leaves guardrails off.
	GuardrailIdentifier string

	// GuardrailVersion is the guardrail version to apply. Ignored unless
	// GuardrailIdentifier is also set.
	GuardrailVersion string

	// PerformanceLatency selects the model's latency profile: "standard" or
	// "optimized". Any other value, including the empty string, leaves the
	// parameter unset so the model's own default applies.
	PerformanceLatency string

	// AdditionalModelRequestFields carries model-specific inference parameters
	// that the Converse API's shared InferenceConfiguration does not cover,
	// such as "top_k". Keys are merged with the thinking configuration this
	// package derives from [core.ChatRequest.Thinking]; a caller-supplied
	// "thinking" key is overwritten when thinking is enabled on the request.
	AdditionalModelRequestFields map[string]any
}

// optionsFor extracts this provider's options from a request, returning the
// zero Options when the key is absent, nil, or holds an unexpected type.
func optionsFor(req *core.ChatRequest) Options {
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
