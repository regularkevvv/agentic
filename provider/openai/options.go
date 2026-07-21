package openai

import "strings"

// ProviderKey is the key under which OpenAI-specific settings are read from
// [core.ChatRequest.ProviderOptions].
const ProviderKey = "openai"

// Options carries OpenAI-specific request settings. Place it in
// ChatRequest.ProviderOptions under the "openai" key:
//
//	req.ProviderOptions = map[string]any{
//	    "openai": openai.Options{ServiceTier: "flex"},
//	}
//
// A value stored under the key that is not an Options (or *Options) is
// ignored, as is a ServiceTier the API does not define, so a request built for
// one provider can be retried against another without failing.
//
// Every field is honored by both [Model] (Chat Completions) and
// [ResponsesModel] (Responses).
type Options struct {
	// Seed requests deterministic sampling: repeated requests with the same
	// seed and parameters return the same result on a best-effort basis.
	// Nil omits the parameter.
	//
	// Chat Completions only — the Responses API has no seed parameter.
	Seed *int64

	// ParallelToolCalls allows the model to emit several tool calls in one
	// turn. Nil omits the parameter, leaving the API default (true).
	ParallelToolCalls *bool

	// ServiceTier selects the processing tier: "auto", "default", "flex",
	// "scale", or "priority". Empty, or any other value, omits the parameter.
	ServiceTier string
}

// optionsFrom extracts the OpenAI Options from a request's ProviderOptions.
// It returns the zero Options when the key is absent or holds a value of an
// unexpected type, so a foreign or malformed entry never fails the request.
func optionsFrom(providerOptions map[string]any) Options {
	raw, ok := providerOptions[ProviderKey]
	if !ok {
		return Options{}
	}
	switch v := raw.(type) {
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

// validServiceTiers are the processing tiers the API accepts. Anything else is
// dropped rather than sent, which would 400.
var validServiceTiers = map[string]bool{
	"auto":     true,
	"default":  true,
	"flex":     true,
	"scale":    true,
	"priority": true,
}

// serviceTier returns the configured tier, or "" when it is unset or not a
// value the API defines.
func (o Options) serviceTier() string {
	if validServiceTiers[o.ServiceTier] {
		return o.ServiceTier
	}
	return ""
}

// reasoningSupport records how a model family handles reasoning, which is what
// decides whether sampling parameters may be sent.
type reasoningSupport struct {
	// enabledByDefault reports that the model reasons when no reasoning
	// effort is requested.
	enabledByDefault bool
	// canBeDisabled reports that the model accepts an effort that turns
	// reasoning off, which re-permits sampling parameters.
	canBeDisabled bool
}

// reasoningPrefix pairs a model-name prefix with its reasoning support.
type reasoningPrefix struct {
	prefix  string
	support reasoningSupport
}

var (
	noReasoning       = reasoningSupport{}
	optInReasoning    = reasoningSupport{canBeDisabled: true}
	alwaysOnReasoning = reasoningSupport{enabledByDefault: true}
)

// reasoningSupportByPrefix lists reasoning support per model-name prefix. The
// first matching prefix wins, so a more specific prefix ("gpt-5-chat") must
// precede the broader one it would otherwise match ("gpt-5"), and every newer
// gpt-5.x family must precede the plain "gpt-5" catch-all. Models matching no
// prefix do not reason.
//
// Mirrors pydantic-ai's _REASONING_SUPPORT_BY_PREFIX
// (pydantic_ai/profiles/openai.py), whose cells were verified against the live
// API: a model reasons by default exactly when it rejects sampling parameters
// with no reasoning effort set.
var reasoningSupportByPrefix = []reasoningPrefix{
	{"gpt-5.6", reasoningSupport{enabledByDefault: true, canBeDisabled: true}},
	{"gpt-5.5-pro", alwaysOnReasoning},
	{"gpt-5.5", reasoningSupport{enabledByDefault: true, canBeDisabled: true}},
	{"gpt-5.4-pro", alwaysOnReasoning},
	{"gpt-5.4", optInReasoning},
	{"gpt-5.3-chat", alwaysOnReasoning},
	{"gpt-5.3", optInReasoning},
	{"gpt-5.2-pro", alwaysOnReasoning},
	{"gpt-5.2-chat", alwaysOnReasoning},
	{"gpt-5.2", optInReasoning},
	{"gpt-5.1-codex", alwaysOnReasoning},
	{"gpt-5.1-chat", alwaysOnReasoning},
	{"gpt-5.1", optInReasoning},
	{"gpt-5-chat", noReasoning},
	{"gpt-5", alwaysOnReasoning},
	{"o", alwaysOnReasoning},
}

// lookupReasoningSupport returns the reasoning support for a model name. A
// leading vendor namespace ("openai/gpt-5" on OpenRouter) is stripped first so
// gateway-qualified names resolve to the same family.
func lookupReasoningSupport(model string) reasoningSupport {
	name := model
	if idx := strings.LastIndex(name, "/"); idx >= 0 {
		name = name[idx+1:]
	}
	for _, entry := range reasoningSupportByPrefix {
		if strings.HasPrefix(name, entry.prefix) {
			return entry.support
		}
	}
	return noReasoning
}

// supportsSamplingParams reports whether temperature and top_p may be sent for
// this model given the request's thinking configuration.
//
// Reasoning models reject sampling parameters while reasoning is active and
// respond 400. A model that can turn reasoning off still accepts them when the
// caller has not asked for thinking.
func supportsSamplingParams(model string, thinkingEnabled bool) bool {
	support := lookupReasoningSupport(model)
	if !support.enabledByDefault && !support.canBeDisabled {
		return true // Not a reasoning model.
	}
	reasoningActive := support.enabledByDefault
	if thinkingEnabled {
		reasoningActive = true
	}
	if support.canBeDisabled && !reasoningActive {
		return true
	}
	return false
}
