package anthropic

import "github.com/regularkevvv/agentic/internal/core"

// ProviderKey is the key this package reads from
// [core.ChatRequest.ProviderOptions]. Every other key is ignored, so a single
// request may carry options for several providers at once.
const ProviderKey = "anthropic"

// providerName identifies thinking blocks produced by this provider. A thinking
// block tagged with any other name is not replayed to Anthropic, which rejects
// signatures it did not issue.
const providerName = "anthropic"

// Options carries Anthropic-specific request settings.
//
// Attach it to a request under [ProviderKey]:
//
//	topK := 40
//	req.ProviderOptions = map[string]any{
//		anthropic.ProviderKey: anthropic.Options{TopK: &topK},
//	}
//
// Both Options and *Options are accepted. A value of any other type under the
// "anthropic" key is ignored rather than failing the request.
type Options struct {
	// TopK restricts sampling to the K most likely tokens. nil leaves the
	// parameter unset, so the model's own default applies. Ignored when
	// adaptive thinking is in use, which rejects sampling overrides.
	TopK *int

	// CacheToolDefs places a cache breakpoint on the final tool definition,
	// caching the whole tool block for the given time-to-live. Valid values
	// are "5m" and "1h"; any other value is ignored and no breakpoint is set.
	CacheToolDefs string
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
