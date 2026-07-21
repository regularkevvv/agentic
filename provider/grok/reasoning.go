package grok

import (
	"encoding/json"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go/packages/respjson"
)

// reasoning_effort values the xAI API accepts. No single model accepts all of
// them; see reasoningEfforts.
const (
	effortNone    = "none"
	effortMinimal = "minimal"
	effortLow     = "low"
	effortMedium  = "medium"
	effortHigh    = "high"
)

// Reasoning-effort sets, one per Grok model family.
//
// Mirrors pydantic-ai's Grok profile (pydantic_ai/profiles/grok.py:
// _GROK_BASIC_REASONING_EFFORTS, _GROK_43_REASONING_EFFORTS,
// _GROK_45_REASONING_EFFORTS) and https://docs.x.ai/developers/models.
var (
	// basicEfforts is what the grok-3-mini family accepts. It has no "none":
	// the model always reasons and cannot be told not to.
	basicEfforts = map[string]bool{effortLow: true, effortHigh: true}

	// grok43Efforts is what Grok 4.3 accepts. It is the only family that
	// takes "none", so it is the only one whose reasoning can be switched off.
	grok43Efforts = map[string]bool{effortNone: true, effortLow: true, effortMedium: true, effortHigh: true}

	// grok45Efforts is what Grok 4.5 accepts. Unlike Grok 4.3 it rejects
	// "none" with HTTP 400 ("This model does not support 'reasoning_effort'
	// value 'none'"), so it always reasons.
	grok45Efforts = map[string]bool{effortLow: true, effortMedium: true, effortHigh: true}
)

// grok43Models are the slugs xAI serves with Grok 4.3, including the retired
// text slugs it redirects there, which accept the same reasoning_effort values
// (https://docs.x.ai/developers/migration/may-15-retirement). grok-code-fast-1
// is deliberately absent: it redirects to grok-build-0.1, not to Grok 4.3.
var grok43Models = map[string]bool{
	"grok-4.3":                    true,
	"grok-4.3-latest":             true,
	"grok-latest":                 true,
	"grok-4-0709":                 true,
	"grok-4-1-fast-reasoning":     true,
	"grok-4-1-fast-non-reasoning": true,
	"grok-4-fast-reasoning":       true,
	"grok-4-fast-non-reasoning":   true,
	"grok-3":                      true,
}

// grok45Models are the slugs xAI serves with Grok 4.5, including the
// grok-build-latest floating alias.
var grok45Models = map[string]bool{
	"grok-4.5":          true,
	"grok-4.5-latest":   true,
	"grok-build-latest": true,
}

// reasoningEfforts returns the reasoning_effort values the named model accepts.
//
// An empty result means the model rejects the parameter outright: xAI answers
// such a request with HTTP 400 rather than ignoring the field, so it must be
// left out of the request body entirely for those models.
func reasoningEfforts(model string) map[string]bool {
	switch {
	case grok43Models[model]:
		return grok43Efforts
	case grok45Models[model]:
		return grok45Efforts
	case strings.HasPrefix(model, "grok-3-mini"):
		return basicEfforts
	default:
		return nil
	}
}

// clampReasoningEffort maps a requested reasoning_effort onto the nearest value
// the named model accepts. The second result is false when there is no usable
// equivalent and the field must be left out of the request.
//
// Leaving it out is not the same as disabling reasoning: a model that accepts
// "none" is switched off by sending "none", while a model that does not accept
// it reasons at its own default when the field is absent. That is the closest
// available behavior, and it is preferable to a request the API rejects.
func clampReasoningEffort(model, effort string) (string, bool) {
	supported := reasoningEfforts(model)
	if len(supported) == 0 {
		return "", false
	}
	if supported[effort] {
		return effort, true
	}

	switch effort {
	case effortMinimal, effortLow:
		if supported[effortLow] {
			return effortLow, true
		}
	case effortMedium, effortHigh:
		// Round up rather than down: a family without "medium" reasons at
		// "high", matching pydantic-ai's _map_reasoning_effort
		// (pydantic_ai/models/xai.py:115).
		if supported[effortHigh] {
			return effortHigh, true
		}
	}
	// "none" for an always-reasoning model, or a value this library does not
	// recognize: omit it rather than risk a 400.
	return "", false
}

// reasoningEffortFor picks the reasoning_effort value to send for a thinking
// config, returning "" when nothing should be sent for this model.
//
// The budget thresholds match the OpenAI provider's, so a request written
// against one reads the same against the other; the result is then clamped to
// what the model actually accepts.
func reasoningEffortFor(model string, cfg *core.ThinkingConfig) string {
	if cfg == nil {
		return ""
	}

	requested := effortMedium
	switch {
	case !cfg.Enabled:
		// Only Grok 4.3 can be told not to reason. For every other family
		// clampReasoningEffort drops this, leaving the model at its default.
		requested = effortNone
	case cfg.BudgetTokens > 20000:
		requested = effortHigh
	case cfg.BudgetTokens > 0 && cfg.BudgetTokens <= 5000:
		requested = effortLow
	}

	effort, ok := clampReasoningEffort(model, requested)
	if !ok {
		return ""
	}
	return effort
}

// extractReasoning pulls thinking text out of the undecoded fields of a
// response message or streaming delta.
//
// The field is not part of the OpenAI schema, so the SDK leaves it in
// JSON.ExtraFields. Providers disagree on the name: xAI and newer vLLM builds
// use "reasoning", DeepSeek and older builds use "reasoning_content". Both are
// probed, "reasoning" first, matching the default pair pydantic-ai checks
// (pydantic_ai/profiles/openai.py: openai_chat_thinking_field). A field that is
// present but not a JSON string is ignored rather than reported.
//
// Presence is decided on the raw payload rather than respjson.Field.Valid: the
// SDK marks every field it has no typed destination for as invalid, so an extra
// field is never "valid" and Valid would discard all of them.
func extractReasoning(extra map[string]respjson.Field) string {
	for _, name := range []string{"reasoning", "reasoning_content"} {
		field, ok := extra[name]
		if !ok {
			continue
		}
		raw := field.Raw()
		if raw == "" {
			continue
		}
		var text string
		if err := json.Unmarshal([]byte(raw), &text); err != nil {
			continue
		}
		if text != "" {
			return text
		}
	}
	return ""
}

// convertFinishReason maps an xAI finish reason onto the normalized core value.
//
// It covers both the OpenAI-compatible reasons and the xAI-specific ones
// (max_output_tokens, cancelled, failed) enumerated in pydantic-ai's
// _FINISH_REASON_MAP (pydantic_ai/models/xai.py:143). Anything else maps to
// core.FinishReasonUnknown, so a stop reason this library does not recognize is
// never reported as a clean completion; the original string is always available
// from ChatResponse.RawFinishReason.
func convertFinishReason(reason string) core.FinishReason {
	switch reason {
	case "stop":
		return core.FinishReasonStop
	case "length", "max_output_tokens":
		return core.FinishReasonLength
	case "tool_calls":
		return core.FinishReasonToolCalls
	case "content_filter":
		return core.FinishReasonContentFilter
	case "cancelled", "failed":
		// xAI reports an aborted generation in band, on a 200 response.
		return core.FinishReasonError
	default:
		return core.FinishReasonUnknown
	}
}
