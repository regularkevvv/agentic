package openrouter

import (
	"encoding/json"
	"fmt"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/respjson"
	"github.com/openai/openai-go/shared"
)

// requestOptions derives per-request SDK options for the parts of OpenRouter's
// request body that have no field on the OpenAI schema, and so must be injected
// into the body directly. Mirrors pydantic-ai models/openrouter.py:650, which
// puts the same objects in extra_body.
func requestOptions(req *core.ChatRequest) []option.RequestOption {
	var opts []option.RequestOption

	if routing := optionsFrom(req).Provider; routing != nil {
		opts = append(opts, option.WithJSONSet("provider", routing))
	}

	if reasoning := buildReasoning(req.Thinking); reasoning != nil {
		opts = append(opts, option.WithJSONSet("reasoning", reasoning))
	}

	return opts
}

// buildReasoning maps the unified thinking config onto OpenRouter's "reasoning"
// object. Effort and max_tokens are mutually exclusive there, so a budget wins
// when one is given and the effort level is used otherwise. See pydantic-ai
// models/openrouter.py:227 (OpenRouterReasoning).
func buildReasoning(cfg *core.ThinkingConfig) map[string]interface{} {
	if cfg == nil || !cfg.Enabled {
		return nil
	}
	// Some reasoning-optional routes require an explicit enabled flag even
	// when effort is set (pydantic-ai models/openrouter.py:663).
	reasoning := map[string]interface{}{"enabled": true}
	if cfg.BudgetTokens > 0 {
		reasoning["max_tokens"] = cfg.BudgetTokens
	} else {
		reasoning["effort"] = "medium"
	}
	return reasoning
}

// reasoningFields are the response fields that may carry reasoning text,
// probed in order. OpenRouter and gpt-oss use "reasoning"; DeepSeek and
// Moonshot use "reasoning_content". Mirrors pydantic-ai
// models/openai.py:1149.
var reasoningFields = []string{"reasoning", "reasoning_content"}

// extractReasoning probes the undecoded fields of a response message or delta
// for reasoning text, returning the field name it came from and the text.
//
// The OpenAI SDK has no typed field for either name, so the values land in
// ExtraFields as raw JSON. A field holding anything other than a string is
// skipped rather than reported, matching pydantic-ai's behavior of warning
// and continuing.
func extractReasoning(extra map[string]respjson.Field) (string, string) {
	for _, name := range reasoningFields {
		field, ok := extra[name]
		if !ok {
			continue
		}
		raw := field.Raw()
		if raw == "" || raw == respjson.Null {
			continue
		}
		var text string
		if err := json.Unmarshal([]byte(raw), &text); err != nil {
			continue
		}
		if text == "" {
			continue
		}
		return name, text
	}
	return "", ""
}

// convertFinishReason normalizes an OpenRouter stop reason.
//
// OpenRouter's documented set is stop, length, tool_calls, content_filter and
// error — it drops OpenAI's function_call and adds error (pydantic-ai
// models/openrouter.py:54 and :507). function_call is still mapped because the
// gateway proxies OpenAI-compatible upstreams that may emit it. Anything else,
// including an absent reason, maps to unknown rather than being reported as a
// clean stop.
func convertFinishReason(reason string) core.FinishReason {
	switch reason {
	case "stop":
		return core.FinishReasonStop
	case "length":
		return core.FinishReasonLength
	case "tool_calls", "function_call":
		return core.FinishReasonToolCalls
	case "content_filter":
		return core.FinishReasonContentFilter
	case "error":
		return core.FinishReasonError
	default:
		return core.FinishReasonUnknown
	}
}

// extractUsage converts SDK usage into core usage, including the reasoning and
// cached-token details OpenRouter passes through from upstream providers.
func extractUsage(u openai.CompletionUsage) core.Usage {
	usage := core.Usage{
		PromptTokens:     int(u.PromptTokens),
		CompletionTokens: int(u.CompletionTokens),
		TotalTokens:      int(u.TotalTokens),
	}
	if u.CompletionTokensDetails.ReasoningTokens > 0 {
		usage.ReasoningTokens = int(u.CompletionTokensDetails.ReasoningTokens)
	}
	if u.PromptTokensDetails.CachedTokens > 0 {
		usage.CacheReadTokens = int(u.PromptTokensDetails.CachedTokens)
	}
	return usage
}

// convertResponseMessage converts an OpenRouter response message to core.
// Reasoning is emitted first, so a message reads in the order the model
// produced it.
func convertResponseMessage(msg openai.ChatCompletionMessage) core.Message {
	out := core.Message{
		Role:    core.RoleAssistant,
		Content: make([]core.Part, 0, 2),
	}

	if id, text := extractReasoning(msg.JSON.ExtraFields); text != "" {
		out.Content = append(out.Content, core.Part{
			Type: core.ContentThinking,
			Thinking: &core.ThinkingBlock{
				Text:         text,
				ID:           id,
				ProviderName: ProviderName,
			},
		})
	}

	if msg.Content != "" {
		out.Content = append(out.Content, core.Part{
			Type: core.ContentText,
			Text: msg.Content,
		})
	}

	for _, tc := range msg.ToolCalls {
		var input map[string]interface{}
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
		out.Content = append(out.Content, core.Part{
			Type: core.ContentToolUse,
			ToolUse: &core.ToolUse{
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			},
		})
	}

	return out
}

// convertMessage converts a core.Message to OpenRouter request format.
func convertMessage(msg core.Message) openai.ChatCompletionMessageParamUnion {
	switch msg.Role {
	case core.RoleSystem:
		return openai.SystemMessage(msg.GetTextContent())

	case core.RoleUser:
		if len(msg.Content) == 1 && msg.Content[0].Type == core.ContentText {
			return openai.UserMessage(msg.Content[0].Text)
		}
		parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(msg.Content))
		for _, part := range msg.Content {
			parts = append(parts, convertContentPart(part))
		}
		return openai.ChatCompletionMessageParamUnion{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfArrayOfContentParts: parts,
				},
			},
		}

	case core.RoleAssistant:
		toolUses := msg.GetToolUses()
		if len(toolUses) > 0 {
			toolCalls := make([]openai.ChatCompletionMessageToolCallParam, len(toolUses))
			for i, tu := range toolUses {
				inputJSON, _ := json.Marshal(tu.Input)
				toolCalls[i] = openai.ChatCompletionMessageToolCallParam{
					ID: tu.ID,
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      tu.Name,
						Arguments: string(inputJSON),
					},
				}
			}
			return openai.ChatCompletionMessageParamUnion{
				OfAssistant: &openai.ChatCompletionAssistantMessageParam{
					Content: openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: openai.String(msg.GetTextContent()),
					},
					ToolCalls: toolCalls,
				},
			}
		}
		return openai.AssistantMessage(msg.GetTextContent())

	case core.RoleTool:
		results := msg.GetToolResults()
		if len(results) > 0 {
			return openai.ToolMessage(results[0].Content, results[0].ToolUseID)
		}
		return openai.ToolMessage(msg.GetTextContent(), "unknown")

	default:
		return openai.UserMessage(msg.GetTextContent())
	}
}

// convertContentPart converts a core.Part to an OpenRouter content part.
func convertContentPart(part core.Part) openai.ChatCompletionContentPartUnionParam {
	switch part.Type {
	case core.ContentText:
		return openai.TextContentPart(part.Text)
	case core.ContentImageURL:
		if part.ImageURL != nil {
			imageURL := openai.ChatCompletionContentPartImageImageURLParam{
				URL: part.ImageURL.URL,
			}
			if part.ImageURL.Detail != "" {
				imageURL.Detail = part.ImageURL.Detail
			}
			return openai.ImageContentPart(imageURL)
		}
	case core.ContentImageData:
		if part.ImageData != nil {
			dataURI := fmt.Sprintf("data:%s;base64,%s", part.ImageData.MediaType, part.ImageData.Data)
			imageURL := openai.ChatCompletionContentPartImageImageURLParam{URL: dataURI}
			if part.ImageData.VendorMetadata != nil {
				if detail, ok := part.ImageData.VendorMetadata["detail"].(string); ok {
					imageURL.Detail = detail
				}
			}
			return openai.ImageContentPart(imageURL)
		}
	case core.ContentAudioURL:
		if part.AudioURL != nil {
			return openai.InputAudioContentPart(openai.ChatCompletionContentPartInputAudioInputAudioParam{
				Data:   part.AudioURL.URL,
				Format: part.AudioURL.Format,
			})
		}
	case core.ContentUploadedFile:
		if part.UploadedFile != nil {
			return openai.FileContentPart(openai.ChatCompletionContentPartFileFileParam{
				FileID: openai.String(part.UploadedFile.FileID),
			})
		}
	case core.ContentCachePoint:
		// Cache points are an Anthropic/Bedrock concept with no OpenAI-schema
		// equivalent; OpenRouter expects cache_control on the content part,
		// which this package does not yet emit.
		return openai.TextContentPart("")
	}
	return openai.TextContentPart(part.Text)
}

// convertTools converts core tools to OpenRouter format.
func convertTools(tools []core.Tool) []openai.ChatCompletionToolParam {
	result := make([]openai.ChatCompletionToolParam, len(tools))
	for i, tool := range tools {
		params := tool.Function.Parameters
		if params == nil {
			params = make(map[string]interface{})
		}
		result[i] = openai.ChatCompletionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        tool.Function.Name,
				Description: openai.String(tool.Function.Description),
				Parameters:  shared.FunctionParameters(params),
			},
		}
	}
	return result
}

// convertToolChoice converts a core.ToolChoice to OpenRouter format.
func convertToolChoice(choice core.ToolChoice) openai.ChatCompletionToolChoiceOptionUnionParam {
	switch choice {
	case core.ToolChoiceNone:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String(string(openai.ChatCompletionToolChoiceOptionAutoNone)),
		}
	case core.ToolChoiceRequired:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String(string(openai.ChatCompletionToolChoiceOptionAutoRequired)),
		}
	default:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String(string(openai.ChatCompletionToolChoiceOptionAutoAuto)),
		}
	}
}

// convertResponseFormat converts a core.ResponseFormat to OpenRouter format.
func convertResponseFormat(rf *core.ResponseFormat) openai.ChatCompletionNewParamsResponseFormatUnion {
	switch rf.Type {
	case "json_object":
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{},
		}
	case "json_schema":
		if rf.JSONSchema != nil {
			schema := openai.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   rf.JSONSchema.Name,
				Schema: rf.JSONSchema.Schema,
			}
			if rf.JSONSchema.Description != "" {
				schema.Description = openai.String(rf.JSONSchema.Description)
			}
			if rf.JSONSchema.Strict != nil {
				schema.Strict = openai.Bool(*rf.JSONSchema.Strict)
			}
			return openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{JSONSchema: schema},
			}
		}
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{},
		}
	default:
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfText: &openai.ResponseFormatTextParam{},
		}
	}
}
