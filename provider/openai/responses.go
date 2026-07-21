package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// providerName identifies thinking blocks this package produced, so they are
// not replayed to a provider that would reject the signature.
const providerName = "openai"

// ResponsesModel implements the core.Model and core.StreamModel interfaces
// using the OpenAI Responses API.
//
// The Responses API is OpenAI's newer, stateful API that supports:
//   - Built-in tools (web search, file search, code interpreter, image generation)
//   - Server-side conversation state via PreviousResponseID
//   - Reasoning configuration with effort + summary controls
//   - Richer structured output via the text config parameter
//
// Use [NewResponses] to create a ResponsesModel, or [NewResponsesFromClient] to
// share a client with an existing [Model].
type ResponsesModel struct {
	client *openai.Client
	model  string
}

// NewResponses creates a new ResponsesModel using the Responses API.
//
// It accepts the same options as [New] for configuring the API key, base URL,
// organization, etc.
//
// Examples:
//
//	model, err := openai.NewResponses("gpt-4o", openai.WithAPIKey("sk-..."))
//	model, err := openai.NewResponses("o3", openai.WithAPIKey("sk-..."))
func NewResponses(model string, opts ...Option) (*ResponsesModel, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	var reqOpts []option.RequestOption
	if cfg.apiKey != "" {
		reqOpts = append(reqOpts, option.WithAPIKey(cfg.apiKey))
	}
	if cfg.baseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(cfg.baseURL))
	}
	if cfg.organization != "" {
		reqOpts = append(reqOpts, option.WithOrganization(cfg.organization))
	}
	reqOpts = append(reqOpts, cfg.extraOpts...)

	client := openai.NewClient(reqOpts...)

	return &ResponsesModel{
		client: &client,
		model:  model,
	}, nil
}

// MustNewResponses is like NewResponses but panics on error.
func MustNewResponses(model string, opts ...Option) *ResponsesModel {
	m, err := NewResponses(model, opts...)
	if err != nil {
		panic(err)
	}
	return m
}

// NewResponsesFromClient creates a ResponsesModel that shares the underlying
// openai.Client with an existing Model. This avoids creating duplicate HTTP
// connections when you need both Chat Completions and Responses models.
func NewResponsesFromClient(model string, m *Model) *ResponsesModel {
	return &ResponsesModel{
		client: m.client,
		model:  model,
	}
}

// Request implements core.Model.
func (m *ResponsesModel) Request(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	params := m.buildParams(req)

	resp, err := m.client.Responses.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("openai responses: %w", err)
	}

	return m.convertResponse(resp), nil
}

// RequestStream implements core.StreamModel.
func (m *ResponsesModel) RequestStream(ctx context.Context, req *core.ChatRequest) (*core.StreamResult, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	params := m.buildParams(req)
	stream := m.client.Responses.NewStreaming(ctx, params)

	ch := make(chan core.StreamEvent, 64)
	sr := core.NewStreamResult(ch)

	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		var usage core.Usage
		finishReason := core.FinishReasonStop
		// Track tool calls by item ID for accumulating argument deltas.
		toolCallIDs := make(map[string]string) // itemID -> callID

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_text.delta":
				ch <- core.StreamEvent{
					Type:  core.StreamEventTextDelta,
					Delta: event.Delta.OfString,
				}

			case "response.function_call_arguments.delta":
				callID := toolCallIDs[event.ItemID]
				ch <- core.StreamEvent{
					Type:       core.StreamEventToolCallDelta,
					Delta:      event.Delta.OfString,
					ToolCallID: callID,
				}

			case "response.output_item.added":
				if event.Item.Type == "function_call" {
					toolCallIDs[event.Item.ID] = event.Item.CallID
					ch <- core.StreamEvent{
						Type: core.StreamEventToolCallStart,
						ToolUse: &core.ToolUse{
							ID:   event.Item.CallID,
							Name: event.Item.Name,
						},
					}
				}

			case "response.output_item.done":
				// The reasoning item's encrypted content is only present
				// once the item completes, after its summary deltas. Emit it
				// as a terminal event on the block so a streamed reasoning
				// turn can be replayed.
				if event.Item.Type == "reasoning" && event.Item.EncryptedContent != "" {
					ch <- core.StreamEvent{
						Type:         core.StreamEventThinkingDelta,
						ThinkingID:   event.Item.ID,
						Signature:    event.Item.EncryptedContent,
						ProviderName: providerName,
					}
				}

			case "response.reasoning_summary_text.delta":
				// On the flattened event union, Text is populated only by the
				// .done variant; the delta text lives in Delta.
				ch <- core.StreamEvent{
					Type:         core.StreamEventThinkingDelta,
					Delta:        event.Delta.OfString,
					ThinkingID:   event.ItemID,
					ProviderName: providerName,
				}

			case "response.refusal.delta":
				// Refusal is populated only by the .done variant.
				finishReason = core.FinishReasonContentFilter
				ch <- core.StreamEvent{
					Type:  core.StreamEventTextDelta,
					Delta: event.Delta.OfString,
				}

			case "response.completed":
				if event.Response.Usage.TotalTokens > 0 {
					usage = extractResponsesUsage(event.Response.Usage)
				}
				if reason, _ := convertResponsesFinishReason(&event.Response); finishReason != core.FinishReasonContentFilter {
					finishReason = reason
				}

			case "response.incomplete", "response.failed":
				// The response ended without a complete answer. Usage is
				// still reported and worth keeping.
				if event.Response.Usage.TotalTokens > 0 {
					usage = extractResponsesUsage(event.Response.Usage)
				}
				reason, raw := convertResponsesFinishReason(&event.Response)
				finishReason = reason
				ch <- core.StreamEvent{
					Type:  core.StreamEventError,
					Error: fmt.Errorf("openai responses stream: response %s: %s", event.Type, raw),
				}

			case "error":
				// An upstream failure reported in-band on a 200 stream.
				finishReason = core.FinishReasonError
				ch <- core.StreamEvent{
					Type:  core.StreamEventError,
					Error: fmt.Errorf("openai responses stream: %s: %s", event.Code, event.Message),
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- core.StreamEvent{
				Type:  core.StreamEventError,
				Error: fmt.Errorf("openai responses stream: %w", err),
			}
			return
		}

		ch <- core.StreamEvent{
			Type:         core.StreamEventDone,
			Usage:        &usage,
			FinishReason: finishReason,
		}
	}()

	return sr, nil
}

// Name implements core.Model.
func (m *ResponsesModel) Name() string {
	return m.model
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

// buildParams converts core.ChatRequest to Responses API params.
func (m *ResponsesModel) buildParams(req *core.ChatRequest) responses.ResponseNewParams {
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(m.model),
	}

	// Convert messages to input items, extracting system instructions. Every
	// system message contributes: keeping only the last silently discards
	// earlier instructions. Joined with a blank line, matching how the agent
	// joins multiple system prompts (agent.go).
	var items responses.ResponseInputParam
	var instructions []string
	for _, msg := range req.Messages {
		if msg.Role == core.RoleSystem {
			if text := msg.GetTextContent(); text != "" {
				instructions = append(instructions, text)
			}
			continue
		}
		items = append(items, convertInputItems(msg)...)
	}
	if len(instructions) > 0 {
		params.Instructions = openai.String(strings.Join(instructions, "\n\n"))
	}
	if len(items) > 0 {
		params.Input.OfInputItemList = items
	}

	thinkingEnabled := req.Thinking != nil && req.Thinking.Enabled

	// Sampling parameters. Reasoning models reject them while reasoning is
	// active and respond 400.
	if supportsSamplingParams(m.model, thinkingEnabled) {
		if req.Temperature != nil {
			params.Temperature = openai.Float(*req.Temperature)
		}
		if req.TopP != nil {
			params.TopP = openai.Float(*req.TopP)
		}
	}
	if req.MaxTokens != nil {
		params.MaxOutputTokens = openai.Int(int64(*req.MaxTokens))
	}

	opts := optionsFrom(req.ProviderOptions)
	if opts.ParallelToolCalls != nil {
		params.ParallelToolCalls = openai.Bool(*opts.ParallelToolCalls)
	}
	if tier := opts.serviceTier(); tier != "" {
		params.ServiceTier = responses.ResponseNewParamsServiceTier(tier)
	}

	// Tools.
	if len(req.Tools) > 0 {
		params.Tools = convertResponsesTools(req.Tools)
		if req.ToolChoice != nil {
			params.ToolChoice = convertResponsesToolChoice(*req.ToolChoice)
		}
	}

	// Response format → text config.
	if req.ResponseFormat != nil {
		params.Text = convertTextConfig(req.ResponseFormat)
	}

	// Reasoning (o-series models).
	if thinkingEnabled {
		params.Reasoning = convertReasoning(req.Thinking)
		// Reasoning items are only replayable when the encrypted content
		// comes back with them. Without this the API omits it, and a
		// reasoning turn fed into the next request is rejected.
		params.Include = append(params.Include, responses.ResponseIncludableReasoningEncryptedContent)
	}

	return params
}

// convertInputItems converts a single core.Message into Responses API input items.
// A single message may produce multiple items (e.g., an assistant message with
// both text and tool calls, followed by tool results).
func convertInputItems(msg core.Message) []responses.ResponseInputItemUnionParam {
	var items []responses.ResponseInputItemUnionParam

	switch msg.Role {
	case core.RoleUser:
		items = append(items, responses.ResponseInputItemUnionParam{
			OfMessage: &responses.EasyInputMessageParam{
				Role:    "user",
				Content: convertInputContent(msg),
			},
		})

	case core.RoleAssistant:
		items = append(items, convertAssistantItems(msg)...)

	case core.RoleTool:
		for _, tr := range msg.GetToolResults() {
			items = append(items, responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: tr.ToolUseID,
					Output: tr.Content,
				},
			})
		}
	}

	return items
}

// convertAssistantItems converts an assistant message into Responses API input
// items in a single ordered pass over its parts.
//
// Order is load-bearing. The API rejects a reasoning item that appears after
// the function calls it produced, so the items must be replayed in the order
// the model emitted them ([reasoning, message, function_call]) rather than
// grouped by kind. Consecutive thinking parts sharing a reasoning-item id are
// re-joined into one reasoning item with several summaries, which is how the
// API sent them; the encrypted content is carried on that single item.
func convertAssistantItems(msg core.Message) []responses.ResponseInputItemUnionParam {
	var items []responses.ResponseInputItemUnionParam
	var textRun []string

	// flushText emits buffered consecutive text parts as one assistant message.
	flushText := func() {
		if len(textRun) == 0 {
			return
		}
		items = append(items, responses.ResponseInputItemUnionParam{
			OfMessage: &responses.EasyInputMessageParam{
				Role:    "assistant",
				Content: responses.EasyInputMessageContentUnionParam{OfString: openai.String(strings.Join(textRun, ""))},
			},
		})
		textRun = nil
	}

	// reasoningIndex maps a reasoning-item id to its position in items, so a
	// later summary with the same id joins the item already emitted instead
	// of starting a new one.
	reasoningIndex := make(map[string]int)

	for _, p := range msg.Content {
		switch {
		case p.Type == core.ContentText:
			if p.Text != "" {
				textRun = append(textRun, p.Text)
			}

		case p.Type == core.ContentToolUse && p.ToolUse != nil:
			flushText()
			argsJSON, _ := json.Marshal(p.ToolUse.Input)
			items = append(items, responses.ResponseInputItemUnionParam{
				OfFunctionCall: &responses.ResponseFunctionToolCallParam{
					CallID:    p.ToolUse.ID,
					Name:      p.ToolUse.Name,
					Arguments: string(argsJSON),
				},
			})

		case p.Type == core.ContentThinking && p.Thinking != nil:
			flushText()
			// Only replay reasoning this provider issued; another
			// provider's block carries a signature OpenAI rejects.
			if p.Thinking.ProviderName != "" && p.Thinking.ProviderName != providerName {
				continue
			}
			summary := responses.ResponseReasoningItemSummaryParam{Text: p.Thinking.Text}
			if idx, ok := reasoningIndex[p.Thinking.ID]; ok && p.Thinking.ID != "" {
				existing := items[idx].OfReasoning
				existing.Summary = append(existing.Summary, summary)
				if !existing.EncryptedContent.Valid() && p.Thinking.Signature != "" {
					existing.EncryptedContent = openai.String(p.Thinking.Signature)
				}
				continue
			}
			item := &responses.ResponseReasoningItemParam{
				ID:      p.Thinking.ID,
				Summary: []responses.ResponseReasoningItemSummaryParam{summary},
			}
			if p.Thinking.Signature != "" {
				item.EncryptedContent = openai.String(p.Thinking.Signature)
			}
			if p.Thinking.ID != "" {
				reasoningIndex[p.Thinking.ID] = len(items)
			}
			items = append(items, responses.ResponseInputItemUnionParam{OfReasoning: item})
		}
	}
	flushText()

	return items
}

// convertInputContent converts user message parts to Responses API content.
func convertInputContent(msg core.Message) responses.EasyInputMessageContentUnionParam {
	// Fast path: single text part.
	if len(msg.Content) == 1 && msg.Content[0].Type == core.ContentText {
		return responses.EasyInputMessageContentUnionParam{
			OfString: openai.String(msg.Content[0].Text),
		}
	}

	// Multi-part: build content list.
	var parts responses.ResponseInputMessageContentListParam
	for _, p := range msg.Content {
		switch p.Type {
		case core.ContentText:
			parts = append(parts, responses.ResponseInputContentUnionParam{
				OfInputText: &responses.ResponseInputTextParam{
					Text: p.Text,
				},
			})
		case core.ContentImageURL:
			if p.ImageURL != nil {
				detail := responses.ResponseInputImageDetailAuto
				if p.ImageURL.Detail != "" {
					detail = responses.ResponseInputImageDetail(p.ImageURL.Detail)
				}
				parts = append(parts, responses.ResponseInputContentUnionParam{
					OfInputImage: &responses.ResponseInputImageParam{
						ImageURL: openai.String(p.ImageURL.URL),
						Detail:   detail,
					},
				})
			}
		case core.ContentImageData:
			if p.ImageData != nil {
				dataURI := fmt.Sprintf("data:%s;base64,%s", p.ImageData.MediaType, p.ImageData.Data)
				parts = append(parts, responses.ResponseInputContentUnionParam{
					OfInputImage: &responses.ResponseInputImageParam{
						ImageURL: openai.String(dataURI),
					},
				})
			}
		case core.ContentAudioURL:
			if p.AudioURL != nil {
				parts = append(parts, responses.ResponseInputContentUnionParam{
					OfInputFile: &responses.ResponseInputFileParam{
						FileURL: openai.String(p.AudioURL.URL),
					},
				})
			}
		case core.ContentDocumentURL:
			if p.DocumentURL != nil {
				parts = append(parts, responses.ResponseInputContentUnionParam{
					OfInputFile: &responses.ResponseInputFileParam{
						FileURL: openai.String(p.DocumentURL.URL),
					},
				})
			}
		case core.ContentUploadedFile:
			if p.UploadedFile != nil {
				parts = append(parts, responses.ResponseInputContentUnionParam{
					OfInputFile: &responses.ResponseInputFileParam{
						FileID: openai.String(p.UploadedFile.FileID),
					},
				})
			}
		}
	}

	return responses.EasyInputMessageContentUnionParam{
		OfInputItemContentList: parts,
	}
}

// convertResponsesTools converts agentic tools to Responses API function tools.
// The Responses API requires strict schemas with additionalProperties: false.
func convertResponsesTools(tools []core.Tool) []responses.ToolUnionParam {
	result := make([]responses.ToolUnionParam, len(tools))
	for i, tool := range tools {
		params := tool.Function.Parameters
		if params == nil {
			params = make(map[string]interface{})
		}
		// Deep copy and add additionalProperties: false (required by Responses API).
		params = ensureAdditionalPropertiesFalse(params)

		ft := &responses.FunctionToolParam{
			Name:       tool.Function.Name,
			Parameters: params,
			Strict:     openai.Bool(true),
		}
		if tool.Function.Description != "" {
			ft.Description = openai.String(tool.Function.Description)
		}
		result[i] = responses.ToolUnionParam{OfFunction: ft}
	}
	return result
}

// normalizeStrictSchema prepares a caller-supplied JSON schema for strict
// structured output.
//
// Strict mode requires every object to carry additionalProperties: false and
// to list every property in "required". Schemas reflected from Go structs
// (agentic's own NewNativeOutput path, via swaggest) emit neither, so a strict
// request built from one is rejected by the API. Schemas sent with strict off
// are returned untouched: forcing every property to be required there would
// change what the caller asked for.
func normalizeStrictSchema(schema map[string]interface{}, strict *bool) map[string]interface{} {
	if schema == nil || strict == nil || !*strict {
		return schema
	}
	return ensureAdditionalPropertiesFalse(schema)
}

// ensureAdditionalPropertiesFalse recursively sets additionalProperties: false
// on all object schemas. The Responses API requires this for strict mode.
func ensureAdditionalPropertiesFalse(schema map[string]interface{}) map[string]interface{} {
	// Copy to avoid mutating the original.
	out := make(map[string]interface{}, len(schema))
	for k, v := range schema {
		out[k] = v
	}

	schemaType, _ := out["type"].(string)
	if schemaType == "object" || out["properties"] != nil {
		out["additionalProperties"] = false

		// Strict mode requires all properties in "required".
		if props, ok := out["properties"].(map[string]interface{}); ok {
			existing := make(map[string]bool)
			if req, ok := out["required"].([]interface{}); ok {
				for _, r := range req {
					if s, ok := r.(string); ok {
						existing[s] = true
					}
				}
			}
			if req, ok := out["required"].([]string); ok {
				for _, r := range req {
					existing[r] = true
				}
			}
			var allRequired []string
			for name := range props {
				allRequired = append(allRequired, name)
			}
			if len(allRequired) > 0 {
				out["required"] = allRequired
			}
		}
	}

	// Recurse into properties.
	if props, ok := out["properties"].(map[string]interface{}); ok {
		newProps := make(map[string]interface{}, len(props))
		for name, prop := range props {
			if propMap, ok := prop.(map[string]interface{}); ok {
				newProps[name] = ensureAdditionalPropertiesFalse(propMap)
			} else {
				newProps[name] = prop
			}
		}
		out["properties"] = newProps
	}

	// Recurse into items (arrays).
	if items, ok := out["items"].(map[string]interface{}); ok {
		out["items"] = ensureAdditionalPropertiesFalse(items)
	}

	// Recurse into the definition pools. A schema reflected from a Go struct
	// with a nested struct field puts that field's object here and references
	// it by $ref, so skipping these leaves the referenced object without
	// additionalProperties and strict mode rejects the whole schema. swaggest
	// emits the draft-07 "definitions"; newer reflectors emit "$defs".
	for _, pool := range []string{"definitions", "$defs"} {
		defs, ok := out[pool].(map[string]interface{})
		if !ok {
			continue
		}
		newDefs := make(map[string]interface{}, len(defs))
		for name, def := range defs {
			if defMap, ok := def.(map[string]interface{}); ok {
				newDefs[name] = ensureAdditionalPropertiesFalse(defMap)
			} else {
				newDefs[name] = def
			}
		}
		out[pool] = newDefs
	}

	// Recurse into the composition keywords. Reflected schemas use these for
	// optional and union-typed fields, and each branch is itself a schema.
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		branches, ok := out[key].([]interface{})
		if !ok {
			continue
		}
		newBranches := make([]interface{}, len(branches))
		for i, branch := range branches {
			if branchMap, ok := branch.(map[string]interface{}); ok {
				newBranches[i] = ensureAdditionalPropertiesFalse(branchMap)
			} else {
				newBranches[i] = branch
			}
		}
		out[key] = newBranches
	}

	return out
}

// convertResponsesToolChoice converts agentic ToolChoice to Responses API format.
func convertResponsesToolChoice(choice core.ToolChoice) responses.ResponseNewParamsToolChoiceUnion {
	switch choice {
	case core.ToolChoiceNone:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsNone),
		}
	case core.ToolChoiceRequired:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsRequired),
		}
	default:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsAuto),
		}
	}
}

// convertTextConfig converts agentic ResponseFormat to Responses API text config.
func convertTextConfig(rf *core.ResponseFormat) responses.ResponseTextConfigParam {
	tc := responses.ResponseTextConfigParam{}
	switch rf.Type {
	case "json_schema":
		if rf.JSONSchema != nil {
			schema := &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   rf.JSONSchema.Name,
				Schema: normalizeStrictSchema(rf.JSONSchema.Schema, rf.JSONSchema.Strict),
			}
			if rf.JSONSchema.Description != "" {
				schema.Description = openai.String(rf.JSONSchema.Description)
			}
			if rf.JSONSchema.Strict != nil {
				schema.Strict = openai.Bool(*rf.JSONSchema.Strict)
			}
			tc.Format.OfJSONSchema = schema
		}
	case "json_object":
		tc.Format.OfJSONObject = &shared.ResponseFormatJSONObjectParam{}
	default:
		tc.Format.OfText = &shared.ResponseFormatTextParam{}
	}
	return tc
}

// convertReasoning converts agentic ThinkingConfig to Responses API reasoning param.
func convertReasoning(thinking *core.ThinkingConfig) shared.ReasoningParam {
	r := shared.ReasoningParam{
		Summary: shared.ReasoningSummaryAuto,
	}
	effort := shared.ReasoningEffortMedium
	if thinking.BudgetTokens > 20000 {
		effort = shared.ReasoningEffortHigh
	} else if thinking.BudgetTokens <= 5000 && thinking.BudgetTokens > 0 {
		effort = shared.ReasoningEffortLow
	}
	r.Effort = effort
	return r
}

// ---------------------------------------------------------------------------
// Response conversion
// ---------------------------------------------------------------------------

// convertResponse converts a Responses API response to agentic types.
func (m *ResponsesModel) convertResponse(resp *responses.Response) *core.ChatResponse {
	msg := core.Message{
		Role:    core.RoleAssistant,
		Content: make([]core.Part, 0),
	}
	refused := false

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				switch c.Type {
				case "output_text":
					msg.Content = append(msg.Content, core.Part{
						Type: core.ContentText,
						Text: c.Text,
					})
				case "refusal":
					refused = true
					msg.Content = append(msg.Content, core.Part{
						Type: core.ContentText,
						Text: c.Refusal,
					})
				}
			}

		case "function_call":
			var input map[string]interface{}
			_ = json.Unmarshal([]byte(item.Arguments), &input)
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentToolUse,
				ToolUse: &core.ToolUse{
					ID:    item.CallID,
					Name:  item.Name,
					Input: input,
				},
			})

		case "reasoning":
			msg.Content = append(msg.Content, convertReasoningItem(item)...)
		}
	}

	finishReason, rawFinishReason := convertResponsesFinishReason(resp)
	// A refusal comes back on a response whose status is "completed";
	// reporting a clean stop would hide the safety filter.
	if refused {
		finishReason = core.FinishReasonContentFilter
	}

	return &core.ChatResponse{
		ID:              resp.ID,
		Model:           string(resp.Model),
		Message:         msg,
		Created:         time.Unix(int64(resp.CreatedAt), 0),
		Usage:           extractResponsesUsage(resp.Usage),
		FinishReason:    finishReason,
		RawFinishReason: rawFinishReason,
	}
}

// convertReasoningItem converts one reasoning output item into thinking parts.
//
// The item id is carried on every part so the parts can be regrouped into a
// single reasoning item when replayed, and the encrypted content is attached
// to the first part only — repeating it on each summary would replay one
// signature as several. An item with no summaries still yields one part:
// dropping it loses the encrypted content, and with it the ability to send the
// reasoning turn back.
func convertReasoningItem(item responses.ResponseOutputItemUnion) []core.Part {
	if len(item.Summary) == 0 {
		return []core.Part{{
			Type: core.ContentThinking,
			Thinking: &core.ThinkingBlock{
				ID:           item.ID,
				Signature:    item.EncryptedContent,
				ProviderName: providerName,
			},
		}}
	}

	parts := make([]core.Part, 0, len(item.Summary))
	for i, summary := range item.Summary {
		block := &core.ThinkingBlock{
			Text:         summary.Text,
			ID:           item.ID,
			ProviderName: providerName,
		}
		if i == 0 {
			block.Signature = item.EncryptedContent
		}
		parts = append(parts, core.Part{Type: core.ContentThinking, Thinking: block})
	}
	return parts
}

// convertResponsesFinishReason maps a Responses API status to the agentic
// FinishReason, and returns the provider's own value alongside it.
//
// The raw value is incomplete_details.reason when the response is incomplete,
// and the response status otherwise, matching pydantic-ai
// (pydantic_ai/models/openai.py: `raw_finish_reason = details.reason if
// (details := response.incomplete_details) else response.status`). A status
// this library does not recognize maps to FinishReasonUnknown; "failed" and
// "cancelled" are genuine failures and map to FinishReasonError, not to a
// clean stop.
func convertResponsesFinishReason(resp *responses.Response) (core.FinishReason, string) {
	if reason := string(resp.IncompleteDetails.Reason); reason != "" {
		switch reason {
		case "max_output_tokens":
			return core.FinishReasonLength, reason
		case "content_filter":
			return core.FinishReasonContentFilter, reason
		default:
			return core.FinishReasonUnknown, reason
		}
	}

	status := string(resp.Status)

	// A response carrying tool calls ended to run them, whatever its status.
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			return core.FinishReasonToolCalls, status
		}
	}

	switch responses.ResponseStatus(status) {
	case responses.ResponseStatusCompleted:
		return core.FinishReasonStop, status
	case responses.ResponseStatusFailed, responses.ResponseStatusCancelled:
		return core.FinishReasonError, status
	default:
		// in_progress, queued, incomplete without details, or anything the
		// API adds later.
		return core.FinishReasonUnknown, status
	}
}

// extractResponsesUsage converts Responses API usage to agentic Usage.
func extractResponsesUsage(u responses.ResponseUsage) core.Usage {
	usage := core.Usage{
		PromptTokens:     int(u.InputTokens),
		CompletionTokens: int(u.OutputTokens),
		TotalTokens:      int(u.TotalTokens),
	}
	if u.InputTokensDetails.CachedTokens > 0 {
		usage.CacheReadTokens = int(u.InputTokensDetails.CachedTokens)
	}
	if u.OutputTokensDetails.ReasoningTokens > 0 {
		usage.ReasoningTokens = int(u.OutputTokensDetails.ReasoningTokens)
	}
	return usage
}

// Compile-time check that ResponsesModel implements core.StreamModel.
var _ core.StreamModel = (*ResponsesModel)(nil)
