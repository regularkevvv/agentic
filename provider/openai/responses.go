package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

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

			case "response.reasoning_summary_text.delta":
				ch <- core.StreamEvent{
					Type:  core.StreamEventThinkingDelta,
					Delta: event.Text,
				}

			case "response.refusal.delta":
				ch <- core.StreamEvent{
					Type:  core.StreamEventTextDelta,
					Delta: event.Refusal,
				}

			case "response.completed":
				if event.Response.Usage.TotalTokens > 0 {
					usage = extractResponsesUsage(event.Response.Usage)
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

		ch <- core.StreamEvent{Type: core.StreamEventDone, Usage: &usage}
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

	// Convert messages to input items, extracting system instructions.
	var items responses.ResponseInputParam
	for _, msg := range req.Messages {
		if msg.Role == core.RoleSystem {
			params.Instructions = openai.String(msg.GetTextContent())
			continue
		}
		items = append(items, convertInputItems(msg)...)
	}
	if len(items) > 0 {
		params.Input.OfInputItemList = items
	}

	// Sampling parameters.
	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if req.MaxTokens != nil {
		params.MaxOutputTokens = openai.Int(int64(*req.MaxTokens))
	}
	if req.TopP != nil {
		params.TopP = openai.Float(*req.TopP)
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
	if req.Thinking != nil && req.Thinking.Enabled {
		params.Reasoning = convertReasoning(req.Thinking)
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
		// Emit text content as an assistant message.
		if text := msg.GetTextContent(); text != "" {
			items = append(items, responses.ResponseInputItemUnionParam{
				OfMessage: &responses.EasyInputMessageParam{
					Role:    "assistant",
					Content: responses.EasyInputMessageContentUnionParam{OfString: openai.String(text)},
				},
			})
		}
		// Emit each tool call as a function_call item.
		for _, tu := range msg.GetToolUses() {
			argsJSON, _ := json.Marshal(tu.Input)
			items = append(items, responses.ResponseInputItemUnionParam{
				OfFunctionCall: &responses.ResponseFunctionToolCallParam{
					CallID:    tu.ID,
					Name:      tu.Name,
					Arguments: string(argsJSON),
				},
			})
		}
		// Emit thinking blocks as reasoning items.
		for _, p := range msg.Content {
			if p.Type == core.ContentThinking && p.Thinking != nil {
				items = append(items, responses.ResponseInputItemUnionParam{
					OfReasoning: &responses.ResponseReasoningItemParam{
						EncryptedContent: openai.String(p.Thinking.Signature),
					},
				})
			}
		}

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
				Schema: rf.JSONSchema.Schema,
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
			for _, summary := range item.Summary {
				msg.Content = append(msg.Content, core.Part{
					Type: core.ContentThinking,
					Thinking: &core.ThinkingBlock{
						Text:         summary.Text,
						Signature:    item.EncryptedContent,
						ProviderName: "openai",
					},
				})
			}
		}
	}

	finishReason := convertResponsesFinishReason(resp)

	chatResp := &core.ChatResponse{
		ID:      resp.ID,
		Model:   string(resp.Model),
		Choices: []core.Choice{{Message: msg, FinishReason: finishReason}},
		Created: time.Unix(int64(resp.CreatedAt), 0),
		Usage:   extractResponsesUsage(resp.Usage),
	}

	return chatResp
}

// convertResponsesFinishReason maps Responses API status to agentic FinishReason.
func convertResponsesFinishReason(resp *responses.Response) core.FinishReason {
	// Check incomplete_details first.
	if resp.IncompleteDetails.Reason != "" {
		switch resp.IncompleteDetails.Reason {
		case "max_output_tokens":
			return core.FinishReasonLength
		case "content_filter":
			return core.FinishReasonContentFilter
		}
	}
	// Check for tool calls in output.
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			return core.FinishReasonToolCalls
		}
	}
	switch resp.Status {
	case "completed":
		return core.FinishReasonStop
	case "failed", "canceled":
		return core.FinishReasonStop
	default:
		return core.FinishReasonStop
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
