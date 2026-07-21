// Package openai provides an OpenAI Model implementation for Agentic.
// It supports both the official OpenAI API and OpenAI-compatible providers
// (OpenRouter, Together AI, DeepSeek, etc.) via the WithBaseURL option.
package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

// Model implements the core.Model and core.StreamModel interfaces
// using the OpenAI ChatCompletions API.
type Model struct {
	client *openai.Client
	model  string
}

// Option configures the OpenAI Model.
type Option func(*config)

type config struct {
	apiKey       string
	baseURL      string
	organization string
	extraOpts    []option.RequestOption
}

// WithAPIKey sets the API key. If not set, the OPENAI_API_KEY env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithBaseURL sets a custom base URL for OpenAI-compatible providers.
//
// Examples:
//
//	openai.New("meta-llama/llama-3.1-405b", openai.WithBaseURL("https://openrouter.ai/api/v1"))
//	openai.New("deepseek-chat", openai.WithBaseURL("https://api.deepseek.com"))
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// WithOrganization sets the OpenAI organization ID.
func WithOrganization(orgID string) Option {
	return func(c *config) { c.organization = orgID }
}

// WithRequestOptions adds raw SDK request options (e.g., custom headers).
// This is used by wrapper providers like OpenRouter to inject additional options.
func WithRequestOptions(opts ...option.RequestOption) Option {
	return func(c *config) { c.extraOpts = append(c.extraOpts, opts...) }
}

// New creates a new OpenAI Model.
//
// Examples:
//
//	// OpenAI
//	model, err := openai.New("gpt-4", openai.WithAPIKey("sk-..."))
//
//	// OpenRouter
//	model, err := openai.New("meta-llama/llama-3.1-405b",
//	    openai.WithAPIKey("or-..."),
//	    openai.WithBaseURL("https://openrouter.ai/api/v1"),
//	)
func New(model string, opts ...Option) (*Model, error) {
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

	return &Model{
		client: &client,
		model:  model,
	}, nil
}

// MustNew is like New but panics on error.
func MustNew(model string, opts ...Option) *Model {
	m, err := New(model, opts...)
	if err != nil {
		panic(err)
	}
	return m
}

// Request implements core.Model.
func (m *Model) Request(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	params := m.buildParams(req)

	resp, err := m.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	return m.convertResponse(resp), nil
}

// RequestStream implements core.StreamModel.
func (m *Model) RequestStream(ctx context.Context, req *core.ChatRequest) (*core.StreamResult, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	params := m.buildParams(req)
	// Ask for a final usage chunk. Without stream_options.include_usage the
	// API sends no usage at all while streaming, and StreamEventDone would
	// report zeros.
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}
	stream := m.client.Chat.Completions.NewStreaming(ctx, params)

	ch := make(chan core.StreamEvent, 64)
	sr := core.NewStreamResult(ch)

	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		// Track tool calls by index for accumulating IDs
		type toolCallInfo struct {
			id   string
			name string
		}
		seenToolCalls := make(map[int64]*toolCallInfo)
		var usage core.Usage
		finishReason := core.FinishReasonStop
		refused := false

		for stream.Next() {
			chunk := stream.Current()

			// Extract usage from final chunk
			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				usage = extractOpenAIUsage(chunk.Usage)
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			if raw := chunk.Choices[0].FinishReason; raw != "" && !refused {
				finishReason = convertFinishReason(raw)
			}

			delta := chunk.Choices[0].Delta

			// Text content
			if delta.Content != "" {
				ch <- core.StreamEvent{
					Type:  core.StreamEventTextDelta,
					Delta: delta.Content,
				}
			}

			// A refusal arrives instead of content, and settles the finish
			// reason regardless of what the final chunk reports.
			if delta.Refusal != "" {
				refused = true
				finishReason = core.FinishReasonContentFilter
				ch <- core.StreamEvent{
					Type:  core.StreamEventTextDelta,
					Delta: delta.Refusal,
				}
			}

			// Tool calls
			for _, tc := range delta.ToolCalls {
				info, seen := seenToolCalls[tc.Index]
				if !seen {
					// First time seeing this tool call index — emit start
					info = &toolCallInfo{
						id:   tc.ID,
						name: tc.Function.Name,
					}
					seenToolCalls[tc.Index] = info

					ch <- core.StreamEvent{
						Type: core.StreamEventToolCallStart,
						ToolUse: &core.ToolUse{
							ID:   tc.ID,
							Name: tc.Function.Name,
						},
					}
				}

				// Emit argument delta if present
				if tc.Function.Arguments != "" {
					ch <- core.StreamEvent{
						Type:       core.StreamEventToolCallDelta,
						Delta:      tc.Function.Arguments,
						ToolCallID: info.id,
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- core.StreamEvent{
				Type:  core.StreamEventError,
				Error: fmt.Errorf("openai stream: %w", err),
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
func (m *Model) Name() string {
	return m.model
}

// buildParams converts core.ChatRequest to OpenAI params.
func (m *Model) buildParams(req *core.ChatRequest) openai.ChatCompletionNewParams {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages))
	for _, msg := range req.Messages {
		messages = append(messages, convertMessages(msg)...)
	}

	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    shared.ChatModel(m.model),
	}

	thinkingEnabled := req.Thinking != nil && req.Thinking.Enabled

	// Reasoning models reject sampling parameters while reasoning is active.
	if supportsSamplingParams(m.model, thinkingEnabled) {
		if req.Temperature != nil {
			params.Temperature = openai.Float(*req.Temperature)
		}
		if req.TopP != nil {
			params.TopP = openai.Float(*req.TopP)
		}
	}
	if req.MaxTokens != nil {
		params.MaxCompletionTokens = openai.Int(int64(*req.MaxTokens))
	}
	if len(req.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: req.StopSequences,
		}
	}

	opts := optionsFrom(req.ProviderOptions)
	if opts.Seed != nil {
		params.Seed = openai.Int(*opts.Seed)
	}
	if opts.ParallelToolCalls != nil {
		params.ParallelToolCalls = openai.Bool(*opts.ParallelToolCalls)
	}
	if tier := opts.serviceTier(); tier != "" {
		params.ServiceTier = openai.ChatCompletionNewParamsServiceTier(tier)
	}

	if len(req.Tools) > 0 {
		params.Tools = convertTools(req.Tools)
		if req.ToolChoice != nil {
			params.ToolChoice = convertToolChoice(*req.ToolChoice)
		}
	}

	if req.ResponseFormat != nil {
		params.ResponseFormat = convertResponseFormat(req.ResponseFormat)
	}

	if thinkingEnabled {
		// OpenAI o1/o3 models use reasoning_effort
		effort := shared.ReasoningEffortMedium
		if req.Thinking.BudgetTokens > 20000 {
			effort = shared.ReasoningEffortHigh
		} else if req.Thinking.BudgetTokens <= 5000 && req.Thinking.BudgetTokens > 0 {
			effort = shared.ReasoningEffortLow
		}
		params.ReasoningEffort = effort
	}

	return params
}

// convertResponseFormat converts core.ResponseFormat to OpenAI format.
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
				Schema: normalizeStrictSchema(rf.JSONSchema.Schema, rf.JSONSchema.Strict),
			}
			if rf.JSONSchema.Description != "" {
				schema.Description = openai.String(rf.JSONSchema.Description)
			}
			if rf.JSONSchema.Strict != nil {
				schema.Strict = openai.Bool(*rf.JSONSchema.Strict)
			}
			return openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
					JSONSchema: schema,
				},
			}
		}
		// Fallback to json_object if no schema provided
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{},
		}
	default:
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfText: &openai.ResponseFormatTextParam{},
		}
	}
}

// convertResponse converts OpenAI response to agentic types.
//
// The API returns one choice per request in every configuration this library
// issues, so only the first is converted; extra choices are dropped.
func (m *Model) convertResponse(resp *openai.ChatCompletion) *core.ChatResponse {
	out := &core.ChatResponse{
		ID:      resp.ID,
		Model:   string(resp.Model),
		Usage:   extractOpenAIUsage(resp.Usage),
		Created: time.Unix(resp.Created, 0),
	}
	if len(resp.Choices) == 0 {
		out.Message = core.Message{Role: core.RoleAssistant, Content: []core.Part{}}
		out.FinishReason = core.FinishReasonError
		return out
	}

	choice := resp.Choices[0]
	out.Message = convertResponseMessage(choice.Message)
	out.RawFinishReason = choice.FinishReason
	out.FinishReason = convertFinishReason(choice.FinishReason)

	// A refusal is returned instead of content, with finish_reason "stop".
	// Reporting a clean stop with an empty message hides the safety filter,
	// so surface the refusal text and the content-filter reason while
	// leaving RawFinishReason as the provider reported it.
	if choice.Message.Refusal != "" {
		out.Message.Content = append(out.Message.Content, core.Part{
			Type: core.ContentText,
			Text: choice.Message.Refusal,
		})
		out.FinishReason = core.FinishReasonContentFilter
	}

	return out
}

// extractOpenAIUsage extracts usage from an OpenAI response, including provider-specific fields.
func extractOpenAIUsage(u openai.CompletionUsage) core.Usage {
	usage := core.Usage{
		PromptTokens:     int(u.PromptTokens),
		CompletionTokens: int(u.CompletionTokens),
		TotalTokens:      int(u.TotalTokens),
	}
	// Extract reasoning tokens from completion_tokens_details (o1/o3 models)
	if u.CompletionTokensDetails.ReasoningTokens > 0 {
		usage.ReasoningTokens = int(u.CompletionTokensDetails.ReasoningTokens)
	}
	// Extract cached prompt tokens
	if u.PromptTokensDetails.CachedTokens > 0 {
		usage.CacheReadTokens = int(u.PromptTokensDetails.CachedTokens)
	}
	return usage
}

// convertMessages converts a core.Message into OpenAI messages.
//
// A tool message carrying several results becomes one OpenAI tool message per
// result: the API keys a result to its call by tool_call_id and accepts only
// one id per message, so the results cannot be merged. Every other role
// produces exactly one message.
func convertMessages(msg core.Message) []openai.ChatCompletionMessageParamUnion {
	switch msg.Role {
	case core.RoleSystem:
		return []openai.ChatCompletionMessageParamUnion{openai.SystemMessage(msg.GetTextContent())}

	case core.RoleUser:
		if len(msg.Content) == 1 && msg.Content[0].Type == core.ContentText {
			return []openai.ChatCompletionMessageParamUnion{openai.UserMessage(msg.Content[0].Text)}
		}
		parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(msg.Content))
		for _, part := range msg.Content {
			converted, ok := convertContentPart(part)
			if !ok {
				continue
			}
			parts = append(parts, converted)
		}
		return []openai.ChatCompletionMessageParamUnion{{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfArrayOfContentParts: parts,
				},
			},
		}}

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
			return []openai.ChatCompletionMessageParamUnion{{
				OfAssistant: &openai.ChatCompletionAssistantMessageParam{
					Content: openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: openai.String(msg.GetTextContent()),
					},
					ToolCalls: toolCalls,
				},
			}}
		}
		return []openai.ChatCompletionMessageParamUnion{openai.AssistantMessage(msg.GetTextContent())}

	case core.RoleTool:
		results := msg.GetToolResults()
		if len(results) == 0 {
			return []openai.ChatCompletionMessageParamUnion{openai.ToolMessage(msg.GetTextContent(), "unknown")}
		}
		out := make([]openai.ChatCompletionMessageParamUnion, len(results))
		for i, tr := range results {
			out[i] = openai.ToolMessage(tr.Content, tr.ToolUseID)
		}
		return out

	default:
		return []openai.ChatCompletionMessageParamUnion{openai.UserMessage(msg.GetTextContent())}
	}
}

// convertContentPart converts core.Part to an OpenAI content part.
//
// The second return value is false for parts OpenAI has no representation for
// (cache points, thinking blocks, unrecognized types); the caller must skip
// them. Emitting an empty text part in their place would send a blank content
// entry the model then has to account for.
func convertContentPart(part core.Part) (openai.ChatCompletionContentPartUnionParam, bool) {
	switch part.Type {
	case core.ContentText:
		return openai.TextContentPart(part.Text), true
	case core.ContentImageURL:
		if part.ImageURL != nil {
			imageURL := openai.ChatCompletionContentPartImageImageURLParam{
				URL: part.ImageURL.URL,
			}
			if part.ImageURL.Detail != "" {
				imageURL.Detail = part.ImageURL.Detail
			}
			return openai.ImageContentPart(imageURL), true
		}
	case core.ContentImageData:
		if part.ImageData != nil {
			dataURI := fmt.Sprintf("data:%s;base64,%s", part.ImageData.MediaType, part.ImageData.Data)
			imageURL := openai.ChatCompletionContentPartImageImageURLParam{
				URL: dataURI,
			}
			// Use vendor_metadata for detail level if present
			if part.ImageData.VendorMetadata != nil {
				if detail, ok := part.ImageData.VendorMetadata["detail"].(string); ok {
					imageURL.Detail = detail
				}
			}
			return openai.ImageContentPart(imageURL), true
		}
	case core.ContentAudioURL:
		if part.AudioURL != nil {
			return openai.InputAudioContentPart(openai.ChatCompletionContentPartInputAudioInputAudioParam{
				Data:   part.AudioURL.URL,
				Format: part.AudioURL.Format,
			}), true
		}
	case core.ContentUploadedFile:
		if part.UploadedFile != nil {
			return openai.FileContentPart(openai.ChatCompletionContentPartFileFileParam{
				FileID: openai.String(part.UploadedFile.FileID),
			}), true
		}
	case core.ContentCachePoint, core.ContentThinking:
		// OpenAI Chat Completions has no representation for either — skip.
		return openai.ChatCompletionContentPartUnionParam{}, false
	}
	// A part whose payload is missing still carries text often enough to be
	// worth sending; a part with neither is dropped.
	if part.Text != "" {
		return openai.TextContentPart(part.Text), true
	}
	return openai.ChatCompletionContentPartUnionParam{}, false
}

// convertTools converts core.Tool to OpenAI format.
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

// convertToolChoice converts core.ToolChoice to OpenAI format.
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

// convertResponseMessage converts OpenAI response message to core.Message.
func convertResponseMessage(msg openai.ChatCompletionMessage) core.Message {
	domainMsg := core.Message{
		Role:    core.MessageRole(msg.Role),
		Content: make([]core.Part, 0),
	}

	if msg.Content != "" {
		domainMsg.Content = append(domainMsg.Content, core.Part{
			Type: core.ContentText,
			Text: msg.Content,
		})
	}

	for _, tc := range msg.ToolCalls {
		var input map[string]interface{}
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
		domainMsg.Content = append(domainMsg.Content, core.Part{
			Type: core.ContentToolUse,
			ToolUse: &core.ToolUse{
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			},
		})
	}

	return domainMsg
}

// convertFinishReason converts an OpenAI finish reason to the agentic type.
//
// The Chat Completions API emits "stop", "length", "tool_calls",
// "content_filter", and the deprecated "function_call". An empty reason means
// the provider reported none, which is treated as a clean stop, matching
// pydantic-ai (pydantic_ai/models/openai.py: `if choice.finish_reason is None:
// choice.finish_reason = 'stop'`). Anything else is a value this library does
// not know and must not be reported as a success. "error" is not an OpenAI
// value but is emitted by OpenAI-compatible gateways this package also backs.
func convertFinishReason(reason string) core.FinishReason {
	switch reason {
	case "stop", "":
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

// Compile-time check that Model implements core.StreamModel (which embeds core.Model).
var _ core.StreamModel = (*Model)(nil)
