// Package anthropic provides an Anthropic Model implementation for Agentic.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Model implements the core.Model and core.StreamModel interfaces
// using the Anthropic Messages API.
type Model struct {
	client *anthropic.Client
	model  string
}

// Option configures the Anthropic Model.
type Option func(*config)

type config struct {
	apiKey  string
	baseURL string
}

// WithAPIKey sets the API key. If not set, the ANTHROPIC_API_KEY env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithBaseURL sets a custom base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// New creates a new Anthropic Model.
//
// Example:
//
//	model, err := anthropic.New("claude-sonnet-4-6", anthropic.WithAPIKey("sk-ant-..."))
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

	client := anthropic.NewClient(reqOpts...)

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

	resp, err := m.client.Messages.New(ctx, params)
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
	stream := m.client.Messages.NewStreaming(ctx, params)

	ch := make(chan core.StreamEvent, 64)
	sr := core.NewStreamResult(ch)

	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		var usage core.Usage

		// Track current content block index to tool call ID mapping
		type blockInfo struct {
			id   string
			kind string // "text", "tool_use", or "thinking"
		}
		blocks := make(map[int64]*blockInfo)

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "message_start":
				// Extract input token usage from the initial message
				if event.Message.Usage.InputTokens > 0 {
					usage.PromptTokens = int(event.Message.Usage.InputTokens)
				}
				if event.Message.Usage.CacheReadInputTokens > 0 {
					usage.CacheReadTokens = int(event.Message.Usage.CacheReadInputTokens)
				}
				if event.Message.Usage.CacheCreationInputTokens > 0 {
					usage.CacheCreationTokens = int(event.Message.Usage.CacheCreationInputTokens)
				}

			case "content_block_start":
				block := event.ContentBlock
				info := &blockInfo{kind: block.Type}

				switch block.Type {
				case "tool_use":
					info.id = block.ID
					blocks[event.Index] = info
					ch <- core.StreamEvent{
						Type: core.StreamEventToolCallStart,
						ToolUse: &core.ToolUse{
							ID:   block.ID,
							Name: block.Name,
						},
					}
				case "thinking":
					blocks[event.Index] = info
				default:
					blocks[event.Index] = info
				}

			case "content_block_delta":
				delta := event.Delta

				switch delta.Type {
				case "thinking_delta":
					if delta.Thinking != "" {
						ch <- core.StreamEvent{
							Type:  core.StreamEventThinkingDelta,
							Delta: delta.Thinking,
						}
					}

				case "text_delta":
					if delta.Text != "" {
						ch <- core.StreamEvent{
							Type:  core.StreamEventTextDelta,
							Delta: delta.Text,
						}
					}

				case "input_json_delta":
					if delta.PartialJSON != "" {
						toolCallID := ""
						if info, ok := blocks[event.Index]; ok {
							toolCallID = info.id
						}
						ch <- core.StreamEvent{
							Type:       core.StreamEventToolCallDelta,
							Delta:      delta.PartialJSON,
							ToolCallID: toolCallID,
						}
					}
				}

			case "message_delta":
				// Extract output token usage
				if event.Usage.OutputTokens > 0 {
					usage.CompletionTokens = int(event.Usage.OutputTokens)
					usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- core.StreamEvent{
				Type:  core.StreamEventError,
				Error: fmt.Errorf("anthropic stream: %w", err),
			}
			return
		}

		ch <- core.StreamEvent{Type: core.StreamEventDone, Usage: &usage}
	}()

	return sr, nil
}

// Name implements core.Model.
func (m *Model) Name() string {
	return m.model
}

// buildParams converts core.ChatRequest to Anthropic params.
func (m *Model) buildParams(req *core.ChatRequest) anthropic.MessageNewParams {
	systemMsgs, convMsgs := separateSystemMessages(req.Messages)

	messages := make([]anthropic.MessageParam, 0, len(convMsgs))
	for _, msg := range convMsgs {
		messages = append(messages, convertMessage(msg))
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(m.model),
		Messages:  messages,
		MaxTokens: 1024, // Required by Anthropic
	}

	if req.MaxTokens != nil {
		params.MaxTokens = int64(*req.MaxTokens)
	}

	if len(systemMsgs) > 0 {
		blocks := make([]anthropic.TextBlockParam, 0, len(systemMsgs))
		for _, sysMsg := range systemMsgs {
			blocks = append(blocks, anthropic.TextBlockParam{
				Text: sysMsg.GetTextContent(),
			})
		}
		params.System = blocks
	}

	if req.Temperature != nil {
		params.Temperature = anthropic.Float(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = anthropic.Float(*req.TopP)
	}

	if len(req.Tools) > 0 {
		params.Tools = convertTools(req.Tools)
		if req.ToolChoice != nil {
			params.ToolChoice = convertToolChoice(*req.ToolChoice)
		}
	}

	if req.ResponseFormat != nil {
		params.OutputConfig = convertResponseFormat(req.ResponseFormat)
	}

	if req.Thinking != nil && req.Thinking.Enabled {
		budget := int64(req.Thinking.BudgetTokens)
		if budget <= 0 {
			budget = 10000 // sensible default
		}
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{
				BudgetTokens: budget,
			},
		}
		// Anthropic requires temperature=1 when thinking is enabled
		params.Temperature = anthropic.Float(1)
	}

	return params
}

// convertResponseFormat converts core.ResponseFormat to Anthropic OutputConfigParam.
// Anthropic supports native JSON schema output via output_config.format.
// Note: "json_object" mode is not supported by Anthropic — it's ignored.
func convertResponseFormat(rf *core.ResponseFormat) anthropic.OutputConfigParam {
	if rf.Type == "json_schema" && rf.JSONSchema != nil {
		return anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{
				Schema: rf.JSONSchema.Schema,
			},
		}
	}
	// Anthropic doesn't support json_object mode; return empty config
	return anthropic.OutputConfigParam{}
}

// convertResponse converts Anthropic response to agentic types.
func (m *Model) convertResponse(resp *anthropic.Message) *core.ChatResponse {
	message := convertResponseMessage(resp.Content, string(resp.Role))

	return &core.ChatResponse{
		ID:    resp.ID,
		Model: m.model,
		Choices: []core.Choice{
			{
				Index:        0,
				Message:      message,
				FinishReason: convertFinishReason(resp.StopReason),
			},
		},
		Usage:   extractAnthropicUsage(resp.Usage),
		Created: time.Now(),
	}
}

// extractAnthropicUsage extracts usage from an Anthropic response, including cache tokens.
func extractAnthropicUsage(u anthropic.Usage) core.Usage {
	usage := core.Usage{
		PromptTokens:        int(u.InputTokens),
		CompletionTokens:    int(u.OutputTokens),
		TotalTokens:         int(u.InputTokens + u.OutputTokens),
		CacheReadTokens:     int(u.CacheReadInputTokens),
		CacheCreationTokens: int(u.CacheCreationInputTokens),
	}
	return usage
}

// convertMessage converts core.Message to Anthropic message format.
func convertMessage(msg core.Message) anthropic.MessageParam {
	role := anthropic.MessageParamRole(msg.Role)
	if msg.Role == core.RoleTool {
		role = anthropic.MessageParamRoleUser
	}

	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(msg.Content))
	for _, part := range msg.Content {
		switch part.Type {
		case core.ContentThinking:
			if part.Thinking != nil {
				if part.Thinking.IsRedacted() {
					// Redacted thinking: send back as redacted_thinking block
					blocks = append(blocks, anthropic.ContentBlockParamUnion{
						OfRedactedThinking: &anthropic.RedactedThinkingBlockParam{
							Data: part.Thinking.Signature,
						},
					})
				} else {
					blocks = append(blocks, anthropic.ContentBlockParamUnion{
						OfThinking: &anthropic.ThinkingBlockParam{
							Thinking:  part.Thinking.Text,
							Signature: part.Thinking.Signature,
						},
					})
				}
			}
		case core.ContentCachePoint:
			// CachePoint: attach cache_control to the previous block.
			// The cache point marks that all content up to the previous block should be cached.
			if part.CachePoint != nil && len(blocks) > 0 {
				ttl := anthropic.CacheControlEphemeralTTLTTL5m
				if part.CachePoint.TTL == "1h" {
					ttl = anthropic.CacheControlEphemeralTTLTTL1h
				}
				if cc := blocks[len(blocks)-1].GetCacheControl(); cc != nil {
					*cc = anthropic.CacheControlEphemeralParam{
						Type: "ephemeral",
						TTL:  ttl,
					}
				}
			}
			// Don't append the cache point itself as a block
			continue
		case core.ContentUploadedFile:
			// UploadedFile is not yet supported in the non-beta Anthropic API.
			// Skip silently — callers should use the Beta API for file uploads.
		case core.ContentText:
			if part.Text != "" {
				blocks = append(blocks, anthropic.NewTextBlock(part.Text))
			}
		case core.ContentImageData:
			if part.ImageData != nil {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfImage: &anthropic.ImageBlockParam{
						Source: anthropic.ImageBlockParamSourceUnion{
							OfBase64: &anthropic.Base64ImageSourceParam{
								Data:      part.ImageData.Data,
								MediaType: anthropic.Base64ImageSourceMediaType(part.ImageData.MediaType),
							},
						},
					},
				})
			}
		case core.ContentImageURL:
			if part.ImageURL != nil {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfImage: &anthropic.ImageBlockParam{
						Source: anthropic.ImageBlockParamSourceUnion{
							OfURL: &anthropic.URLImageSourceParam{
								URL: part.ImageURL.URL,
							},
						},
					},
				})
			}
		case core.ContentDocumentURL:
			if part.DocumentURL != nil {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfDocument: &anthropic.DocumentBlockParam{
						Source: anthropic.DocumentBlockParamSourceUnion{
							OfURL: &anthropic.URLPDFSourceParam{
								URL: part.DocumentURL.URL,
							},
						},
					},
				})
			}
		case core.ContentToolUse:
			if part.ToolUse != nil {
				blocks = append(blocks, anthropic.NewToolUseBlock(
					part.ToolUse.ID,
					part.ToolUse.Input,
					part.ToolUse.Name,
				))
			}
		case core.ContentToolResult:
			if part.ToolResult != nil {
				blocks = append(blocks, anthropic.NewToolResultBlock(
					part.ToolResult.ToolUseID,
					part.ToolResult.Content,
					part.ToolResult.IsError,
				))
			}
		}
	}

	return anthropic.MessageParam{
		Role:    role,
		Content: blocks,
	}
}

// convertTools converts core.Tool to Anthropic format.
func convertTools(tools []core.Tool) []anthropic.ToolUnionParam {
	result := make([]anthropic.ToolUnionParam, len(tools))
	for i, tool := range tools {
		params := tool.Function.Parameters
		if params == nil {
			params = make(map[string]interface{})
		}

		var properties interface{}
		var required []string

		if props, ok := params["properties"]; ok {
			properties = props
		} else {
			properties = params
		}

		if req, ok := params["required"].([]interface{}); ok {
			required = make([]string, len(req))
			for j, r := range req {
				if s, ok := r.(string); ok {
					required[j] = s
				}
			}
		} else if req, ok := params["required"].([]string); ok {
			required = req
		}

		inputSchema := anthropic.ToolInputSchemaParam{
			Type:       "object",
			Properties: properties,
			Required:   required,
		}
		inputSchema.ExtraFields = map[string]any{
			"additionalProperties": false,
		}

		result[i] = anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Function.Name,
				Description: anthropic.String(tool.Function.Description),
				InputSchema: inputSchema,
			},
		}
	}
	return result
}

// convertToolChoice converts core.ToolChoice to Anthropic format.
func convertToolChoice(choice core.ToolChoice) anthropic.ToolChoiceUnionParam {
	switch choice {
	case core.ToolChoiceRequired:
		return anthropic.ToolChoiceUnionParam{
			OfAny: &anthropic.ToolChoiceAnyParam{},
		}
	default:
		return anthropic.ToolChoiceUnionParam{
			OfAuto: &anthropic.ToolChoiceAutoParam{},
		}
	}
}

// convertResponseMessage converts Anthropic response content to core.Message.
func convertResponseMessage(content []anthropic.ContentBlockUnion, role string) core.Message {
	msg := core.Message{
		Role:    core.MessageRole(role),
		Content: make([]core.Part, 0, len(content)),
	}

	for _, block := range content {
		switch block.Type {
		case "thinking":
			thinkingBlock := block.AsThinking()
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentThinking,
				Thinking: &core.ThinkingBlock{
					Text:         thinkingBlock.Thinking,
					Signature:    thinkingBlock.Signature,
					ProviderName: "anthropic",
				},
			})
		case "redacted_thinking":
			redacted := block.AsRedactedThinking()
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentThinking,
				Thinking: &core.ThinkingBlock{
					Text:         "",
					ID:           "redacted_thinking",
					Signature:    redacted.Data,
					ProviderName: "anthropic",
				},
			})
		case "text":
			textBlock := block.AsText()
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentText,
				Text: textBlock.Text,
			})
		case "tool_use":
			toolUseBlock := block.AsToolUse()
			var input map[string]interface{}
			_ = json.Unmarshal(toolUseBlock.Input, &input)
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentToolUse,
				ToolUse: &core.ToolUse{
					ID:    toolUseBlock.ID,
					Name:  toolUseBlock.Name,
					Input: input,
				},
			})
		}
	}

	return msg
}

// convertFinishReason converts Anthropic stop reason to agentic type.
func convertFinishReason(reason anthropic.StopReason) core.FinishReason {
	switch reason {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return core.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return core.FinishReasonLength
	case anthropic.StopReasonToolUse:
		return core.FinishReasonToolCalls
	default:
		return core.FinishReasonStop
	}
}

// separateSystemMessages splits system messages from conversation messages.
// Anthropic requires system messages to be passed separately.
func separateSystemMessages(messages []core.Message) ([]core.Message, []core.Message) {
	var system, conversation []core.Message
	for _, msg := range messages {
		if msg.Role == core.RoleSystem {
			system = append(system, msg)
		} else {
			conversation = append(conversation, msg)
		}
	}
	return system, conversation
}

// Compile-time check that Model implements core.StreamModel (which embeds core.Model).
var _ core.StreamModel = (*Model)(nil)
