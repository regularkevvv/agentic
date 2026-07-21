// Package grok provides a Grok (xAI) Model implementation for Agentic.
// Grok models are served by the xAI API, which exposes an OpenAI-compatible
// endpoint at https://api.x.ai/v1.
//
// The wire format is OpenAI Chat Completions, but xAI differs from OpenAI
// itself in ways that matter and are handled here:
//
//   - Only some models accept the reasoning_effort parameter, and the accepted
//     values differ per family. Sending an unsupported one is a 400, not a
//     silently ignored field, so it is gated per model.
//   - Reasoning text is returned in a non-standard "reasoning" field.
//   - Generation failures are reported in band through xAI-specific finish
//     reasons ("cancelled", "failed") that have no OpenAI equivalent.
//
// Because of those differences this package speaks to the API directly rather
// than delegating to the OpenAI provider.
package grok

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

// DefaultBaseURL is the xAI API endpoint.
const DefaultBaseURL = "https://api.x.ai/v1"

// ProviderName identifies this provider in ChatRequest.ProviderOptions and on
// the thinking blocks and stream events it produces.
const ProviderName = "grok"

// ErrNoChoices is returned when xAI answers with a well-formed body that
// carries no choice.
var ErrNoChoices = errors.New("grok: response contained no choices")

// Options are the Grok-specific settings a caller may attach to a request
// through core.ChatRequest.ProviderOptions under the key ProviderName.
//
//	req.ProviderOptions = map[string]any{
//		"grok": grok.Options{ReasoningEffort: "high"},
//	}
type Options struct {
	// ReasoningEffort overrides the reasoning_effort derived from
	// ChatRequest.Thinking. Accepted values are "none", "low", "medium" and
	// "high"; a value the target model does not accept is clamped to the
	// nearest one it does, or dropped when it has no equivalent.
	ReasoningEffort string
}

// Model implements core.Model and core.StreamModel against the xAI API.
type Model struct {
	client *openai.Client
	model  string
}

// Option configures the Grok Model.
type Option func(*config)

type config struct {
	apiKey    string
	baseURL   string
	extraOpts []option.RequestOption
}

// WithAPIKey sets the API key. If not set, the GROK_API_KEY or XAI_API_KEY
// env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithBaseURL overrides the default xAI base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// WithRequestOptions adds raw SDK request options (custom headers, a custom
// HTTP client, middleware). Options are applied after the ones this package
// derives from the other Option values, so they win on any conflict.
func WithRequestOptions(opts ...option.RequestOption) Option {
	return func(c *config) { c.extraOpts = append(c.extraOpts, opts...) }
}

// New creates a new Grok Model.
//
// Examples:
//
//	model, err := grok.New("grok-4.3")
//	model, err := grok.New("grok-4.5", grok.WithAPIKey("xai-..."))
func New(model string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("GROK_API_KEY")
	}
	if apiKey == "" {
		apiKey = os.Getenv("XAI_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New("grok: API key not set (use WithAPIKey or set GROK_API_KEY / XAI_API_KEY)")
	}

	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	reqOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	reqOpts = append(reqOpts, cfg.extraOpts...)

	client := openai.NewClient(reqOpts...)

	return &Model{client: &client, model: model}, nil
}

// MustNew is like New but panics on error.
func MustNew(model string, opts ...Option) *Model {
	m, err := New(model, opts...)
	if err != nil {
		panic(err)
	}
	return m
}

// Name implements core.Model.
func (m *Model) Name() string {
	return m.model
}

// Request implements core.Model.
func (m *Model) Request(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	resp, err := m.client.Chat.Completions.New(ctx, m.buildParams(req))
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, ErrNoChoices
	}

	choice := resp.Choices[0]

	return &core.ChatResponse{
		ID:              resp.ID,
		Model:           string(resp.Model),
		Message:         convertResponseMessage(choice.Message),
		Usage:           extractUsage(resp.Usage),
		Created:         time.Unix(resp.Created, 0),
		FinishReason:    convertFinishReason(choice.FinishReason),
		RawFinishReason: choice.FinishReason,
	}, nil
}

// RequestStream implements core.StreamModel.
func (m *Model) RequestStream(ctx context.Context, req *core.ChatRequest) (*core.StreamResult, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	params := m.buildParams(req)
	// xAI reports usage on a streamed turn only when asked to, so without this
	// StreamEventDone would carry zeros.
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}

	stream := m.client.Chat.Completions.NewStreaming(ctx, params)

	ch := make(chan core.StreamEvent, 64)
	sr := core.NewStreamResult(ch)

	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		// Tool call IDs arrive only on the first chunk for a given index, so
		// later argument deltas have to be correlated back to it.
		seenToolCalls := make(map[int64]string)
		var usage core.Usage
		var rawFinish string

		for stream.Next() {
			chunk := stream.Current()

			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				usage = extractUsage(chunk.Usage)
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			if choice.FinishReason != "" {
				rawFinish = choice.FinishReason
			}

			// Reasoning text arrives in a non-standard delta field, ahead of
			// the answer it explains.
			if text := extractReasoning(choice.Delta.JSON.ExtraFields); text != "" {
				ch <- core.StreamEvent{
					Type:         core.StreamEventThinkingDelta,
					Delta:        text,
					ProviderName: ProviderName,
				}
			}

			if choice.Delta.Content != "" {
				ch <- core.StreamEvent{
					Type:  core.StreamEventTextDelta,
					Delta: choice.Delta.Content,
				}
			}

			for _, tc := range choice.Delta.ToolCalls {
				id, seen := seenToolCalls[tc.Index]
				if !seen {
					id = tc.ID
					seenToolCalls[tc.Index] = id
					ch <- core.StreamEvent{
						Type: core.StreamEventToolCallStart,
						ToolUse: &core.ToolUse{
							ID:   tc.ID,
							Name: tc.Function.Name,
						},
					}
				}

				if tc.Function.Arguments != "" {
					ch <- core.StreamEvent{
						Type:       core.StreamEventToolCallDelta,
						Delta:      tc.Function.Arguments,
						ToolCallID: id,
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- core.StreamEvent{
				Type:  core.StreamEventError,
				Error: fmt.Errorf("grok stream: %w", err),
			}
			return
		}

		ch <- core.StreamEvent{
			Type:         core.StreamEventDone,
			Usage:        &usage,
			FinishReason: convertFinishReason(rawFinish),
		}
	}()

	return sr, nil
}

// buildParams converts a core.ChatRequest into xAI chat completion params.
func (m *Model) buildParams(req *core.ChatRequest) openai.ChatCompletionNewParams {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages))
	for _, msg := range req.Messages {
		messages = append(messages, convertMessage(msg))
	}

	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    shared.ChatModel(m.model),
	}

	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if req.MaxTokens != nil {
		params.MaxCompletionTokens = openai.Int(int64(*req.MaxTokens))
	}
	if req.TopP != nil {
		params.TopP = openai.Float(*req.TopP)
	}
	if len(req.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: req.StopSequences,
		}
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

	if effort := m.resolveReasoningEffort(req); effort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(effort)
	}

	return params
}

// resolveReasoningEffort determines the reasoning_effort to send, preferring an
// explicit Options override over the request's ThinkingConfig. The result is
// always a value the target model accepts, or "" when it accepts none.
func (m *Model) resolveReasoningEffort(req *core.ChatRequest) string {
	if opts, ok := providerOptions(req); ok && opts.ReasoningEffort != "" {
		effort, keep := clampReasoningEffort(m.model, opts.ReasoningEffort)
		if !keep {
			return ""
		}
		return effort
	}
	return reasoningEffortFor(m.model, req.Thinking)
}

// providerOptions reads this provider's entry from ChatRequest.ProviderOptions.
// Entries belonging to other providers, and an entry of an unexpected type, are
// ignored rather than failing the request.
func providerOptions(req *core.ChatRequest) (Options, bool) {
	raw, ok := req.ProviderOptions[ProviderName]
	if !ok {
		return Options{}, false
	}
	switch opts := raw.(type) {
	case Options:
		return opts, true
	case *Options:
		if opts == nil {
			return Options{}, false
		}
		return *opts, true
	default:
		return Options{}, false
	}
}

// convertResponseMessage converts an xAI response message to a core.Message.
// A thinking part, when the model produced one, is placed ahead of the answer
// text it explains.
func convertResponseMessage(msg openai.ChatCompletionMessage) core.Message {
	out := core.Message{
		Role:    core.MessageRole(msg.Role),
		Content: make([]core.Part, 0, 2),
	}

	if text := extractReasoning(msg.JSON.ExtraFields); text != "" {
		out.Content = append(out.Content, core.Part{
			Type: core.ContentThinking,
			Thinking: &core.ThinkingBlock{
				Text:         text,
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

// extractUsage converts SDK usage into core.Usage, including the reasoning and
// cached-prompt token details xAI reports alongside the headline counts.
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

// convertMessage converts a core.Message to the xAI request format.
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

// convertContentPart converts a core.Part to an xAI content part.
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
			imageURL := openai.ChatCompletionContentPartImageImageURLParam{
				URL: dataURI,
			}
			if part.ImageData.VendorMetadata != nil {
				if detail, ok := part.ImageData.VendorMetadata["detail"].(string); ok {
					imageURL.Detail = detail
				}
			}
			return openai.ImageContentPart(imageURL)
		}
	case core.ContentCachePoint:
		// xAI caches prompts automatically and has no explicit cache marker.
		return openai.TextContentPart("")
	}
	return openai.TextContentPart(part.Text)
}

// convertTools converts core tools to the xAI request format.
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

// convertToolChoice converts a core.ToolChoice to the xAI request format.
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

// convertResponseFormat converts a core.ResponseFormat to the xAI request format.
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
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
					JSONSchema: schema,
				},
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

// Compile-time checks that Model implements both core interfaces.
var (
	_ core.Model       = (*Model)(nil)
	_ core.StreamModel = (*Model)(nil)
)
