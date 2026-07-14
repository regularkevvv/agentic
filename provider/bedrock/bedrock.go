// Package bedrock provides an AWS Bedrock Model implementation for Agentic.
// It uses the Bedrock Runtime Converse API, which provides a unified interface
// for all foundation models on Bedrock (Anthropic Claude, Meta Llama, Mistral, etc.).
//
// Features: streaming, tool/function calling, multimodal inputs (images, documents),
// thinking/reasoning tokens, and usage tracking with cache token support.
package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// Model implements the core.Model and core.StreamModel interfaces
// using the AWS Bedrock Runtime Converse API.
type Model struct {
	client  runtimeClient
	modelID string
}

type runtimeClient interface {
	Converse(ctx context.Context, params *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(ctx context.Context, params *bedrockruntime.ConverseStreamInput, optFns ...func(*bedrockruntime.Options)) (converseStream, error)
}

type converseStream interface {
	Events() <-chan types.ConverseStreamOutput
	Close() error
	Err() error
}

type sdkRuntimeClient struct {
	client *bedrockruntime.Client
}

func (c *sdkRuntimeClient) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	return c.client.Converse(ctx, params, optFns...)
}

func (c *sdkRuntimeClient) ConverseStream(
	ctx context.Context,
	params *bedrockruntime.ConverseStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (converseStream, error) {
	resp, err := c.client.ConverseStream(ctx, params, optFns...)
	if err != nil {
		return nil, err
	}
	return resp.GetStream(), nil
}

// Option configures the Bedrock Model.
type Option func(*config)

type config struct {
	region          string
	profile         string
	accessKeyID     string
	secretAccessKey string
	sessionToken    string
	client          runtimeClient
}

// WithRegion sets the AWS region. If not set, the AWS_DEFAULT_REGION env var is used.
func WithRegion(region string) Option {
	return func(c *config) { c.region = region }
}

// WithProfile sets the AWS credentials profile name.
func WithProfile(profile string) Option {
	return func(c *config) { c.profile = profile }
}

// WithCredentials sets explicit AWS credentials.
func WithCredentials(accessKeyID, secretAccessKey, sessionToken string) Option {
	return func(c *config) {
		c.accessKeyID = accessKeyID
		c.secretAccessKey = secretAccessKey
		c.sessionToken = sessionToken
	}
}

// WithClient sets a pre-configured Bedrock Runtime client.
// When set, all other connection options are ignored.
func WithClient(client *bedrockruntime.Client) Option {
	return func(c *config) { c.client = &sdkRuntimeClient{client: client} }
}

// New creates a new Bedrock Model.
//
// The modelID should be a Bedrock model ID such as:
//   - "anthropic.claude-sonnet-4-6"
//   - "meta.llama3-1-70b-instruct-v1:0"
//   - "mistral.mistral-large-2402-v1:0"
//
// Examples:
//
//	model, err := bedrock.New("anthropic.claude-sonnet-4-6",
//	    bedrock.WithRegion("us-east-1"),
//	)
//
//	model, err := bedrock.New("anthropic.claude-sonnet-4-6",
//	    bedrock.WithRegion("us-east-1"),
//	    bedrock.WithProfile("my-profile"),
//	)
func New(modelID string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	if cfg.client != nil {
		return &Model{client: cfg.client, modelID: modelID}, nil
	}

	region := cfg.region
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		region = os.Getenv("AWS_REGION")
	}
	if region == "" {
		return nil, fmt.Errorf("bedrock: region not set (use WithRegion or set AWS_DEFAULT_REGION)")
	}

	var awsOpts []func(*awsconfig.LoadOptions) error
	awsOpts = append(awsOpts, awsconfig.WithRegion(region))

	if cfg.profile != "" {
		awsOpts = append(awsOpts, awsconfig.WithSharedConfigProfile(cfg.profile))
	}

	if cfg.accessKeyID != "" {
		awsOpts = append(awsOpts, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(cfg.accessKeyID, cfg.secretAccessKey, cfg.sessionToken),
		))
	}

	awsCfg, err := awsconfig.LoadDefaultConfig(context.Background(), awsOpts...)
	if err != nil {
		return nil, fmt.Errorf("bedrock: failed to load AWS config: %w", err)
	}

	client := bedrockruntime.NewFromConfig(awsCfg)

	return &Model{
		client:  &sdkRuntimeClient{client: client},
		modelID: modelID,
	}, nil
}

// MustNew is like New but panics on error.
func MustNew(modelID string, opts ...Option) *Model {
	m, err := New(modelID, opts...)
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

	input := m.buildInput(req)

	resp, err := m.client.Converse(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("bedrock: %w", err)
	}

	return m.convertResponse(resp), nil
}

// RequestStream implements core.StreamModel.
func (m *Model) RequestStream(ctx context.Context, req *core.ChatRequest) (*core.StreamResult, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	streamInput := m.buildStreamInput(req)

	stream, err := m.client.ConverseStream(ctx, streamInput)
	if err != nil {
		return nil, fmt.Errorf("bedrock: %w", err)
	}

	ch := make(chan core.StreamEvent, 64)
	sr := core.NewStreamResult(ch)

	go func() {
		defer close(ch)

		defer func() { _ = stream.Close() }()

		var usage core.Usage
		// Track active tool calls by content block index
		toolCalls := make(map[int32]string) // index -> toolUseID

		for event := range stream.Events() {
			switch v := event.(type) {
			case *types.ConverseStreamOutputMemberContentBlockStart:
				if v.Value.Start != nil {
					if tu, ok := v.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
						idx := aws.ToInt32(v.Value.ContentBlockIndex)
						toolUseID := aws.ToString(tu.Value.ToolUseId)
						toolCalls[idx] = toolUseID
						ch <- core.StreamEvent{
							Type: core.StreamEventToolCallStart,
							ToolUse: &core.ToolUse{
								ID:   toolUseID,
								Name: aws.ToString(tu.Value.Name),
							},
						}
					}
				}

			case *types.ConverseStreamOutputMemberContentBlockDelta:
				if v.Value.Delta != nil {
					switch d := v.Value.Delta.(type) {
					case *types.ContentBlockDeltaMemberText:
						ch <- core.StreamEvent{
							Type:  core.StreamEventTextDelta,
							Delta: d.Value,
						}
					case *types.ContentBlockDeltaMemberToolUse:
						idx := aws.ToInt32(v.Value.ContentBlockIndex)
						toolUseID := toolCalls[idx]
						ch <- core.StreamEvent{
							Type:       core.StreamEventToolCallDelta,
							Delta:      aws.ToString(d.Value.Input),
							ToolCallID: toolUseID,
						}
					case *types.ContentBlockDeltaMemberReasoningContent:
						if textDelta, ok := d.Value.(*types.ReasoningContentBlockDeltaMemberText); ok {
							ch <- core.StreamEvent{
								Type:  core.StreamEventThinkingDelta,
								Delta: textDelta.Value,
							}
						}
					}
				}

			case *types.ConverseStreamOutputMemberMetadata:
				if v.Value.Usage != nil {
					usage = extractUsage(v.Value.Usage)
				}

			case *types.ConverseStreamOutputMemberMessageStop:
				// Message complete — done event sent after loop

			case *types.UnknownUnionMember:
				// Ignore unknown event types
			}
		}

		if err := stream.Err(); err != nil {
			ch <- core.StreamEvent{
				Type:  core.StreamEventError,
				Error: fmt.Errorf("bedrock stream: %w", err),
			}
			return
		}

		ch <- core.StreamEvent{Type: core.StreamEventDone, Usage: &usage}
	}()

	return sr, nil
}

// Name implements core.Model.
func (m *Model) Name() string {
	return m.modelID
}

// converseParams holds the shared parameters for Converse and ConverseStream.
type converseParams struct {
	messages      []types.Message
	system        []types.SystemContentBlock
	inferenceConf *types.InferenceConfiguration
	toolConfig    *types.ToolConfiguration
	additionalReq document.Interface
}

// buildParams extracts the shared request parameters from core.ChatRequest.
func (m *Model) buildParams(req *core.ChatRequest) converseParams {
	var p converseParams

	// Convert messages, extracting system prompt
	for _, msg := range req.Messages {
		if msg.Role == core.RoleSystem {
			p.system = convertSystemBlocks(msg)
			continue
		}
		if converted := convertMessage(msg); converted != nil {
			p.messages = append(p.messages, *converted)
		}
	}

	// Inference configuration
	infCfg := &types.InferenceConfiguration{}
	hasInfCfg := false

	if req.Temperature != nil {
		t := float32(*req.Temperature)
		infCfg.Temperature = &t
		hasInfCfg = true
	}
	if req.MaxTokens != nil {
		mt := int32(*req.MaxTokens)
		infCfg.MaxTokens = &mt
		hasInfCfg = true
	}
	if req.TopP != nil {
		tp := float32(*req.TopP)
		infCfg.TopP = &tp
		hasInfCfg = true
	}

	if hasInfCfg {
		p.inferenceConf = infCfg
	}

	// Tools
	if len(req.Tools) > 0 {
		p.toolConfig = convertToolConfig(req.Tools, req.ToolChoice)
	}

	// Thinking (via AdditionalModelRequestFields for Anthropic models)
	if req.Thinking != nil && req.Thinking.Enabled {
		thinking := map[string]interface{}{
			"thinking": map[string]interface{}{
				"type":          "enabled",
				"budget_tokens": req.Thinking.BudgetTokens,
			},
		}
		p.additionalReq = document.NewLazyDocument(thinking)
	}

	return p
}

// buildInput converts core.ChatRequest to Bedrock ConverseInput.
func (m *Model) buildInput(req *core.ChatRequest) *bedrockruntime.ConverseInput {
	p := m.buildParams(req)
	return &bedrockruntime.ConverseInput{
		ModelId:                      aws.String(m.modelID),
		Messages:                     p.messages,
		System:                       p.system,
		InferenceConfig:              p.inferenceConf,
		ToolConfig:                   p.toolConfig,
		AdditionalModelRequestFields: p.additionalReq,
	}
}

// buildStreamInput converts core.ChatRequest to Bedrock ConverseStreamInput.
func (m *Model) buildStreamInput(req *core.ChatRequest) *bedrockruntime.ConverseStreamInput {
	p := m.buildParams(req)
	return &bedrockruntime.ConverseStreamInput{
		ModelId:                      aws.String(m.modelID),
		Messages:                     p.messages,
		System:                       p.system,
		InferenceConfig:              p.inferenceConf,
		ToolConfig:                   p.toolConfig,
		AdditionalModelRequestFields: p.additionalReq,
	}
}

// convertSystemBlocks converts a system message to Bedrock system content blocks.
func convertSystemBlocks(msg core.Message) []types.SystemContentBlock {
	text := msg.GetTextContent()
	if text == "" {
		return nil
	}
	return []types.SystemContentBlock{
		&types.SystemContentBlockMemberText{Value: text},
	}
}

// convertMessage converts core.Message to Bedrock Message.
func convertMessage(msg core.Message) *types.Message {
	var blocks []types.ContentBlock

	for _, p := range msg.Content {
		switch p.Type {
		case core.ContentText:
			if p.Text != "" {
				blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Text})
			}

		case core.ContentToolUse:
			if p.ToolUse != nil {
				inputDoc := document.NewLazyDocument(p.ToolUse.Input)
				blocks = append(blocks, &types.ContentBlockMemberToolUse{
					Value: types.ToolUseBlock{
						ToolUseId: aws.String(p.ToolUse.ID),
						Name:      aws.String(p.ToolUse.Name),
						Input:     inputDoc,
					},
				})
			}

		case core.ContentToolResult:
			if p.ToolResult != nil {
				status := types.ToolResultStatusSuccess
				if p.ToolResult.IsError {
					status = types.ToolResultStatusError
				}
				blocks = append(blocks, &types.ContentBlockMemberToolResult{
					Value: types.ToolResultBlock{
						ToolUseId: aws.String(p.ToolResult.ToolUseID),
						Status:    status,
						Content: []types.ToolResultContentBlock{
							&types.ToolResultContentBlockMemberText{Value: p.ToolResult.Content},
						},
					},
				})
			}

		case core.ContentImageURL:
			// Bedrock requires inline image data; URL-based images need pre-fetching
			// Skip for now — users should use ImageData instead
			continue

		case core.ContentImageData:
			if p.ImageData != nil {
				data, err := base64.StdEncoding.DecodeString(p.ImageData.Data)
				if err == nil {
					format := imageFormat(p.ImageData.MediaType)
					blocks = append(blocks, &types.ContentBlockMemberImage{
						Value: types.ImageBlock{
							Source: &types.ImageSourceMemberBytes{Value: data},
							Format: format,
						},
					})
				}
			}

		case core.ContentDocumentURL:
			// Bedrock document blocks require inline bytes; skip URL-based
			continue

		case core.ContentThinking:
			// Pass through thinking blocks for multi-turn with thinking
			if p.Thinking != nil && p.Thinking.Text != "" {
				blocks = append(blocks, &types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberReasoningText{
						Value: types.ReasoningTextBlock{
							Text:      aws.String(p.Thinking.Text),
							Signature: aws.String(p.Thinking.Signature),
						},
					},
				})
			}

		case core.ContentAudioURL, core.ContentVideoURL, core.ContentCachePoint, core.ContentUploadedFile:
			// Not supported by Bedrock Converse API — skip
			continue
		}
	}

	if len(blocks) == 0 {
		return nil
	}

	role := convertRole(msg.Role)
	return &types.Message{
		Role:    role,
		Content: blocks,
	}
}

// convertRole maps agentic role to Bedrock role.
func convertRole(role core.MessageRole) types.ConversationRole {
	switch role {
	case core.RoleAssistant:
		return types.ConversationRoleAssistant
	default:
		return types.ConversationRoleUser
	}
}

// convertToolConfig converts agentic tools to Bedrock ToolConfiguration.
func convertToolConfig(tools []core.Tool, choice *core.ToolChoice) *types.ToolConfiguration {
	tc := &types.ToolConfiguration{}

	for _, tool := range tools {
		spec := types.ToolSpecification{
			Name:        aws.String(tool.Function.Name),
			Description: aws.String(tool.Function.Description),
		}
		if tool.Function.Parameters != nil {
			spec.InputSchema = &types.ToolInputSchemaMemberJson{
				Value: document.NewLazyDocument(tool.Function.Parameters),
			}
		}
		tc.Tools = append(tc.Tools, &types.ToolMemberToolSpec{Value: spec})
	}

	if choice != nil {
		switch *choice {
		case core.ToolChoiceNone:
			// Bedrock doesn't have a "none" — omit tool config
			return nil
		case core.ToolChoiceRequired:
			tc.ToolChoice = &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}
		default:
			tc.ToolChoice = &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}
		}
	}

	return tc
}

// convertResponse converts Bedrock ConverseOutput to agentic ChatResponse.
func (m *Model) convertResponse(resp *bedrockruntime.ConverseOutput) *core.ChatResponse {
	chatResp := &core.ChatResponse{
		Model:   m.modelID,
		Created: time.Now(),
	}

	msg := convertOutputMessage(resp.Output)
	chatResp.Choices = []core.Choice{{
		Index:        0,
		Message:      msg,
		FinishReason: convertStopReason(resp.StopReason),
	}}

	if resp.Usage != nil {
		chatResp.Usage = extractUsage(resp.Usage)
	}

	return chatResp
}

// convertOutputMessage converts Bedrock output to agentic Message.
func convertOutputMessage(output types.ConverseOutput) core.Message {
	msg := core.Message{
		Role:    core.RoleAssistant,
		Content: make([]core.Part, 0),
	}

	msgOutput, ok := output.(*types.ConverseOutputMemberMessage)
	if !ok || msgOutput == nil {
		return msg
	}

	for _, block := range msgOutput.Value.Content {
		switch b := block.(type) {
		case *types.ContentBlockMemberText:
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentText,
				Text: b.Value,
			})

		case *types.ContentBlockMemberToolUse:
			var input map[string]interface{}
			if b.Value.Input != nil {
				inputBytes, _ := json.Marshal(b.Value.Input)
				_ = json.Unmarshal(inputBytes, &input)
			}
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentToolUse,
				ToolUse: &core.ToolUse{
					ID:    aws.ToString(b.Value.ToolUseId),
					Name:  aws.ToString(b.Value.Name),
					Input: input,
				},
			})

		case *types.ContentBlockMemberReasoningContent:
			if b.Value != nil {
				if rt, ok := b.Value.(*types.ReasoningContentBlockMemberReasoningText); ok {
					msg.Content = append(msg.Content, core.Part{
						Type: core.ContentThinking,
						Thinking: &core.ThinkingBlock{
							Text:         aws.ToString(rt.Value.Text),
							Signature:    aws.ToString(rt.Value.Signature),
							ProviderName: "bedrock",
						},
					})
				}
			}
		}
	}

	return msg
}

// convertStopReason converts Bedrock stop reason to agentic FinishReason.
func convertStopReason(reason types.StopReason) core.FinishReason {
	switch reason {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return core.FinishReasonStop
	case types.StopReasonMaxTokens:
		return core.FinishReasonLength
	case types.StopReasonToolUse:
		return core.FinishReasonToolCalls
	case types.StopReasonContentFiltered, types.StopReasonGuardrailIntervened:
		return core.FinishReasonContentFilter
	default:
		return core.FinishReasonStop
	}
}

// extractUsage converts Bedrock TokenUsage to agentic Usage.
func extractUsage(tu *types.TokenUsage) core.Usage {
	usage := core.Usage{
		PromptTokens:     int(aws.ToInt32(tu.InputTokens)),
		CompletionTokens: int(aws.ToInt32(tu.OutputTokens)),
		TotalTokens:      int(aws.ToInt32(tu.TotalTokens)),
	}
	if tu.CacheReadInputTokens != nil {
		usage.CacheReadTokens = int(aws.ToInt32(tu.CacheReadInputTokens))
	}
	if tu.CacheWriteInputTokens != nil {
		usage.CacheCreationTokens = int(aws.ToInt32(tu.CacheWriteInputTokens))
	}
	return usage
}

// imageFormat maps MIME type to Bedrock ImageFormat.
func imageFormat(mediaType string) types.ImageFormat {
	switch mediaType {
	case "image/png":
		return types.ImageFormatPng
	case "image/gif":
		return types.ImageFormatGif
	case "image/webp":
		return types.ImageFormatWebp
	default:
		return types.ImageFormatJpeg
	}
}

// Compile-time check that Model implements core.StreamModel (which embeds core.Model).
var _ core.StreamModel = (*Model)(nil)
