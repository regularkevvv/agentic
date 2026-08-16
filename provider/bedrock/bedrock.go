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
	"strings"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsmiddleware "github.com/aws/aws-sdk-go-v2/aws/middleware"
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
	// rawClient holds the same client as the client field, undecorated.
	// Embedder calls InvokeModel, which the Converse-shaped runtimeClient
	// interface does not carry.
	rawClient *bedrockruntime.Client
	// embedConcurrency and titanNormalize are read only by Embedder.
	embedConcurrency int
	titanNormalize   *bool
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
	return func(c *config) {
		c.client = &sdkRuntimeClient{client: client}
		c.rawClient = client
	}
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

	awsCfg, err := resolveAWSConfig(cfg)
	if err != nil {
		return nil, err
	}

	client := bedrockruntime.NewFromConfig(awsCfg)

	return &Model{
		client:  &sdkRuntimeClient{client: client},
		modelID: modelID,
	}, nil
}

// resolveAWSConfig builds the AWS config the Bedrock Runtime client is
// constructed from, applying the region, profile and credential options in
// order. It is shared by New and NewEmbedder so both resolve credentials
// identically.
func resolveAWSConfig(cfg *config) (aws.Config, error) {
	region := cfg.region
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		region = os.Getenv("AWS_REGION")
	}
	if region == "" {
		return aws.Config{}, fmt.Errorf("bedrock: region not set (use WithRegion or set AWS_DEFAULT_REGION)")
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
		return aws.Config{}, fmt.Errorf("bedrock: failed to load AWS config: %w", err)
	}
	return awsCfg, nil
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

	input, err := m.buildInput(req)
	if err != nil {
		return nil, err
	}

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

	streamInput, err := m.buildStreamInput(req)
	if err != nil {
		return nil, err
	}

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
		var stopReason types.StopReason
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
						switch r := d.Value.(type) {
						case *types.ReasoningContentBlockDeltaMemberText:
							// Text arrives unsigned; the signature follows as
							// its own delta, so leave ProviderName unset until
							// there is a signature to attribute.
							ch <- core.StreamEvent{
								Type:  core.StreamEventThinkingDelta,
								Delta: r.Value,
							}
						case *types.ReasoningContentBlockDeltaMemberSignature:
							// Terminal event for the block. Bedrock rejects a
							// reasoning block replayed without its signature,
							// so this has to reach the caller.
							ch <- core.StreamEvent{
								Type:         core.StreamEventThinkingDelta,
								Signature:    r.Value,
								ProviderName: providerName,
							}
						case *types.ReasoningContentBlockDeltaMemberRedactedContent:
							// Encrypted reasoning: no readable text, only the
							// ciphertext to replay verbatim.
							ch <- core.StreamEvent{
								Type:         core.StreamEventThinkingDelta,
								Signature:    string(r.Value),
								ProviderName: providerName,
								ThinkingID:   redactedThinkingID,
							}
						}
					}
				}

			case *types.ConverseStreamOutputMemberMetadata:
				if v.Value.Usage != nil {
					usage = extractUsage(v.Value.Usage)
				}

			case *types.ConverseStreamOutputMemberMessageStop:
				// Message complete — the reason is replayed on the done event
				// sent after the loop.
				stopReason = v.Value.StopReason

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

		ch <- core.StreamEvent{
			Type:         core.StreamEventDone,
			Usage:        &usage,
			FinishReason: convertStopReason(stopReason),
		}
	}()

	return sr, nil
}

// Name implements core.Model.
func (m *Model) Name() string {
	return m.modelID
}

// ModelMetadata reports semantic provider and transport identity.
func (m *Model) ModelMetadata() core.ModelMetadata {
	return core.ModelMetadata{Provider: "aws.bedrock", Operation: "chat"}
}

// converseParams holds the shared parameters for Converse and ConverseStream.
type converseParams struct {
	messages      []types.Message
	system        []types.SystemContentBlock
	inferenceConf *types.InferenceConfiguration
	toolConfig    *types.ToolConfiguration
	additionalReq document.Interface
	performance   *types.PerformanceConfiguration

	// Guardrail identity, kept as plain strings because Converse and
	// ConverseStream take differently-typed guardrail configs.
	guardrailID      string
	guardrailVersion string
}

// guardrailConfig builds the non-streaming guardrail config, or nil when no
// guardrail was requested.
func (p converseParams) guardrailConfig() *types.GuardrailConfiguration {
	if p.guardrailID == "" {
		return nil
	}
	cfg := &types.GuardrailConfiguration{GuardrailIdentifier: aws.String(p.guardrailID)}
	if p.guardrailVersion != "" {
		cfg.GuardrailVersion = aws.String(p.guardrailVersion)
	}
	return cfg
}

// guardrailStreamConfig builds the streaming guardrail config, or nil when no
// guardrail was requested.
func (p converseParams) guardrailStreamConfig() *types.GuardrailStreamConfiguration {
	if p.guardrailID == "" {
		return nil
	}
	cfg := &types.GuardrailStreamConfiguration{GuardrailIdentifier: aws.String(p.guardrailID)}
	if p.guardrailVersion != "" {
		cfg.GuardrailVersion = aws.String(p.guardrailVersion)
	}
	return cfg
}

// buildParams extracts the shared request parameters from core.ChatRequest.
//
// It fails rather than silently dropping anything the Converse API cannot
// express, so a caller never receives a response that quietly ignored part of
// the request.
func (m *Model) buildParams(req *core.ChatRequest) (converseParams, error) {
	var p converseParams

	// The Converse API expresses structured output through outputConfig, which
	// only a subset of Bedrock's models accept, and this package has no
	// per-model capability table to gate it on. Honoring the field for some
	// models and not others is worse than declining it outright.
	if req.ResponseFormat != nil {
		return p, fmt.Errorf(
			"bedrock: ResponseFormat is not supported; use tool-based structured output (core.OutputModeTool) instead",
		)
	}

	opts := optionsFor(req)

	// Convert messages, accumulating the system prompt. Bedrock takes the
	// system prompt as a single top-level field, so every system message in
	// the conversation has to be folded into it — keeping only the last one
	// would silently discard earlier instructions.
	var systemText []string
	var systemCache *core.CachePoint
	for _, msg := range req.Messages {
		if msg.Role == core.RoleSystem {
			if text := msg.GetTextContent(); text != "" {
				systemText = append(systemText, text)
			}
			if cp := lastCachePoint(msg); cp != nil {
				systemCache = cp
			}
			continue
		}
		converted, err := convertMessage(msg)
		if err != nil {
			return converseParams{}, err
		}
		if converted != nil {
			p.messages = append(p.messages, *converted)
		}
	}
	p.system = convertSystemBlocks(strings.Join(systemText, "\n\n"), systemCache)
	if req.PromptCache != nil && req.PromptCache.Enabled() {
		cachePoint := &core.CachePoint{TTL: req.PromptCache.TTL()}
		if systemCache == nil && len(p.system) > 0 {
			p.system = append(p.system, &types.SystemContentBlockMemberCachePoint{Value: cachePointBlock(cachePoint)})
		}
		appendGeneratedCachePoint(p.messages, cachePoint)
	}

	// Every message may have been dropped for holding only content Bedrock
	// cannot carry. An empty Messages array is a ValidationException from the
	// service, so fail here where the cause is still visible.
	if len(p.messages) == 0 {
		return converseParams{}, fmt.Errorf(
			"bedrock: no messages left after conversion; the request carries only content the Converse API cannot represent",
		)
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
	if len(req.StopSequences) > 0 {
		infCfg.StopSequences = req.StopSequences
		hasInfCfg = true
	}

	if hasInfCfg {
		p.inferenceConf = infCfg
	}

	thinkingEnabled := req.Thinking != nil && req.Thinking.Enabled

	// Tools
	if len(req.Tools) > 0 {
		p.toolConfig = convertToolConfig(req.Tools, req.ToolChoice, thinkingEnabled)
	}

	// Model-specific inference parameters, including thinking, which Bedrock
	// routes through AdditionalModelRequestFields for Anthropic models.
	extra := make(map[string]interface{}, len(opts.AdditionalModelRequestFields)+1)
	for k, v := range opts.AdditionalModelRequestFields {
		extra[k] = v
	}
	if thinkingEnabled {
		extra["thinking"] = map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": req.Thinking.BudgetTokens,
		}
	}
	if len(extra) > 0 {
		p.additionalReq = document.NewLazyDocument(extra)
	}

	if opts.GuardrailIdentifier != "" {
		p.guardrailID = opts.GuardrailIdentifier
		p.guardrailVersion = opts.GuardrailVersion
	}

	switch opts.PerformanceLatency {
	case string(types.PerformanceConfigLatencyStandard), string(types.PerformanceConfigLatencyOptimized):
		p.performance = &types.PerformanceConfiguration{
			Latency: types.PerformanceConfigLatency(opts.PerformanceLatency),
		}
	}

	return p, nil
}

// buildInput converts core.ChatRequest to Bedrock ConverseInput.
func (m *Model) buildInput(req *core.ChatRequest) (*bedrockruntime.ConverseInput, error) {
	p, err := m.buildParams(req)
	if err != nil {
		return nil, err
	}
	return &bedrockruntime.ConverseInput{
		ModelId:                      aws.String(m.modelID),
		Messages:                     p.messages,
		System:                       p.system,
		InferenceConfig:              p.inferenceConf,
		ToolConfig:                   p.toolConfig,
		AdditionalModelRequestFields: p.additionalReq,
		GuardrailConfig:              p.guardrailConfig(),
		PerformanceConfig:            p.performance,
	}, nil
}

// buildStreamInput converts core.ChatRequest to Bedrock ConverseStreamInput.
func (m *Model) buildStreamInput(req *core.ChatRequest) (*bedrockruntime.ConverseStreamInput, error) {
	p, err := m.buildParams(req)
	if err != nil {
		return nil, err
	}
	return &bedrockruntime.ConverseStreamInput{
		ModelId:                      aws.String(m.modelID),
		Messages:                     p.messages,
		System:                       p.system,
		InferenceConfig:              p.inferenceConf,
		ToolConfig:                   p.toolConfig,
		AdditionalModelRequestFields: p.additionalReq,
		GuardrailConfig:              p.guardrailStreamConfig(),
		PerformanceConfig:            p.performance,
	}, nil
}

// lastCachePoint returns the final cache point marker in a message, or nil when
// it carries none.
func lastCachePoint(msg core.Message) *core.CachePoint {
	var found *core.CachePoint
	for _, p := range msg.Content {
		if p.Type == core.ContentCachePoint && p.CachePoint != nil {
			found = p.CachePoint
		}
	}
	return found
}

// cachePointBlock builds a Bedrock cache point from a core cache point marker.
// A TTL outside the set Bedrock accepts is left unset so the service default
// applies rather than the request being rejected.
func cachePointBlock(cp *core.CachePoint) types.CachePointBlock {
	block := types.CachePointBlock{Type: types.CachePointTypeDefault}
	switch cp.TTL {
	case string(types.CacheTTLFiveMinutes):
		block.Ttl = types.CacheTTLFiveMinutes
	case string(types.CacheTTLOneHour):
		block.Ttl = types.CacheTTLOneHour
	}
	return block
}

// appendGeneratedCachePoint marks the longest stable conversational prefix.
// It does not duplicate an explicit trailing marker supplied by an application.
func appendGeneratedCachePoint(messages []types.Message, cp *core.CachePoint) {
	if len(messages) == 0 || cp == nil {
		return
	}
	last := &messages[len(messages)-1]
	if len(last.Content) == 0 {
		return
	}
	if _, explicit := last.Content[len(last.Content)-1].(*types.ContentBlockMemberCachePoint); explicit {
		return
	}
	last.Content = append(last.Content, &types.ContentBlockMemberCachePoint{Value: cachePointBlock(cp)})
}

// convertSystemBlocks builds the Bedrock system content blocks from the joined
// system prompt text and an optional trailing cache point.
func convertSystemBlocks(text string, cp *core.CachePoint) []types.SystemContentBlock {
	if text == "" {
		return nil
	}
	blocks := []types.SystemContentBlock{
		&types.SystemContentBlockMemberText{Value: text},
	}
	if cp != nil {
		blocks = append(blocks, &types.SystemContentBlockMemberCachePoint{Value: cachePointBlock(cp)})
	}
	return blocks
}

// convertMessage converts core.Message to Bedrock Message. It returns nil when
// the message holds no content the Converse API can carry.
func convertMessage(msg core.Message) (*types.Message, error) {
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
				if err != nil {
					// Dropping the image would send a prompt that refers to a
					// picture the model never receives, so surface the decode
					// failure instead.
					return nil, fmt.Errorf("bedrock: decode image data (%s): %w", p.ImageData.MediaType, err)
				}
				blocks = append(blocks, &types.ContentBlockMemberImage{
					Value: types.ImageBlock{
						Source: &types.ImageSourceMemberBytes{Value: data},
						Format: imageFormat(p.ImageData.MediaType),
					},
				})
			}

		case core.ContentDocumentURL:
			// Bedrock document blocks require inline bytes; skip URL-based
			continue

		case core.ContentThinking:
			if p.Thinking == nil {
				continue
			}
			// Bedrock verifies the signature on every replayed reasoning block
			// and rejects one that is missing or was issued by a different
			// provider. A block failing either test degrades to plain text, so
			// a conversation that passed through another model keeps its
			// reasoning visible instead of 400-ing.
			if p.Thinking.ProviderName != providerName || p.Thinking.Signature == "" {
				if p.Thinking.Text != "" {
					blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Thinking.Text})
				}
				continue
			}
			if p.Thinking.IsRedacted() {
				// Redacted reasoning travels as the opaque ciphertext the model
				// returned, which this library carries in Signature.
				blocks = append(blocks, &types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberRedactedContent{
						Value: []byte(p.Thinking.Signature),
					},
				})
				continue
			}
			blocks = append(blocks, &types.ContentBlockMemberReasoningContent{
				Value: &types.ReasoningContentBlockMemberReasoningText{
					Value: types.ReasoningTextBlock{
						Text:      aws.String(p.Thinking.Text),
						Signature: aws.String(p.Thinking.Signature),
					},
				},
			})

		case core.ContentCachePoint:
			// Bedrock carries a cache point as its own content block. It marks
			// everything before it as cacheable, so one that opens a message or
			// immediately follows another has nothing to cache and is rejected
			// by the service; drop those rather than fail the request.
			if p.CachePoint == nil || len(blocks) == 0 {
				continue
			}
			if _, dup := blocks[len(blocks)-1].(*types.ContentBlockMemberCachePoint); dup {
				continue
			}
			blocks = append(blocks, &types.ContentBlockMemberCachePoint{
				Value: cachePointBlock(p.CachePoint),
			})

		case core.ContentAudioURL, core.ContentVideoURL, core.ContentUploadedFile:
			// Not supported by Bedrock Converse API — skip
			continue
		}
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	role := convertRole(msg.Role)
	return &types.Message{
		Role:    role,
		Content: blocks,
	}, nil
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
//
// thinkingEnabled downgrades a forced tool choice to auto: Bedrock rejects
// toolChoice.any while extended thinking is on, and degrading keeps the request
// answerable instead of failing it outright.
func convertToolConfig(tools []core.Tool, choice *core.ToolChoice, thinkingEnabled bool) *types.ToolConfiguration {
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
			if thinkingEnabled {
				tc.ToolChoice = &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}
			} else {
				tc.ToolChoice = &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}
			}
		default:
			tc.ToolChoice = &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}
		}
	}

	return tc
}

// convertResponse converts Bedrock ConverseOutput to agentic ChatResponse.
func (m *Model) convertResponse(resp *bedrockruntime.ConverseOutput) *core.ChatResponse {
	// Converse returns no response body id; the AWS request id is the only
	// identifier that correlates a response with CloudWatch and with support
	// cases, so it stands in as the response id.
	requestID, _ := awsmiddleware.GetRequestIDMetadata(resp.ResultMetadata)

	chatResp := &core.ChatResponse{
		ID:              requestID,
		Model:           m.modelID,
		Created:         time.Now(),
		Message:         convertOutputMessage(resp.Output),
		FinishReason:    convertStopReason(resp.StopReason),
		RawFinishReason: string(resp.StopReason),
	}

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
			switch r := b.Value.(type) {
			case *types.ReasoningContentBlockMemberReasoningText:
				signature := aws.ToString(r.Value.Signature)
				block := &core.ThinkingBlock{
					Text:      aws.ToString(r.Value.Text),
					Signature: signature,
				}
				// Only a signed block can be replayed to Bedrock, so only a
				// signed block claims this provider. Tagging an unsigned one
				// would make it look replayable when it is not.
				if signature != "" {
					block.ProviderName = providerName
				}
				msg.Content = append(msg.Content, core.Part{
					Type:     core.ContentThinking,
					Thinking: block,
				})

			case *types.ReasoningContentBlockMemberRedactedContent:
				// Encrypted reasoning carries no readable text; the ciphertext
				// is what has to be replayed, so it lives in Signature.
				msg.Content = append(msg.Content, core.Part{
					Type: core.ContentThinking,
					Thinking: &core.ThinkingBlock{
						ID:           redactedThinkingID,
						Signature:    string(r.Value),
						ProviderName: providerName,
					},
				})
			}
		}
	}

	return msg
}

// convertStopReason converts Bedrock stop reason to agentic FinishReason.
//
// A value this library does not recognize maps to FinishReasonUnknown rather
// than being reported as a clean stop; callers that need the original string
// read ChatResponse.RawFinishReason.
func convertStopReason(reason types.StopReason) core.FinishReason {
	switch reason {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return core.FinishReasonStop
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		// Both mean the answer was cut short for want of room: max_tokens hit
		// the output ceiling, model_context_window_exceeded hit the total one.
		return core.FinishReasonLength
	case types.StopReasonToolUse:
		return core.FinishReasonToolCalls
	case types.StopReasonContentFiltered, types.StopReasonGuardrailIntervened:
		return core.FinishReasonContentFilter
	case types.StopReasonMalformedModelOutput, types.StopReasonMalformedToolUse:
		// The model aborted mid-generation: whatever content arrived is a
		// fragment, not an answer.
		return core.FinishReasonError
	default:
		return core.FinishReasonUnknown
	}
}

// extractUsage converts Bedrock TokenUsage to agentic Usage.
func extractUsage(tu *types.TokenUsage) core.Usage {
	cacheRead := int(aws.ToInt32(tu.CacheReadInputTokens))
	cacheCreation := int(aws.ToInt32(tu.CacheWriteInputTokens))
	prompt := int(aws.ToInt32(tu.InputTokens)) + cacheRead + cacheCreation
	completion := int(aws.ToInt32(tu.OutputTokens))
	usage := core.Usage{
		PromptTokens:        prompt,
		CompletionTokens:    completion,
		TotalTokens:         prompt + completion,
		CacheReadTokens:     cacheRead,
		CacheCreationTokens: cacheCreation,
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
