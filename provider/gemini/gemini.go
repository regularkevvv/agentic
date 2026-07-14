// Package gemini provides a Google Gemini Model implementation for Agentic.
// It supports both the Google Generative Language API (Gemini API) and Vertex AI
// through the unified google.golang.org/genai SDK.
//
// Features: streaming, tool/function calling, multimodal inputs, thinking tokens,
// structured output via response schema, and prompt caching.
package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"google.golang.org/genai"
)

// Model implements the core.Model and core.StreamModel interfaces
// using the Google Gemini API.
type Model struct {
	client *genai.Client
	model  string
}

// Option configures the Gemini Model.
type Option func(*config)

type config struct {
	apiKey   string
	project  string
	location string
	vertexAI bool
}

// WithAPIKey sets the Gemini API key. If not set, the GEMINI_API_KEY
// or GOOGLE_API_KEY env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithVertexAI configures the model to use Vertex AI instead of the Gemini API.
func WithVertexAI(project, location string) Option {
	return func(c *config) {
		c.project = project
		c.location = location
		c.vertexAI = true
	}
}

// New creates a new Gemini Model.
//
// Examples:
//
//	// Gemini API
//	model, err := gemini.New("gemini-2.5-pro", gemini.WithAPIKey("..."))
//
//	// Vertex AI
//	model, err := gemini.New("gemini-2.5-pro",
//	    gemini.WithVertexAI("my-project", "us-central1"),
//	)
func New(model string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	clientCfg := &genai.ClientConfig{}

	if cfg.vertexAI {
		clientCfg.Backend = genai.BackendVertexAI
		clientCfg.Project = cfg.project
		clientCfg.Location = cfg.location
		if clientCfg.Location == "" {
			clientCfg.Location = os.Getenv("GOOGLE_CLOUD_LOCATION")
		}
		if clientCfg.Location == "" {
			clientCfg.Location = "us-central1"
		}
		if clientCfg.Project == "" {
			clientCfg.Project = os.Getenv("GOOGLE_CLOUD_PROJECT")
		}
	} else {
		clientCfg.Backend = genai.BackendGeminiAPI
		clientCfg.APIKey = cfg.apiKey
		if clientCfg.APIKey == "" {
			clientCfg.APIKey = os.Getenv("GEMINI_API_KEY")
		}
		if clientCfg.APIKey == "" {
			clientCfg.APIKey = os.Getenv("GOOGLE_API_KEY")
		}
		if clientCfg.APIKey == "" {
			return nil, fmt.Errorf("gemini: API key not set (use WithAPIKey or set GEMINI_API_KEY)")
		}
	}

	client, err := genai.NewClient(context.Background(), clientCfg)
	if err != nil {
		return nil, fmt.Errorf("gemini: failed to create client: %w", err)
	}

	return &Model{
		client: client,
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

	contents, genCfg := m.buildRequest(req)

	resp, err := m.client.Models.GenerateContent(ctx, m.model, contents, genCfg)
	if err != nil {
		return nil, fmt.Errorf("gemini: %w", err)
	}

	return m.convertResponse(resp), nil
}

// RequestStream implements core.StreamModel.
func (m *Model) RequestStream(ctx context.Context, req *core.ChatRequest) (*core.StreamResult, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	contents, genCfg := m.buildRequest(req)

	ch := make(chan core.StreamEvent, 64)
	sr := core.NewStreamResult(ch)

	go func() {
		defer close(ch)

		var usage core.Usage

		for chunk, err := range m.client.Models.GenerateContentStream(ctx, m.model, contents, genCfg) {
			if err != nil {
				ch <- core.StreamEvent{
					Type:  core.StreamEventError,
					Error: fmt.Errorf("gemini stream: %w", err),
				}
				return
			}

			// Extract usage from chunk
			if chunk.UsageMetadata != nil {
				usage = extractUsage(chunk.UsageMetadata)
			}

			if len(chunk.Candidates) == 0 {
				continue
			}

			candidate := chunk.Candidates[0]
			if candidate.Content == nil {
				continue
			}

			for _, part := range candidate.Content.Parts {
				if part.FunctionCall != nil {
					// Emit tool call as start + delta with full args
					argsJSON, _ := json.Marshal(part.FunctionCall.Args)
					toolUseID := fmt.Sprintf("call_%s_%d", part.FunctionCall.Name, time.Now().UnixNano())

					ch <- core.StreamEvent{
						Type: core.StreamEventToolCallStart,
						ToolUse: &core.ToolUse{
							ID:   toolUseID,
							Name: part.FunctionCall.Name,
						},
					}
					ch <- core.StreamEvent{
						Type:       core.StreamEventToolCallDelta,
						Delta:      string(argsJSON),
						ToolCallID: toolUseID,
					}
					continue
				}

				if part.Thought {
					ch <- core.StreamEvent{
						Type:  core.StreamEventThinkingDelta,
						Delta: part.Text,
					}
					continue
				}

				if part.Text != "" {
					ch <- core.StreamEvent{
						Type:  core.StreamEventTextDelta,
						Delta: part.Text,
					}
				}
			}
		}

		ch <- core.StreamEvent{Type: core.StreamEventDone, Usage: &usage}
	}()

	return sr, nil
}

// Name implements core.Model.
func (m *Model) Name() string {
	return m.model
}

// buildRequest converts core.ChatRequest to Gemini API parameters.
func (m *Model) buildRequest(req *core.ChatRequest) ([]*genai.Content, *genai.GenerateContentConfig) {
	cfg := &genai.GenerateContentConfig{}

	// Convert messages, extracting system prompt
	var contents []*genai.Content
	for _, msg := range req.Messages {
		if msg.Role == core.RoleSystem {
			cfg.SystemInstruction = convertSystemMessage(msg)
			continue
		}
		contents = append(contents, convertMessage(msg))
	}

	// Generation parameters
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
	}
	if req.MaxTokens != nil {
		cfg.MaxOutputTokens = int32(*req.MaxTokens)
	}
	if req.TopP != nil {
		tp := float32(*req.TopP)
		cfg.TopP = &tp
	}
	if req.FrequencyPenalty != nil {
		fp := float32(*req.FrequencyPenalty)
		cfg.FrequencyPenalty = &fp
	}
	if req.PresencePenalty != nil {
		pp := float32(*req.PresencePenalty)
		cfg.PresencePenalty = &pp
	}

	// Tools
	if len(req.Tools) > 0 {
		cfg.Tools = convertTools(req.Tools)
		if req.ToolChoice != nil {
			cfg.ToolConfig = convertToolConfig(*req.ToolChoice)
		}
	}

	// Response format
	if req.ResponseFormat != nil {
		cfg.ResponseMIMEType = "application/json"
		if req.ResponseFormat.JSONSchema != nil && req.ResponseFormat.JSONSchema.Schema != nil {
			cfg.ResponseSchema = convertSchema(req.ResponseFormat.JSONSchema.Schema)
		}
	}

	// Thinking
	if req.Thinking != nil && req.Thinking.Enabled {
		cfg.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: true,
		}
		if req.Thinking.BudgetTokens > 0 {
			budget := int32(req.Thinking.BudgetTokens)
			cfg.ThinkingConfig.ThinkingBudget = &budget
		}
	}

	return contents, cfg
}

// convertSystemMessage converts a system message to Gemini Content.
func convertSystemMessage(msg core.Message) *genai.Content {
	return genai.NewContentFromText(msg.GetTextContent(), "user")
}

// convertMessage converts core.Message to Gemini Content.
func convertMessage(msg core.Message) *genai.Content {
	var parts []*genai.Part

	for _, p := range msg.Content {
		switch p.Type {
		case core.ContentText:
			parts = append(parts, genai.NewPartFromText(p.Text))

		case core.ContentToolUse:
			if p.ToolUse != nil {
				parts = append(parts, genai.NewPartFromFunctionCall(p.ToolUse.Name, p.ToolUse.Input))
			}

		case core.ContentToolResult:
			if p.ToolResult != nil {
				var response map[string]any
				_ = json.Unmarshal([]byte(p.ToolResult.Content), &response)
				if response == nil {
					response = map[string]any{"result": p.ToolResult.Content}
				}
				parts = append(parts, genai.NewPartFromFunctionResponse(p.ToolResult.ToolUseID, response))
			}

		case core.ContentImageURL:
			if p.ImageURL != nil {
				parts = append(parts, genai.NewPartFromURI(p.ImageURL.URL, "image/jpeg"))
			}

		case core.ContentImageData:
			if p.ImageData != nil {
				data, err := base64.StdEncoding.DecodeString(p.ImageData.Data)
				if err == nil {
					parts = append(parts, genai.NewPartFromBytes(data, p.ImageData.MediaType))
				}
			}

		case core.ContentAudioURL:
			if p.AudioURL != nil {
				mimeType := "audio/mp3"
				if p.AudioURL.Format != "" {
					mimeType = "audio/" + p.AudioURL.Format
				}
				parts = append(parts, genai.NewPartFromURI(p.AudioURL.URL, mimeType))
			}

		case core.ContentVideoURL:
			if p.VideoURL != nil {
				parts = append(parts, genai.NewPartFromURI(p.VideoURL.URL, "video/mp4"))
			}

		case core.ContentDocumentURL:
			if p.DocumentURL != nil {
				mimeType := p.DocumentURL.MediaType
				if mimeType == "" {
					mimeType = "application/pdf"
				}
				parts = append(parts, genai.NewPartFromURI(p.DocumentURL.URL, mimeType))
			}

		case core.ContentCachePoint, core.ContentUploadedFile:
			// Not directly supported — skip
			continue
		}
	}

	if len(parts) == 0 {
		parts = append(parts, genai.NewPartFromText(""))
	}

	role := genai.Role(convertRole(msg.Role))
	return genai.NewContentFromParts(parts, role)
}

// convertRole maps agentic role to Gemini role string.
func convertRole(role core.MessageRole) string {
	switch role {
	case core.RoleAssistant:
		return "model"
	case core.RoleTool:
		return "user"
	default:
		return "user"
	}
}

// convertTools converts agentic tools to Gemini format.
func convertTools(tools []core.Tool) []*genai.Tool {
	var decls []*genai.FunctionDeclaration
	for _, tool := range tools {
		decl := &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		}
		if tool.Function.Parameters != nil {
			decl.Parameters = convertSchema(tool.Function.Parameters)
		}
		decls = append(decls, decl)
	}
	return []*genai.Tool{{FunctionDeclarations: decls}}
}

// convertToolConfig converts agentic ToolChoice to Gemini ToolConfig.
func convertToolConfig(choice core.ToolChoice) *genai.ToolConfig {
	tc := &genai.ToolConfig{
		FunctionCallingConfig: &genai.FunctionCallingConfig{},
	}
	switch choice {
	case core.ToolChoiceNone:
		tc.FunctionCallingConfig.Mode = genai.FunctionCallingConfigModeNone
	case core.ToolChoiceRequired:
		tc.FunctionCallingConfig.Mode = genai.FunctionCallingConfigModeAny
	default:
		tc.FunctionCallingConfig.Mode = genai.FunctionCallingConfigModeAuto
	}
	return tc
}

// convertSchema converts a JSON Schema map to a Gemini Schema.
func convertSchema(schema map[string]interface{}) *genai.Schema {
	s := &genai.Schema{}

	if t, ok := schema["type"].(string); ok {
		s.Type = genai.Type(t)
	}

	if desc, ok := schema["description"].(string); ok {
		s.Description = desc
	}

	if props, ok := schema["properties"].(map[string]interface{}); ok {
		s.Properties = make(map[string]*genai.Schema)
		for name, prop := range props {
			if propMap, ok := prop.(map[string]interface{}); ok {
				s.Properties[name] = convertSchema(propMap)
			}
		}
	}

	if req, ok := schema["required"].([]interface{}); ok {
		for _, r := range req {
			if str, ok := r.(string); ok {
				s.Required = append(s.Required, str)
			}
		}
	}

	if items, ok := schema["items"].(map[string]interface{}); ok {
		s.Items = convertSchema(items)
	}

	if enum, ok := schema["enum"].([]interface{}); ok {
		for _, e := range enum {
			if str, ok := e.(string); ok {
				s.Enum = append(s.Enum, str)
			}
		}
	}

	return s
}

// convertResponse converts a Gemini response to agentic types.
func (m *Model) convertResponse(resp *genai.GenerateContentResponse) *core.ChatResponse {
	choices := make([]core.Choice, 0, len(resp.Candidates))
	for i, candidate := range resp.Candidates {
		choices = append(choices, core.Choice{
			Index:        i,
			Message:      convertCandidateMessage(candidate),
			FinishReason: convertFinishReason(candidate.FinishReason),
		})
	}

	chatResp := &core.ChatResponse{
		Model:   m.model,
		Choices: choices,
		Created: time.Now(),
	}

	if resp.UsageMetadata != nil {
		chatResp.Usage = extractUsage(resp.UsageMetadata)
	}

	return chatResp
}

// convertCandidateMessage converts a Gemini candidate to agentic Message.
func convertCandidateMessage(candidate *genai.Candidate) core.Message {
	msg := core.Message{
		Role:    core.RoleAssistant,
		Content: make([]core.Part, 0),
	}

	if candidate.Content == nil {
		return msg
	}

	for _, part := range candidate.Content.Parts {
		if part.FunctionCall != nil {
			inputJSON, _ := json.Marshal(part.FunctionCall.Args)
			var input map[string]interface{}
			_ = json.Unmarshal(inputJSON, &input)

			toolUseID := fmt.Sprintf("call_%s_%d", part.FunctionCall.Name, time.Now().UnixNano())
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentToolUse,
				ToolUse: &core.ToolUse{
					ID:    toolUseID,
					Name:  part.FunctionCall.Name,
					Input: input,
				},
			})
			continue
		}

		if part.Thought {
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentThinking,
				Thinking: &core.ThinkingBlock{
					Text:         part.Text,
					ProviderName: "gemini",
				},
			})
			continue
		}

		if part.Text != "" {
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentText,
				Text: part.Text,
			})
		}
	}

	return msg
}

// convertFinishReason converts Gemini finish reason to agentic type.
func convertFinishReason(reason genai.FinishReason) core.FinishReason {
	switch reason {
	case genai.FinishReasonStop:
		return core.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		return core.FinishReasonLength
	case genai.FinishReasonSafety, genai.FinishReasonBlocklist, genai.FinishReasonProhibitedContent, genai.FinishReasonSPII:
		return core.FinishReasonContentFilter
	default:
		return core.FinishReasonStop
	}
}

// extractUsage converts Gemini usage metadata to agentic Usage.
func extractUsage(meta *genai.GenerateContentResponseUsageMetadata) core.Usage {
	usage := core.Usage{
		PromptTokens:     int(meta.PromptTokenCount),
		CompletionTokens: int(meta.CandidatesTokenCount),
		TotalTokens:      int(meta.PromptTokenCount + meta.CandidatesTokenCount),
	}
	if meta.CachedContentTokenCount > 0 {
		usage.CacheReadTokens = int(meta.CachedContentTokenCount)
	}
	return usage
}

// Compile-time check that Model implements core.StreamModel (which embeds core.Model).
var _ core.StreamModel = (*Model)(nil)
