// Package gemini provides a Google Gemini Model implementation for Agentic.
// It supports both the Google Generative Language API (Gemini API) and Vertex AI
// through the unified google.golang.org/genai SDK.
//
// Features: streaming, tool/function calling, multimodal inputs, thinking tokens,
// thought-signature round-tripping, structured output via response schema, and
// prompt caching.
package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"google.golang.org/genai"
)

// ErrPromptBlocked is returned when Gemini rejects the prompt outright — the
// response carries no candidate, only a block reason in promptFeedback.
// Errors wrapping it can be matched with [errors.Is].
var ErrPromptBlocked = errors.New("gemini: prompt blocked")

// synthesizedIDPrefix marks a tool-call id this package invented because the
// API supplied none (pre-Gemini-3 models omit FunctionCall.ID). Synthesized ids
// are never echoed back to the API, which rejects ids it did not issue; they
// exist only so callers have a stable local handle for the call.
const synthesizedIDPrefix = "gemini_call_"

// skipThoughtSignatureValidator is the documented placeholder that lets a
// function call replay without a real thought signature. Gemini requires the
// first function call of a turn to carry a signature; this value tells the
// server to skip that validation on both the Gemini API and Vertex AI.
// See https://ai.google.dev/gemini-api/docs/thought-signatures.
const skipThoughtSignatureValidator = "skip_thought_signature_validator"

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
	taskType string
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

	client, err := newGenAIClient(cfg)
	if err != nil {
		return nil, err
	}

	return &Model{
		client: client,
		model:  model,
	}, nil
}

// newGenAIClient builds the underlying SDK client from an already-applied
// config, resolving the Vertex AI project and location or the Gemini API key
// from the environment when the caller left them unset.
//
// Both New and NewEmbedder use it, so a client-construction fix reaches the
// chat model and the embedder together.
func newGenAIClient(cfg *config) (*genai.Client, error) {
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
	return client, nil
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

	return m.convertResponse(resp)
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
		finishReason := core.FinishReasonUnknown
		calls := 0

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

			if blockErr := promptBlockError(chunk.PromptFeedback); blockErr != nil {
				ch <- core.StreamEvent{Type: core.StreamEventError, Error: blockErr}
				return
			}

			if len(chunk.Candidates) == 0 {
				continue
			}

			candidate := chunk.Candidates[0]
			if candidate.FinishReason != "" {
				finishReason = convertFinishReason(candidate.FinishReason)
			}
			if candidate.Content == nil {
				continue
			}

			for _, part := range candidate.Content.Parts {
				if part.FunctionCall != nil {
					// Emit tool call as start + delta with full args
					argsJSON, _ := json.Marshal(part.FunctionCall.Args)
					toolUseID := toolCallID(part.FunctionCall, calls)
					calls++

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
						Type:         core.StreamEventThinkingDelta,
						Delta:        part.Text,
						Signature:    encodeSignature(part.ThoughtSignature),
						ProviderName: providerName,
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

		ch <- core.StreamEvent{Type: core.StreamEventDone, Usage: &usage, FinishReason: finishReason}
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

	// Convert messages, accumulating every system message into the single
	// system instruction Gemini accepts.
	var contents []*genai.Content
	var system []string
	for _, msg := range req.Messages {
		if msg.Role == core.RoleSystem {
			if text := msg.GetTextContent(); text != "" {
				system = append(system, text)
			}
			continue
		}
		contents = append(contents, convertMessage(msg))
	}
	if len(system) > 0 {
		cfg.SystemInstruction = genai.NewContentFromText(strings.Join(system, "\n\n"), "user")
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
	if len(req.StopSequences) > 0 {
		cfg.StopSequences = req.StopSequences
	}

	opts := optionsFor(req)
	if opts.TopK != nil {
		tk := float32(*opts.TopK)
		cfg.TopK = &tk
	}
	if opts.Seed != nil {
		seed := int32(*opts.Seed)
		cfg.Seed = &seed
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
		if req.ResponseFormat.JSONSchema != nil {
			if schema, ok := jsonSchema(req.ResponseFormat.JSONSchema.Schema); ok {
				cfg.ResponseJsonSchema = schema
			}
		}
	}

	// Thinking. An explicit disable has to be sent as a zero budget: omitting
	// the config leaves the model's own default (thinking on) in place.
	if req.Thinking != nil {
		if req.Thinking.Enabled {
			cfg.ThinkingConfig = &genai.ThinkingConfig{IncludeThoughts: true}
			if req.Thinking.BudgetTokens > 0 {
				budget := int32(req.Thinking.BudgetTokens)
				cfg.ThinkingConfig.ThinkingBudget = &budget
			}
		} else {
			var off int32
			cfg.ThinkingConfig = &genai.ThinkingConfig{ThinkingBudget: &off}
		}
	}

	return contents, cfg
}

// convertMessage converts core.Message to Gemini Content.
//
// Thought signatures travel with the part that follows the thinking block that
// produced them, and the first function call of a turn must carry one, so parts
// are built in order with a pending signature carried forward.
// See https://ai.google.dev/gemini-api/docs/thought-signatures.
func convertMessage(msg core.Message) *genai.Content {
	var parts []*genai.Part
	var pending []byte
	needsCallSignature := true

	// attach moves any pending thought signature onto this part.
	attach := func(p *genai.Part) *genai.Part {
		if pending != nil {
			p.ThoughtSignature = pending
			pending = nil
		}
		return p
	}

	for _, p := range msg.Content {
		switch p.Type {
		case core.ContentText:
			parts = append(parts, attach(&genai.Part{Text: p.Text}))

		case core.ContentThinking:
			if p.Thinking == nil {
				continue
			}
			// A signature we issued attaches to the *next* part rather than
			// being replayed as a thought part of its own.
			if p.Thinking.ProviderName == providerName && p.Thinking.Signature != "" {
				if sig, err := base64.StdEncoding.DecodeString(p.Thinking.Signature); err == nil {
					pending = sig
					continue
				}
			}
			if p.Thinking.Text == "" {
				continue
			}
			parts = append(parts, attach(&genai.Part{Text: p.Thinking.Text, Thought: true}))

		case core.ContentToolUse:
			if p.ToolUse == nil {
				continue
			}
			part := attach(&genai.Part{FunctionCall: &genai.FunctionCall{
				ID:   providerToolCallID(p.ToolUse.ID),
				Name: p.ToolUse.Name,
				Args: p.ToolUse.Input,
			}})
			if part.ThoughtSignature == nil && needsCallSignature {
				part.ThoughtSignature = []byte(skipThoughtSignatureValidator)
			}
			needsCallSignature = false
			parts = append(parts, part)

		case core.ContentToolResult:
			if p.ToolResult == nil {
				continue
			}
			// Gemini matches a response to its call by function name, not by
			// id, so the name is the correlation key. The id is echoed only
			// when the API issued it.
			parts = append(parts, attach(&genai.Part{FunctionResponse: &genai.FunctionResponse{
				ID:       providerToolCallID(p.ToolResult.ToolUseID),
				Name:     toolResultName(p.ToolResult),
				Response: toolResultResponse(p.ToolResult),
			}}))

		case core.ContentImageURL:
			if p.ImageURL != nil {
				// Gemini requires a MIME type alongside a file URI and does
				// not sniff it, so an unset MediaType falls back to JPEG.
				// That default is wrong for a PNG or WebP, hence MediaType.
				mimeType := p.ImageURL.MediaType
				if mimeType == "" {
					mimeType = "image/jpeg"
				}
				parts = append(parts, attach(genai.NewPartFromURI(p.ImageURL.URL, mimeType)))
			}

		case core.ContentImageData:
			if p.ImageData != nil {
				data, err := base64.StdEncoding.DecodeString(p.ImageData.Data)
				if err == nil {
					parts = append(parts, attach(genai.NewPartFromBytes(data, p.ImageData.MediaType)))
				}
			}

		case core.ContentAudioURL:
			if p.AudioURL != nil {
				mimeType := "audio/mp3"
				if p.AudioURL.Format != "" {
					mimeType = "audio/" + p.AudioURL.Format
				}
				parts = append(parts, attach(genai.NewPartFromURI(p.AudioURL.URL, mimeType)))
			}

		case core.ContentVideoURL:
			if p.VideoURL != nil {
				// As with images, the MIME type is declared rather than
				// sniffed; MP4 remains the fallback for an unset MediaType.
				mimeType := p.VideoURL.MediaType
				if mimeType == "" {
					mimeType = "video/mp4"
				}
				parts = append(parts, attach(genai.NewPartFromURI(p.VideoURL.URL, mimeType)))
			}

		case core.ContentDocumentURL:
			if p.DocumentURL != nil {
				mimeType := p.DocumentURL.MediaType
				if mimeType == "" {
					mimeType = "application/pdf"
				}
				parts = append(parts, attach(genai.NewPartFromURI(p.DocumentURL.URL, mimeType)))
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

// toolResultName returns the function name a tool result answers, falling back
// to the tool-use id for results built without one. A result whose name does
// not match a declared function cannot be correlated by Gemini, so the fallback
// is a best effort rather than a fix.
func toolResultName(result *core.ToolResult) string {
	if result.Name != "" {
		return result.Name
	}
	return result.ToolUseID
}

// toolResultResponse shapes a tool result into the response object Gemini
// expects: the "error" key for failures, "output" for anything that is not
// already a JSON object.
func toolResultResponse(result *core.ToolResult) map[string]any {
	if result.IsError {
		return map[string]any{"error": result.Content}
	}
	var response map[string]any
	if err := json.Unmarshal([]byte(result.Content), &response); err == nil && response != nil {
		return response
	}
	return map[string]any{"output": result.Content}
}

// toolCallID returns a stable identifier for a function call: the id the API
// issued (Gemini 3 and later) or, failing that, one derived from the call's
// position in the response so the same response always converts identically.
func toolCallID(call *genai.FunctionCall, index int) string {
	if call.ID != "" {
		return call.ID
	}
	return fmt.Sprintf("%s%d_%s", synthesizedIDPrefix, index, call.Name)
}

// providerToolCallID returns id if the API issued it, or "" if this package
// synthesized it. Gemini rejects ids it did not issue, so synthesized ones must
// not be sent back.
func providerToolCallID(id string) string {
	if strings.HasPrefix(id, synthesizedIDPrefix) {
		return ""
	}
	return id
}

// encodeSignature renders a raw thought signature as base64 for transport
// through core.ThinkingBlock.Signature.
func encodeSignature(sig []byte) string {
	if len(sig) == 0 {
		return ""
	}
	return base64.StdEncoding.EncodeToString(sig)
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
		if schema, ok := jsonSchema(tool.Function.Parameters); ok {
			decl.ParametersJsonSchema = schema
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

// jsonSchema prepares a JSON Schema for the SDK's schema-passthrough fields
// (FunctionDeclaration.ParametersJsonSchema, GenerateContentConfig.ResponseJsonSchema),
// which forward the document as written and so preserve $ref, $defs, anyOf,
// format and the numeric/string constraints a hand-rolled conversion drops.
//
// The second result is false when there is nothing usable to send: an empty
// schema, or one holding values that do not survive JSON encoding (channels,
// functions, cycles). Sending such a schema would fail the whole request, so it
// is dropped and the request proceeds unconstrained.
func jsonSchema(schema map[string]interface{}) (any, bool) {
	if len(schema) == 0 {
		return nil, false
	}
	if _, err := json.Marshal(schema); err != nil {
		return nil, false
	}
	return schema, true
}

// convertResponse converts a Gemini response to agentic types.
func (m *Model) convertResponse(resp *genai.GenerateContentResponse) (*core.ChatResponse, error) {
	if err := promptBlockError(resp.PromptFeedback); err != nil {
		return nil, err
	}

	chatResp := &core.ChatResponse{
		ID:           resp.ResponseID,
		Model:        m.model,
		Created:      time.Now(),
		FinishReason: core.FinishReasonUnknown,
	}
	// The server reports which model actually served the request, which may be
	// a concrete version behind an alias such as "gemini-2.5-pro".
	if resp.ModelVersion != "" {
		chatResp.Model = resp.ModelVersion
	}
	if !resp.CreateTime.IsZero() {
		chatResp.Created = resp.CreateTime
	}

	if len(resp.Candidates) > 0 {
		candidate := resp.Candidates[0]
		chatResp.Message = convertCandidateMessage(candidate)
		chatResp.FinishReason = convertFinishReason(candidate.FinishReason)
		chatResp.RawFinishReason = string(candidate.FinishReason)
	}

	if resp.UsageMetadata != nil {
		chatResp.Usage = extractUsage(resp.UsageMetadata)
	}

	return chatResp, nil
}

// promptBlockError reports a prompt rejected before generation started. Gemini
// answers such a request with HTTP 200 and no candidate, so without this check
// the caller sees an empty, successful-looking response.
func promptBlockError(feedback *genai.GenerateContentResponsePromptFeedback) error {
	if feedback == nil || feedback.BlockReason == "" {
		return nil
	}
	if feedback.BlockReasonMessage != "" {
		return fmt.Errorf("%w: %s: %s", ErrPromptBlocked, feedback.BlockReason, feedback.BlockReasonMessage)
	}
	return fmt.Errorf("%w: %s", ErrPromptBlocked, feedback.BlockReason)
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

	calls := 0
	for _, part := range candidate.Content.Parts {
		if part.FunctionCall != nil {
			inputJSON, _ := json.Marshal(part.FunctionCall.Args)
			var input map[string]interface{}
			_ = json.Unmarshal(inputJSON, &input)

			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentToolUse,
				ToolUse: &core.ToolUse{
					ID:    toolCallID(part.FunctionCall, calls),
					Name:  part.FunctionCall.Name,
					Input: input,
				},
			})
			calls++
			continue
		}

		if part.Thought {
			msg.Content = append(msg.Content, core.Part{
				Type: core.ContentThinking,
				Thinking: &core.ThinkingBlock{
					Text:         part.Text,
					Signature:    encodeSignature(part.ThoughtSignature),
					ProviderName: providerName,
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

// convertFinishReason converts a Gemini finish reason to the agentic type.
//
// Values Gemini may report but this library does not recognize map to
// core.FinishReasonUnknown; the caller reads ChatResponse.RawFinishReason for
// the original string.
func convertFinishReason(reason genai.FinishReason) core.FinishReason {
	switch reason {
	case genai.FinishReasonStop:
		return core.FinishReasonStop

	case genai.FinishReasonMaxTokens:
		return core.FinishReasonLength

	// Content the model was not allowed to return: safety categories, the
	// terminology blocklist, recitation of training data, personal
	// identifiers, and their image-generation equivalents.
	case genai.FinishReasonSafety,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonRecitation,
		genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent,
		genai.FinishReasonImageRecitation:
		return core.FinishReasonContentFilter

	// Generation aborted: the model emitted something unusable, or the request
	// could not be served in the language or modality asked for.
	case genai.FinishReasonMalformedFunctionCall,
		genai.FinishReasonUnexpectedToolCall,
		genai.FinishReasonLanguage,
		genai.FinishReasonNoImage,
		genai.FinishReasonImageOther:
		return core.FinishReasonError

	default:
		// Includes FINISH_REASON_UNSPECIFIED, OTHER, and any value added to
		// the API after this mapping was written.
		return core.FinishReasonUnknown
	}
}

// extractUsage converts Gemini usage metadata to agentic Usage.
//
// Thinking tokens are billed as output but reported separately from
// CandidatesTokenCount, so they are added to the completion total rather than
// silently dropped.
func extractUsage(meta *genai.GenerateContentResponseUsageMetadata) core.Usage {
	completion := int(meta.CandidatesTokenCount + meta.ThoughtsTokenCount)
	usage := core.Usage{
		PromptTokens:     int(meta.PromptTokenCount),
		CompletionTokens: completion,
		TotalTokens:      int(meta.PromptTokenCount) + completion,
		ReasoningTokens:  int(meta.ThoughtsTokenCount),
	}
	if meta.TotalTokenCount > 0 {
		usage.TotalTokens = int(meta.TotalTokenCount)
	}
	if meta.CachedContentTokenCount > 0 {
		usage.CacheReadTokens = int(meta.CachedContentTokenCount)
	}
	return usage
}

// Compile-time check that Model implements core.StreamModel (which embeds core.Model).
var _ core.StreamModel = (*Model)(nil)
