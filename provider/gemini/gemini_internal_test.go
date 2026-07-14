package gemini

import (
	"context"
	"testing"

	"google.golang.org/genai"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/internal/core"
)

func TestWithVertexAIOption(t *testing.T) {
	cfg := &config{}
	WithVertexAI("project-1", "us-west1")(cfg)
	if !cfg.vertexAI || cfg.project != "project-1" || cfg.location != "us-west1" {
		t.Fatalf("unexpected vertex config %#v", cfg)
	}
}

func TestGeminiRequestValidationErrors(t *testing.T) {
	model := &Model{model: "gemini-2.5-pro"}

	if _, err := model.Request(context.Background(), &core.ChatRequest{}); err == nil {
		t.Fatal("expected Request to fail validation")
	}
	if _, err := model.RequestStream(context.Background(), &core.ChatRequest{}); err == nil {
		t.Fatal("expected RequestStream to fail validation")
	}
}

func TestBuildRequestCoversConversions(t *testing.T) {
	model := &Model{model: "gemini-2.5-pro"}
	temperature := 0.6
	maxTokens := 128
	topP := 0.9
	frequencyPenalty := 0.1
	presencePenalty := 0.2
	toolChoice := core.ToolChoiceRequired

	contents, cfg := model.buildRequest(&core.ChatRequest{
		Model: "gemini-2.5-pro",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "system"),
			{
				Role: core.RoleUser,
				Content: []core.Part{
					{Type: core.ContentText, Text: "hello"},
					{Type: core.ContentImageURL, ImageURL: &core.ImageURL{URL: "https://example.com/image.jpg"}},
					agentic.ImageDataPart([]byte("img"), "image/png"),
					{Type: core.ContentAudioURL, AudioURL: &core.AudioURL{URL: "https://example.com/audio.mp3", Format: "mp3"}},
					{Type: core.ContentVideoURL, VideoURL: &core.VideoURL{URL: "https://example.com/video.mp4"}},
					{Type: core.ContentDocumentURL, DocumentURL: &core.DocumentURL{URL: "https://example.com/file.pdf"}},
					{Type: core.ContentUploadedFile, UploadedFile: &core.UploadedFile{FileID: "file_123"}},
					{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{}},
				},
			},
			{
				Role: core.RoleAssistant,
				Content: []core.Part{
					{Type: core.ContentText, Text: "working"},
					{Type: core.ContentToolUse, ToolUse: &core.ToolUse{ID: "call_1", Name: "lookup", Input: map[string]interface{}{"city": "Lima"}}},
					{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{Text: "reasoning"}},
				},
			},
			core.NewToolResultMessage("call_1", `{"temp":72}`, false),
		},
		Temperature:      &temperature,
		MaxTokens:        &maxTokens,
		TopP:             &topP,
		FrequencyPenalty: &frequencyPenalty,
		PresencePenalty:  &presencePenalty,
		Tools: []core.Tool{{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "lookup",
				Description: "Lookup weather",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{"type": "string"},
					},
				},
			},
		}},
		ToolChoice: &toolChoice,
		ResponseFormat: &core.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &core.JSONSchemaFormat{
				Name:   "weather",
				Schema: map[string]interface{}{"type": "object"},
			},
		},
		Thinking: &core.ThinkingConfig{Enabled: true, BudgetTokens: 123},
	})

	if len(contents) != 3 {
		t.Fatalf("expected 3 converted messages, got %d", len(contents))
	}
	if cfg.SystemInstruction == nil || cfg.SystemInstruction.Parts[0].Text != "system" {
		t.Fatalf("expected system instruction, got %#v", cfg.SystemInstruction)
	}
	if cfg.Temperature == nil || *cfg.Temperature != float32(temperature) {
		t.Fatalf("unexpected temperature %#v", cfg.Temperature)
	}
	if cfg.MaxOutputTokens != int32(maxTokens) {
		t.Fatalf("expected max output tokens %d, got %d", maxTokens, cfg.MaxOutputTokens)
	}
	if cfg.TopP == nil || *cfg.TopP != float32(topP) {
		t.Fatalf("unexpected top_p %#v", cfg.TopP)
	}
	if cfg.FrequencyPenalty == nil || *cfg.FrequencyPenalty != float32(frequencyPenalty) {
		t.Fatalf("unexpected frequency penalty %#v", cfg.FrequencyPenalty)
	}
	if cfg.PresencePenalty == nil || *cfg.PresencePenalty != float32(presencePenalty) {
		t.Fatalf("unexpected presence penalty %#v", cfg.PresencePenalty)
	}
	if len(cfg.Tools) != 1 || len(cfg.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("expected one function declaration, got %#v", cfg.Tools)
	}
	if cfg.ToolConfig == nil || cfg.ToolConfig.FunctionCallingConfig.Mode != genai.FunctionCallingConfigModeAny {
		t.Fatalf("unexpected tool config %#v", cfg.ToolConfig)
	}
	if cfg.ResponseMIMEType != "application/json" || cfg.ResponseSchema == nil {
		t.Fatalf("expected JSON response schema, got %#v", cfg)
	}
	if cfg.ThinkingConfig == nil || !cfg.ThinkingConfig.IncludeThoughts || cfg.ThinkingConfig.ThinkingBudget == nil || *cfg.ThinkingConfig.ThinkingBudget != 123 {
		t.Fatalf("unexpected thinking config %#v", cfg.ThinkingConfig)
	}
}

func TestConvertMessageFallbackAndToolConfig(t *testing.T) {
	fallback := convertMessage(core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{}},
			{Type: core.ContentUploadedFile, UploadedFile: &core.UploadedFile{FileID: "file_123"}},
			{Type: core.ContentImageData, ImageData: &core.ImageData{Data: "!!!", MediaType: "image/png"}},
		},
	})
	if len(fallback.Parts) != 1 || fallback.Parts[0].Text != "" {
		t.Fatalf("expected empty text fallback, got %#v", fallback.Parts)
	}

	if got := convertToolConfig(core.ToolChoiceNone); got.FunctionCallingConfig.Mode != genai.FunctionCallingConfigModeNone {
		t.Fatalf("expected none tool config, got %#v", got)
	}
	if got := convertToolConfig(core.ToolChoiceAuto); got.FunctionCallingConfig.Mode != genai.FunctionCallingConfigModeAuto {
		t.Fatalf("expected auto tool config, got %#v", got)
	}
}

func TestGeminiConvertResponseAndUsage(t *testing.T) {
	model := &Model{model: "gemini-2.5-pro"}
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					genai.NewPartFromText("answer"),
					{Thought: true, Text: "thinking"},
					genai.NewPartFromFunctionCall("lookup", map[string]any{"city": "Lima"}),
				},
			},
			FinishReason: genai.FinishReasonMaxTokens,
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        100,
			CandidatesTokenCount:    40,
			CachedContentTokenCount: 8,
		},
	}

	chatResp := model.convertResponse(resp)
	if chatResp.Model != "gemini-2.5-pro" {
		t.Fatalf("unexpected model %q", chatResp.Model)
	}
	if chatResp.Choices[0].FinishReason != core.FinishReasonLength {
		t.Fatalf("expected length finish reason, got %q", chatResp.Choices[0].FinishReason)
	}
	if chatResp.Usage.CacheReadTokens != 8 || chatResp.Usage.TotalTokens != 140 {
		t.Fatalf("unexpected usage %#v", chatResp.Usage)
	}
	msg := chatResp.Choices[0].Message
	if msg.GetTextContent() != "answer" {
		t.Fatalf("unexpected text content %q", msg.GetTextContent())
	}
	if msg.GetThinkingContent() != "thinking" {
		t.Fatalf("unexpected thinking content %q", msg.GetThinkingContent())
	}
	if len(msg.GetToolUses()) != 1 || msg.GetToolUses()[0].Name != "lookup" {
		t.Fatalf("unexpected tool uses %#v", msg.GetToolUses())
	}
}

func TestGeminiFinishReasonMapping(t *testing.T) {
	if got := convertFinishReason(genai.FinishReasonStop); got != core.FinishReasonStop {
		t.Fatalf("expected stop finish reason, got %q", got)
	}
	if got := convertFinishReason(genai.FinishReasonSafety); got != core.FinishReasonContentFilter {
		t.Fatalf("expected content filter finish reason, got %q", got)
	}
	if got := convertFinishReason(genai.FinishReason("other")); got != core.FinishReasonStop {
		t.Fatalf("expected default stop finish reason, got %q", got)
	}
}
