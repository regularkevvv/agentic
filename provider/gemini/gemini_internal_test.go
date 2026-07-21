package gemini

import (
	"context"
	"encoding/base64"
	"errors"
	"testing"
	"time"

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
	topK := 40
	seed := 7
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
			core.NewToolResultMessageFor("call_1", "lookup", `{"temp":72}`, false),
		},
		Temperature:   &temperature,
		MaxTokens:     &maxTokens,
		TopP:          &topP,
		StopSequences: []string{"STOP!", "HALT"},
		ProviderOptions: map[string]any{
			ProviderKey: Options{TopK: &topK, Seed: &seed},
		},
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
	if len(cfg.StopSequences) != 2 || cfg.StopSequences[0] != "STOP!" {
		t.Fatalf("unexpected stop sequences %#v", cfg.StopSequences)
	}
	if cfg.TopK == nil || *cfg.TopK != float32(topK) {
		t.Fatalf("unexpected top_k %#v", cfg.TopK)
	}
	if cfg.Seed == nil || *cfg.Seed != int32(seed) {
		t.Fatalf("unexpected seed %#v", cfg.Seed)
	}
	if len(cfg.Tools) != 1 || len(cfg.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("expected one function declaration, got %#v", cfg.Tools)
	}
	if cfg.Tools[0].FunctionDeclarations[0].ParametersJsonSchema == nil {
		t.Fatal("expected parameters to be forwarded as a JSON Schema")
	}
	if cfg.ToolConfig == nil || cfg.ToolConfig.FunctionCallingConfig.Mode != genai.FunctionCallingConfigModeAny {
		t.Fatalf("unexpected tool config %#v", cfg.ToolConfig)
	}
	if cfg.ResponseMIMEType != "application/json" || cfg.ResponseJsonSchema == nil {
		t.Fatalf("expected JSON response schema, got %#v", cfg)
	}
	if cfg.ThinkingConfig == nil || !cfg.ThinkingConfig.IncludeThoughts || cfg.ThinkingConfig.ThinkingBudget == nil || *cfg.ThinkingConfig.ThinkingBudget != 123 {
		t.Fatalf("unexpected thinking config %#v", cfg.ThinkingConfig)
	}
}

func TestBuildRequestJoinsEverySystemMessage(t *testing.T) {
	model := &Model{model: "gemini-2.5-pro"}

	_, cfg := model.buildRequest(&core.ChatRequest{
		Model: "gemini-2.5-pro",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "first"),
			core.NewTextMessage(core.RoleSystem, "second"),
			core.NewTextMessage(core.RoleUser, "hi"),
		},
	})

	if cfg.SystemInstruction == nil {
		t.Fatal("expected a system instruction")
	}
	if got := cfg.SystemInstruction.Parts[0].Text; got != "first\n\nsecond" {
		t.Fatalf("expected both system messages joined, got %q", got)
	}
}

func TestBuildRequestThinkingConfig(t *testing.T) {
	model := &Model{model: "gemini-2.5-pro"}
	messages := []core.Message{core.NewTextMessage(core.RoleUser, "hi")}

	tests := []struct {
		name     string
		thinking *core.ThinkingConfig
		want     func(*genai.ThinkingConfig) bool
	}{
		{
			name:     "unset leaves the model default",
			thinking: nil,
			want:     func(tc *genai.ThinkingConfig) bool { return tc == nil },
		},
		{
			name:     "enabled without budget",
			thinking: &core.ThinkingConfig{Enabled: true},
			want: func(tc *genai.ThinkingConfig) bool {
				return tc != nil && tc.IncludeThoughts && tc.ThinkingBudget == nil
			},
		},
		{
			name:     "disabled sends a zero budget",
			thinking: &core.ThinkingConfig{Enabled: false},
			want: func(tc *genai.ThinkingConfig) bool {
				return tc != nil && !tc.IncludeThoughts && tc.ThinkingBudget != nil && *tc.ThinkingBudget == 0
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, cfg := model.buildRequest(&core.ChatRequest{
				Model:    "gemini-2.5-pro",
				Messages: messages,
				Thinking: tt.thinking,
			})
			if !tt.want(cfg.ThinkingConfig) {
				t.Fatalf("unexpected thinking config %#v", cfg.ThinkingConfig)
			}
		})
	}
}

func TestConvertMessageToolResultUsesFunctionName(t *testing.T) {
	content := convertMessage(core.NewToolResultMessageFor("call_abc", "get_weather", `{"temp":72}`, false))

	if len(content.Parts) != 1 || content.Parts[0].FunctionResponse == nil {
		t.Fatalf("expected one function response part, got %#v", content.Parts)
	}
	fr := content.Parts[0].FunctionResponse
	if fr.Name != "get_weather" {
		t.Fatalf("expected function response named after the tool, got %q", fr.Name)
	}
	if fr.ID != "call_abc" {
		t.Fatalf("expected the provider-issued id to be echoed, got %q", fr.ID)
	}
	if fr.Response["temp"] != float64(72) {
		t.Fatalf("unexpected response payload %#v", fr.Response)
	}
}

func TestConvertMessageToolResultVariants(t *testing.T) {
	tests := []struct {
		name     string
		result   core.ToolResult
		wantName string
		wantID   string
		wantResp map[string]any
	}{
		{
			name:     "name missing falls back to the tool use id",
			result:   core.ToolResult{ToolUseID: "lookup", Content: `{"ok":true}`},
			wantName: "lookup",
			wantID:   "lookup",
			wantResp: map[string]any{"ok": true},
		},
		{
			name:     "non-JSON content is wrapped as output",
			result:   core.ToolResult{ToolUseID: "id-1", Name: "lookup", Content: "72F"},
			wantName: "lookup",
			wantID:   "id-1",
			wantResp: map[string]any{"output": "72F"},
		},
		{
			name:     "errors use the error key",
			result:   core.ToolResult{ToolUseID: "id-1", Name: "lookup", Content: "boom", IsError: true},
			wantName: "lookup",
			wantID:   "id-1",
			wantResp: map[string]any{"error": "boom"},
		},
		{
			name:     "synthesized ids are not echoed back",
			result:   core.ToolResult{ToolUseID: synthesizedIDPrefix + "0_lookup", Name: "lookup", Content: "{}"},
			wantName: "lookup",
			wantID:   "",
			wantResp: map[string]any{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.result
			content := convertMessage(core.Message{
				Role:    core.RoleTool,
				Content: []core.Part{{Type: core.ContentToolResult, ToolResult: &result}},
			})
			fr := content.Parts[0].FunctionResponse
			if fr.Name != tt.wantName {
				t.Fatalf("expected name %q, got %q", tt.wantName, fr.Name)
			}
			if fr.ID != tt.wantID {
				t.Fatalf("expected id %q, got %q", tt.wantID, fr.ID)
			}
			if len(fr.Response) != len(tt.wantResp) {
				t.Fatalf("expected response %#v, got %#v", tt.wantResp, fr.Response)
			}
			for k, v := range tt.wantResp {
				if fr.Response[k] != v {
					t.Fatalf("expected response[%q] = %#v, got %#v", k, v, fr.Response[k])
				}
			}
		})
	}
}

func TestConvertMessageIsStableAcrossCalls(t *testing.T) {
	msg := core.NewToolResultMessageFor("call_abc", "get_weather", `{"temp":72}`, false)

	first := convertMessage(msg)
	second := convertMessage(msg)

	if first.Parts[0].FunctionResponse.Name != second.Parts[0].FunctionResponse.Name ||
		first.Parts[0].FunctionResponse.ID != second.Parts[0].FunctionResponse.ID {
		t.Fatalf("expected a stable conversion, got %#v then %#v",
			first.Parts[0].FunctionResponse, second.Parts[0].FunctionResponse)
	}
}

func TestConvertMessageReplaysThoughtSignature(t *testing.T) {
	signature := base64.StdEncoding.EncodeToString([]byte("sig-bytes"))

	content := convertMessage(core.Message{
		Role: core.RoleAssistant,
		Content: []core.Part{
			{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{
				Text:         "reasoning",
				Signature:    signature,
				ProviderName: providerName,
			}},
			{Type: core.ContentToolUse, ToolUse: &core.ToolUse{ID: "call_1", Name: "lookup"}},
			{Type: core.ContentToolUse, ToolUse: &core.ToolUse{ID: "call_2", Name: "lookup"}},
		},
	})

	if len(content.Parts) != 2 {
		t.Fatalf("expected the signed thinking block to fold into the next part, got %#v", content.Parts)
	}
	if string(content.Parts[0].ThoughtSignature) != "sig-bytes" {
		t.Fatalf("expected the signature on the first function call, got %q", content.Parts[0].ThoughtSignature)
	}
	if content.Parts[1].ThoughtSignature != nil {
		t.Fatalf("expected no signature on the second function call, got %q", content.Parts[1].ThoughtSignature)
	}
}

func TestConvertMessageThinkingWithoutSignature(t *testing.T) {
	tests := []struct {
		name     string
		thinking *core.ThinkingBlock
		wantText string
	}{
		{
			name:     "unsigned thinking replays as a thought part",
			thinking: &core.ThinkingBlock{Text: "reasoning"},
			wantText: "reasoning",
		},
		{
			name:     "another provider's signature is not replayed as a signature",
			thinking: &core.ThinkingBlock{Text: "reasoning", Signature: "abc", ProviderName: "anthropic"},
			wantText: "reasoning",
		},
		{
			name:     "an undecodable signature falls back to the text",
			thinking: &core.ThinkingBlock{Text: "reasoning", Signature: "!!not-base64!!", ProviderName: providerName},
			wantText: "reasoning",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content := convertMessage(core.Message{
				Role:    core.RoleAssistant,
				Content: []core.Part{{Type: core.ContentThinking, Thinking: tt.thinking}},
			})
			if len(content.Parts) != 1 || !content.Parts[0].Thought || content.Parts[0].Text != tt.wantText {
				t.Fatalf("unexpected parts %#v", content.Parts)
			}
		})
	}
}

func TestConvertMessageFirstFunctionCallCarriesSignaturePlaceholder(t *testing.T) {
	content := convertMessage(core.Message{
		Role: core.RoleAssistant,
		Content: []core.Part{
			{Type: core.ContentToolUse, ToolUse: &core.ToolUse{ID: "call_1", Name: "lookup"}},
		},
	})

	if string(content.Parts[0].ThoughtSignature) != skipThoughtSignatureValidator {
		t.Fatalf("expected the documented placeholder signature, got %q", content.Parts[0].ThoughtSignature)
	}
	if content.Parts[0].FunctionCall.ID != "call_1" {
		t.Fatalf("expected the provider-issued call id, got %q", content.Parts[0].FunctionCall.ID)
	}
}

func TestConvertMessageSkipsNilPayloads(t *testing.T) {
	content := convertMessage(core.Message{
		Role: core.RoleAssistant,
		Content: []core.Part{
			{Type: core.ContentThinking},
			{Type: core.ContentToolUse},
			{Type: core.ContentToolResult},
		},
	})

	if len(content.Parts) != 1 || content.Parts[0].Text != "" {
		t.Fatalf("expected the empty-text fallback, got %#v", content.Parts)
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
	created := time.Date(2026, 7, 19, 10, 0, 0, 0, time.UTC)
	resp := &genai.GenerateContentResponse{
		ResponseID:   "resp-123",
		ModelVersion: "gemini-2.5-pro-002",
		CreateTime:   created,
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					genai.NewPartFromText("answer"),
					{Thought: true, Text: "thinking", ThoughtSignature: []byte("sig-bytes")},
					{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "lookup", Args: map[string]any{"city": "Lima"}}},
				},
			},
			FinishReason: genai.FinishReasonMaxTokens,
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        100,
			CandidatesTokenCount:    40,
			ThoughtsTokenCount:      12,
			CachedContentTokenCount: 8,
		},
	}

	chatResp, err := model.convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if chatResp.ID != "resp-123" {
		t.Fatalf("expected the provider response id, got %q", chatResp.ID)
	}
	if chatResp.Model != "gemini-2.5-pro-002" {
		t.Fatalf("expected the resolved model version, got %q", chatResp.Model)
	}
	if !chatResp.Created.Equal(created) {
		t.Fatalf("expected the server timestamp, got %v", chatResp.Created)
	}
	if chatResp.FinishReason != core.FinishReasonLength {
		t.Fatalf("expected length finish reason, got %q", chatResp.FinishReason)
	}
	if chatResp.RawFinishReason != "MAX_TOKENS" {
		t.Fatalf("expected the raw finish reason, got %q", chatResp.RawFinishReason)
	}
	if chatResp.Usage.CacheReadTokens != 8 || chatResp.Usage.ReasoningTokens != 12 {
		t.Fatalf("unexpected usage %#v", chatResp.Usage)
	}
	if chatResp.Usage.CompletionTokens != 52 || chatResp.Usage.TotalTokens != 152 {
		t.Fatalf("expected thinking tokens counted as output, got %#v", chatResp.Usage)
	}
	msg := chatResp.Message
	if msg.GetTextContent() != "answer" {
		t.Fatalf("unexpected text content %q", msg.GetTextContent())
	}
	if msg.GetThinkingContent() != "thinking" {
		t.Fatalf("unexpected thinking content %q", msg.GetThinkingContent())
	}
	if got := msg.Content[1].Thinking.Signature; got != base64.StdEncoding.EncodeToString([]byte("sig-bytes")) {
		t.Fatalf("expected the thought signature captured, got %q", got)
	}
	if msg.Content[1].Thinking.ProviderName != providerName {
		t.Fatalf("unexpected thinking provider %q", msg.Content[1].Thinking.ProviderName)
	}
	uses := msg.GetToolUses()
	if len(uses) != 1 || uses[0].Name != "lookup" {
		t.Fatalf("unexpected tool uses %#v", uses)
	}
	if uses[0].ID != "fc-1" {
		t.Fatalf("expected the provider-issued call id, got %q", uses[0].ID)
	}
}

func TestGeminiConvertResponseSynthesizesStableToolCallIDs(t *testing.T) {
	model := &Model{model: "gemini-2.5-pro"}
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("lookup", map[string]any{"city": "Lima"}),
				genai.NewPartFromFunctionCall("lookup", map[string]any{"city": "Cusco"}),
			}},
			FinishReason: genai.FinishReasonStop,
		}},
	}

	first, err := model.convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	second, err := model.convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}

	firstIDs, secondIDs := first.Message.GetToolUses(), second.Message.GetToolUses()
	if firstIDs[0].ID != secondIDs[0].ID || firstIDs[1].ID != secondIDs[1].ID {
		t.Fatalf("expected stable ids, got %v then %v", firstIDs, secondIDs)
	}
	if firstIDs[0].ID == firstIDs[1].ID {
		t.Fatalf("expected distinct ids per call, got %q twice", firstIDs[0].ID)
	}
	if first.Model != "gemini-2.5-pro" {
		t.Fatalf("expected the configured model when the server reports none, got %q", first.Model)
	}
}

func TestGeminiConvertResponseBlockedPrompt(t *testing.T) {
	model := &Model{model: "gemini-2.5-pro"}

	tests := []struct {
		name     string
		feedback *genai.GenerateContentResponsePromptFeedback
		wantErr  bool
	}{
		{
			name:     "blocked with a reason",
			feedback: &genai.GenerateContentResponsePromptFeedback{BlockReason: genai.BlockedReasonSafety},
			wantErr:  true,
		},
		{
			name: "blocked with an explanation",
			feedback: &genai.GenerateContentResponsePromptFeedback{
				BlockReason:        genai.BlockedReasonProhibitedContent,
				BlockReasonMessage: "prompt violates policy",
			},
			wantErr: true,
		},
		{
			name:     "feedback without a block reason is not an error",
			feedback: &genai.GenerateContentResponsePromptFeedback{},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := model.convertResponse(&genai.GenerateContentResponse{PromptFeedback: tt.feedback})
			if tt.wantErr {
				if !errors.Is(err, ErrPromptBlocked) {
					t.Fatalf("expected ErrPromptBlocked, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error %v", err)
			}
			if resp.FinishReason != core.FinishReasonUnknown {
				t.Fatalf("expected unknown finish reason for a candidate-less response, got %q", resp.FinishReason)
			}
		})
	}
}

func TestGeminiFinishReasonMapping(t *testing.T) {
	tests := []struct {
		reason genai.FinishReason
		want   core.FinishReason
	}{
		{genai.FinishReasonStop, core.FinishReasonStop},
		{genai.FinishReasonMaxTokens, core.FinishReasonLength},
		{genai.FinishReasonSafety, core.FinishReasonContentFilter},
		{genai.FinishReasonBlocklist, core.FinishReasonContentFilter},
		{genai.FinishReasonProhibitedContent, core.FinishReasonContentFilter},
		{genai.FinishReasonSPII, core.FinishReasonContentFilter},
		{genai.FinishReasonRecitation, core.FinishReasonContentFilter},
		{genai.FinishReasonImageSafety, core.FinishReasonContentFilter},
		{genai.FinishReasonImageProhibitedContent, core.FinishReasonContentFilter},
		{genai.FinishReasonImageRecitation, core.FinishReasonContentFilter},
		{genai.FinishReasonMalformedFunctionCall, core.FinishReasonError},
		{genai.FinishReasonUnexpectedToolCall, core.FinishReasonError},
		{genai.FinishReasonLanguage, core.FinishReasonError},
		{genai.FinishReasonNoImage, core.FinishReasonError},
		{genai.FinishReasonImageOther, core.FinishReasonError},
		{genai.FinishReasonOther, core.FinishReasonUnknown},
		{genai.FinishReasonUnspecified, core.FinishReasonUnknown},
		{genai.FinishReason("SOMETHING_NEW"), core.FinishReasonUnknown},
		{genai.FinishReason(""), core.FinishReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(string(tt.reason), func(t *testing.T) {
			if got := convertFinishReason(tt.reason); got != tt.want {
				t.Fatalf("convertFinishReason(%q) = %q, want %q", tt.reason, got, tt.want)
			}
		})
	}
}

func TestExtractUsagePrefersReportedTotal(t *testing.T) {
	usage := extractUsage(&genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     10,
		CandidatesTokenCount: 5,
		ThoughtsTokenCount:   3,
		TotalTokenCount:      21,
	})

	if usage.TotalTokens != 21 {
		t.Fatalf("expected the reported total, got %d", usage.TotalTokens)
	}
	if usage.CompletionTokens != 8 {
		t.Fatalf("expected thinking tokens folded into completion, got %d", usage.CompletionTokens)
	}
}

func TestOptionsFor(t *testing.T) {
	topK := 40
	tests := []struct {
		name string
		req  *core.ChatRequest
		want Options
	}{
		{name: "nil request", req: nil},
		{name: "no provider options", req: &core.ChatRequest{}},
		{name: "another provider's key", req: &core.ChatRequest{ProviderOptions: map[string]any{"anthropic": 1}}},
		{name: "wrong type", req: &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: "nope"}}},
		{name: "nil pointer", req: &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: (*Options)(nil)}}},
		{
			name: "value",
			req:  &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: Options{TopK: &topK}}},
			want: Options{TopK: &topK},
		},
		{
			name: "pointer",
			req:  &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: &Options{TopK: &topK}}},
			want: Options{TopK: &topK},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := optionsFor(tt.req); got != tt.want {
				t.Fatalf("optionsFor() = %#v, want %#v", got, tt.want)
			}
		})
	}
}
