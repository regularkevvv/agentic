package openai

import (
	"context"
	"testing"

	sdkopenai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestWithRequestOptions(t *testing.T) {
	cfg := &config{}
	WithRequestOptions(option.WithHeader("X-Test", "1"))(cfg)
	if len(cfg.extraOpts) != 1 {
		t.Fatalf("expected 1 extra option, got %d", len(cfg.extraOpts))
	}
}

func TestOpenAIRequestValidationErrors(t *testing.T) {
	model := &Model{model: "gpt-4o"}

	if _, err := model.Request(context.Background(), &core.ChatRequest{}); err == nil {
		t.Fatal("expected Request to fail validation")
	}
	if _, err := model.RequestStream(context.Background(), &core.ChatRequest{}); err == nil {
		t.Fatal("expected RequestStream to fail validation")
	}
}

func TestBuildParamsAppliesOptionalFields(t *testing.T) {
	model := &Model{model: "gpt-4o"}
	temperature := 0.6
	maxTokens := 128
	topP := 0.9
	toolChoice := core.ToolChoiceRequired
	strict := true

	params := model.buildParams(&core.ChatRequest{
		Model: "gpt-4o",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "system"),
			core.NewTextMessage(core.RoleUser, "hello"),
		},
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
		TopP:        &topP,
		Tools: []core.Tool{{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "lookup_weather",
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
				Name:        "weather",
				Description: "Weather response",
				Schema:      map[string]interface{}{"type": "object"},
				Strict:      &strict,
			},
		},
		Thinking: &core.ThinkingConfig{Enabled: true, BudgetTokens: 3000},
	})

	if params.Model != shared.ChatModel("gpt-4o") {
		t.Fatalf("unexpected model %q", params.Model)
	}
	if got := params.Temperature.Or(0); got != temperature {
		t.Fatalf("expected temperature %.1f, got %.1f", temperature, got)
	}
	if got := params.MaxCompletionTokens.Or(0); got != int64(maxTokens) {
		t.Fatalf("expected max tokens %d, got %d", maxTokens, got)
	}
	if got := params.TopP.Or(0); got != topP {
		t.Fatalf("expected top_p %.1f, got %.1f", topP, got)
	}
	if len(params.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(params.Tools))
	}
	if params.ToolChoice.OfAuto.Or("") != string(sdkopenai.ChatCompletionToolChoiceOptionAutoRequired) {
		t.Fatalf("unexpected tool choice %#v", params.ToolChoice)
	}
	if params.ResponseFormat.OfJSONSchema == nil {
		t.Fatalf("expected JSON schema response format")
	}
	if params.ReasoningEffort != shared.ReasoningEffortLow {
		t.Fatalf("expected low reasoning effort, got %q", params.ReasoningEffort)
	}
}

func TestBuildParamsUsesMediumReasoningEffortForDefaultThinkingBudget(t *testing.T) {
	model := &Model{model: "gpt-4o"}

	params := model.buildParams(&core.ChatRequest{
		Model: "gpt-4o",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
		Thinking: &core.ThinkingConfig{Enabled: true, BudgetTokens: 10000},
	})

	if params.ReasoningEffort != shared.ReasoningEffortMedium {
		t.Fatalf("expected medium reasoning effort, got %q", params.ReasoningEffort)
	}
}

func TestConvertResponseFormat(t *testing.T) {
	t.Run("json_object", func(t *testing.T) {
		got := convertResponseFormat(&core.ResponseFormat{Type: "json_object"})
		if got.OfJSONObject == nil {
			t.Fatal("expected json object response format")
		}
	})

	t.Run("json_schema", func(t *testing.T) {
		strict := true
		got := convertResponseFormat(&core.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &core.JSONSchemaFormat{
				Name:        "output",
				Description: "Structured output",
				Schema:      map[string]interface{}{"type": "object"},
				Strict:      &strict,
			},
		})
		if got.OfJSONSchema == nil {
			t.Fatal("expected json schema response format")
		}
	})

	t.Run("json_schema_without_schema_falls_back", func(t *testing.T) {
		got := convertResponseFormat(&core.ResponseFormat{Type: "json_schema"})
		if got.OfJSONObject == nil {
			t.Fatal("expected fallback to json object response format")
		}
	})

	t.Run("default_text", func(t *testing.T) {
		got := convertResponseFormat(&core.ResponseFormat{Type: "text"})
		if got.OfText == nil {
			t.Fatal("expected text response format")
		}
	})
}

func TestConvertContentPartAdditionalModalities(t *testing.T) {
	imageData, ok := convertContentPart(core.Part{
		Type: core.ContentImageData,
		ImageData: &core.ImageData{
			Data:           "AQID",
			MediaType:      "image/png",
			VendorMetadata: map[string]interface{}{"detail": "high"},
		},
	})
	if !ok || imageData.OfImageURL == nil {
		t.Fatal("expected image data to convert to image URL part")
	}

	audio, ok := convertContentPart(core.Part{
		Type:     core.ContentAudioURL,
		AudioURL: &core.AudioURL{URL: "https://example.com/audio.mp3", Format: "mp3"},
	})
	if !ok || audio.OfInputAudio == nil {
		t.Fatal("expected audio URL to convert to input audio part")
	}

	file, ok := convertContentPart(core.Part{
		Type:         core.ContentUploadedFile,
		UploadedFile: &core.UploadedFile{FileID: "file_123"},
	})
	if !ok || file.OfFile == nil {
		t.Fatal("expected uploaded file to convert to file part")
	}
}

// TestConvertContentPartSkipsUnrepresentableParts pins that parts OpenAI has
// no representation for are dropped rather than emitted as an empty text part,
// which the doc on convertContentPart has always promised.
func TestConvertContentPartSkipsUnrepresentableParts(t *testing.T) {
	tests := []struct {
		name string
		part core.Part
	}{
		{"cache point", core.Part{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{TTL: "5m"}}},
		{"thinking", core.Part{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{Text: "hidden"}}},
		{"unknown type", core.Part{Type: core.ContentType("something_new")}},
		{"video without text", core.Part{Type: core.ContentVideoURL, VideoURL: &core.VideoURL{URL: "https://example.com/v.mp4"}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := convertContentPart(tt.part)
			if ok {
				t.Fatalf("expected part to be skipped, got %#v", got)
			}
		})
	}
}

// TestConvertMessagesUserSkipsCachePoint pins that a skipped part leaves no
// blank content entry behind in the emitted user message.
func TestConvertMessagesUserSkipsCachePoint(t *testing.T) {
	msgs := convertMessages(core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentText, Text: "hello"},
			{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{}},
			{Type: core.ContentText, Text: "world"},
		},
	})

	if len(msgs) != 1 || msgs[0].OfUser == nil {
		t.Fatalf("expected a single user message, got %#v", msgs)
	}
	parts := msgs[0].OfUser.Content.OfArrayOfContentParts
	if len(parts) != 2 {
		t.Fatalf("expected the cache point to be dropped, got %d parts", len(parts))
	}
	for i, p := range parts {
		if p.OfText == nil || p.OfText.Text == "" {
			t.Fatalf("part %d is an empty text entry: %#v", i, p)
		}
	}
}

func TestExtractOpenAIUsageIncludesReasoningAndCache(t *testing.T) {
	usage := extractOpenAIUsage(sdkopenai.CompletionUsage{
		PromptTokens:     100,
		CompletionTokens: 40,
		TotalTokens:      140,
		CompletionTokensDetails: sdkopenai.CompletionUsageCompletionTokensDetails{
			ReasoningTokens: 12,
		},
		PromptTokensDetails: sdkopenai.CompletionUsagePromptTokensDetails{
			CachedTokens: 8,
		},
	})

	if usage.PromptTokens != 100 || usage.CompletionTokens != 40 || usage.TotalTokens != 140 {
		t.Fatalf("unexpected usage totals: %#v", usage)
	}
	if usage.ReasoningTokens != 12 {
		t.Fatalf("expected reasoning tokens, got %#v", usage)
	}
	if usage.CacheReadTokens != 8 {
		t.Fatalf("expected cache read tokens, got %#v", usage)
	}
}

// TestConvertContentPartFallsBackToText covers a part whose typed payload is
// missing but which still carries text worth sending.
func TestConvertContentPartFallsBackToText(t *testing.T) {
	got, ok := convertContentPart(core.Part{Type: core.ContentVideoURL, Text: "a video"})
	if !ok || got.OfText == nil || got.OfText.Text != "a video" {
		t.Fatalf("expected a text fallback carrying the part's text, got %#v (ok=%v)", got, ok)
	}
}
