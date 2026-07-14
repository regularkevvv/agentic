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
	frequencyPenalty := 0.1
	presencePenalty := 0.2
	toolChoice := core.ToolChoiceRequired
	strict := true

	params := model.buildParams(&core.ChatRequest{
		Model: "gpt-4o",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "system"),
			core.NewTextMessage(core.RoleUser, "hello"),
		},
		Temperature:      &temperature,
		MaxTokens:        &maxTokens,
		TopP:             &topP,
		FrequencyPenalty: &frequencyPenalty,
		PresencePenalty:  &presencePenalty,
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
	if got := params.FrequencyPenalty.Or(0); got != frequencyPenalty {
		t.Fatalf("expected frequency penalty %.1f, got %.1f", frequencyPenalty, got)
	}
	if got := params.PresencePenalty.Or(0); got != presencePenalty {
		t.Fatalf("expected presence penalty %.1f, got %.1f", presencePenalty, got)
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
	imageData := convertContentPart(core.Part{
		Type: core.ContentImageData,
		ImageData: &core.ImageData{
			Data:           "AQID",
			MediaType:      "image/png",
			VendorMetadata: map[string]interface{}{"detail": "high"},
		},
	})
	if imageData.OfImageURL == nil {
		t.Fatal("expected image data to convert to image URL part")
	}

	audio := convertContentPart(core.Part{
		Type:     core.ContentAudioURL,
		AudioURL: &core.AudioURL{URL: "https://example.com/audio.mp3", Format: "mp3"},
	})
	if audio.OfInputAudio == nil {
		t.Fatal("expected audio URL to convert to input audio part")
	}

	file := convertContentPart(core.Part{
		Type:         core.ContentUploadedFile,
		UploadedFile: &core.UploadedFile{FileID: "file_123"},
	})
	if file.OfFile == nil {
		t.Fatal("expected uploaded file to convert to file part")
	}

	cachePoint := convertContentPart(core.Part{Type: core.ContentCachePoint})
	if cachePoint.OfText == nil {
		t.Fatal("expected cache point to be skipped as an empty text part")
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
