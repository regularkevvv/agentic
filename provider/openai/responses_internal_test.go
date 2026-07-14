package openai

import (
	"context"
	"testing"

	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestResponsesRequestValidationErrors(t *testing.T) {
	model := &ResponsesModel{model: "gpt-4.1"}

	if _, err := model.Request(context.Background(), &core.ChatRequest{}); err == nil {
		t.Fatal("expected Request to fail validation")
	}
	if _, err := model.RequestStream(context.Background(), &core.ChatRequest{}); err == nil {
		t.Fatal("expected RequestStream to fail validation")
	}
}

func TestNewResponsesWithOptions(t *testing.T) {
	model, err := NewResponses(
		"gpt-4.1",
		WithAPIKey("test-key"),
		WithBaseURL("https://example.com/v1"),
		WithOrganization("org-123"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gpt-4.1" {
		t.Fatalf("expected model name to be preserved, got %q", model.Name())
	}
}

func TestResponsesBuildParams(t *testing.T) {
	model := &ResponsesModel{model: "gpt-4.1"}
	temperature := 0.6
	maxTokens := 256
	topP := 0.8
	toolChoice := core.ToolChoiceRequired
	strict := true

	params := model.buildParams(&core.ChatRequest{
		Model: "gpt-4.1",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "system"),
			{
				Role: core.RoleUser,
				Content: []core.Part{
					{Type: core.ContentText, Text: "hello"},
					{Type: core.ContentImageURL, ImageURL: &core.ImageURL{URL: "https://example.com/image.png", Detail: "high"}},
				},
			},
			{
				Role: core.RoleAssistant,
				Content: []core.Part{
					{Type: core.ContentText, Text: "working"},
					{Type: core.ContentToolUse, ToolUse: &core.ToolUse{ID: "call_1", Name: "lookup", Input: map[string]interface{}{"city": "Lima"}}},
					{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{Signature: "enc"}},
				},
			},
			core.NewToolResultMessage("call_1", `{"temp":72}`, false),
		},
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
		TopP:        &topP,
		Tools: []core.Tool{{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "lookup",
				Description: "Lookup a city",
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
				Name:   "city_lookup",
				Schema: map[string]interface{}{"type": "object"},
				Strict: &strict,
			},
		},
		Thinking: &core.ThinkingConfig{Enabled: true, BudgetTokens: 25000},
	})

	if params.Instructions.Or("") != "system" {
		t.Fatalf("expected system instructions, got %#v", params.Instructions)
	}
	if params.Temperature.Or(0) != temperature {
		t.Fatalf("expected temperature %.1f, got %.1f", temperature, params.Temperature.Or(0))
	}
	if params.MaxOutputTokens.Or(0) != int64(maxTokens) {
		t.Fatalf("expected max output tokens %d, got %d", maxTokens, params.MaxOutputTokens.Or(0))
	}
	if params.TopP.Or(0) != topP {
		t.Fatalf("expected top_p %.1f, got %.1f", topP, params.TopP.Or(0))
	}
	if len(params.Input.OfInputItemList) != 5 {
		t.Fatalf("expected 5 input items, got %d", len(params.Input.OfInputItemList))
	}
	if len(params.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(params.Tools))
	}
	if params.ToolChoice.OfToolChoiceMode.Or("") != responses.ToolChoiceOptionsRequired {
		t.Fatalf("unexpected tool choice %#v", params.ToolChoice)
	}
	if params.Text.Format.OfJSONSchema == nil {
		t.Fatalf("expected JSON schema text format")
	}
	if params.Reasoning.Effort != shared.ReasoningEffortHigh {
		t.Fatalf("expected high reasoning effort, got %q", params.Reasoning.Effort)
	}
}

func TestConvertInputContentMultipart(t *testing.T) {
	content := convertInputContent(core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentText, Text: "hello"},
			{Type: core.ContentImageURL, ImageURL: &core.ImageURL{URL: "https://example.com/image.png"}},
			{Type: core.ContentImageData, ImageData: &core.ImageData{MediaType: "image/png", Data: "AQID"}},
			{Type: core.ContentAudioURL, AudioURL: &core.AudioURL{URL: "https://example.com/audio.mp3"}},
			{Type: core.ContentDocumentURL, DocumentURL: &core.DocumentURL{URL: "https://example.com/file.pdf"}},
			{Type: core.ContentUploadedFile, UploadedFile: &core.UploadedFile{FileID: "file_123"}},
		},
	})

	if len(content.OfInputItemContentList) != 6 {
		t.Fatalf("expected 6 content parts, got %d", len(content.OfInputItemContentList))
	}
	if content.OfInputItemContentList[1].OfInputImage == nil {
		t.Fatalf("expected image URL content part")
	}
	if content.OfInputItemContentList[2].OfInputImage == nil {
		t.Fatalf("expected inline image content part")
	}
	if content.OfInputItemContentList[3].OfInputFile == nil || content.OfInputItemContentList[4].OfInputFile == nil || content.OfInputItemContentList[5].OfInputFile == nil {
		t.Fatalf("expected file-based content parts")
	}
}

func TestResponsesConvertResponseAndFinishReason(t *testing.T) {
	model := &ResponsesModel{model: "gpt-4.1"}
	resp := &responses.Response{
		ID:        "resp_123",
		Model:     shared.ResponsesModel("gpt-4.1"),
		CreatedAt: 123,
		Status:    "completed",
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "answer"},
					{Type: "refusal", Refusal: " but limited"},
				},
			},
			{
				Type:      "function_call",
				CallID:    "call_1",
				Name:      "lookup",
				Arguments: `{"city":"Lima"}`,
			},
			{
				Type:             "reasoning",
				EncryptedContent: "enc",
				Summary: []responses.ResponseReasoningItemSummary{{
					Text: "reasoning summary",
				}},
			},
		},
		Usage: responses.ResponseUsage{
			InputTokens:  100,
			OutputTokens: 40,
			TotalTokens:  140,
		},
	}

	chatResp := model.convertResponse(resp)
	if chatResp.ID != "resp_123" || chatResp.Model != "gpt-4.1" {
		t.Fatalf("unexpected response metadata: %#v", chatResp)
	}
	if chatResp.Choices[0].FinishReason != core.FinishReasonToolCalls {
		t.Fatalf("expected tool-calls finish reason, got %q", chatResp.Choices[0].FinishReason)
	}
	msg := chatResp.Choices[0].Message
	if msg.GetTextContent() != "answer but limited" {
		t.Fatalf("unexpected text content %q", msg.GetTextContent())
	}
	if len(msg.GetToolUses()) != 1 || msg.GetToolUses()[0].Name != "lookup" {
		t.Fatalf("unexpected tool uses %#v", msg.GetToolUses())
	}
	if msg.GetThinkingContent() != "reasoning summary" {
		t.Fatalf("unexpected thinking content %q", msg.GetThinkingContent())
	}
}

func TestConvertResponsesFinishReason(t *testing.T) {
	if got := convertResponsesFinishReason(&responses.Response{
		IncompleteDetails: responses.ResponseIncompleteDetails{Reason: "max_output_tokens"},
	}); got != core.FinishReasonLength {
		t.Fatalf("expected length finish reason, got %q", got)
	}

	if got := convertResponsesFinishReason(&responses.Response{
		IncompleteDetails: responses.ResponseIncompleteDetails{Reason: "content_filter"},
	}); got != core.FinishReasonContentFilter {
		t.Fatalf("expected content filter finish reason, got %q", got)
	}

	if got := convertResponsesFinishReason(&responses.Response{Status: "completed"}); got != core.FinishReasonStop {
		t.Fatalf("expected stop finish reason, got %q", got)
	}

	if got := convertResponsesFinishReason(&responses.Response{Status: "failed"}); got != core.FinishReasonStop {
		t.Fatalf("expected failed status to map to stop, got %q", got)
	}

	if got := convertResponsesFinishReason(&responses.Response{Status: "canceled"}); got != core.FinishReasonStop {
		t.Fatalf("expected canceled status to map to stop, got %q", got)
	}

	if got := convertResponsesFinishReason(&responses.Response{Status: "unknown"}); got != core.FinishReasonStop {
		t.Fatalf("expected unknown status to map to stop, got %q", got)
	}
}

func TestResponsesSchemaHelpers(t *testing.T) {
	t.Run("ensureAdditionalPropertiesFalse recurses through nested objects and arrays", func(t *testing.T) {
		schema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name":    map[string]interface{}{"type": "string"},
				"literal": "value",
				"meta": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"id": map[string]interface{}{"type": "string"},
					},
					"required": []interface{}{"id"},
				},
				"tags": map[string]interface{}{
					"type": "array",
					"items": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"label": map[string]interface{}{"type": "string"},
						},
					},
				},
			},
			"required": []string{"name"},
		}

		normalized := ensureAdditionalPropertiesFalse(schema)
		if normalized["additionalProperties"] != false {
			t.Fatalf("expected root additionalProperties=false, got %#v", normalized)
		}

		required := normalized["required"].([]string)
		if len(required) != 4 {
			t.Fatalf("expected all root properties to become required, got %#v", required)
		}

		meta := normalized["properties"].(map[string]interface{})["meta"].(map[string]interface{})
		if meta["additionalProperties"] != false {
			t.Fatalf("expected nested object additionalProperties=false, got %#v", meta)
		}

		tags := normalized["properties"].(map[string]interface{})["tags"].(map[string]interface{})
		items := tags["items"].(map[string]interface{})
		if items["additionalProperties"] != false {
			t.Fatalf("expected array items additionalProperties=false, got %#v", items)
		}
		if normalized["properties"].(map[string]interface{})["literal"] != "value" {
			t.Fatalf("expected non-map property values to be preserved, got %#v", normalized["properties"])
		}

		// Original input should not be mutated.
		if _, ok := schema["additionalProperties"]; ok {
			t.Fatalf("expected original schema to remain unchanged, got %#v", schema)
		}
	})

	t.Run("convertResponsesTools handles nil params and empty description", func(t *testing.T) {
		tools := convertResponsesTools([]core.Tool{
			{
				Type: core.ToolTypeFunction,
				Function: core.Function{
					Name:        "noop",
					Description: "",
					Parameters:  nil,
				},
			},
			{
				Type: core.ToolTypeFunction,
				Function: core.Function{
					Name:        "strict_no_desc",
					Description: "",
					Parameters:  map[string]interface{}{"type": "object"},
				},
			},
		})

		if len(tools) != 2 || tools[0].OfFunction == nil || tools[1].OfFunction == nil {
			t.Fatalf("expected two function tools, got %#v", tools)
		}
		if tools[0].OfFunction.Parameters == nil {
			t.Fatalf("expected nil params to become an empty map, got %#v", tools[0].OfFunction.Parameters)
		}
		if tools[1].OfFunction.Parameters["additionalProperties"] != false {
			t.Fatalf("expected strict schema normalization, got %#v", tools[1].OfFunction.Parameters)
		}
	})

	t.Run("convertTextConfig includes optional schema metadata", func(t *testing.T) {
		strict := true
		tc := convertTextConfig(&core.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &core.JSONSchemaFormat{
				Name:        "output",
				Description: "Structured output",
				Schema:      map[string]interface{}{"type": "object"},
				Strict:      &strict,
			},
		})

		if tc.Format.OfJSONSchema == nil {
			t.Fatal("expected JSON schema text config")
		}
		if tc.Format.OfJSONSchema.Description.Value != "Structured output" {
			t.Fatalf("expected description to be preserved, got %#v", tc.Format.OfJSONSchema.Description)
		}
		if tc.Format.OfJSONSchema.Strict.Value != true {
			t.Fatalf("expected strict flag to be preserved, got %#v", tc.Format.OfJSONSchema.Strict)
		}
	})
}
