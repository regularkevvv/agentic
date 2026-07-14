package openai

import (
	"context"
	"testing"

	"github.com/openai/openai-go"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNew(t *testing.T) {
	model, err := New("gpt-4o")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gpt-4o" {
		t.Errorf("expected name %q, got %q", "gpt-4o", model.Name())
	}
}

func TestNewWithOptions(t *testing.T) {
	model, err := New("gpt-4o",
		WithAPIKey("test-key"),
		WithBaseURL("https://custom.api.com"),
		WithOrganization("org-123"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gpt-4o" {
		t.Errorf("expected name %q, got %q", "gpt-4o", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := MustNew("gpt-4o", WithAPIKey("test-key"))
	if model.Name() != "gpt-4o" {
		t.Errorf("expected name %q, got %q", "gpt-4o", model.Name())
	}
}

func TestConvertMessageSystem(t *testing.T) {
	msg := core.NewTextMessage(core.RoleSystem, "system prompt")
	param := convertMessage(msg)
	if param.OfSystem == nil {
		t.Error("expected OfSystem for system message")
	}
}

func TestConvertMessageUser(t *testing.T) {
	msg := core.NewTextMessage(core.RoleUser, "hello")
	param := convertMessage(msg)
	if param.OfUser == nil {
		t.Error("expected OfUser for user message")
	}
}

func TestConvertMessageUserMultiPart(t *testing.T) {
	msg := core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentText, Text: "hello"},
			{Type: core.ContentText, Text: " world"},
		},
	}
	param := convertMessage(msg)
	if param.OfUser == nil {
		t.Error("expected OfUser for multi-part user message")
	}
}

func TestConvertMessageAssistant(t *testing.T) {
	msg := core.NewTextMessage(core.RoleAssistant, "response")
	param := convertMessage(msg)
	if param.OfAssistant == nil {
		t.Error("expected OfAssistant for assistant message")
	}
}

func TestConvertMessageAssistantWithToolCalls(t *testing.T) {
	msg := core.NewToolUseMessage(core.ToolUse{
		ID:    "c1",
		Name:  "test",
		Input: map[string]interface{}{"key": "val"},
	})
	param := convertMessage(msg)
	if param.OfAssistant == nil {
		t.Error("expected OfAssistant for tool use message")
	}
	if len(param.OfAssistant.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(param.OfAssistant.ToolCalls))
	}
}

func TestConvertMessageTool(t *testing.T) {
	msg := core.NewToolResultMessage("c1", "result", false)
	param := convertMessage(msg)
	if param.OfTool == nil {
		t.Error("expected OfTool for tool result message")
	}
}

func TestConvertMessageToolNoResults(t *testing.T) {
	// Tool message with no results
	msg := core.Message{
		Role:    core.RoleTool,
		Content: []core.Part{{Type: core.ContentText, Text: "raw text"}},
	}
	param := convertMessage(msg)
	if param.OfTool == nil {
		t.Error("expected OfTool for tool message without results")
	}
}

func TestConvertMessageDefault(t *testing.T) {
	// Unknown role falls to default (user message)
	msg := core.Message{
		Role:    core.MessageRole("custom"),
		Content: []core.Part{{Type: core.ContentText, Text: "hi"}},
	}
	param := convertMessage(msg)
	if param.OfUser == nil {
		t.Error("expected OfUser for unknown role (default)")
	}
}

func TestConvertContentPart(t *testing.T) {
	textPart := core.Part{Type: core.ContentText, Text: "hello"}
	result := convertContentPart(textPart)
	if result.OfText == nil {
		t.Error("expected OfText for text content part")
	}
}

func TestConvertContentPartImage(t *testing.T) {
	imgPart := core.Part{
		Type:     core.ContentImageURL,
		ImageURL: &core.ImageURL{URL: "https://example.com/img.png", Detail: "high"},
	}
	result := convertContentPart(imgPart)
	if result.OfImageURL == nil {
		t.Error("expected OfImageURL for image content part")
	}
}

func TestConvertContentPartImageNoDetail(t *testing.T) {
	imgPart := core.Part{
		Type:     core.ContentImageURL,
		ImageURL: &core.ImageURL{URL: "https://example.com/img.png"},
	}
	result := convertContentPart(imgPart)
	if result.OfImageURL == nil {
		t.Error("expected OfImageURL for image content part without detail")
	}
}

func TestConvertContentPartImageNil(t *testing.T) {
	imgPart := core.Part{
		Type:     core.ContentImageURL,
		ImageURL: nil,
	}
	// Should fallback to text
	result := convertContentPart(imgPart)
	if result.OfText == nil {
		t.Error("expected OfText fallback for nil image URL")
	}
}

func TestConvertTools(t *testing.T) {
	tools := []core.Tool{
		{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "test",
				Description: "test tool",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		},
	}

	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].Function.Name != "test" {
		t.Errorf("expected name %q, got %q", "test", result[0].Function.Name)
	}
}

func TestConvertToolsNilParams(t *testing.T) {
	tools := []core.Tool{
		{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "simple",
				Description: "simple tool",
				Parameters:  nil,
			},
		},
	}

	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
}

func TestConvertToolChoice(t *testing.T) {
	// Just verify these don't panic — the return type uses param.Opt
	convertToolChoice(core.ToolChoiceNone)
	convertToolChoice(core.ToolChoiceRequired)
	convertToolChoice(core.ToolChoiceAuto)
}

func TestConvertFinishReason(t *testing.T) {
	tests := []struct {
		input    string
		expected core.FinishReason
	}{
		{"stop", core.FinishReasonStop},
		{"length", core.FinishReasonLength},
		{"tool_calls", core.FinishReasonToolCalls},
		{"content_filter", core.FinishReasonContentFilter},
		{"unknown", core.FinishReasonStop},
	}

	for _, tt := range tests {
		result := convertFinishReason(tt.input)
		if result != tt.expected {
			t.Errorf("convertFinishReason(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestConvertResponseMessage(t *testing.T) {
	msg := openai.ChatCompletionMessage{
		Role:    "assistant",
		Content: "hello world",
	}

	result := convertResponseMessage(msg)
	if result.Role != core.RoleAssistant {
		t.Errorf("expected role assistant, got %q", result.Role)
	}
	if result.GetTextContent() != "hello world" {
		t.Errorf("expected %q, got %q", "hello world", result.GetTextContent())
	}
}

func TestConvertResponseMessageEmpty(t *testing.T) {
	msg := openai.ChatCompletionMessage{
		Role:    "assistant",
		Content: "",
	}

	result := convertResponseMessage(msg)
	if len(result.Content) != 0 {
		t.Errorf("expected 0 content parts for empty content, got %d", len(result.Content))
	}
}

func TestConvertResponseMessageWithToolCalls(t *testing.T) {
	msg := openai.ChatCompletionMessage{
		Role: "assistant",
		ToolCalls: []openai.ChatCompletionMessageToolCall{
			{
				ID: "c1",
				Function: openai.ChatCompletionMessageToolCallFunction{
					Name:      "test_tool",
					Arguments: `{"key":"val"}`,
				},
			},
		},
	}

	result := convertResponseMessage(msg)
	uses := result.GetToolUses()
	if len(uses) != 1 {
		t.Fatalf("expected 1 tool use, got %d", len(uses))
	}
	if uses[0].Name != "test_tool" {
		t.Errorf("expected name %q, got %q", "test_tool", uses[0].Name)
	}
	if uses[0].Input["key"] != "val" {
		t.Errorf("expected input key=val, got %v", uses[0].Input)
	}
}

func TestBuildParams(t *testing.T) {
	model, _ := New("gpt-4o", WithAPIKey("test-key"))

	temp := 0.7
	maxTokens := 500
	topP := 0.9
	freqPenalty := 0.5
	presPenalty := 0.3

	tc := core.ToolChoiceRequired
	req := &core.ChatRequest{
		Model: "gpt-4o",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
		Temperature:      &temp,
		MaxTokens:        &maxTokens,
		TopP:             &topP,
		FrequencyPenalty: &freqPenalty,
		PresencePenalty:  &presPenalty,
		Tools: []core.Tool{
			{
				Type: core.ToolTypeFunction,
				Function: core.Function{
					Name:        "test",
					Description: "test",
					Parameters:  map[string]interface{}{"type": "object"},
				},
			},
		},
		ToolChoice: &tc,
	}

	params := model.buildParams(req)
	if len(params.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(params.Messages))
	}
	if len(params.Tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(params.Tools))
	}
}

func TestConvertResponse(t *testing.T) {
	model, _ := New("gpt-4o", WithAPIKey("test-key"))

	resp := &openai.ChatCompletion{
		ID:    "test-id",
		Model: "gpt-4o",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "hello",
				},
				FinishReason: "stop",
			},
			{
				Index: 1,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "world",
				},
				FinishReason: "length",
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
		Created:           1234567890,
		SystemFingerprint: "fp_test",
	}

	result := model.convertResponse(resp)
	if result.ID != "test-id" {
		t.Errorf("expected ID %q, got %q", "test-id", result.ID)
	}
	if result.Model != "gpt-4o" {
		t.Errorf("expected model %q, got %q", "gpt-4o", result.Model)
	}
	if len(result.Choices) != 2 {
		t.Fatalf("expected 2 choices, got %d", len(result.Choices))
	}
	if result.Choices[0].Message.GetTextContent() != "hello" {
		t.Errorf("expected %q, got %q", "hello", result.Choices[0].Message.GetTextContent())
	}
	if result.Choices[0].FinishReason != core.FinishReasonStop {
		t.Errorf("expected stop, got %q", result.Choices[0].FinishReason)
	}
	if result.Choices[1].FinishReason != core.FinishReasonLength {
		t.Errorf("expected length, got %q", result.Choices[1].FinishReason)
	}
	if result.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", result.Usage.TotalTokens)
	}
	if result.SystemFingerprint != "fp_test" {
		t.Errorf("expected fingerprint %q, got %q", "fp_test", result.SystemFingerprint)
	}
}

func TestRequestValidationError(t *testing.T) {
	model, _ := New("gpt-4o", WithAPIKey("test-key"))

	// Empty model should fail validation
	_, err := model.Request(context.TODO(), &core.ChatRequest{
		Model:    "",
		Messages: nil,
	})
	if err == nil {
		t.Error("expected validation error")
	}
}

func TestBuildParamsMinimal(t *testing.T) {
	model, _ := New("gpt-4o", WithAPIKey("test-key"))

	req := &core.ChatRequest{
		Model: "gpt-4o",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
	}

	params := model.buildParams(req)
	if len(params.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(params.Messages))
	}
}
