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
	params := convertMessages(msg)
	if len(params) != 1 || params[0].OfSystem == nil {
		t.Error("expected OfSystem for system message")
	}
}

func TestConvertMessageUser(t *testing.T) {
	msg := core.NewTextMessage(core.RoleUser, "hello")
	params := convertMessages(msg)
	if len(params) != 1 || params[0].OfUser == nil {
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
	params := convertMessages(msg)
	if len(params) != 1 || params[0].OfUser == nil {
		t.Error("expected OfUser for multi-part user message")
	}
}

func TestConvertMessageAssistant(t *testing.T) {
	msg := core.NewTextMessage(core.RoleAssistant, "response")
	params := convertMessages(msg)
	if len(params) != 1 || params[0].OfAssistant == nil {
		t.Error("expected OfAssistant for assistant message")
	}
}

func TestConvertMessageAssistantWithToolCalls(t *testing.T) {
	msg := core.NewToolUseMessage(core.ToolUse{
		ID:    "c1",
		Name:  "test",
		Input: map[string]interface{}{"key": "val"},
	})
	params := convertMessages(msg)
	if len(params) != 1 || params[0].OfAssistant == nil {
		t.Fatal("expected OfAssistant for tool use message")
	}
	if len(params[0].OfAssistant.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(params[0].OfAssistant.ToolCalls))
	}
}

func TestConvertMessageTool(t *testing.T) {
	msg := core.NewToolResultMessage("c1", "result", false)
	params := convertMessages(msg)
	if len(params) != 1 || params[0].OfTool == nil {
		t.Error("expected OfTool for tool result message")
	}
}

func TestConvertMessageToolNoResults(t *testing.T) {
	// Tool message with no results
	msg := core.Message{
		Role:    core.RoleTool,
		Content: []core.Part{{Type: core.ContentText, Text: "raw text"}},
	}
	params := convertMessages(msg)
	if len(params) != 1 || params[0].OfTool == nil {
		t.Error("expected OfTool for tool message without results")
	}
}

func TestConvertMessageDefault(t *testing.T) {
	// Unknown role falls to default (user message)
	msg := core.Message{
		Role:    core.MessageRole("custom"),
		Content: []core.Part{{Type: core.ContentText, Text: "hi"}},
	}
	params := convertMessages(msg)
	if len(params) != 1 || params[0].OfUser == nil {
		t.Error("expected OfUser for unknown role (default)")
	}
}

func TestConvertContentPart(t *testing.T) {
	textPart := core.Part{Type: core.ContentText, Text: "hello"}
	result, ok := convertContentPart(textPart)
	if !ok || result.OfText == nil {
		t.Error("expected OfText for text content part")
	}
}

func TestConvertContentPartImage(t *testing.T) {
	imgPart := core.Part{
		Type:     core.ContentImageURL,
		ImageURL: &core.ImageURL{URL: "https://example.com/img.png", Detail: "high"},
	}
	result, ok := convertContentPart(imgPart)
	if !ok || result.OfImageURL == nil {
		t.Error("expected OfImageURL for image content part")
	}
}

func TestConvertContentPartImageNoDetail(t *testing.T) {
	imgPart := core.Part{
		Type:     core.ContentImageURL,
		ImageURL: &core.ImageURL{URL: "https://example.com/img.png"},
	}
	result, ok := convertContentPart(imgPart)
	if !ok || result.OfImageURL == nil {
		t.Error("expected OfImageURL for image content part without detail")
	}
}

func TestConvertContentPartImageNil(t *testing.T) {
	imgPart := core.Part{
		Type:     core.ContentImageURL,
		ImageURL: nil,
	}
	// An image part with no payload and no text carries nothing to send.
	result, ok := convertContentPart(imgPart)
	if ok {
		t.Errorf("expected an image part with no URL to be skipped, got %#v", result)
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
		name     string
		input    string
		expected core.FinishReason
	}{
		{"stop", "stop", core.FinishReasonStop},
		{"length", "length", core.FinishReasonLength},
		{"tool calls", "tool_calls", core.FinishReasonToolCalls},
		{"deprecated function call", "function_call", core.FinishReasonToolCalls},
		{"content filter", "content_filter", core.FinishReasonContentFilter},
		{"gateway error", "error", core.FinishReasonError},
		// A reason none reported is a clean stop, matching pydantic-ai.
		{"absent", "", core.FinishReasonStop},
		// An unrecognized reason must never be reported as a success: the
		// caller sees FinishReasonUnknown and inspects RawFinishReason.
		{"unrecognized", "something_new", core.FinishReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if result := convertFinishReason(tt.input); result != tt.expected {
				t.Errorf("convertFinishReason(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestConvertResponsePreservesRawFinishReason(t *testing.T) {
	model, _ := New("gpt-4o", WithAPIKey("test-key"))

	result := model.convertResponse(&openai.ChatCompletion{
		ID:    "test-id",
		Model: "gpt-4o",
		Choices: []openai.ChatCompletionChoice{{
			Message:      openai.ChatCompletionMessage{Role: "assistant", Content: "hi"},
			FinishReason: "some_new_reason",
		}},
	})

	if result.FinishReason != core.FinishReasonUnknown {
		t.Errorf("expected unknown finish reason, got %q", result.FinishReason)
	}
	if result.RawFinishReason != "some_new_reason" {
		t.Errorf("expected the provider's reason to pass through, got %q", result.RawFinishReason)
	}
}

// TestConvertResponseSurfacesRefusal pins that a safety refusal is not
// reported as an empty, clean stop.
func TestConvertResponseSurfacesRefusal(t *testing.T) {
	model, _ := New("gpt-4o", WithAPIKey("test-key"))

	result := model.convertResponse(&openai.ChatCompletion{
		ID:    "test-id",
		Model: "gpt-4o",
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{
				Role:    "assistant",
				Refusal: "I can't help with that.",
			},
			FinishReason: "stop",
		}},
	})

	if result.FinishReason != core.FinishReasonContentFilter {
		t.Errorf("expected content_filter finish reason, got %q", result.FinishReason)
	}
	if result.RawFinishReason != "stop" {
		t.Errorf("expected raw reason to stay lossless, got %q", result.RawFinishReason)
	}
	if got := result.Message.GetTextContent(); got != "I can't help with that." {
		t.Errorf("expected the refusal text to be preserved, got %q", got)
	}
}

func TestConvertResponseWithoutChoices(t *testing.T) {
	model, _ := New("gpt-4o", WithAPIKey("test-key"))

	result := model.convertResponse(&openai.ChatCompletion{ID: "test-id", Model: "gpt-4o"})
	if result.FinishReason != core.FinishReasonError {
		t.Errorf("expected a choiceless response to report an error, got %q", result.FinishReason)
	}
	if len(result.Message.Content) != 0 {
		t.Errorf("expected an empty message, got %#v", result.Message)
	}
}

// TestConvertMessagesToolMultipleResults pins that every result in a tool
// message is sent, not only the first.
func TestConvertMessagesToolMultipleResults(t *testing.T) {
	msg := core.Message{
		Role: core.RoleTool,
		Content: []core.Part{
			{Type: core.ContentToolResult, ToolResult: &core.ToolResult{ToolUseID: "c1", Content: "first"}},
			{Type: core.ContentToolResult, ToolResult: &core.ToolResult{ToolUseID: "c2", Content: "second"}},
		},
	}

	params := convertMessages(msg)
	if len(params) != 2 {
		t.Fatalf("expected one tool message per result, got %d", len(params))
	}
	for i, want := range []struct{ id, content string }{{"c1", "first"}, {"c2", "second"}} {
		tool := params[i].OfTool
		if tool == nil {
			t.Fatalf("message %d is not a tool message: %#v", i, params[i])
		}
		if tool.ToolCallID != want.id {
			t.Errorf("message %d: expected tool_call_id %q, got %q", i, want.id, tool.ToolCallID)
		}
		if tool.Content.OfString.Or("") != want.content {
			t.Errorf("message %d: expected content %q, got %q", i, want.content, tool.Content.OfString.Or(""))
		}
	}
}

func TestBuildParamsSamplingParamsAndReasoningModels(t *testing.T) {
	temp := 0.7
	topP := 0.9

	tests := []struct {
		name            string
		model           string
		thinking        bool
		wantSamplingSet bool
	}{
		{"non-reasoning model keeps sampling params", "gpt-4o", false, true},
		{"o-series rejects sampling params", "o3", false, false},
		{"o-series rejects them with thinking on too", "o1-mini", true, false},
		{"gpt-5 reasons by default", "gpt-5", false, false},
		{"gpt-5-chat does not reason", "gpt-5-chat-latest", false, true},
		{"gpt-5.4 is opt-in, so keeps them when thinking is off", "gpt-5.4", false, true},
		{"gpt-5.4 drops them once thinking is on", "gpt-5.4", true, false},
		{"gateway-qualified name resolves to the same family", "openai/gpt-5", false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, _ := New(tt.model, WithAPIKey("test-key"))
			req := &core.ChatRequest{
				Model:       tt.model,
				Messages:    []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
				Temperature: &temp,
				TopP:        &topP,
			}
			if tt.thinking {
				req.Thinking = &core.ThinkingConfig{Enabled: true}
			}

			params := model.buildParams(req)
			if got := params.Temperature.Valid(); got != tt.wantSamplingSet {
				t.Errorf("temperature sent = %v, want %v", got, tt.wantSamplingSet)
			}
			if got := params.TopP.Valid(); got != tt.wantSamplingSet {
				t.Errorf("top_p sent = %v, want %v", got, tt.wantSamplingSet)
			}
		})
	}
}

func TestBuildParamsStopSequencesAndProviderOptions(t *testing.T) {
	model, _ := New("gpt-4o", WithAPIKey("test-key"))
	seed := int64(42)
	parallel := false

	params := model.buildParams(&core.ChatRequest{
		Model:         "gpt-4o",
		Messages:      []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		StopSequences: []string{"END", "\n\n"},
		ProviderOptions: map[string]any{
			"openai":    Options{Seed: &seed, ParallelToolCalls: &parallel, ServiceTier: "flex"},
			"anthropic": struct{ TopK int }{TopK: 40},
		},
	})

	if len(params.Stop.OfStringArray) != 2 || params.Stop.OfStringArray[0] != "END" {
		t.Errorf("expected stop sequences to be wired, got %#v", params.Stop)
	}
	if params.Seed.Or(0) != seed {
		t.Errorf("expected seed %d, got %d", seed, params.Seed.Or(0))
	}
	if params.ParallelToolCalls.Or(true) != false {
		t.Errorf("expected parallel_tool_calls=false, got %#v", params.ParallelToolCalls)
	}
	if params.ServiceTier != openai.ChatCompletionNewParamsServiceTierFlex {
		t.Errorf("expected flex service tier, got %q", params.ServiceTier)
	}
}

func TestBuildParamsIgnoresMalformedProviderOptions(t *testing.T) {
	model, _ := New("gpt-4o", WithAPIKey("test-key"))
	seed := int64(7)

	tests := []struct {
		name     string
		options  map[string]any
		wantSeed int64
	}{
		{"absent", nil, 0},
		{"wrong type under our key", map[string]any{"openai": "not-options"}, 0},
		{"nil pointer", map[string]any{"openai": (*Options)(nil)}, 0},
		{"pointer form is accepted", map[string]any{"openai": &Options{Seed: &seed}}, seed},
		{"unknown service tier is dropped", map[string]any{"openai": Options{ServiceTier: "turbo"}}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := model.buildParams(&core.ChatRequest{
				Model:           "gpt-4o",
				Messages:        []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
				ProviderOptions: tt.options,
			})
			if params.Seed.Or(0) != tt.wantSeed {
				t.Errorf("expected seed %d, got %d", tt.wantSeed, params.Seed.Or(0))
			}
			if params.ServiceTier != "" {
				t.Errorf("expected no service tier, got %q", params.ServiceTier)
			}
		})
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

	tc := core.ToolChoiceRequired
	req := &core.ChatRequest{
		Model: "gpt-4o",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
		Temperature: &temp,
		MaxTokens:   &maxTokens,
		TopP:        &topP,
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
		Created: 1234567890,
	}

	result := model.convertResponse(resp)
	if result.ID != "test-id" {
		t.Errorf("expected ID %q, got %q", "test-id", result.ID)
	}
	if result.Model != "gpt-4o" {
		t.Errorf("expected model %q, got %q", "gpt-4o", result.Model)
	}
	// Only the first choice is converted; the API returns one per request.
	if result.Message.GetTextContent() != "hello" {
		t.Errorf("expected %q, got %q", "hello", result.Message.GetTextContent())
	}
	if result.FinishReason != core.FinishReasonStop {
		t.Errorf("expected stop, got %q", result.FinishReason)
	}
	if result.RawFinishReason != "stop" {
		t.Errorf("expected raw reason %q, got %q", "stop", result.RawFinishReason)
	}
	if result.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", result.Usage.TotalTokens)
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
