package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNew(t *testing.T) {
	model, err := New("claude-sonnet-4-20250514")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "claude-sonnet-4-20250514" {
		t.Errorf("expected name %q, got %q", "claude-sonnet-4-20250514", model.Name())
	}
}

func TestNewWithOptions(t *testing.T) {
	model, err := New("claude-sonnet-4-20250514",
		WithAPIKey("test-key"),
		WithBaseURL("https://custom.api.com"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "claude-sonnet-4-20250514" {
		t.Errorf("expected name %q, got %q", "claude-sonnet-4-20250514", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := MustNew("claude-sonnet-4-20250514", WithAPIKey("test-key"))
	if model.Name() != "claude-sonnet-4-20250514" {
		t.Errorf("expected name %q, got %q", "claude-sonnet-4-20250514", model.Name())
	}
}

func TestConvertMessage(t *testing.T) {
	// Test text message
	msg := core.NewTextMessage(core.RoleUser, "hello")
	param := convertMessage(msg)
	if param.Role != anthropic.MessageParamRoleUser {
		t.Errorf("expected role user, got %q", param.Role)
	}

	// Test tool role conversion
	toolMsg := core.NewToolResultMessage("c1", "result", false)
	toolParam := convertMessage(toolMsg)
	if toolParam.Role != anthropic.MessageParamRoleUser {
		t.Errorf("expected tool role converted to user, got %q", toolParam.Role)
	}

	// Test assistant with tool use
	tuMsg := core.NewToolUseMessage(core.ToolUse{
		ID:    "c1",
		Name:  "test",
		Input: map[string]interface{}{"key": "val"},
	})
	tuParam := convertMessage(tuMsg)
	if tuParam.Role != anthropic.MessageParamRoleAssistant {
		t.Errorf("expected assistant role, got %q", tuParam.Role)
	}
	if len(tuParam.Content) != 1 {
		t.Errorf("expected 1 content block, got %d", len(tuParam.Content))
	}
}

func TestConvertMessageEmptyText(t *testing.T) {
	// Empty text should not be added
	msg := core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentText, Text: ""},
		},
	}
	param := convertMessage(msg)
	if len(param.Content) != 0 {
		t.Errorf("expected 0 content blocks for empty text, got %d", len(param.Content))
	}
}

func TestConvertTools(t *testing.T) {
	tools := []core.Tool{
		{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "test_tool",
				Description: "A test tool",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{"type": "string"},
					},
					"required": []interface{}{"query"},
				},
			},
		},
	}

	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].OfTool.Name != "test_tool" {
		t.Errorf("expected name %q, got %q", "test_tool", result[0].OfTool.Name)
	}
}

func TestConvertToolsNilParams(t *testing.T) {
	tools := []core.Tool{
		{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "simple",
				Description: "Simple tool",
				Parameters:  nil,
			},
		},
	}

	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
}

func TestConvertToolsStringRequired(t *testing.T) {
	tools := []core.Tool{
		{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "test",
				Description: "test",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
					"required":   []string{"field1"},
				},
			},
		},
	}

	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
}

func TestConvertToolsNoProperties(t *testing.T) {
	tools := []core.Tool{
		{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "test",
				Description: "test",
				Parameters: map[string]interface{}{
					"type": "object",
				},
			},
		},
	}

	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
}

func TestConvertToolChoice(t *testing.T) {
	// Required
	choice := convertToolChoice(core.ToolChoiceRequired)
	if choice.OfAny == nil {
		t.Error("expected OfAny for required tool choice")
	}

	// Auto (default)
	choice2 := convertToolChoice(core.ToolChoiceAuto)
	if choice2.OfAuto == nil {
		t.Error("expected OfAuto for auto tool choice")
	}

	// None (falls to default)
	choice3 := convertToolChoice(core.ToolChoiceNone)
	if choice3.OfAuto == nil {
		t.Error("expected OfAuto for none tool choice (default)")
	}
}

func TestConvertFinishReason(t *testing.T) {
	tests := []struct {
		input    anthropic.StopReason
		expected core.FinishReason
	}{
		{anthropic.StopReasonEndTurn, core.FinishReasonStop},
		{anthropic.StopReasonStopSequence, core.FinishReasonStop},
		{anthropic.StopReasonMaxTokens, core.FinishReasonLength},
		{anthropic.StopReasonToolUse, core.FinishReasonToolCalls},
		{anthropic.StopReason("unknown"), core.FinishReasonStop},
	}

	for _, tt := range tests {
		result := convertFinishReason(tt.input)
		if result != tt.expected {
			t.Errorf("convertFinishReason(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestSeparateSystemMessages(t *testing.T) {
	messages := []core.Message{
		core.NewTextMessage(core.RoleSystem, "system prompt"),
		core.NewTextMessage(core.RoleUser, "user message"),
		core.NewTextMessage(core.RoleAssistant, "assistant response"),
	}

	system, conversation := separateSystemMessages(messages)
	if len(system) != 1 {
		t.Errorf("expected 1 system message, got %d", len(system))
	}
	if len(conversation) != 2 {
		t.Errorf("expected 2 conversation messages, got %d", len(conversation))
	}
}

func TestSeparateSystemMessagesNoSystem(t *testing.T) {
	messages := []core.Message{
		core.NewTextMessage(core.RoleUser, "user message"),
	}

	system, conversation := separateSystemMessages(messages)
	if len(system) != 0 {
		t.Errorf("expected 0 system messages, got %d", len(system))
	}
	if len(conversation) != 1 {
		t.Errorf("expected 1 conversation message, got %d", len(conversation))
	}
}

func TestBuildParams(t *testing.T) {
	model, _ := New("claude-sonnet-4-20250514", WithAPIKey("test-key"))

	temp := 0.7
	maxTokens := 500
	topP := 0.9

	req := &core.ChatRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "system"),
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
		ToolChoice: func() *core.ToolChoice { tc := core.ToolChoiceRequired; return &tc }(),
	}

	params := model.buildParams(req)
	if params.MaxTokens != 500 {
		t.Errorf("expected max tokens 500, got %d", params.MaxTokens)
	}
	if len(params.System) != 1 {
		t.Errorf("expected 1 system block, got %d", len(params.System))
	}
	if len(params.Messages) != 1 {
		t.Errorf("expected 1 conversation message, got %d", len(params.Messages))
	}
	if len(params.Tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(params.Tools))
	}
}

func TestBuildParamsDefaults(t *testing.T) {
	model, _ := New("claude-sonnet-4-20250514", WithAPIKey("test-key"))

	req := &core.ChatRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
	}

	params := model.buildParams(req)
	if params.MaxTokens != 1024 {
		t.Errorf("expected default max tokens 1024, got %d", params.MaxTokens)
	}
}

func TestBuildParamsThinkingAndResponseFormat(t *testing.T) {
	model, _ := New("claude-sonnet-4-20250514", WithAPIKey("test-key"))
	temp := 0.2

	params := model.buildParams(&core.ChatRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
		Temperature: &temp,
		ResponseFormat: &core.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &core.JSONSchemaFormat{
				Schema: map[string]interface{}{"type": "object"},
			},
		},
		Thinking: &core.ThinkingConfig{Enabled: true},
	})

	if params.OutputConfig.Format.Schema["type"] != "object" {
		t.Fatalf("expected output schema to be preserved, got %#v", params.OutputConfig)
	}
	if params.Thinking.OfEnabled == nil || params.Thinking.OfEnabled.BudgetTokens != 10000 {
		t.Fatalf("expected thinking budget default, got %#v", params.Thinking)
	}
	if got := params.Temperature.Value; got != 1 {
		t.Fatalf("expected thinking to force temperature=1, got %#v", got)
	}
}

func TestConvertResponseMessageEmpty(t *testing.T) {
	// Test with empty content
	content := []anthropic.ContentBlockUnion{}
	msg := convertResponseMessage(content, "assistant")
	if msg.Role != core.RoleAssistant {
		t.Errorf("expected role assistant, got %q", msg.Role)
	}
	if msg.GetTextContent() != "" {
		t.Errorf("expected empty text, got %q", msg.GetTextContent())
	}
}

func TestConvertResponseMessageUnknownType(t *testing.T) {
	// Unknown type should be skipped
	content := []anthropic.ContentBlockUnion{
		{Type: "unknown_block_type"},
	}
	msg := convertResponseMessage(content, "assistant")
	if len(msg.Content) != 0 {
		t.Errorf("expected 0 content parts for unknown type, got %d", len(msg.Content))
	}
}

func TestConvertResponseFull(t *testing.T) {
	model, _ := New("claude-sonnet-4-20250514", WithAPIKey("test-key"))

	resp := &anthropic.Message{
		ID:    "msg_test",
		Model: "claude-sonnet-4-20250514",
		Content: []anthropic.ContentBlockUnion{
			{Type: "text", Text: "hello"},
		},
		StopReason: anthropic.StopReasonEndTurn,
		Usage: anthropic.Usage{
			InputTokens:  10,
			OutputTokens: 5,
		},
		Role: "assistant",
	}

	result := model.convertResponse(resp)
	if result.ID != "msg_test" {
		t.Errorf("expected ID %q, got %q", "msg_test", result.ID)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}
	if result.Choices[0].FinishReason != core.FinishReasonStop {
		t.Errorf("expected stop, got %q", result.Choices[0].FinishReason)
	}
	if result.Usage.PromptTokens != 10 {
		t.Errorf("expected 10 prompt tokens, got %d", result.Usage.PromptTokens)
	}
	if result.Usage.CompletionTokens != 5 {
		t.Errorf("expected 5 completion tokens, got %d", result.Usage.CompletionTokens)
	}
	if result.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", result.Usage.TotalTokens)
	}
}

func TestConvertResponseMessageTextType(t *testing.T) {
	// Construct a ContentBlockUnion with type "text" — since AsText()
	// accesses internal fields, we test the function handles the branch
	content := []anthropic.ContentBlockUnion{
		{Type: "text"},
		{Type: "tool_use"},
	}
	msg := convertResponseMessage(content, "assistant")
	// The text/tool blocks may not produce content due to uninitialized internal fields,
	// but at least the function shouldn't panic
	if msg.Role != core.RoleAssistant {
		t.Errorf("expected assistant role, got %q", msg.Role)
	}
}

func TestRequestValidationError(t *testing.T) {
	model, _ := New("claude-sonnet-4-20250514", WithAPIKey("test-key"))

	// Empty model should fail validation
	_, err := model.Request(context.TODO(), &core.ChatRequest{
		Model:    "",
		Messages: nil,
	})
	if err == nil {
		t.Error("expected validation error")
	}
}

func TestRequestServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer server.Close()

	model, err := New("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = model.Request(context.Background(), &core.ChatRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
	})
	if err == nil {
		t.Fatal("expected request error")
	}
}
