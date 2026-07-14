package openai

import (
	"testing"

	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNewResponses(t *testing.T) {
	model, err := NewResponses("gpt-4o", WithAPIKey("sk-test"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "gpt-4o" {
		t.Errorf("expected name %q, got %q", "gpt-4o", model.Name())
	}
}

func TestMustNewResponses(t *testing.T) {
	model := MustNewResponses("o3", WithAPIKey("sk-test"))
	if model.Name() != "o3" {
		t.Errorf("expected name %q, got %q", "o3", model.Name())
	}
}

func TestNewResponsesFromClient(t *testing.T) {
	chatModel := MustNew("gpt-4o", WithAPIKey("sk-test"))
	respModel := NewResponsesFromClient("o3", chatModel)

	if respModel.Name() != "o3" {
		t.Errorf("expected name %q, got %q", "o3", respModel.Name())
	}
}

func TestResponsesImplementsInterfaces(t *testing.T) {
	model, _ := NewResponses("gpt-4o", WithAPIKey("sk-test"))

	var _ core.Model = model
	var _ core.StreamModel = model
}

func TestConvertResponsesTools(t *testing.T) {
	tools := []core.Tool{
		{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "get_weather",
				Description: "Get weather for a city",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{"type": "string"},
					},
					"required": []interface{}{"city"},
				},
			},
		},
	}

	result := convertResponsesTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].OfFunction == nil {
		t.Fatal("expected function tool")
	}
	if result[0].OfFunction.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", result[0].OfFunction.Name)
	}
}

func TestConvertResponsesToolChoice(t *testing.T) {
	tests := []struct {
		choice   core.ToolChoice
		expected responses.ToolChoiceOptions
	}{
		{core.ToolChoiceNone, responses.ToolChoiceOptionsNone},
		{core.ToolChoiceAuto, responses.ToolChoiceOptionsAuto},
		{core.ToolChoiceRequired, responses.ToolChoiceOptionsRequired},
	}

	for _, tt := range tests {
		tc := convertResponsesToolChoice(tt.choice)
		got := tc.OfToolChoiceMode.Or("")
		if got != tt.expected {
			t.Errorf("convertResponsesToolChoice(%q) = %q, want %q", tt.choice, got, tt.expected)
		}
	}
}

func TestConvertTextConfig(t *testing.T) {
	t.Run("json_schema", func(t *testing.T) {
		rf := &core.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &core.JSONSchemaFormat{
				Name:   "test_output",
				Schema: map[string]interface{}{"type": "object"},
			},
		}
		tc := convertTextConfig(rf)
		if tc.Format.OfJSONSchema == nil {
			t.Fatal("expected json_schema format")
		}
		if tc.Format.OfJSONSchema.Name != "test_output" {
			t.Errorf("expected name 'test_output', got %q", tc.Format.OfJSONSchema.Name)
		}
	})

	t.Run("json_object", func(t *testing.T) {
		rf := &core.ResponseFormat{Type: "json_object"}
		tc := convertTextConfig(rf)
		if tc.Format.OfJSONObject == nil {
			t.Fatal("expected json_object format")
		}
	})

	t.Run("text", func(t *testing.T) {
		rf := &core.ResponseFormat{Type: "text"}
		tc := convertTextConfig(rf)
		if tc.Format.OfText == nil {
			t.Fatal("expected text format")
		}
	})
}

func TestConvertReasoning(t *testing.T) {
	tests := []struct {
		budget   int
		expected shared.ReasoningEffort
	}{
		{0, shared.ReasoningEffortMedium},
		{3000, shared.ReasoningEffortLow},
		{10000, shared.ReasoningEffortMedium},
		{50000, shared.ReasoningEffortHigh},
	}

	for _, tt := range tests {
		r := convertReasoning(&core.ThinkingConfig{
			Enabled:      true,
			BudgetTokens: tt.budget,
		})
		if r.Effort != tt.expected {
			t.Errorf("budget %d: got effort %q, want %q", tt.budget, r.Effort, tt.expected)
		}
		if r.Summary != shared.ReasoningSummaryAuto {
			t.Errorf("expected summary 'auto', got %q", r.Summary)
		}
	}
}

func TestConvertInputItems(t *testing.T) {
	t.Run("user message", func(t *testing.T) {
		msg := core.Message{
			Role: core.RoleUser,
			Content: []core.Part{
				{Type: core.ContentText, Text: "Hello"},
			},
		}
		items := convertInputItems(msg)
		if len(items) != 1 {
			t.Fatalf("expected 1 item, got %d", len(items))
		}
		if items[0].OfMessage == nil {
			t.Fatal("expected message item")
		}
		if string(items[0].OfMessage.Role) != "user" {
			t.Errorf("expected role 'user', got %q", items[0].OfMessage.Role)
		}
	})

	t.Run("assistant with tool call", func(t *testing.T) {
		msg := core.Message{
			Role: core.RoleAssistant,
			Content: []core.Part{
				{Type: core.ContentText, Text: "Let me check"},
				{
					Type: core.ContentToolUse,
					ToolUse: &core.ToolUse{
						ID:    "call_123",
						Name:  "get_weather",
						Input: map[string]interface{}{"city": "NYC"},
					},
				},
			},
		}
		items := convertInputItems(msg)
		if len(items) != 2 {
			t.Fatalf("expected 2 items (text + function_call), got %d", len(items))
		}
		if items[0].OfMessage == nil {
			t.Fatal("expected first item to be message")
		}
		if items[1].OfFunctionCall == nil {
			t.Fatal("expected second item to be function call")
		}
		if items[1].OfFunctionCall.CallID != "call_123" {
			t.Errorf("expected call_id 'call_123', got %q", items[1].OfFunctionCall.CallID)
		}
	})

	t.Run("tool result", func(t *testing.T) {
		msg := core.Message{
			Role: core.RoleTool,
			Content: []core.Part{
				{
					Type: core.ContentToolResult,
					ToolResult: &core.ToolResult{
						ToolUseID: "call_123",
						Content:   `{"temp": 72}`,
					},
				},
			},
		}
		items := convertInputItems(msg)
		if len(items) != 1 {
			t.Fatalf("expected 1 item, got %d", len(items))
		}
		if items[0].OfFunctionCallOutput == nil {
			t.Fatal("expected function call output item")
		}
		if items[0].OfFunctionCallOutput.CallID != "call_123" {
			t.Errorf("expected call_id 'call_123', got %q", items[0].OfFunctionCallOutput.CallID)
		}
	})
}

func TestExtractResponsesUsage(t *testing.T) {
	u := responses.ResponseUsage{
		InputTokens:  100,
		OutputTokens: 50,
		TotalTokens:  150,
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: 30,
		},
		OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
			ReasoningTokens: 10,
		},
	}

	usage := extractResponsesUsage(u)
	if usage.PromptTokens != 100 {
		t.Errorf("expected PromptTokens 100, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 50 {
		t.Errorf("expected CompletionTokens 50, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 150 {
		t.Errorf("expected TotalTokens 150, got %d", usage.TotalTokens)
	}
	if usage.CacheReadTokens != 30 {
		t.Errorf("expected CacheReadTokens 30, got %d", usage.CacheReadTokens)
	}
	if usage.ReasoningTokens != 10 {
		t.Errorf("expected ReasoningTokens 10, got %d", usage.ReasoningTokens)
	}
}
