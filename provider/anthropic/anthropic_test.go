package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
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

// TestConvertToolsNoProperties pins the schema emitted for a zero-argument
// tool. Rebuilding the schema from the raw parameter map used to fall back to
// "properties = params", turning the declared {"type":"object"} into a phantom
// argument named "type".
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

	schema := result[0].OfTool.InputSchema
	props, ok := schema.Properties.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map properties, got %#v", schema.Properties)
	}
	if len(props) != 0 {
		t.Errorf("expected empty properties for a zero-argument tool, got %#v", props)
	}
	if len(schema.Required) != 0 {
		t.Errorf("expected no required fields, got %#v", schema.Required)
	}
	if _, injected := schema.ExtraFields["additionalProperties"]; injected {
		t.Errorf("additionalProperties must not be injected, got %#v", schema.ExtraFields)
	}
}

func TestConvertToolsPreservesDefsAndCallerConstraints(t *testing.T) {
	defs := map[string]interface{}{
		"Address": map[string]interface{}{"type": "object"},
	}
	tools := []core.Tool{
		{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name: "ship",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"to": map[string]interface{}{"$ref": "#/$defs/Address"},
					},
					"required":             []interface{}{"to"},
					"$defs":                defs,
					"additionalProperties": true,
					"description":          "shipping request",
				},
			},
		},
	}

	schema := convertTools(tools)[0].OfTool.InputSchema

	if got := schema.ExtraFields["$defs"]; !reflect.DeepEqual(got, defs) {
		t.Errorf("$defs must be forwarded so $ref resolves, got %#v", got)
	}
	if got := schema.ExtraFields["additionalProperties"]; got != true {
		t.Errorf("caller's additionalProperties must stand, got %#v", got)
	}
	if got := schema.ExtraFields["description"]; got != "shipping request" {
		t.Errorf("unknown schema keys must be forwarded, got %#v", got)
	}
	if _, dup := schema.ExtraFields["type"]; dup {
		t.Error(`"type" must not be duplicated into ExtraFields`)
	}
	if !reflect.DeepEqual(schema.Required, []string{"to"}) {
		t.Errorf("unexpected required %#v", schema.Required)
	}
}

func TestConvertToolChoice(t *testing.T) {
	tests := []struct {
		name     string
		choice   core.ToolChoice
		thinking bool
		want     string
	}{
		{"required", core.ToolChoiceRequired, false, "any"},
		{"auto", core.ToolChoiceAuto, false, "auto"},
		// "none" used to fall through to auto, silently letting the model
		// call tools the caller had disabled.
		{"none", core.ToolChoiceNone, false, "none"},
		{"none with thinking", core.ToolChoiceNone, true, "none"},
		// Anthropic rejects a forced tool choice while thinking is on.
		{"required downgrades under thinking", core.ToolChoiceRequired, true, "auto"},
		{"auto with thinking", core.ToolChoiceAuto, true, "auto"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertToolChoice(tt.choice, tt.thinking)
			var kind string
			switch {
			case got.OfAny != nil:
				kind = "any"
			case got.OfNone != nil:
				kind = "none"
			case got.OfAuto != nil:
				kind = "auto"
			}
			if kind != tt.want {
				t.Errorf("convertToolChoice(%q, %v) = %q, want %q", tt.choice, tt.thinking, kind, tt.want)
			}
		})
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
		{anthropic.StopReasonRefusal, core.FinishReasonContentFilter},
		{anthropic.StopReasonPauseTurn, core.FinishReasonError},
		// An unrecognized reason must not be reported as a clean stop.
		{anthropic.StopReason("some_future_reason"), core.FinishReasonUnknown},
		{anthropic.StopReason(""), core.FinishReason("")},
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
	// The 1024 default ceiling sits below the 10000 budget, which Anthropic
	// rejects outright. The floor must clear the budget.
	if params.MaxTokens <= params.Thinking.OfEnabled.BudgetTokens {
		t.Fatalf("max tokens %d must exceed thinking budget %d",
			params.MaxTokens, params.Thinking.OfEnabled.BudgetTokens)
	}
}

func TestBuildParamsThinkingMaxTokensFloor(t *testing.T) {
	tests := []struct {
		name      string
		budget    int
		maxTokens *int
		want      int64
	}{
		{"default budget lifts default max", 0, nil, 10000 + thinkingAnswerHeadroom},
		{"explicit budget lifts default max", 4000, nil, 4000 + thinkingAnswerHeadroom},
		{"caller max below budget is lifted", 4000, intPtr(500), 4000 + thinkingAnswerHeadroom},
		{"caller max equal to budget is lifted", 4000, intPtr(4000), 4000 + thinkingAnswerHeadroom},
		{"caller max above budget is respected", 4000, intPtr(9000), 9000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, _ := New("claude-sonnet-4-20250514", WithAPIKey("test-key"))
			params := model.buildParams(&core.ChatRequest{
				Model:     "claude-sonnet-4-20250514",
				Messages:  []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
				MaxTokens: tt.maxTokens,
				Thinking:  &core.ThinkingConfig{Enabled: true, BudgetTokens: tt.budget},
			})
			if params.MaxTokens != tt.want {
				t.Errorf("max tokens = %d, want %d", params.MaxTokens, tt.want)
			}
			if params.MaxTokens <= params.Thinking.OfEnabled.BudgetTokens {
				t.Errorf("max tokens %d does not exceed budget %d",
					params.MaxTokens, params.Thinking.OfEnabled.BudgetTokens)
			}
		})
	}
}

func TestBuildParamsAdaptiveThinkingModels(t *testing.T) {
	temp := 0.3
	topP := 0.8
	topK := 40

	for _, name := range []string{
		"claude-opus-4-7",
		"claude-opus-4-8",
		"claude-sonnet-5-20260101",
		"claude-fable-5",
		"claude-mythos-5",
	} {
		t.Run(name, func(t *testing.T) {
			model, _ := New(name, WithAPIKey("test-key"))
			params := model.buildParams(&core.ChatRequest{
				Model:       name,
				Messages:    []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
				Temperature: &temp,
				TopP:        &topP,
				Thinking:    &core.ThinkingConfig{Enabled: true, BudgetTokens: 5000},
				ProviderOptions: map[string]any{
					ProviderKey: Options{TopK: &topK},
				},
			})

			if params.Thinking.OfAdaptive == nil {
				t.Fatalf("expected adaptive thinking, got %#v", params.Thinking)
			}
			if params.Thinking.OfEnabled != nil {
				t.Error("adaptive models must not receive a budget thinking config")
			}
			if params.Temperature.Valid() {
				t.Errorf("temperature must be omitted, got %#v", params.Temperature)
			}
			if params.TopP.Valid() {
				t.Errorf("top_p must be omitted, got %#v", params.TopP)
			}
			if params.TopK.Valid() {
				t.Errorf("top_k must be omitted, got %#v", params.TopK)
			}
		})
	}
}

func TestBuildParamsBudgetThinkingModelUnaffected(t *testing.T) {
	// A model outside the adaptive families keeps the budget form.
	model, _ := New("claude-sonnet-4-6", WithAPIKey("test-key"))
	params := model.buildParams(&core.ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		Thinking: &core.ThinkingConfig{Enabled: true, BudgetTokens: 2000},
	})
	if params.Thinking.OfEnabled == nil || params.Thinking.OfAdaptive != nil {
		t.Fatalf("expected budget thinking, got %#v", params.Thinking)
	}
}

func TestBuildParamsStopSequencesAndProviderOptions(t *testing.T) {
	model, _ := New("claude-sonnet-4-20250514", WithAPIKey("test-key"))
	topK := 25

	base := func(opts map[string]any) *core.ChatRequest {
		return &core.ChatRequest{
			Model:         "claude-sonnet-4-20250514",
			Messages:      []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
			StopSequences: []string{"STOP", "END"},
			Tools: []core.Tool{
				{Type: core.ToolTypeFunction, Function: core.Function{Name: "a"}},
				{Type: core.ToolTypeFunction, Function: core.Function{Name: "b"}},
			},
			ProviderOptions: opts,
		}
	}

	params := model.buildParams(base(map[string]any{
		ProviderKey: Options{TopK: &topK, CacheToolDefs: "1h"},
	}))
	if len(params.StopSequences) != 2 || params.StopSequences[0] != "STOP" {
		t.Errorf("unexpected stop sequences %#v", params.StopSequences)
	}
	if params.TopK.Value != 25 {
		t.Errorf("expected top_k 25, got %#v", params.TopK)
	}
	if cc := params.Tools[1].OfTool.CacheControl; cc.TTL != anthropic.CacheControlEphemeralTTLTTL1h {
		t.Errorf("expected 1h cache breakpoint on the last tool, got %#v", cc)
	}
	if cc := params.Tools[0].OfTool.CacheControl; cc.TTL != "" {
		t.Errorf("only the last tool carries the breakpoint, got %#v", cc)
	}

	// A pointer value is accepted too.
	ptrParams := model.buildParams(base(map[string]any{
		ProviderKey: &Options{CacheToolDefs: "5m"},
	}))
	if cc := ptrParams.Tools[1].OfTool.CacheControl; cc.TTL != anthropic.CacheControlEphemeralTTLTTL5m {
		t.Errorf("expected 5m cache breakpoint, got %#v", cc)
	}
}

func TestBuildParamsIgnoresForeignAndMistypedOptions(t *testing.T) {
	model, _ := New("claude-sonnet-4-20250514", WithAPIKey("test-key"))

	for _, opts := range []map[string]any{
		{"openai": struct{ TopK int }{5}},
		{ProviderKey: "not-an-options-struct"},
		{ProviderKey: nil},
		{ProviderKey: (*Options)(nil)},
	} {
		params := model.buildParams(&core.ChatRequest{
			Model:    "claude-sonnet-4-20250514",
			Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
			Tools: []core.Tool{
				{Type: core.ToolTypeFunction, Function: core.Function{Name: "a"}},
			},
			ProviderOptions: opts,
		})
		if params.TopK.Valid() {
			t.Errorf("top_k must stay unset for %#v, got %#v", opts, params.TopK)
		}
		if cc := params.Tools[0].OfTool.CacheControl; cc.TTL != "" {
			t.Errorf("no cache breakpoint expected for %#v, got %#v", opts, cc)
		}
	}
}

func TestApplyToolCacheControlIgnoresUnknownTTL(t *testing.T) {
	tools := convertTools([]core.Tool{
		{Type: core.ToolTypeFunction, Function: core.Function{Name: "a"}},
	})
	applyToolCacheControl(tools, "7d")
	if cc := tools[0].OfTool.CacheControl; cc.TTL != "" {
		t.Errorf("unknown ttl must be ignored, got %#v", cc)
	}
	applyToolCacheControl(nil, "5m") // must not panic
}

func intPtr(v int) *int { return &v }

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

	// Built from JSON: ContentBlockUnion.AsText reads the raw payload, so a
	// struct literal would produce an empty text block.
	var textBlock anthropic.ContentBlockUnion
	if err := json.Unmarshal([]byte(`{"type":"text","text":"hello"}`), &textBlock); err != nil {
		t.Fatalf("unmarshal content block: %v", err)
	}

	resp := &anthropic.Message{
		ID:         "msg_test",
		Model:      "claude-sonnet-4-20250514",
		Content:    []anthropic.ContentBlockUnion{textBlock},
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
	if result.Message.GetTextContent() != "hello" {
		t.Errorf("expected message text %q, got %q", "hello", result.Message.GetTextContent())
	}
	if result.FinishReason != core.FinishReasonStop {
		t.Errorf("expected stop, got %q", result.FinishReason)
	}
	if result.RawFinishReason != "end_turn" {
		t.Errorf("expected raw finish reason %q, got %q", "end_turn", result.RawFinishReason)
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
