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
				ID:               "rs_1",
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
	if chatResp.FinishReason != core.FinishReasonToolCalls {
		t.Fatalf("expected tool-calls finish reason, got %q", chatResp.FinishReason)
	}
	if chatResp.RawFinishReason != "completed" {
		t.Fatalf("expected the raw status to pass through, got %q", chatResp.RawFinishReason)
	}
	msg := chatResp.Message
	if msg.GetTextContent() != "answer" {
		t.Fatalf("unexpected text content %q", msg.GetTextContent())
	}
	if len(msg.GetToolUses()) != 1 || msg.GetToolUses()[0].Name != "lookup" {
		t.Fatalf("unexpected tool uses %#v", msg.GetToolUses())
	}
	if msg.GetThinkingContent() != "reasoning summary" {
		t.Fatalf("unexpected thinking content %q", msg.GetThinkingContent())
	}
}

// TestResponsesConvertResponseCapturesReasoningIdentity pins the halves of the
// reasoning round-trip that a replayed turn needs: the item id, the encrypted
// content on the first summary only, and an item that carries no summary at
// all.
func TestResponsesConvertResponseCapturesReasoningIdentity(t *testing.T) {
	model := &ResponsesModel{model: "o3"}

	resp := &responses.Response{
		Status: "completed",
		Output: []responses.ResponseOutputItemUnion{
			{
				Type:             "reasoning",
				ID:               "rs_1",
				EncryptedContent: "enc_abc",
				Summary: []responses.ResponseReasoningItemSummary{
					{Text: "first"},
					{Text: "second"},
				},
			},
			{
				Type:             "reasoning",
				ID:               "rs_2",
				EncryptedContent: "enc_def",
			},
		},
	}

	parts := model.convertResponse(resp).Message.Content
	if len(parts) != 3 {
		t.Fatalf("expected 3 thinking parts, got %d: %#v", len(parts), parts)
	}

	for i, want := range []struct {
		id, text, signature string
	}{
		{"rs_1", "first", "enc_abc"},
		// The signature belongs to the item, not to each summary: repeating
		// it would replay one signature as several.
		{"rs_1", "second", ""},
		// An empty summary must not drop the item — that loses the
		// encrypted content and with it the ability to replay the turn.
		{"rs_2", "", "enc_def"},
	} {
		block := parts[i].Thinking
		if block == nil {
			t.Fatalf("part %d is not a thinking part: %#v", i, parts[i])
		}
		if block.ID != want.id || block.Text != want.text || block.Signature != want.signature {
			t.Errorf("part %d = %+v, want id=%q text=%q signature=%q", i, *block, want.id, want.text, want.signature)
		}
		if block.ProviderName != "openai" {
			t.Errorf("part %d: expected provider name to be recorded, got %q", i, block.ProviderName)
		}
	}
}

// TestResponsesConvertResponseSurfacesRefusal pins that a refusal is not
// reported as a clean stop.
func TestResponsesConvertResponseSurfacesRefusal(t *testing.T) {
	model := &ResponsesModel{model: "gpt-4.1"}

	out := model.convertResponse(&responses.Response{
		Status: "completed",
		Output: []responses.ResponseOutputItemUnion{{
			Type: "message",
			Content: []responses.ResponseOutputMessageContentUnion{
				{Type: "refusal", Refusal: "I can't help with that."},
			},
		}},
	})

	if out.FinishReason != core.FinishReasonContentFilter {
		t.Errorf("expected content_filter finish reason, got %q", out.FinishReason)
	}
	if out.RawFinishReason != "completed" {
		t.Errorf("expected raw status to stay lossless, got %q", out.RawFinishReason)
	}
	if got := out.Message.GetTextContent(); got != "I can't help with that." {
		t.Errorf("expected the refusal text to be preserved, got %q", got)
	}
}

func TestConvertResponsesFinishReason(t *testing.T) {
	tests := []struct {
		name    string
		resp    *responses.Response
		want    core.FinishReason
		wantRaw string
	}{
		{
			name:    "truncated at max output tokens",
			resp:    &responses.Response{IncompleteDetails: responses.ResponseIncompleteDetails{Reason: "max_output_tokens"}},
			want:    core.FinishReasonLength,
			wantRaw: "max_output_tokens",
		},
		{
			name:    "content filter",
			resp:    &responses.Response{IncompleteDetails: responses.ResponseIncompleteDetails{Reason: "content_filter"}},
			want:    core.FinishReasonContentFilter,
			wantRaw: "content_filter",
		},
		{
			name:    "unrecognized incomplete reason",
			resp:    &responses.Response{IncompleteDetails: responses.ResponseIncompleteDetails{Reason: "something_new"}},
			want:    core.FinishReasonUnknown,
			wantRaw: "something_new",
		},
		{
			name:    "completed",
			resp:    &responses.Response{Status: "completed"},
			want:    core.FinishReasonStop,
			wantRaw: "completed",
		},
		{
			name:    "tool calls",
			resp:    &responses.Response{Status: "completed", Output: []responses.ResponseOutputItemUnion{{Type: "function_call"}}},
			want:    core.FinishReasonToolCalls,
			wantRaw: "completed",
		},
		// A generation that aborted is a failure, not a clean stop.
		{
			name:    "failed",
			resp:    &responses.Response{Status: "failed"},
			want:    core.FinishReasonError,
			wantRaw: "failed",
		},
		{
			name:    "cancelled",
			resp:    &responses.Response{Status: "cancelled"},
			want:    core.FinishReasonError,
			wantRaw: "cancelled",
		},
		{
			name:    "unrecognized status",
			resp:    &responses.Response{Status: "something_new"},
			want:    core.FinishReasonUnknown,
			wantRaw: "something_new",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, raw := convertResponsesFinishReason(tt.resp)
			if got != tt.want {
				t.Errorf("finish reason = %q, want %q", got, tt.want)
			}
			if raw != tt.wantRaw {
				t.Errorf("raw finish reason = %q, want %q", raw, tt.wantRaw)
			}
		})
	}
}

// TestResponsesBuildParamsJoinsSystemMessages pins that every system message
// reaches the model: keeping only the last silently drops instructions.
func TestResponsesBuildParamsJoinsSystemMessages(t *testing.T) {
	model := &ResponsesModel{model: "gpt-4.1"}

	params := model.buildParams(&core.ChatRequest{
		Model: "gpt-4.1",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "be terse"),
			core.NewTextMessage(core.RoleSystem, "answer in Spanish"),
			core.NewTextMessage(core.RoleUser, "hello"),
		},
	})

	if got := params.Instructions.Or(""); got != "be terse\n\nanswer in Spanish" {
		t.Fatalf("expected both system messages joined, got %q", got)
	}
}

// TestConvertAssistantItemsOrdering pins the replay order the API requires:
// a reasoning item must precede the function calls it produced.
func TestConvertAssistantItemsOrdering(t *testing.T) {
	items := convertAssistantItems(core.Message{
		Role: core.RoleAssistant,
		Content: []core.Part{
			{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{ID: "rs_1", Text: "first", Signature: "enc", ProviderName: "openai"}},
			{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{ID: "rs_1", Text: "second", ProviderName: "openai"}},
			{Type: core.ContentText, Text: "let me look"},
			{Type: core.ContentToolUse, ToolUse: &core.ToolUse{ID: "call_1", Name: "lookup", Input: map[string]interface{}{"city": "Lima"}}},
		},
	})

	if len(items) != 3 {
		t.Fatalf("expected reasoning, message, function_call, got %d items: %#v", len(items), items)
	}
	reasoning := items[0].OfReasoning
	if reasoning == nil {
		t.Fatalf("expected the reasoning item first, got %#v", items[0])
	}
	if reasoning.ID != "rs_1" {
		t.Errorf("expected the reasoning item id to be replayed, got %q", reasoning.ID)
	}
	// Both summaries share one id and must regroup into a single item.
	if len(reasoning.Summary) != 2 {
		t.Errorf("expected 2 summaries on one reasoning item, got %d", len(reasoning.Summary))
	}
	if reasoning.EncryptedContent.Or("") != "enc" {
		t.Errorf("expected the encrypted content to be carried, got %#v", reasoning.EncryptedContent)
	}
	if items[1].OfMessage == nil {
		t.Errorf("expected the assistant message second, got %#v", items[1])
	}
	if items[2].OfFunctionCall == nil {
		t.Errorf("expected the function call last, got %#v", items[2])
	}
}

// TestConvertAssistantItemsSkipsForeignThinking pins that another provider's
// reasoning block is not replayed to OpenAI, which would reject its signature.
func TestConvertAssistantItemsSkipsForeignThinking(t *testing.T) {
	items := convertAssistantItems(core.Message{
		Role: core.RoleAssistant,
		Content: []core.Part{
			{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{Text: "hmm", Signature: "sig", ProviderName: "anthropic"}},
			{Type: core.ContentText, Text: "answer"},
		},
	})

	if len(items) != 1 || items[0].OfMessage == nil {
		t.Fatalf("expected only the assistant message, got %#v", items)
	}
}

// TestNormalizeStrictSchema pins that strict structured output is normalized
// on both response-format paths. Schemas reflected from Go structs carry
// neither additionalProperties nor a complete required array, which the API
// rejects in strict mode.
func TestNormalizeStrictSchema(t *testing.T) {
	reflected := func() map[string]interface{} {
		return map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"title":  map[string]interface{}{"type": "string"},
				"rating": map[string]interface{}{"type": "number"},
			},
		}
	}
	strict, loose := true, false

	t.Run("chat completions response_format", func(t *testing.T) {
		got := convertResponseFormat(&core.ResponseFormat{
			Type:       "json_schema",
			JSONSchema: &core.JSONSchemaFormat{Name: "review", Schema: reflected(), Strict: &strict},
		})
		schema := got.OfJSONSchema.JSONSchema.Schema.(map[string]interface{})
		if schema["additionalProperties"] != false {
			t.Errorf("expected additionalProperties=false, got %#v", schema)
		}
		if required, _ := schema["required"].([]string); len(required) != 2 {
			t.Errorf("expected both properties required, got %#v", schema["required"])
		}
	})

	t.Run("responses text config", func(t *testing.T) {
		got := convertTextConfig(&core.ResponseFormat{
			Type:       "json_schema",
			JSONSchema: &core.JSONSchemaFormat{Name: "review", Schema: reflected(), Strict: &strict},
		})
		schema := got.Format.OfJSONSchema.Schema
		if schema["additionalProperties"] != false {
			t.Errorf("expected additionalProperties=false, got %#v", schema)
		}
		if required, _ := schema["required"].([]string); len(required) != 2 {
			t.Errorf("expected both properties required, got %#v", schema["required"])
		}
	})

	t.Run("non-strict schemas are left alone", func(t *testing.T) {
		for _, strictFlag := range []*bool{&loose, nil} {
			got := convertTextConfig(&core.ResponseFormat{
				Type:       "json_schema",
				JSONSchema: &core.JSONSchemaFormat{Name: "review", Schema: reflected(), Strict: strictFlag},
			})
			schema := got.Format.OfJSONSchema.Schema
			if _, ok := schema["additionalProperties"]; ok {
				t.Errorf("expected a non-strict schema to pass through untouched, got %#v", schema)
			}
		}
	})
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

	// A schema reflected from a Go struct with a nested struct field puts that
	// field's object in a definition pool and references it by $ref. Skipping
	// the pool leaves the referenced object without additionalProperties, and
	// the API rejects the whole schema — which is what happened on agentic's
	// own NewNativeOutput path until this recursion was added.
	t.Run("ensureAdditionalPropertiesFalse recurses into definition pools", func(t *testing.T) {
		for _, pool := range []string{"definitions", "$defs"} {
			t.Run(pool, func(t *testing.T) {
				schema := map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"address": map[string]interface{}{"$ref": "#/" + pool + "/Address"},
					},
					pool: map[string]interface{}{
						"Address": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"city": map[string]interface{}{"type": "string"},
							},
						},
						"literal": "value",
					},
				}

				normalized := ensureAdditionalPropertiesFalse(schema)
				defs, ok := normalized[pool].(map[string]interface{})
				if !ok {
					t.Fatalf("expected %q to survive normalization, got %#v", pool, normalized)
				}
				address, ok := defs["Address"].(map[string]interface{})
				if !ok {
					t.Fatalf("expected Address definition, got %#v", defs)
				}
				if address["additionalProperties"] != false {
					t.Errorf("definition object missing additionalProperties=false: %#v", address)
				}
				if req, ok := address["required"].([]string); !ok || len(req) != 1 || req[0] != "city" {
					t.Errorf("definition object required = %#v, want [city]", address["required"])
				}
				if defs["literal"] != "value" {
					t.Errorf("non-map definition values must be preserved, got %#v", defs["literal"])
				}
			})
		}
	})

	t.Run("ensureAdditionalPropertiesFalse recurses into composition keywords", func(t *testing.T) {
		for _, key := range []string{"anyOf", "oneOf", "allOf"} {
			t.Run(key, func(t *testing.T) {
				schema := map[string]interface{}{
					key: []interface{}{
						map[string]interface{}{
							"type":       "object",
							"properties": map[string]interface{}{"id": map[string]interface{}{"type": "string"}},
						},
						"literal",
					},
				}

				normalized := ensureAdditionalPropertiesFalse(schema)
				branches, ok := normalized[key].([]interface{})
				if !ok || len(branches) != 2 {
					t.Fatalf("expected %q to survive with 2 branches, got %#v", key, normalized[key])
				}
				branch, ok := branches[0].(map[string]interface{})
				if !ok {
					t.Fatalf("expected first branch to be an object, got %#v", branches[0])
				}
				if branch["additionalProperties"] != false {
					t.Errorf("%s branch missing additionalProperties=false: %#v", key, branch)
				}
				if branches[1] != "literal" {
					t.Errorf("non-map branches must be preserved, got %#v", branches[1])
				}
			})
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

// TestConvertAssistantItemsSignatureArrivesLate covers a reasoning group whose
// encrypted content is carried on a later summary than the first.
func TestConvertAssistantItemsSignatureArrivesLate(t *testing.T) {
	items := convertAssistantItems(core.Message{
		Role: core.RoleAssistant,
		Content: []core.Part{
			{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{ID: "rs_1", Text: "first"}},
			{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{ID: "rs_1", Text: "second", Signature: "enc"}},
		},
	})

	if len(items) != 1 || items[0].OfReasoning == nil {
		t.Fatalf("expected one reasoning item, got %#v", items)
	}
	if got := items[0].OfReasoning.EncryptedContent.Or(""); got != "enc" {
		t.Errorf("expected the late signature to be adopted, got %q", got)
	}
}

// TestResponsesBuildParamsProviderOptions pins that the Responses path honors
// the same "openai" ProviderOptions key as Chat Completions.
func TestResponsesBuildParamsProviderOptions(t *testing.T) {
	model := &ResponsesModel{model: "gpt-4.1"}
	parallel := false

	params := model.buildParams(&core.ChatRequest{
		Model:    "gpt-4.1",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		ProviderOptions: map[string]any{
			"openai": Options{ParallelToolCalls: &parallel, ServiceTier: "priority"},
		},
	})

	if params.ParallelToolCalls.Or(true) != false {
		t.Errorf("expected parallel_tool_calls=false, got %#v", params.ParallelToolCalls)
	}
	if params.ServiceTier != responses.ResponseNewParamsServiceTierPriority {
		t.Errorf("expected priority service tier, got %q", params.ServiceTier)
	}
}
