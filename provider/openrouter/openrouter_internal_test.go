package openrouter

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/respjson"
)

// marshalParams renders built params the way the SDK would send them, so tests
// can assert on the wire shape without a server.
func marshalParams(t *testing.T, params openai.ChatCompletionNewParams) map[string]any {
	t.Helper()
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	return out
}

func TestBuildParamsOmitsMaxTokensWhenUnset(t *testing.T) {
	m := &Model{model: "m"}
	body := marshalParams(t, m.buildParams(&core.ChatRequest{
		Model:    "m",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
	}))

	if _, present := body["max_tokens"]; present {
		t.Errorf("max_tokens must be omitted when unset, got %#v", body)
	}
	if _, present := body["max_completion_tokens"]; present {
		t.Errorf("max_completion_tokens must never be sent, got %#v", body)
	}
}

func TestBuildParamsPromptCacheKey(t *testing.T) {
	m := &Model{model: "m"}
	long := strings.Repeat("é", 80)
	body := marshalParams(t, m.buildParams(&core.ChatRequest{
		Model: "m", Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
		PromptCache: &core.PromptCacheConfig{Key: long, Retention: core.PromptCacheShort},
	}))
	key, ok := body["prompt_cache_key"].(string)
	if !ok || len([]rune(key)) != 64 {
		t.Fatalf("prompt_cache_key = %#v", body["prompt_cache_key"])
	}
	without := marshalParams(t, m.buildParams(&core.ChatRequest{
		Model: "m", Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
		PromptCache: &core.PromptCacheConfig{Retention: core.PromptCacheNone},
	}))
	if _, present := without["prompt_cache_key"]; present {
		t.Fatalf("disabled cache key present: %#v", without)
	}
}

func TestBuildParamsSamplingAndFormat(t *testing.T) {
	m := &Model{model: "m"}
	temp, topP := 0.5, 0.9
	strict := true
	choice := core.ToolChoiceRequired

	body := marshalParams(t, m.buildParams(&core.ChatRequest{
		Model:       "m",
		Messages:    []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
		Temperature: &temp,
		TopP:        &topP,
		Tools: []core.Tool{{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "lookup",
				Description: "look something up",
				Parameters:  map[string]interface{}{"type": "object"},
			},
		}},
		ToolChoice: &choice,
		ResponseFormat: &core.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &core.JSONSchemaFormat{
				Name:        "out",
				Description: "the output",
				Schema:      map[string]interface{}{"type": "object"},
				Strict:      &strict,
			},
		},
	}))

	if body["temperature"] != 0.5 || body["top_p"] != 0.9 {
		t.Errorf("sampling params not forwarded: %#v", body)
	}
	if body["tool_choice"] != "required" {
		t.Errorf("tool_choice = %#v, want required", body["tool_choice"])
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %#v, want one entry", body["tools"])
	}
	format, _ := body["response_format"].(map[string]any)
	if format["type"] != "json_schema" {
		t.Errorf("response_format = %#v", format)
	}
}

func TestConvertToolChoice(t *testing.T) {
	tests := []struct {
		name   string
		choice core.ToolChoice
		want   string
	}{
		{name: "none", choice: core.ToolChoiceNone, want: "none"},
		{name: "required", choice: core.ToolChoiceRequired, want: "required"},
		{name: "auto", choice: core.ToolChoiceAuto, want: "auto"},
		{name: "unrecognized falls back to auto", choice: core.ToolChoice("bogus"), want: "auto"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertToolChoice(tt.choice)
			if got.OfAuto.Value != tt.want {
				t.Errorf("convertToolChoice(%q) = %q, want %q", tt.choice, got.OfAuto.Value, tt.want)
			}
		})
	}
}

func TestConvertToolsDefaultsNilParameters(t *testing.T) {
	tools := convertTools([]core.Tool{{
		Type:     core.ToolTypeFunction,
		Function: core.Function{Name: "noargs"},
	}})

	if len(tools) != 1 {
		t.Fatalf("expected one tool, got %d", len(tools))
	}
	if tools[0].Function.Parameters == nil {
		t.Error("nil parameters must become an empty schema, not a null")
	}
}

func TestConvertResponseFormat(t *testing.T) {
	tests := []struct {
		name string
		in   *core.ResponseFormat
		want string
	}{
		{name: "json object", in: &core.ResponseFormat{Type: "json_object"}, want: "json_object"},
		{name: "text", in: &core.ResponseFormat{Type: "text"}, want: "text"},
		{name: "unrecognized falls back to text", in: &core.ResponseFormat{Type: "bogus"}, want: "text"},
		{
			name: "json schema without a schema falls back to json object",
			in:   &core.ResponseFormat{Type: "json_schema"},
			want: "json_object",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw, err := json.Marshal(convertResponseFormat(tt.in))
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got["type"] != tt.want {
				t.Errorf("type = %#v, want %q", got["type"], tt.want)
			}
		})
	}
}

func TestConvertMessageRoles(t *testing.T) {
	tests := []struct {
		name    string
		msg     core.Message
		wantKey string
	}{
		{
			name:    "system",
			msg:     core.NewTextMessage(core.RoleSystem, "be brief"),
			wantKey: "system",
		},
		{
			name:    "single-part user takes the string fast path",
			msg:     core.NewTextMessage(core.RoleUser, "hello"),
			wantKey: "user",
		},
		{
			name:    "assistant text",
			msg:     core.NewTextMessage(core.RoleAssistant, "hi"),
			wantKey: "assistant",
		},
		{
			name:    "assistant with tool calls",
			msg:     core.NewToolUseMessage(core.ToolUse{ID: "call_1", Name: "lookup", Input: map[string]interface{}{"q": "x"}}),
			wantKey: "assistant",
		},
		{
			name:    "tool result",
			msg:     core.NewToolResultMessageFor("call_1", "lookup", "42", false),
			wantKey: "tool",
		},
		{
			name:    "unknown role is sent as user",
			msg:     core.NewTextMessage(core.MessageRole("wizard"), "abracadabra"),
			wantKey: "user",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw, err := json.Marshal(convertMessage(tt.msg))
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got["role"] != tt.wantKey {
				t.Errorf("role = %#v, want %q", got["role"], tt.wantKey)
			}
		})
	}
}

// A tool-role message with no tool-result part still has to carry a tool_call_id,
// because the API rejects a tool message without one.
func TestConvertMessageToolWithoutResultPart(t *testing.T) {
	raw, err := json.Marshal(convertMessage(core.NewTextMessage(core.RoleTool, "orphaned")))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["tool_call_id"] != "unknown" {
		t.Errorf("tool_call_id = %#v, want the placeholder", got["tool_call_id"])
	}
}

func TestConvertMessageMultipartUser(t *testing.T) {
	msg := core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentText, Text: "describe"},
			{Type: core.ContentImageURL, ImageURL: &core.ImageURL{URL: "https://img.example/a.png", Detail: "high"}},
			{Type: core.ContentImageData, ImageData: &core.ImageData{
				Data:           "AAAA",
				MediaType:      "image/png",
				VendorMetadata: map[string]interface{}{"detail": "low"},
			}},
			{Type: core.ContentAudioURL, AudioURL: &core.AudioURL{URL: "ZGF0YQ==", Format: "wav"}},
			{Type: core.ContentUploadedFile, UploadedFile: &core.UploadedFile{FileID: "file-1"}},
			{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{TTL: "5m"}},
		},
	}

	raw, err := json.Marshal(convertMessage(msg))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(got.Content) != 6 {
		t.Fatalf("expected 6 content parts, got %d: %#v", len(got.Content), got.Content)
	}
	if got.Content[0]["type"] != "text" {
		t.Errorf("part 0 = %#v", got.Content[0])
	}
	image, _ := got.Content[1]["image_url"].(map[string]any)
	if image["detail"] != "high" {
		t.Errorf("image detail not forwarded: %#v", got.Content[1])
	}
	inline, _ := got.Content[2]["image_url"].(map[string]any)
	if inline["url"] != "data:image/png;base64,AAAA" {
		t.Errorf("inline image not encoded as a data URI: %#v", got.Content[2])
	}
	if inline["detail"] != "low" {
		t.Errorf("vendor detail not forwarded: %#v", got.Content[2])
	}
	if got.Content[3]["type"] != "input_audio" {
		t.Errorf("part 3 = %#v", got.Content[3])
	}
	if got.Content[4]["type"] != "file" {
		t.Errorf("part 4 = %#v", got.Content[4])
	}
	if got.Content[5]["type"] != "text" {
		t.Errorf("cache point must degrade to an empty text part: %#v", got.Content[5])
	}
}

// A typed part whose payload pointer is nil must not panic; it degrades to text.
func TestConvertContentPartNilPayloads(t *testing.T) {
	tests := []struct {
		name string
		part core.Part
	}{
		{name: "image url", part: core.Part{Type: core.ContentImageURL, Text: "fallback"}},
		{name: "image data", part: core.Part{Type: core.ContentImageData, Text: "fallback"}},
		{name: "audio url", part: core.Part{Type: core.ContentAudioURL, Text: "fallback"}},
		{name: "uploaded file", part: core.Part{Type: core.ContentUploadedFile, Text: "fallback"}},
		{name: "unhandled type", part: core.Part{Type: core.ContentVideoURL, Text: "fallback"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertContentPart(tt.part)
			if got.OfText == nil || got.OfText.Text != "fallback" {
				t.Errorf("expected a text fallback, got %#v", got)
			}
		})
	}
}

func TestExtractUsage(t *testing.T) {
	var u openai.CompletionUsage
	if err := json.Unmarshal([]byte(`{
		"prompt_tokens":10,"completion_tokens":4,"total_tokens":14,
		"completion_tokens_details":{"reasoning_tokens":3},
		"prompt_tokens_details":{"cached_tokens":6}
	}`), &u); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}

	got := extractUsage(u)
	if got.PromptTokens != 10 || got.CompletionTokens != 4 || got.TotalTokens != 14 {
		t.Errorf("token counts = %#v", got)
	}
	if got.ReasoningTokens != 3 {
		t.Errorf("ReasoningTokens = %d, want 3", got.ReasoningTokens)
	}
	if got.CacheReadTokens != 6 {
		t.Errorf("CacheReadTokens = %d, want 6", got.CacheReadTokens)
	}
}

func TestExtractUsageWithoutDetails(t *testing.T) {
	var u openai.CompletionUsage
	if err := json.Unmarshal([]byte(`{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}`), &u); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}

	got := extractUsage(u)
	if got.ReasoningTokens != 0 || got.CacheReadTokens != 0 {
		t.Errorf("absent details must stay zero, got %#v", got)
	}
}

// extraFieldsOf decodes a message body and returns its undecoded fields, which
// is where reasoning text lands because the SDK has no typed field for it.
func extraFieldsOf(t *testing.T, body string) map[string]respjson.Field {
	t.Helper()
	var msg openai.ChatCompletionMessage
	if err := json.Unmarshal([]byte(body), &msg); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	return msg.JSON.ExtraFields
}

func TestExtractReasoning(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		wantID   string
		wantText string
	}{
		{
			name:     "reasoning",
			body:     `{"role":"assistant","content":"a","reasoning":"because"}`,
			wantID:   "reasoning",
			wantText: "because",
		},
		{
			name:     "reasoning_content",
			body:     `{"role":"assistant","content":"a","reasoning_content":"because"}`,
			wantID:   "reasoning_content",
			wantText: "because",
		},
		{
			name:     "reasoning wins over reasoning_content",
			body:     `{"role":"assistant","content":"a","reasoning":"first","reasoning_content":"second"}`,
			wantID:   "reasoning",
			wantText: "first",
		},
		{
			name: "absent",
			body: `{"role":"assistant","content":"a"}`,
		},
		{
			name: "null is not reasoning",
			body: `{"role":"assistant","content":"a","reasoning":null}`,
		},
		{
			name: "empty string is not reasoning",
			body: `{"role":"assistant","content":"a","reasoning":""}`,
		},
		{
			name: "non-string value is skipped rather than reported",
			body: `{"role":"assistant","content":"a","reasoning":{"text":"because"}}`,
		},
		{
			name:     "falls through a non-string to the next field",
			body:     `{"role":"assistant","content":"a","reasoning":123,"reasoning_content":"because"}`,
			wantID:   "reasoning_content",
			wantText: "because",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotID, gotText := extractReasoning(extraFieldsOf(t, tt.body))
			if gotID != tt.wantID || gotText != tt.wantText {
				t.Errorf("extractReasoning = (%q, %q), want (%q, %q)", gotID, gotText, tt.wantID, tt.wantText)
			}
		})
	}
}

func TestConvertResponseMessageWithToolCalls(t *testing.T) {
	var msg openai.ChatCompletionMessage
	if err := json.Unmarshal([]byte(`{
		"role":"assistant","content":"",
		"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Lima\"}"}}]
	}`), &msg); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}

	got := convertResponseMessage(msg)
	if got.Role != core.RoleAssistant {
		t.Errorf("role = %q", got.Role)
	}
	uses := got.GetToolUses()
	if len(uses) != 1 {
		t.Fatalf("expected one tool use, got %#v", got.Content)
	}
	if uses[0].ID != "call_1" || uses[0].Name != "lookup" {
		t.Errorf("unexpected tool use %#v", uses[0])
	}
	if uses[0].Input["city"] != "Lima" {
		t.Errorf("arguments not decoded: %#v", uses[0].Input)
	}
	// Empty content must not produce a stray empty text part.
	if got.GetTextContent() != "" {
		t.Errorf("unexpected text content %q", got.GetTextContent())
	}
}

func TestOptionsFrom(t *testing.T) {
	routing := &ProviderRouting{Order: []string{"anthropic"}}

	tests := []struct {
		name string
		req  *core.ChatRequest
		want *ProviderRouting
	}{
		{name: "nil request", req: nil},
		{name: "nil map", req: &core.ChatRequest{}},
		{name: "absent key", req: &core.ChatRequest{ProviderOptions: map[string]any{"anthropic": 1}}},
		{name: "wrong type", req: &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: "order"}}},
		{name: "typed nil pointer", req: &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: (*Options)(nil)}}},
		{
			name: "value",
			req:  &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: Options{Provider: routing}}},
			want: routing,
		},
		{
			name: "pointer",
			req:  &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: &Options{Provider: routing}}},
			want: routing,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := optionsFrom(tt.req).Provider; got != tt.want {
				t.Errorf("Provider = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestRequestOptionsEmptyWhenNothingToInject(t *testing.T) {
	if opts := requestOptions(&core.ChatRequest{}); len(opts) != 0 {
		t.Errorf("expected no request options, got %d", len(opts))
	}
	// A present Options value with no routing must still inject nothing.
	req := &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: Options{}}}
	if opts := requestOptions(req); len(opts) != 0 {
		t.Errorf("expected no request options, got %d", len(opts))
	}
}

func TestBuildReasoning(t *testing.T) {
	tests := []struct {
		name string
		cfg  *core.ThinkingConfig
		want map[string]interface{}
	}{
		{name: "nil"},
		{name: "disabled", cfg: &core.ThinkingConfig{}},
		{
			name: "enabled without a budget uses effort",
			cfg:  &core.ThinkingConfig{Enabled: true},
			want: map[string]interface{}{"enabled": true, "effort": "medium"},
		},
		{
			name: "budget wins over effort",
			cfg:  &core.ThinkingConfig{Enabled: true, BudgetTokens: 1024},
			want: map[string]interface{}{"enabled": true, "max_tokens": 1024},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildReasoning(tt.cfg)
			if tt.want == nil {
				if got != nil {
					t.Fatalf("expected nil, got %#v", got)
				}
				return
			}
			if len(got) != len(tt.want) {
				t.Fatalf("buildReasoning = %#v, want %#v", got, tt.want)
			}
			for k, v := range tt.want {
				if got[k] != v {
					t.Errorf("key %q = %#v, want %#v", k, got[k], v)
				}
			}
		})
	}
}

func TestDecodeInBandError(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		wantNil  bool
		wantCode int
		wantMsg  string
	}{
		{name: "empty body", raw: "", wantNil: true},
		{name: "not json", raw: "<html>502</html>", wantNil: true},
		{name: "no error key", raw: `{"id":"x","choices":[]}`, wantNil: true},
		{name: "null error", raw: `{"error":null}`, wantNil: true},
		{name: "empty error object", raw: `{"error":{}}`, wantNil: true},
		{
			name:     "numeric code",
			raw:      `{"error":{"code":429,"message":"slow down"}}`,
			wantCode: 429,
			wantMsg:  "slow down",
		},
		{
			name:     "quoted code",
			raw:      `{"error":{"code":"429","message":"slow down"}}`,
			wantCode: 429,
			wantMsg:  "slow down",
		},
		{
			name:    "non-numeric code degrades to zero",
			raw:     `{"error":{"code":"rate_limited","message":"slow down"}}`,
			wantMsg: "slow down",
		},
		{
			name:    "message only",
			raw:     `{"error":{"message":"something broke"}}`,
			wantMsg: "something broke",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := decodeInBandError(tt.raw)
			if tt.wantNil {
				if got != nil {
					t.Fatalf("expected nil, got %#v", got)
				}
				return
			}
			if got == nil {
				t.Fatal("expected an APIError, got nil")
			}
			if got.Code != tt.wantCode {
				t.Errorf("Code = %d, want %d", got.Code, tt.wantCode)
			}
			if got.Message != tt.wantMsg {
				t.Errorf("Message = %q, want %q", got.Message, tt.wantMsg)
			}
		})
	}
}

func TestStreamError(t *testing.T) {
	sdkErr := func(body string) error {
		return fmt.Errorf("%s%s", sdkStreamErrorPrefix, body)
	}

	tests := []struct {
		name     string
		err      error
		wantAPI  bool
		wantCode int
	}{
		{name: "transport failure is wrapped", err: errors.New("connection reset")},
		{name: "unparsable payload is wrapped", err: sdkErr("not json")},
		{name: "empty error object is wrapped", err: sdkErr("{}")},
		{
			name:     "error object becomes typed",
			err:      sdkErr(`{"code":502,"message":"upstream went away"}`),
			wantAPI:  true,
			wantCode: 502,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := streamError(tt.err)

			var apiErr *APIError
			isAPI := errors.As(got, &apiErr)
			if isAPI != tt.wantAPI {
				t.Fatalf("errors.As(*APIError) = %v, want %v (err %v)", isAPI, tt.wantAPI, got)
			}
			if tt.wantAPI {
				if apiErr.Code != tt.wantCode {
					t.Errorf("Code = %d, want %d", apiErr.Code, tt.wantCode)
				}
				return
			}
			if !errors.Is(got, tt.err) {
				t.Errorf("wrapped error must keep the original: %v", got)
			}
		})
	}
}

func TestRawFinishReason(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "null choices defaults to error",
			body: `{"id":"x","choices":null}`,
			want: "error",
		},
		{
			name: "empty finish reason defaults to error",
			body: `{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":""}]}`,
			want: "error",
		},
		{
			name: "reported reason is preserved",
			body: `{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"content_filter"}]}`,
			want: "content_filter",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var resp openai.ChatCompletion
			if err := json.Unmarshal([]byte(tt.body), &resp); err != nil {
				t.Fatalf("unmarshal response: %v", err)
			}
			if got := rawFinishReason(&resp); got != tt.want {
				t.Errorf("rawFinishReason = %q, want %q", got, tt.want)
			}
		})
	}
}
